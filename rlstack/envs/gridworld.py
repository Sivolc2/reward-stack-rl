"""Tier 3 - ResourceWorld: a lightweight, top-down, multi-agent survival world.

A deliberately small, fast, procedurally-generated grid (think "Crafter, but
tabular") in which several agents wander, eat regrowing food, avoid hazards, and
bump into each other. It is the open-ended setting where a *stack* of drives
should pay off: there is no single win condition, so a one-note "maximise food"
policy and a multi-drive policy produce visibly different behaviour.

Design choices that make the hypothesis testable:

* **The environment never hands out a scalar reward.** It only emits *events*
  per agent ("you ate X", "a hazard is adjacent", "N neighbours nearby", "you
  are at cell P"). Each agent's steering subsystem turns those into reward, so
  swapping the drive stack swaps the whole reward structure with zero env
  changes.
* **Internal state is part of the observation.** The agent sees a coarse bucket
  of its own hunger level, so the *optimal* action genuinely differs by internal
  state — the precondition for goal-switching to emerge rather than be scripted.
* **Food regrows and hazards persist**, giving the "things grow and die"
  dynamics the team wanted, while staying cheap enough to fast-forward.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..agent import BrainAgent

# actions
STAY, NORTH, EAST, SOUTH, WEST = range(5)
_DELTA = {
    STAY: (0, 0),
    NORTH: (0, -1),
    EAST: (1, 0),
    SOUTH: (0, 1),
    WEST: (-1, 0),
}

# tiles
EMPTY, FOOD, HAZARD = 0, 1, 2


def _direction_bucket(dx: int, dy: int) -> int:
    """Map a relative offset to one of 9 buckets (8 compass dirs + 'here/none')."""
    if dx == 0 and dy == 0:
        return 0
    ang = np.arctan2(dy, dx)  # screen coords: +y is south
    octant = int((np.round(ang / (np.pi / 4))) % 8)
    return 1 + octant


class ResourceWorld:
    n_actions = 5

    def __init__(
        self,
        width: int = 12,
        height: int = 12,
        *,
        n_food: int = 14,
        n_hazards: int = 6,
        food_value: float = 1.0,
        regrow_delay: int = 25,
        hazard_damage: float = 0.34,
        vision_radius: int = 4,
        social_radius: int = 2,
        seed: int = 0,
    ) -> None:
        self.W = width
        self.H = height
        self.n_food = n_food
        self.n_hazards = n_hazards
        self.food_value = food_value
        self.regrow_delay = regrow_delay
        self.hazard_damage = hazard_damage
        self.vision_radius = vision_radius
        self.social_radius = social_radius
        self.rng = np.random.default_rng(seed)
        self.agents: list["BrainAgent"] = []
        self.positions: list[tuple[int, int]] = []
        self._grid = np.zeros((self.H, self.W), dtype=np.int8)
        self._regrow_timer = np.zeros((self.H, self.W), dtype=np.int32)
        self.deaths = np.zeros(0, dtype=np.int64)

    # --- world generation ----------------------------------------------------
    def _random_empty_cell(self) -> tuple[int, int]:
        while True:
            x = int(self.rng.integers(self.W))
            y = int(self.rng.integers(self.H))
            if self._grid[y, x] == EMPTY:
                return x, y

    def reset(self, agents: Sequence["BrainAgent"]) -> None:
        self.agents = list(agents)
        self._grid[:] = EMPTY
        self._regrow_timer[:] = 0
        self.deaths = np.zeros(len(self.agents), dtype=np.int64)
        # Scatter hazards first (clustered a little to form "dangerous regions").
        for _ in range(self.n_hazards):
            x, y = self._random_empty_cell()
            self._grid[y, x] = HAZARD
        # Then food.
        for _ in range(self.n_food):
            x, y = self._random_empty_cell()
            self._grid[y, x] = FOOD
        # Place agents.
        self.positions = []
        for ag in self.agents:
            ag.reset_episode()
            self.positions.append(self._random_empty_cell())

    # --- observation ---------------------------------------------------------
    def _nearest(self, pos: tuple[int, int], tile: int) -> tuple[int, int] | None:
        x0, y0 = pos
        best = None
        best_d = None
        r = self.vision_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = x0 + dx, y0 + dy
                if 0 <= x < self.W and 0 <= y < self.H and self._grid[y, x] == tile:
                    d = abs(dx) + abs(dy)
                    if best_d is None or d < best_d:
                        best_d, best = d, (dx, dy)
        return best

    def _nearest_agent(self, idx: int) -> tuple[int, int] | None:
        x0, y0 = self.positions[idx]
        best = None
        best_d = None
        for j, (x, y) in enumerate(self.positions):
            if j == idx:
                continue
            dx, dy = x - x0, y - y0
            d = abs(dx) + abs(dy)
            if d <= self.vision_radius and (best_d is None or d < best_d):
                best_d, best = d, (dx, dy)
        return best

    def _observe(self, idx: int) -> tuple:
        pos = self.positions[idx]
        food = self._nearest(pos, FOOD)
        hazard = self._nearest(pos, HAZARD)
        other = self._nearest_agent(idx)
        food_dir = _direction_bucket(*food) if food else 0
        hazard_dir = _direction_bucket(*hazard) if hazard else 0
        agent_dir = _direction_bucket(*other) if other else 0
        # Internal hunger level -> 4 buckets (the key piece of internal state).
        levels = self.agents[idx].steering.levels()
        hunger = levels.get("hunger", 1.0)
        hunger_bucket = min(3, int(hunger * 4))
        return ("grid", food_dir, hazard_dir, agent_dir, hunger_bucket)

    def observations(self) -> list[tuple]:
        return [self._observe(i) for i in range(len(self.agents))]

    # --- dynamics ------------------------------------------------------------
    def _count_neighbors(self, idx: int) -> int:
        x0, y0 = self.positions[idx]
        n = 0
        for j, (x, y) in enumerate(self.positions):
            if j != idx and abs(x - x0) + abs(y - y0) <= self.social_radius:
                n += 1
        return n

    def step(self, actions: Sequence[int]) -> list[dict]:
        """Apply one joint action; return a per-agent event context dict.

        The returned contexts are what each agent's steering subsystem consumes
        to compute reward (the env itself assigns no scalar reward).
        """
        contexts: list[dict] = []
        # Move.
        for i, a in enumerate(actions):
            dx, dy = _DELTA[int(a)]
            x, y = self.positions[i]
            nx, ny = min(max(x + dx, 0), self.W - 1), min(max(y + dy, 0), self.H - 1)
            self.positions[i] = (nx, ny)

        # Resolve tile effects + build contexts.
        for i in range(len(self.agents)):
            x, y = self.positions[i]
            ate = 0.0
            got_hurt = False
            if self._grid[y, x] == FOOD:
                ate = self.food_value
                self._grid[y, x] = EMPTY
                self._regrow_timer[y, x] = self.regrow_delay
            elif self._grid[y, x] == HAZARD:
                got_hurt = True

            # Hazard proximity in [0,1]: 1 if a hazard is adjacent/under us.
            hz = self._nearest((x, y), HAZARD)
            if hz is None:
                proximity = 0.0
            else:
                d = abs(hz[0]) + abs(hz[1])
                proximity = max(0.0, 1.0 - d / (self.vision_radius + 1))

            contexts.append(
                {
                    "ate": ate,
                    "got_hurt": got_hurt,
                    "hazard_proximity": proximity,
                    "neighbors": self._count_neighbors(i),
                    "pos": (x, y),
                }
            )

        # Regrow food.
        regrowing = self._regrow_timer > 0
        self._regrow_timer[regrowing] -= 1
        ready = (self._regrow_timer == 0) & regrowing
        ys, xs = np.where(ready)
        for y, x in zip(ys.tolist(), xs.tolist()):
            if self._grid[y, x] == EMPTY:
                self._grid[y, x] = FOOD

        return contexts

    def respawn_dead(self) -> None:
        """Respawn any agent whose homeostatic drive hit zero (counts a death)."""
        for i, ag in enumerate(self.agents):
            if ag.dead:
                self.deaths[i] += 1
                self.positions[i] = self._random_empty_cell()
                # Reset just the drives (keep what the learner has learned).
                ag.steering.reset()

    # --- presentation --------------------------------------------------------
    def render(self) -> str:
        glyph = {EMPTY: "·", FOOD: "♣", HAZARD: "▲"}
        chars = [[glyph[int(self._grid[y, x])] for x in range(self.W)] for y in range(self.H)]
        for i, (x, y) in enumerate(self.positions):
            chars[y][x] = str(i % 10)
        return "\n".join("".join(row) for row in chars)
