"""Tier 2 - TugGame: cooperative joint-action control (the "bungee" game).

Three agents each hold a rope tied to a shared elastic band. No single agent can
move the payload; the *actuator is the joint effect of everyone's pull*. The
band's centre is the mean of the three anchor points, and the "bottle" is
gripped only when the three anchors straddle it (a real triangulation
constraint) and the band centre is on top of it. Once gripped, the bottle
follows the band centre. The goal is to carry the bottle from A to B.

This is the cleanest cooperative-control test in the suite: the reward is shared
and only flows when the agents *coordinate*, so it directly probes whether a
cooperative/social drive stacked on a shared task reward produces coordination
that a purely self-interested reward would not.

State and action spaces are discretised so the same tabular learner used
elsewhere applies. Each agent moves its own anchor (stay/N/E/S/W). Independent
Q-learners can reach partial coordination here; it is meant as an environment +
baseline, not a solved benchmark.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..agent import BrainAgent

STAY, NORTH, EAST, SOUTH, WEST = range(5)
_DELTA = {STAY: (0, 0), NORTH: (0, -1), EAST: (1, 0), SOUTH: (0, 1), WEST: (-1, 0)}


def _sign(v: float) -> int:
    return 0 if v == 0 else (1 if v > 0 else -1)


class TugGame:
    n_actions = 5

    def __init__(
        self,
        size: int = 9,
        *,
        n_pullers: int = 3,
        grip_radius: float = 1.6,
        spread_min: float = 1.2,
        max_steps: int = 60,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.n_pullers = n_pullers
        self.grip_radius = grip_radius
        self.spread_min = spread_min
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.agents: list["BrainAgent"] = []
        self.anchors = np.zeros((n_pullers, 2), dtype=np.float64)
        self.bottle = np.zeros(2, dtype=np.float64)
        self.goal = np.zeros(2, dtype=np.float64)
        self.t = 0

    # --- geometry ------------------------------------------------------------
    @property
    def band_center(self) -> np.ndarray:
        return self.anchors.mean(axis=0)

    @property
    def spread(self) -> float:
        """Mean distance of anchors from the band centre (triangulation size)."""
        return float(np.linalg.norm(self.anchors - self.band_center, axis=1).mean())

    def _gripped(self) -> bool:
        on_bottle = np.linalg.norm(self.band_center - self.bottle) <= self.grip_radius
        return bool(on_bottle and self.spread >= self.spread_min)

    # --- lifecycle -----------------------------------------------------------
    def reset(self, agents: Sequence["BrainAgent"]) -> None:
        assert len(agents) == self.n_pullers
        self.agents = list(agents)
        for ag in self.agents:
            ag.reset_episode()
        c = self.size / 2.0
        # Start anchors in a loose triangle around the bottle at A.
        self.bottle = np.array([c * 0.5, c], dtype=np.float64)
        self.goal = np.array([c * 1.5, c], dtype=np.float64)
        offsets = np.array([[-1.5, -1.5], [1.5, -1.5], [0.0, 1.8]])
        self.anchors = self.bottle + offsets[: self.n_pullers]
        self.t = 0

    def _dist_to_goal(self) -> float:
        return float(np.linalg.norm(self.bottle - self.goal))

    def _observe(self, idx: int) -> tuple:
        bc = self.band_center
        # Coarse signs: where the goal is relative to the bottle, where the band
        # centre is relative to the bottle, and where this anchor is relative to
        # the band centre. Enough to coordinate, small enough to be tabular.
        gb = self.goal - self.bottle
        cb = bc - self.bottle
        ac = self.anchors[idx] - bc
        return (
            "tug",
            _sign(gb[0]), _sign(gb[1]),
            _sign(cb[0]), _sign(cb[1]),
            _sign(ac[0]), _sign(ac[1]),
            int(self._gripped()),
        )

    def observations(self) -> list[tuple]:
        return [self._observe(i) for i in range(self.n_pullers)]

    def step(self, actions: Sequence[int]) -> tuple[list[dict], bool]:
        self.t += 1
        for i, a in enumerate(actions):
            dx, dy = _DELTA[int(a)]
            self.anchors[i, 0] = np.clip(self.anchors[i, 0] + dx, 0, self.size - 1)
            self.anchors[i, 1] = np.clip(self.anchors[i, 1] + dy, 0, self.size - 1)

        prev = self._dist_to_goal()
        gripped = self._gripped()
        if gripped:
            self.bottle = self.band_center.copy()
        progress = prev - self._dist_to_goal()  # >0 means we moved it closer

        delivered = self._dist_to_goal() <= self.grip_radius
        done = delivered or self.t >= self.max_steps

        # Shared cooperative reward: everyone gets the same task signal, so there
        # is no individual incentive that can be satisfied without the group.
        shared = 2.0 * progress + (5.0 if delivered else 0.0)
        contexts = [
            {
                "extrinsic": shared,
                "cooperation": 1.0 if gripped else 0.0,
                "neighbors": self.n_pullers - 1,
                "pos": tuple(np.round(self.anchors[i]).astype(int).tolist()),
            }
            for i in range(self.n_pullers)
        ]
        return contexts, done

    def render(self) -> str:
        grid = [["·" for _ in range(self.size)] for _ in range(self.size)]

        def place(p, ch):
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < self.size and 0 <= y < self.size:
                grid[y][x] = ch

        place(self.goal, "B")
        for i in range(self.n_pullers):
            place(self.anchors[i], str(i))
        place(self.band_center, "+")
        place(self.bottle, "O")
        return "\n".join("".join(r) for r in grid)
