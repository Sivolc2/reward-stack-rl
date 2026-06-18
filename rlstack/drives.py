"""Drives: the individual reward functions that make up the steering subsystem.

The central thesis of this project (after Steven Byrnes / Astera Institute) is
that a brain is not driven by *one* reward function but by a *stack* of many
innate, genetically-hardwired "drives". Each drive:

  * owns optional internal state (e.g. a homeostatic variable like energy),
  * emits an instantaneous scalar ``reward`` in response to world events, and
  * reports a dynamic ``urgency`` — how loudly it is currently demanding the
    organism's attention.

The :class:`~rlstack.steering.SteeringSubsystem` arbitrates between drives using
those urgencies, so the *same* external event produces *different* net reward
depending on the agent's internal state (hungry vs. sated, safe vs. threatened).
That state-dependence is what turns a fixed environment into a source of
dynamic, goal-switching behaviour — the phenomenon we are trying to reproduce.

All drives read from a plain ``ctx`` dict supplied by the environment each step,
falling back to sensible defaults, so the same drive works across environments.

Homeostatic drives implement the drive-reduction reward of Keramati & Gutkin
(2014): reward equals the *reduction* in distance-to-setpoint, which provably
induces homeostasis without hand-tuned reward shaping.
"""
from __future__ import annotations

from typing import Any, Mapping


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


class Drive:
    """Base class for a single innate reward function.

    Subclasses override :meth:`step`, :meth:`reward` and :meth:`urgency`.
    ``ctx`` is a mapping of per-step facts produced by the environment, e.g.
    ``{"ate": True, "near_hazard": False, "neighbors": 2, "pos": (3, 4)}``.
    """

    name: str = "drive"

    def reset(self) -> None:  # pragma: no cover - trivial
        """Reset internal state at the start of an episode."""

    def step(self, ctx: Mapping[str, Any]) -> None:  # pragma: no cover - trivial
        """Advance internal state given this step's context (before reward)."""

    def reward(self, ctx: Mapping[str, Any]) -> float:
        """Instantaneous reward contribution for this step."""
        return 0.0

    def urgency(self, ctx: Mapping[str, Any]) -> float:
        """Dynamic weight in [0, inf): how much this drive wants attention now."""
        return 1.0

    # Drives may expose a normalised internal level in [0, 1] for observations.
    def level(self) -> float:
        return 1.0


class HomeostaticDrive(Drive):
    """A drive that maintains an internal variable near a setpoint.

    Models hunger/energy, temperature, fatigue, etc. The internal ``_level``
    decays every step and is replenished when ``replenish_key`` is truthy in the
    context (e.g. the agent ate). Reward follows drive-reduction theory: the
    decrease in squared distance-to-setpoint. Urgency is the current deficit,
    so a starving agent weights food far more heavily than a sated one.
    """

    def __init__(
        self,
        name: str,
        replenish_key: str,
        *,
        setpoint: float = 1.0,
        decay: float = 0.03,
        replenish: float = 0.6,
        urgency_gain: float = 6.0,
        death_level: float = 0.0,
        death_penalty: float = 1.0,
        damage_key: str = "got_hurt",
        damage: float = 0.25,
    ) -> None:
        self.name = name
        self.replenish_key = replenish_key
        self.setpoint = setpoint
        self.decay = decay
        self.replenish = replenish
        self.urgency_gain = urgency_gain
        self.death_level = death_level
        self.death_penalty = death_penalty
        self.damage_key = damage_key
        self.damage = damage
        self._level = setpoint
        self._prev_deficit = 0.0
        self.dead = False

    def reset(self) -> None:
        self._level = self.setpoint
        self._prev_deficit = abs(self.setpoint - self._level)
        self.dead = False

    def _deficit(self) -> float:
        return abs(self.setpoint - self._level)

    def step(self, ctx: Mapping[str, Any]) -> None:
        self._prev_deficit = self._deficit()
        self._level = _clamp(self._level - self.decay)
        if ctx.get(self.replenish_key):
            amount = ctx.get(self.replenish_key)
            gain = self.replenish * (amount if isinstance(amount, (int, float)) else 1.0)
            self._level = _clamp(self._level + gain)
        if ctx.get(self.damage_key):
            # Hazards physically drain energy for *every* agent, whether or not
            # it has a dedicated fear drive to see them coming.
            self._level = _clamp(self._level - self.damage)
        if self._level <= self.death_level:
            self.dead = True

    def reward(self, ctx: Mapping[str, Any]) -> float:
        # Drive-reduction: positive when we move *toward* the setpoint.
        r = self._prev_deficit ** 2 - self._deficit() ** 2
        if self.dead:
            r -= self.death_penalty
        return r

    def urgency(self, ctx: Mapping[str, Any]) -> float:
        # The hungrier we are, the louder this drive shouts.
        return 0.05 + self.urgency_gain * self._deficit()

    def level(self) -> float:
        return self._level


class SafetyDrive(Drive):
    """Fear: punishes proximity to / contact with hazards.

    Mirrors Byrnes' example of amygdala fear circuits hardwired to a visual
    "skittering predator" detector. Urgency spikes when a hazard is near, so a
    threatened agent will abandon foraging to flee.
    """

    def __init__(
        self,
        name: str = "safety",
        *,
        proximity_key: str = "hazard_proximity",
        hit_key: str = "got_hurt",
        hit_penalty: float = 1.0,
        fear_gain: float = 0.3,
        urgency_gain: float = 5.0,
    ) -> None:
        self.name = name
        self.proximity_key = proximity_key
        self.hit_key = hit_key
        self.hit_penalty = hit_penalty
        self.fear_gain = fear_gain
        self.urgency_gain = urgency_gain
        self._proximity = 0.0

    def reset(self) -> None:
        self._proximity = 0.0

    def step(self, ctx: Mapping[str, Any]) -> None:
        # proximity in [0, 1]: 1 == adjacent hazard, 0 == none in sight.
        self._proximity = float(ctx.get(self.proximity_key, 0.0))

    def reward(self, ctx: Mapping[str, Any]) -> float:
        r = -self.fear_gain * self._proximity
        if ctx.get(self.hit_key):
            r -= self.hit_penalty
        return r

    def urgency(self, ctx: Mapping[str, Any]) -> float:
        return 0.05 + self.urgency_gain * self._proximity

    def level(self) -> float:
        return 1.0 - self._proximity


class SocialDrive(Drive):
    """Affiliation: rewards being near other agents and cooperating.

    Reads ``neighbors`` (count within a radius) and an optional
    ``cooperation`` signal (e.g. a successful joint action / mutual cooperation
    in a social dilemma).
    """

    def __init__(
        self,
        name: str = "social",
        *,
        neighbor_key: str = "neighbors",
        coop_key: str = "cooperation",
        affiliation_gain: float = 0.15,
        coop_gain: float = 0.5,
        target_neighbors: int = 2,
        urgency_base: float = 0.6,
    ) -> None:
        self.name = name
        self.neighbor_key = neighbor_key
        self.coop_key = coop_key
        self.affiliation_gain = affiliation_gain
        self.coop_gain = coop_gain
        self.target_neighbors = target_neighbors
        self.urgency_base = urgency_base
        self._loneliness = 0.0

    def reset(self) -> None:
        self._loneliness = 0.0

    def step(self, ctx: Mapping[str, Any]) -> None:
        n = float(ctx.get(self.neighbor_key, 0))
        # Loneliness grows when below the target company level.
        self._loneliness = _clamp((self.target_neighbors - n) / max(self.target_neighbors, 1))

    def reward(self, ctx: Mapping[str, Any]) -> float:
        n = float(ctx.get(self.neighbor_key, 0))
        # Diminishing returns: good to have company, crowding adds nothing.
        affil = self.affiliation_gain * min(n, self.target_neighbors)
        coop = self.coop_gain * float(ctx.get(self.coop_key, 0.0))
        return affil + coop

    def urgency(self, ctx: Mapping[str, Any]) -> float:
        return 0.05 + self.urgency_base * self._loneliness

    def level(self) -> float:
        return 1.0 - self._loneliness


class CuriosityDrive(Drive):
    """Intrinsic motivation via *recency-based* novelty (a lightweight ICM/RND stand-in).

    A place is "novel" if the agent has not visited it *recently*: reward grows
    with the time since the cell was last seen, saturating at ``horizon`` steps.
    Unlike raw visit counts (which saturate over a long life and stop providing a
    gradient), recency-based novelty sustains exploration indefinitely — revisit
    somewhere you've neglected and it feels fresh again. This is the "boredom"
    that keeps an agent wandering, and the seed of open-ended behaviour. Urgency
    rises when the recent past has felt familiar (restlessness).
    """

    def __init__(
        self,
        name: str = "curiosity",
        *,
        novelty_key: str = "pos",
        gain: float = 0.5,
        urgency_gain: float = 2.0,
        horizon: float = 120.0,
    ) -> None:
        self.name = name
        self.novelty_key = novelty_key
        self.gain = gain
        self.urgency_gain = urgency_gain
        self.horizon = horizon
        self._last_visit: dict[Any, int] = {}
        self._t = 0
        self._pending = 1.0
        self._recent_novelty = 1.0

    def reset(self) -> None:
        self._last_visit.clear()
        self._t = 0
        self._pending = 1.0
        self._recent_novelty = 1.0

    def step(self, ctx: Mapping[str, Any]) -> None:
        self._t += 1
        key = ctx.get(self.novelty_key)
        if key is None:
            self._pending = 0.0
            return
        last = self._last_visit.get(key, self._t - int(self.horizon))
        gap = self._t - last
        self._pending = min(1.0, gap / self.horizon)  # 1 == not seen for a long time
        self._last_visit[key] = self._t
        self._recent_novelty = 0.95 * self._recent_novelty + 0.05 * self._pending

    def reward(self, ctx: Mapping[str, Any]) -> float:
        return self.gain * self._pending

    def urgency(self, ctx: Mapping[str, Any]) -> float:
        # When the recent past has felt familiar, curiosity's pull *rises*.
        return 0.05 + self.urgency_gain * (1.0 - self._recent_novelty)

    def level(self) -> float:
        return self._recent_novelty


class ExtrinsicDrive(Drive):
    """A plain pass-through of an environment-supplied reward.

    Used for game payoffs (e.g. prisoner's-dilemma points) and as the
    single-reward *baseline*: an agent whose entire steering subsystem is one
    ExtrinsicDrive is a classic monolithic-reward RL agent.
    """

    def __init__(
        self,
        name: str = "payoff",
        *,
        reward_key: str = "extrinsic",
        urgency_value: float = 1.0,
    ) -> None:
        self.name = name
        self.reward_key = reward_key
        self.urgency_value = urgency_value

    def reward(self, ctx: Mapping[str, Any]) -> float:
        return float(ctx.get(self.reward_key, 0.0))

    def urgency(self, ctx: Mapping[str, Any]) -> float:
        return self.urgency_value


class ReciprocityDrive(Drive):
    """Tit-for-tat-flavoured social reward for matrix games.

    Rewards mutual cooperation and punishes *being* the defector against a
    cooperator (guilt) and being defected on (resentment). This is the kind of
    innate fairness drive that, stacked on top of raw payoff, can tip a
    prisoner's dilemma toward cooperation without changing the game's payoffs.
    """

    def __init__(
        self,
        name: str = "reciprocity",
        *,
        mutual_coop_bonus: float = 1.6,
        guilt: float = 1.4,
        resentment: float = 0.5,
        urgency_value: float = 1.0,
    ) -> None:
        self.name = name
        self.mutual_coop_bonus = mutual_coop_bonus
        self.guilt = guilt
        self.resentment = resentment
        self.urgency_value = urgency_value

    def reward(self, ctx: Mapping[str, Any]) -> float:
        my = ctx.get("my_move")          # 1 == cooperate, 0 == defect
        opp = ctx.get("opp_move")
        if my is None or opp is None:
            return 0.0
        r = 0.0
        if my == 1 and opp == 1:
            r += self.mutual_coop_bonus
        if my == 0 and opp == 1:
            r -= self.guilt
        if my == 1 and opp == 0:
            r -= self.resentment
        return r

    def urgency(self, ctx: Mapping[str, Any]) -> float:
        return self.urgency_value


class AcquisitionDrive(Drive):
    """Wealth-seeking: rewards *gaining* a fungible resource (e.g. gold), with
    urgency that scales with how poor the agent is relative to a target.

    This is the economic drive in the stack. Unlike a homeostatic drive (which
    defends a setpoint and is satisfied at it), acquisition keeps wanting more as
    long as the agent is below target, and is rewarded by the *increase* in
    wealth — so it pushes agents toward trade/accumulation. Reading ``wealth``
    from the context lets the same drive sit on any environment that exposes a
    currency (Neural MMO's ``gold``, a market balance, etc.).
    """

    def __init__(
        self,
        name: str = "acquisition",
        *,
        wealth_key: str = "wealth",
        target: float = 10.0,
        gain: float = 0.2,
        urgency_gain: float = 1.2,
    ) -> None:
        self.name = name
        self.wealth_key = wealth_key
        self.target = target
        self.gain = gain
        self.urgency_gain = urgency_gain
        self._wealth = 0.0
        self._delta = 0.0

    def reset(self) -> None:
        self._wealth = 0.0
        self._delta = 0.0

    def step(self, ctx: Mapping[str, Any]) -> None:
        new = float(ctx.get(self.wealth_key, self._wealth))
        self._delta = new - self._wealth
        self._wealth = new

    def reward(self, ctx: Mapping[str, Any]) -> float:
        # Rewarded by gaining wealth (and mildly stung by losing it).
        return self.gain * self._delta

    def urgency(self, ctx: Mapping[str, Any]) -> float:
        poverty = max(0.0, (self.target - self._wealth) / self.target)
        return 0.05 + self.urgency_gain * poverty

    def level(self) -> float:
        return _clamp(self._wealth / self.target)
