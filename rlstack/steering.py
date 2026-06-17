"""The Steering Subsystem: arbitration over a stack of drives.

In Byrnes' decomposition the brain has a *Learning Subsystem* (the neocortex —
a from-scratch learner, here our Q-learner) and a *Steering Subsystem* (the
brainstem/hypothalamus — innate circuitry that decides what is rewarding). This
module is the steering subsystem: it takes the stack of :class:`~rlstack.drives.Drive`
objects and collapses their many reward signals into the single scalar the
learner actually optimises.

Three arbitration modes let us run the key experimental contrast:

``"dynamic"``  Urgency-weighted reward — each drive's contribution is scaled by
               its *current* urgency (then normalised). This is the hypothesis
               under test: a stack with state-dependent arbitration.
``"softmax"``  Like dynamic, but urgencies pass through a softmax (temperature
               ``tau``) so the most urgent drive dominates — a sharper "winner
               mostly takes the steering wheel" arbitration.
``"sum"``      Plain (optionally fixed-weighted) sum of drive rewards. This is
               the "one reward with many hand-weighted terms" control: a stack
               *without* dynamic arbitration.

A single-drive steering subsystem (e.g. just an ExtrinsicDrive or just a
HomeostaticDrive) is the monolithic-reward baseline.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .drives import Drive


class SteeringSubsystem:
    def __init__(
        self,
        drives: Sequence[Drive],
        *,
        mode: str = "dynamic",
        tau: float = 1.0,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        if mode not in ("dynamic", "softmax", "sum"):
            raise ValueError(f"unknown steering mode: {mode!r}")
        self.drives = list(drives)
        self.mode = mode
        self.tau = tau
        self.weights = dict(weights) if weights else {}
        self.last_info: dict[str, Any] = {}

    def reset(self) -> None:
        for d in self.drives:
            d.reset()
        self.last_info = {}

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.drives]

    def evaluate(self, ctx: Mapping[str, Any]) -> float:
        """Advance every drive, then return the single arbitrated reward.

        Side effect: stores a per-drive breakdown (reward, urgency, the
        currently dominant drive) in :attr:`last_info` for metrics/inspection.
        """
        for d in self.drives:
            d.step(ctx)

        rewards = {d.name: d.reward(ctx) for d in self.drives}
        urgencies = {d.name: max(d.urgency(ctx), 0.0) for d in self.drives}

        if self.mode == "sum":
            total = sum(self.weights.get(n, 1.0) * rewards[n] for n in rewards)
        elif self.mode == "dynamic":
            norm = sum(urgencies.values()) or 1.0
            total = sum(rewards[n] * urgencies[n] / norm for n in rewards)
        else:  # softmax
            ws = self._softmax(urgencies)
            total = sum(rewards[n] * ws[n] for n in rewards)

        dominant = max(urgencies, key=urgencies.get) if urgencies else None
        self.last_info = {
            "reward": total,
            "rewards": rewards,
            "urgencies": urgencies,
            "dominant": dominant,
        }
        return total

    def _softmax(self, urgencies: Mapping[str, float]) -> dict[str, float]:
        items = list(urgencies.items())
        mx = max(v for _, v in items)
        exps = {k: math.exp((v - mx) / self.tau) for k, v in items}
        z = sum(exps.values()) or 1.0
        return {k: v / z for k, v in exps.items()}

    # --- observation helpers -------------------------------------------------
    def levels(self) -> dict[str, float]:
        """Normalised internal level of each drive (for building observations)."""
        return {d.name: d.level() for d in self.drives}

    def any_dead(self) -> bool:
        return any(getattr(d, "dead", False) for d in self.drives)
