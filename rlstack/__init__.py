"""reward-stack-rl: testing whether a *stack* of reward functions produces more
dynamic, human-like, cooperative behaviour than a single monolithic reward.

The public surface is small on purpose. Build agents with the factory helpers
below, drop them into an environment from :mod:`rlstack.envs`, and measure with
:mod:`rlstack.metrics`.
"""
from __future__ import annotations

from .agent import BrainAgent, QLearningAgent
from .drives import (
    AcquisitionDrive,
    CuriosityDrive,
    Drive,
    ExtrinsicDrive,
    HomeostaticDrive,
    ReciprocityDrive,
    SafetyDrive,
    SocialDrive,
)
from .steering import SteeringSubsystem

__all__ = [
    "BrainAgent",
    "QLearningAgent",
    "SteeringSubsystem",
    "Drive",
    "HomeostaticDrive",
    "SafetyDrive",
    "SocialDrive",
    "CuriosityDrive",
    "ExtrinsicDrive",
    "ReciprocityDrive",
    "AcquisitionDrive",
    "make_survival_agent",
    "make_stacked_agent",
    "make_agent",
    "make_ipd_agent",
    "GRID_DRIVES",
]


def _learner_kwargs(seed: int) -> dict:
    return dict(alpha=0.15, gamma=0.95, epsilon=0.25, epsilon_min=0.02,
               epsilon_decay=0.9997, seed=seed)


# --- ResourceWorld agents ----------------------------------------------------
# The available drives for the ResourceWorld, by name, so experiments can build
# any subset (for ablations) from a single source of truth.
def _grid_drive(name: str) -> Drive:
    return {
        "hunger": lambda: HomeostaticDrive("hunger", "ate"),
        "safety": lambda: SafetyDrive("safety"),
        "social": lambda: SocialDrive("social"),
        "curiosity": lambda: CuriosityDrive("curiosity"),
    }[name]()


GRID_DRIVES = ("hunger", "safety", "social", "curiosity")


def make_agent(
    agent_id: int,
    n_actions: int,
    drive_names: tuple[str, ...] = GRID_DRIVES,
    *,
    seed: int = 0,
    mode: str = "dynamic",
) -> BrainAgent:
    """Build a ResourceWorld agent from any subset of drives (for ablations)."""
    steering = SteeringSubsystem([_grid_drive(n) for n in drive_names], mode=mode)
    return BrainAgent(agent_id, n_actions, steering, **_learner_kwargs(seed))


def make_survival_agent(agent_id: int, n_actions: int, *, seed: int = 0) -> BrainAgent:
    """Baseline: a single homeostatic survival drive (monolithic reward)."""
    return make_agent(agent_id, n_actions, ("hunger",), seed=seed)


def make_stacked_agent(
    agent_id: int,
    n_actions: int,
    *,
    seed: int = 0,
    mode: str = "dynamic",
) -> BrainAgent:
    """The hypothesis agent: a full stack of hunger + safety + social + curiosity."""
    return make_agent(agent_id, n_actions, GRID_DRIVES, seed=seed, mode=mode)


# --- IPD agents --------------------------------------------------------------
def make_ipd_agent(
    agent_id: int,
    *,
    stacked: bool,
    seed: int = 0,
    mode: str = "sum",
) -> BrainAgent:
    """IPD agent. ``stacked=False`` -> payoff only (defector-leaning baseline);
    ``stacked=True`` -> payoff + reciprocity drive (can learn to cooperate).

    Uses ``sum`` arbitration: an innate fairness term *adds to* game payoff
    rather than competing with it as a homeostatic drive would."""
    drives = [ExtrinsicDrive("payoff", reward_key="extrinsic")]
    if stacked:
        drives.append(ReciprocityDrive("reciprocity"))
    steering = SteeringSubsystem(drives, mode=mode)
    return BrainAgent(agent_id, 2, steering, **_learner_kwargs(seed))
