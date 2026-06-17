"""Behavioural metrics that operationalise "dynamic, human-like" behaviour.

The hypothesis predicts that a *stack* of dynamically-arbitrated drives yields
behaviour that is richer than a single reward: more varied actions, more goal
*switching*, broader exploration, and (in social settings) more cooperation.
These functions turn raw episode traces into the numbers that let us check that.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np


def shannon_entropy(counts: Iterable[float], *, base: float = 2.0) -> float:
    """Shannon entropy of a distribution given raw counts (0 if degenerate)."""
    arr = np.asarray(list(counts), dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return 0.0
    p = arr[arr > 0] / total
    return float(-(p * (np.log(p) / np.log(base))).sum())


def action_entropy(action_counts: Sequence[int]) -> float:
    """Behavioural diversity: entropy over the agent's action distribution.

    Low for a one-note policy ("always forage"); high for a policy that flexibly
    deploys its whole repertoire.
    """
    return shannon_entropy(action_counts)


def goal_switch_rate(dominant_history: Sequence[str]) -> float:
    """Fraction of steps on which the dominant drive changed.

    A direct proxy for the steering subsystem handing the wheel between drives —
    the 'untangling/retangling of reward functions' the project is chasing.
    """
    if len(dominant_history) < 2:
        return 0.0
    switches = sum(
        1 for a, b in zip(dominant_history, dominant_history[1:]) if a != b
    )
    return switches / (len(dominant_history) - 1)


def dominant_distribution(dominant_history: Sequence[str]) -> dict[str, float]:
    """How the agent's 'attention' was split across drives over an episode."""
    c = Counter(dominant_history)
    total = sum(c.values()) or 1
    return {k: v / total for k, v in c.items()}


def coverage(visited: Iterable, total_cells: int) -> float:
    """Exploration: fraction of the world's cells the agent visited."""
    if total_cells <= 0:
        return 0.0
    return len(set(visited)) / total_cells


def cooperation_rate(moves: Sequence[int]) -> float:
    """Fraction of cooperative moves (1 == cooperate) in a social dilemma."""
    if not moves:
        return 0.0
    return float(np.mean(moves))


def gini(values: Sequence[float]) -> float:
    """Inequality of an outcome distribution (0 == perfectly equal).

    Useful for asking whether cooperation also produced *fair* outcomes.
    """
    arr = np.sort(np.asarray(values, dtype=np.float64))
    n = arr.size
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * arr).sum() - (n + 1) * arr.sum()) / (n * arr.sum()))


def summarize_episode(agent) -> dict[str, float]:
    """Bundle the standard per-agent behavioural metrics from a BrainAgent."""
    return {
        "action_entropy": action_entropy(agent.action_counts),
        "goal_switch_rate": goal_switch_rate(agent.dominant_history),
    }
