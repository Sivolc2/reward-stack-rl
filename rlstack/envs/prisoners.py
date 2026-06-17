"""Tier 1 - Iterated Prisoner's Dilemma (the "defection game").

The simplest possible test of cooperation vs. defection, and trivially
evaluable: just measure the fraction of cooperative moves. The payoffs are the
canonical T > R > P > S with 2R > T + S, so *defection strictly dominates a
single round* yet *mutual cooperation beats mutual defection* — exactly the
Harvard-Business-School "defection game" the team described.

The environment is reward-agnostic: it only reports the game *payoff* and the
two moves. Whether an agent learns to defect or cooperate then depends entirely
on its steering subsystem:

  * a single ``ExtrinsicDrive`` (payoff only)  -> learns to defect (baseline)
  * payoff + ``ReciprocityDrive`` (a fairness drive)  -> can learn cooperation

That contrast is the whole point: an innate social drive *stacked on top of*
unchanged game payoffs can flip the equilibrium.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..agent import BrainAgent

DEFECT, COOPERATE = 0, 1


class IteratedPrisonersDilemma:
    """Plays repeated K-round matches between pairs of :class:`BrainAgent`.

    Parameters
    ----------
    payoffs:
        ``(T, R, P, S)`` = (temptation, reward, punishment, sucker). Default
        ``(5, 3, 1, 0)`` is the classic Axelrod tournament payoff.
    rounds_per_match:
        How many rounds a fixed pair plays before memory resets (the repeated
        game that lets reciprocity get a foothold).
    """

    n_actions = 2

    def __init__(
        self,
        payoffs: tuple[float, float, float, float] = (5.0, 3.0, 1.0, 0.0),
        *,
        rounds_per_match: int = 16,
        seed: int = 0,
    ) -> None:
        self.T, self.R, self.P, self.S = payoffs
        self.rounds_per_match = rounds_per_match
        self.rng = np.random.default_rng(seed)

    def _payoff(self, mine: int, theirs: int) -> float:
        if mine == COOPERATE and theirs == COOPERATE:
            return self.R
        if mine == COOPERATE and theirs == DEFECT:
            return self.S
        if mine == DEFECT and theirs == COOPERATE:
            return self.T
        return self.P

    @staticmethod
    def _state(my_last: int, opp_last: int) -> tuple:
        # memory-1 state; -1 == "first round of the match".
        return ("ipd", my_last, opp_last)

    def play_match(self, a: "BrainAgent", b: "BrainAgent", *, train: bool = True):
        """Run one K-round match; agents act, earn their own reward, and learn.

        Returns a dict of per-agent cooperation counts and total game payoff.
        """
        a.reset_episode()
        b.reset_episode()
        a_last, b_last = -1, -1
        stats = {
            "a_coop": 0,
            "b_coop": 0,
            "a_payoff": 0.0,
            "b_payoff": 0.0,
            "a_moves": [],
            "b_moves": [],
        }

        for t in range(self.rounds_per_match):
            sa = self._state(a_last, b_last)
            sb = self._state(b_last, a_last)
            move_a = a.act(sa, greedy=not train)
            move_b = b.act(sb, greedy=not train)

            pay_a = self._payoff(move_a, move_b)
            pay_b = self._payoff(move_b, move_a)

            ctx_a = {"extrinsic": pay_a, "my_move": move_a, "opp_move": move_b}
            ctx_b = {"extrinsic": pay_b, "my_move": move_b, "opp_move": move_a}
            r_a = a.reward_from(ctx_a)
            r_b = b.reward_from(ctx_b)

            done = t == self.rounds_per_match - 1
            next_sa = self._state(move_a, move_b)
            next_sb = self._state(move_b, move_a)
            if train:
                a.observe(sa, r_a, next_sa, done)
                b.observe(sb, r_b, next_sb, done)

            a_last, b_last = move_a, move_b
            stats["a_coop"] += move_a
            stats["b_coop"] += move_b
            stats["a_payoff"] += pay_a
            stats["b_payoff"] += pay_b
            stats["a_moves"].append(move_a)
            stats["b_moves"].append(move_b)

        return stats
