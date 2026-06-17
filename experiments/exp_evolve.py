"""Experiment 4 - generational drift: does cooperation evolve?

The team's long-horizon idea is that the parameters of the drive stack should
*drift across generations*, the way evolution tuned our innate drives. Here we
evolve a single gene — the strength of the innate ReciprocityDrive — under
purely *selfish* selection: a genome's fitness is the realised game payoff its
(homogeneous) population earns in repeated round-robin IPD. No cooperation is
rewarded directly.

The Axelrod-style prediction: because a population of reciprocators sustains
mutual cooperation (≈R/round) while defectors collapse to mutual defection
(≈P/round), selfish payoff selection should *raise* the reciprocity gene and
cooperation should emerge as a by-product.

Run:  python -m experiments.exp_evolve
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlstack.agent import BrainAgent  # noqa: E402
from rlstack.drives import ExtrinsicDrive, ReciprocityDrive  # noqa: E402
from rlstack.envs import IteratedPrisonersDilemma  # noqa: E402
from rlstack.evolution import Genome, evolve  # noqa: E402
from rlstack.steering import SteeringSubsystem  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _make_pop(reciprocity: float, n: int, seed: int) -> list[BrainAgent]:
    pop = []
    for i in range(n):
        drives = [ExtrinsicDrive("payoff", reward_key="extrinsic")]
        if reciprocity > 0:
            drives.append(
                ReciprocityDrive(
                    "reciprocity",
                    mutual_coop_bonus=reciprocity,
                    guilt=0.875 * reciprocity,
                    resentment=0.3 * reciprocity,
                )
            )
        steering = SteeringSubsystem(drives, mode="sum")
        pop.append(BrainAgent(i, 2, steering, alpha=0.2, gamma=0.95, epsilon=0.25,
                              epsilon_min=0.02, epsilon_decay=0.999, seed=seed + i))
    return pop


def _train_and_measure(reciprocity: float, *, n: int = 6, generations: int = 45,
                       seed: int = 0) -> tuple[float, float]:
    """Return (mean payoff/move, mean cooperation rate) for a homogeneous pop."""
    env = IteratedPrisonersDilemma(rounds_per_match=14, seed=seed)
    pop = _make_pop(reciprocity, n, seed)
    pairs = list(itertools.combinations(range(n), 2))
    rng = np.random.default_rng(seed)
    for _ in range(generations):
        rng.shuffle(pairs)
        for i, j in pairs:
            env.play_match(pop[i], pop[j], train=True)
    # Greedy evaluation.
    coop, moves, payoff = 0, 0, 0.0
    for i, j in pairs:
        s = env.play_match(pop[i], pop[j], train=False)
        coop += s["a_coop"] + s["b_coop"]
        payoff += s["a_payoff"] + s["b_payoff"]
        moves += 2 * env.rounds_per_match
    return payoff / moves, coop / moves


def fitness_fn(genome: Genome, seed: int) -> float:
    payoff, _ = _train_and_measure(genome.genes["reciprocity"], seed=seed)
    return payoff


def main() -> dict:
    print("=" * 64)
    print("EXPERIMENT 4 - Evolving the reciprocity gene (selfish selection)")
    print("=" * 64)
    seed0 = Genome({"reciprocity": 0.3})
    best, history = evolve(seed0, fitness_fn, generations=12, population=10,
                           mutation_scale=0.35, seed=11)

    print(f"\n{'gen':>4}{'mean payoff':>14}{'best payoff':>14}{'reciprocity gene':>20}")
    print("-" * 52)
    for h in history:
        print(f"{h['generation']:>4}{h['mean_fitness']:>14.2f}{h['best_fitness']:>14.2f}"
              f"{h['gene_means']['reciprocity']:>20.2f}")

    # Confirm the by-product: cooperation at the evolved gene vs the seed gene.
    _, coop_seed = _train_and_measure(0.2, seed=99)
    _, coop_evolved = _train_and_measure(best.genes["reciprocity"], seed=99)
    print(f"\nreciprocity gene drifted: {seed0.genes['reciprocity']:.2f} "
          f"-> {best.genes['reciprocity']:.2f}")
    print(f"cooperation rate (by-product): {coop_seed:.0%} -> {coop_evolved:.0%}")

    out = {"history": history, "best_reciprocity": best.genes["reciprocity"],
           "coop_seed": coop_seed, "coop_evolved": coop_evolved}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "evolve.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved -> results/evolve.json")
    return out


if __name__ == "__main__":
    main()
