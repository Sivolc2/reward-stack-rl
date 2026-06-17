"""Experiment 1 - does a reciprocity drive flip the prisoner's dilemma?

Two populations play repeated round-robin IPD tournaments and learn across them:

  * BASELINE  agents whose only drive is the game payoff (ExtrinsicDrive).
  * STACKED   agents with payoff + an innate ReciprocityDrive on top.

The game's payoffs are identical for both. If the hypothesis holds, the stacked
population should converge to markedly higher mutual cooperation while the
baseline drifts toward defection.

Run:  python -m experiments.exp_prisoners
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlstack import make_ipd_agent  # noqa: E402
from rlstack.envs import IteratedPrisonersDilemma  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run_population(stacked: bool, *, n_agents: int = 6, generations: int = 120,
                   seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    env = IteratedPrisonersDilemma(rounds_per_match=16, seed=seed)
    agents = [make_ipd_agent(i, stacked=stacked, seed=seed + i) for i in range(n_agents)]
    pairs = list(itertools.combinations(range(n_agents), 2))

    coop_curve: list[float] = []
    payoff_curve: list[float] = []
    for gen in range(generations):
        rng.shuffle(pairs)
        gen_coop, gen_moves, gen_payoff = 0, 0, 0.0
        for i, j in pairs:
            s = env.play_match(agents[i], agents[j], train=True)
            gen_coop += s["a_coop"] + s["b_coop"]
            gen_moves += 2 * env.rounds_per_match
            gen_payoff += s["a_payoff"] + s["b_payoff"]
        coop_curve.append(gen_coop / gen_moves)
        payoff_curve.append(gen_payoff / gen_moves)

    # Greedy evaluation (no exploration) for a clean final read.
    eval_coop, eval_moves = 0, 0
    for i, j in pairs:
        s = env.play_match(agents[i], agents[j], train=False)
        eval_coop += s["a_coop"] + s["b_coop"]
        eval_moves += 2 * env.rounds_per_match
    return {
        "stacked": stacked,
        "coop_curve": coop_curve,
        "payoff_curve": payoff_curve,
        "final_coop_rate": eval_coop / eval_moves,
        "final_train_coop": float(np.mean(coop_curve[-10:])),
        "final_payoff_per_move": float(np.mean(payoff_curve[-10:])),
    }


def main() -> dict:
    print("=" * 64)
    print("EXPERIMENT 1 - Iterated Prisoner's Dilemma (the defection game)")
    print("=" * 64)
    baseline = run_population(stacked=False, seed=1)
    stacked = run_population(stacked=True, seed=1)

    print(f"\n{'agent type':<28}{'final coop rate':>18}{'payoff/move':>16}")
    print("-" * 62)
    print(f"{'baseline (payoff only)':<28}{baseline['final_coop_rate']:>18.1%}"
          f"{baseline['final_payoff_per_move']:>16.2f}")
    print(f"{'stacked (+reciprocity)':<28}{stacked['final_coop_rate']:>18.1%}"
          f"{stacked['final_payoff_per_move']:>16.2f}")

    # ASCII trend of cooperation over training.
    print("\ncooperation rate over training (baseline=., stacked=#):")
    print(_ascii_curves(baseline["coop_curve"], stacked["coop_curve"]))

    out = {"baseline": baseline, "stacked": stacked}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "prisoners.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> results/prisoners.json")
    return out


def _ascii_curves(a: list[float], b: list[float], *, width: int = 56, height: int = 10) -> str:
    def sample(curve):
        idx = np.linspace(0, len(curve) - 1, width).astype(int)
        return [curve[i] for i in idx]
    sa, sb = sample(a), sample(b)
    rows = []
    for h in range(height, -1, -1):
        thresh = h / height
        line = []
        for ca, cb in zip(sa, sb):
            if cb >= thresh and ca >= thresh:
                line.append("@")
            elif cb >= thresh:
                line.append("#")
            elif ca >= thresh:
                line.append(".")
            else:
                line.append(" ")
        rows.append(f"{thresh:4.1f} |" + "".join(line))
    rows.append("     +" + "-" * width)
    rows.append("      0" + " " * (width - 8) + "generations")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
