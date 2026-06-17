"""Run the whole experiment loop end to end and print a consolidated summary.

This is the "run the loop, test it, present it" step: it executes all four
experiments, then prints a one-screen scorecard of the headline findings.

Run:  python -m experiments.run_all
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import exp_evolve, exp_gridworld, exp_prisoners, exp_tug  # noqa: E402


def main() -> None:
    t0 = time.time()
    ipd = exp_prisoners.main()
    print()
    grid = exp_gridworld.main()
    print()
    tug = exp_tug.main()
    print()
    evo = exp_evolve.main()

    print("\n" + "=" * 64)
    print("CONSOLIDATED SCORECARD")
    print("=" * 64)
    print("1. Prisoner's dilemma — a reciprocity drive on unchanged payoffs:")
    print(f"     cooperation  {ipd['baseline']['final_coop_rate']:.0%} (payoff only)"
          f"  ->  {ipd['stacked']['final_coop_rate']:.0%} (+reciprocity)")
    print("2. ResourceWorld — each added drive moves its own behaviour:")
    print(f"     hazard exposure  {grid['hunger']['hazard_exposure']:.0%} -> "
          f"{grid['+safety']['hazard_exposure']:.0%} (+safety)")
    print(f"     clustering       {grid['hunger']['clustering']:.2f} -> "
          f"{grid['+social']['clustering']:.2f} (+social)")
    print(f"     goal-switching   {grid['hunger']['goal_switch_rate']:.0%} -> "
          f"{grid['full stack']['goal_switch_rate']:.0%} (full stack)")
    print("3. TugGame — cooperation from a shared actuator:")
    print(f"     delivery rate  {tug['plain']['pre_delivery_rate']:.0%} -> "
          f"{tug['plain']['post_delivery_rate']:.0%} after learning")
    print("4. Generational drift — selfish selection grows reciprocity:")
    print(f"     reciprocity gene {evo['history'][0]['gene_means']['reciprocity']:.2f} -> "
          f"{evo['best_reciprocity']:.2f};  cooperation "
          f"{evo['coop_seed']:.0%} -> {evo['coop_evolved']:.0%}")
    print(f"\ntotal wall-clock: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
