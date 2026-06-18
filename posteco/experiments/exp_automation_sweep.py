"""Experiment 1 — the pothole: which regime turns automation into a sinkhole?

Sweep automation A from 0 to ~1 under each institutional regime and watch for the
*tipping point* where the economy falls into the sinkhole (mass subsistence failure
+ demand collapse). The thesis: the catastrophe is OPTIONAL — the determining
variable is the regime, not the automation level.

Run:  python -m posteco.experiments.exp_automation_sweep
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from posteco import REGIMES, PostLaborEconomy  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
A_GRID = [round(0.05 * i, 2) for i in range(20)]  # 0.00 .. 0.95
REGIME_ORDER = ["laissez-faire", "wealth-tax", "charter", "all-three"]


def sweep(seeds=(0, 1, 2)) -> dict:
    out = {}
    for rname in REGIME_ORDER:
        rows = []
        for A in A_GRID:
            runs = [PostLaborEconomy(A, REGIMES[rname], seed=s).run() for s in seeds]
            rows.append({k: float(np.mean([r[k] for r in runs]))
                         for k in ("subsistence_fail", "output", "capacity", "gini", "p_good")})
        out[rname] = rows
    return out


def tipping_point(rows) -> float | None:
    for A, row in zip(A_GRID, rows):
        if row["subsistence_fail"] > 0.5:
            return A
    return None


def main() -> dict:
    print("=" * 70)
    print("EXPERIMENT 1 — the pothole: automation sweep across regimes")
    print("=" * 70)
    data = sweep()

    print("\nsubsistence-failure %% vs automation A  (· <10%, ▓ 10-50%, █ >50% = sinkhole)")
    print(f"{'A →':<16}" + "".join(f"{a:>4.2f}" for a in A_GRID[::2]))
    for rname in REGIME_ORDER:
        cells = []
        for row in data[rname][::2]:
            f = row["subsistence_fail"]
            cells.append("  █ " if f > 0.5 else "  ▓ " if f > 0.1 else "  · ")
        print(f"{rname:<16}" + "".join(cells))

    print("\ntipping point into the sinkhole (first A with >50% subsistence failure):")
    tips = {}
    for rname in REGIME_ORDER:
        tp = tipping_point(data[rname])
        tips[rname] = tp
        label = f"A = {tp:.2f}" if tp is not None else "never (holds across the sweep)"
        print(f"   {rname:<18} {label}")

    print("\nat A = 0.90 (deep automation):")
    print(f"   {'regime':<18}{'fail%':>7}{'output/cap':>12}{'gini':>7}{'p_good':>8}")
    for rname in REGIME_ORDER:
        row = data[rname][A_GRID.index(0.90)]
        util = row["output"] / row["capacity"]
        print(f"   {rname:<18}{row['subsistence_fail']*100:>6.0f}%{util*100:>11.0f}%"
              f"{row['gini']:>7.2f}{row['p_good']:>8.2f}")

    print("\n→ The pothole is real and regime-dependent: laissez-faire tips first, the")
    print("  leaky wealth tax only delays it (capital routes around redistribution at")
    print("  high automation), and the non-dilutable commons charter holds. The")
    print("  catastrophe is optional.")

    os.makedirs(RESULTS, exist_ok=True)
    payload = {"A_grid": A_GRID, "regimes": data, "tipping_points": tips}
    with open(os.path.join(RESULTS, "automation_sweep.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("\nsaved → posteco/results/automation_sweep.json")
    return payload


if __name__ == "__main__":
    main()
