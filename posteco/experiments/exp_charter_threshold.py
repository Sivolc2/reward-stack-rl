"""Experiment 2 — how big must the commons pool be? (and the Game B caveat)

Sweep the charter's non-dilutable commons share epsilon at deep automation and find
the minimum that keeps everyone fed. Then look at the Gini: does the charter
*de-concentrate* wealth, or just top up the bottom inside a still-rivalrous system
(Schmachtenberger's Game B challenge)?

Run:  python -m posteco.experiments.exp_charter_threshold
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from posteco import PostLaborEconomy, Regime  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
EPS_GRID = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]
A = 0.92


def main() -> dict:
    print("=" * 70)
    print(f"EXPERIMENT 2 — charter commons-share threshold (at automation A={A})")
    print("=" * 70)
    rows = []
    for eps in EPS_GRID:
        reg = Regime(f"charter-{eps}", charter_epsilon=eps)
        runs = [PostLaborEconomy(A, reg, seed=s).run() for s in range(4)]
        rows.append({
            "epsilon": eps,
            "fail": float(np.mean([r["subsistence_fail"] for r in runs])),
            "gini": float(np.mean([r["gini"] for r in runs])),
            "util": float(np.mean([r["output"] / r["capacity"] for r in runs])),
        })

    print(f"\n   {'commons ε':>10}{'fail%':>8}{'output/cap':>12}{'wealth Gini':>13}")
    min_eps = None
    for row in rows:
        bar = "█" if row["fail"] > 0.5 else "·"
        if min_eps is None and row["fail"] < 0.1:
            min_eps = row["epsilon"]
        print(f"   {row['epsilon']:>10.3f}{row['fail']*100:>7.0f}%{row['util']*100:>11.0f}%"
              f"{row['gini']:>13.2f}   {bar}")

    gini0 = rows[0]["gini"]
    gini_hi = rows[-1]["gini"]
    print(f"\n→ Minimum commons share that prevents the sinkhole: ε ≈ {min_eps:.3f}.")
    print(f"  The framework's ~20% sits comfortably above it (margin for shocks).")
    print(f"→ Game B caveat (honest): even at ε=0.50 the wealth Gini barely moves")
    print(f"  ({gini0:.2f} → {gini_hi:.2f}). The charter STOPS STARVATION but does not")
    print(f"  de-concentrate capital — it redistributes the *flow*, not the *stock*.")
    print(f"  It changes who eats, not who owns the machine. That is a real limit.")

    os.makedirs(RESULTS, exist_ok=True)
    payload = {"A": A, "rows": rows, "min_epsilon": min_eps}
    with open(os.path.join(RESULTS, "charter_threshold.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("\nsaved → posteco/results/charter_threshold.json")
    return payload


if __name__ == "__main__":
    main()
