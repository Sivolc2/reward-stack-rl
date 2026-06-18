"""Experiment 3 — the fear that creates what it fears (the demand channel).

Hold the economy and institutions fixed (a charter economy where needs are met) and
vary only the shared *belief* p_good — the expectation that the post-labor economy
goes well "for me." Fear suppresses spending and investment (precautionary hoarding);
a credible vision unlocks it. This is Moloch at the wallet: even when nobody starves,
collective fear leaves a large slice of capacity idle — prosperity destroyed by a
coordination failure, not by scarcity.

This isolates what "vision production" buys, and is the reward-stack mechanism at
civilizational scale (belief reweighting the acquisition/social drives).

Run:  python -m posteco.experiments.exp_fear_vs_vision
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from posteco import REGIMES, PostLaborEconomy  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
BELIEFS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
A = 0.85


def main() -> dict:
    print("=" * 70)
    print("EXPERIMENT 3 — fear vs vision: the demand channel (charter economy)")
    print("=" * 70)
    rows = []
    for b in BELIEFS:
        runs = [PostLaborEconomy(A, REGIMES["charter"], fix_belief=b, seed=s).run()
                for s in range(4)]
        rows.append({
            "belief": b,
            "util": float(np.mean([r["output"] / r["capacity"] for r in runs])),
            "idle": float(np.mean([r["demand_gap"] for r in runs])),
            "median_consumption": float(np.mean([r["median_consumption"] for r in runs])),
        })

    print(f"\n   {'belief p_good':>13}{'capacity used':>15}{'idle capacity':>15}{'median cons':>13}")
    for row in rows:
        bar = "▇" * int(round(row["util"] * 24))
        print(f"   {row['belief']:>13.1f}{row['util']*100:>13.0f}% {bar:<24}"
              f"{row['idle']*100:>5.0f}%   {row['median_consumption']:>10.2f}")

    lo, hi = rows[0], rows[-1]
    print(f"\n→ Pure fear (p_good={lo['belief']}) leaves {lo['idle']*100:.0f}% of capacity idle;")
    print(f"  vision (p_good={hi['belief']}) cuts that to {hi['idle']*100:.0f}%. Same factories, same")
    print(f"  institutions — {(hi['util']-lo['util'])*100:.0f} points of prosperity created or destroyed")
    print(f"  purely by what people believe. 'You cannot make things without customers';")
    print(f"  fear makes people stop being customers. This is why vision comes first.")
    print(f"\n  Caveat: here belief is an exogenous knob isolating the demand channel. The")
    print(f"  essay's stronger claim — that vision enables the *political adoption* of the")
    print(f"  charter — is a political-economy mechanism this ABM does not model.")

    os.makedirs(RESULTS, exist_ok=True)
    payload = {"A": A, "rows": rows}
    with open(os.path.join(RESULTS, "fear_vs_vision.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("\nsaved → posteco/results/fear_vs_vision.json")
    return payload


if __name__ == "__main__":
    main()
