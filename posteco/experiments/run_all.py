"""Run all three post-labor experiments and print a consolidated takeaway.

Run:  python -m posteco.experiments.run_all
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from posteco.experiments import (  # noqa: E402
    exp_automation_sweep, exp_charter_threshold, exp_fear_vs_vision,
)


def main() -> None:
    sweep = exp_automation_sweep.main()
    print()
    thresh = exp_charter_threshold.main()
    print()
    fear = exp_fear_vs_vision.main()

    print("\n" + "=" * 70)
    print("CONSOLIDATED TAKEAWAY — a mechanism sandbox for 'Potholes on the Way to Utopia'")
    print("=" * 70)
    tp = sweep["tipping_points"]
    print(f"1. The pothole is optional. Laissez-faire tips into the sinkhole at A≈{tp['laissez-faire']},")
    print(f"   the leaky wealth tax only delays it to A≈{tp['wealth-tax']}, the charter never tips.")
    print(f"2. The commons share needs ε≈{thresh['min_epsilon']:.2f} to hold; ~20% has margin — BUT it")
    print(f"   stops starvation without de-concentrating capital (the Game B limit).")
    fr = fear["rows"]
    print(f"3. Fear alone idles {fr[0]['idle']*100:.0f}% of capacity vs {fr[-1]['idle']*100:.0f}% under vision — prosperity")
    print(f"   created or destroyed by belief. Vision is load-bearing, not decoration.")
    print("\nThis is a MECHANISM model, conditional on automation rising — not a forecast")
    print("(see README: Hanson's timing objection, and 'mechanism, not magnitude').")


if __name__ == "__main__":
    main()
