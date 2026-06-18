"""Sanity tests for the post-labor economy ABM. Run: python -m unittest -v

These pin the qualitative properties the writeup relies on (not the exact numbers,
which are illustrative): a healthy pre-automation economy, the laissez-faire
sinkhole at high automation, and the charter holding where the leaky tax does not.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posteco import REGIMES, PostLaborEconomy, Regime  # noqa: E402


class TestPostLaborEconomy(unittest.TestCase):
    def test_runs_and_metric_bounds(self):
        r = PostLaborEconomy(0.5, REGIMES["charter"], periods=40, seed=0).run()
        for key in ("gini", "subsistence_fail", "demand_gap", "p_good"):
            self.assertGreaterEqual(r[key], -1e-9)
            self.assertLessEqual(r[key], 1.0 + 1e-9)

    def test_deterministic_under_seed(self):
        a = PostLaborEconomy(0.7, REGIMES["charter"], seed=3).run()
        b = PostLaborEconomy(0.7, REGIMES["charter"], seed=3).run()
        self.assertEqual(a["gini"], b["gini"])
        self.assertEqual(a["subsistence_fail"], b["subsistence_fail"])

    def test_pre_automation_is_healthy(self):
        # At A=0 even laissez-faire should keep almost everyone above subsistence.
        r = PostLaborEconomy(0.0, REGIMES["laissez-faire"], seed=1).run()
        self.assertLess(r["subsistence_fail"], 0.1)

    def test_laissez_faire_sinkhole_at_high_automation(self):
        r = PostLaborEconomy(0.9, REGIMES["laissez-faire"], seed=1).run()
        self.assertGreater(r["subsistence_fail"], 0.5)        # mass failure
        self.assertGreater(r["demand_gap"], 0.2)              # idle capacity

    def test_charter_holds_where_tax_fails(self):
        charter = PostLaborEconomy(0.92, REGIMES["charter"], seed=1).run()
        tax = PostLaborEconomy(0.92, REGIMES["wealth-tax"], seed=1).run()
        self.assertLess(charter["subsistence_fail"], 0.1)     # charter holds
        self.assertGreater(tax["subsistence_fail"], 0.5)      # leaky tax tips

    def test_charter_threshold_monotone(self):
        # More commons share never makes the sinkhole worse.
        low = PostLaborEconomy(0.92, Regime("c", charter_epsilon=0.05), seed=2).run()
        high = PostLaborEconomy(0.92, Regime("c", charter_epsilon=0.20), seed=2).run()
        self.assertGreaterEqual(low["subsistence_fail"], high["subsistence_fail"])

    def test_fear_idles_capacity(self):
        # Pinned fear leaves more capacity idle than pinned vision, needs met both ways.
        fear = PostLaborEconomy(0.85, REGIMES["charter"], fix_belief=0.1, seed=0).run()
        vision = PostLaborEconomy(0.85, REGIMES["charter"], fix_belief=0.9, seed=0).run()
        self.assertGreater(fear["demand_gap"], vision["demand_gap"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
