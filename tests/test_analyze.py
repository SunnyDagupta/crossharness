"""Unit tests for analyze.py aggregation math."""
from __future__ import annotations

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze import aggregate, proportion_se, rule_of_three, two_proportion_se


_ROWS = [
    # none condition — never passes by construction
    {"task": "w1", "family": "W", "condition": "none",      "passed": False, "calls": []},
    {"task": "w1", "family": "W", "condition": "none",      "passed": False, "calls": []},
    {"task": "e1", "family": "E", "condition": "none",      "passed": False, "calls": []},
    # syntactic — some passes, all native calls
    {"task": "w1", "family": "W", "condition": "syntactic", "passed": True,  "calls": ["harness_native"]},
    {"task": "w1", "family": "W", "condition": "syntactic", "passed": False, "calls": []},
    {"task": "e1", "family": "E", "condition": "syntactic", "passed": False, "calls": ["harness_native"]},
    # full — same native calls, one generic deviation
    {"task": "w1", "family": "W", "condition": "full",      "passed": True,  "calls": ["harness_native"]},
    {"task": "w1", "family": "W", "condition": "full",      "passed": False, "calls": []},
    {"task": "e1", "family": "E", "condition": "full",      "passed": False, "calls": ["mappable_generic"]},
]


class TestAggregationMath(unittest.TestCase):

    def test_none_condition_always_zero(self):
        agg = aggregate(_ROWS)
        self.assertEqual(agg["none"]["ok"], 0)
        self.assertEqual(agg["none"]["p"], 0.0)

    def test_syntactic_success_rate(self):
        agg = aggregate(_ROWS)
        self.assertEqual(agg["syntactic"]["ok"], 1)
        self.assertEqual(agg["syntactic"]["n"], 3)
        self.assertAlmostEqual(agg["syntactic"]["p"], 1 / 3)

    def test_deviation_counted_correctly(self):
        agg = aggregate(_ROWS)
        self.assertEqual(agg["full"]["calls_deviation"], 1)
        self.assertEqual(agg["full"]["calls_native"], 1)
        self.assertEqual(agg["full"]["calls_total"], 2)

    def test_syntactic_zero_deviation(self):
        agg = aggregate(_ROWS)
        self.assertEqual(agg["syntactic"]["calls_deviation"], 0)

    def test_family_breakdown(self):
        agg = aggregate(_ROWS)
        self.assertIn("W", agg["families"])
        w_full = agg["families"]["W"]["full"]
        self.assertEqual(w_full["ok"], 1)
        self.assertEqual(w_full["n"], 2)


class TestStatFunctions(unittest.TestCase):

    def test_proportion_se_known_value(self):
        se = proportion_se(0.5, 100)
        self.assertAlmostEqual(se, 0.05, places=6)

    def test_proportion_se_zero_n(self):
        self.assertTrue(math.isnan(proportion_se(0.5, 0)))

    def test_two_proportion_se_equal_props(self):
        # pooled SE for p1==p2==0.25, n=120 each matches hand calculation
        se = two_proportion_se(0.25, 120, 0.25, 120)
        expected = math.sqrt(0.25 * 0.75 * 2 / 120)
        self.assertAlmostEqual(se, expected, places=10)

    def test_rule_of_three(self):
        self.assertAlmostEqual(rule_of_three(57), 3 / 57)
        self.assertAlmostEqual(rule_of_three(77), 3 / 77)

    def test_rule_of_three_zero_n(self):
        self.assertTrue(math.isnan(rule_of_three(0)))


if __name__ == "__main__":
    unittest.main()
