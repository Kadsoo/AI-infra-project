"""Continuum Eq.2 regression tests for the Kairos compat TTLPolicy (paper v5 Sec 4.2).

Locks: cold start ln(B) under Exp(1); K=100 per-tool/global fallback;
enumeration over {0} U unique durations maximizing P(tau)*B - tau.
Pure stdlib; runs without GPU/torch/vLLM.

NOTE: runs inside a Kairos checkout (https://github.com/MaxizYB/Kairos),
e.g. `python -m unittest tests.test_ttl_policy -v`. Kept here as the
audit artifact proving the in-repo Continuum implements paper v5 Eq.2.
"""
import math
import unittest
from bisect import bisect_right

from kairos._runtime.compat.policy import TTLPolicy


def brute_force(samples, benefit):
    xs = sorted(samples)
    best, bu = 0.0, 0.0
    for tau in sorted({0.0, *xs}):
        u = bisect_right(xs, tau) / len(xs) * benefit - tau
        if u > bu:
            best, bu = tau, u
    return best


class ContinuumTTLTests(unittest.TestCase):
    def test_cold_start_is_log_benefit(self):
        p = TTLPolicy(mode="paper")
        tau, src = p.ttl("unseen", 4.0)
        self.assertAlmostEqual(tau, math.log(4.0))
        self.assertEqual(src, "cold-exp-unit-mean")
        tau0, _ = p.ttl("unseen", 0.5)
        self.assertEqual(tau0, 0.0)

    def test_cold_threshold_k100(self):
        p = TTLPolicy(mode="paper")
        for i in range(100):
            p.observe("f", 0.5 + (i % 5) * 0.1, 0)
        tau, src = p.ttl("f", 3.0)
        self.assertEqual(src, "cold-exp-unit-mean")  # 100 <= k -> still cold
        p.observe("f", 0.6, 0)
        tau, src = p.ttl("f", 3.0)
        self.assertEqual(src, "per-tool-ecdf")

    def test_per_tool_matches_bruteforce(self):
        p = TTLPolicy(mode="paper")
        samples = [0.4, 0.5, 0.6, 0.5, 0.45, 0.55] * 25  # 150 > k
        for t in samples:
            p.observe("cd", t, 0)
        for benefit in (1.0, 3.0, 10.0):
            tau, src = p.ttl("cd", benefit)
            self.assertEqual(src, "per-tool-ecdf")
            self.assertAlmostEqual(tau, brute_force(samples, benefit))

    def test_global_fallback(self):
        p = TTLPolicy(mode="paper")
        for i in range(120):
            p.observe("other", 1.0 + (i % 4) * 0.1, 0)
        tau, src = p.ttl("newtool", 3.0)
        self.assertEqual(src, "global-ecdf")
        self.assertGreaterEqual(tau, 0.0)

    def test_zero_benefit_pins_nothing(self):
        p = TTLPolicy(mode="paper")
        for i in range(120):
            p.observe("f", 0.5, 0)
        tau, _ = p.ttl("f", 0.0)
        self.assertEqual(tau, 0.0)

    def test_fixed_and_zero_modes(self):
        self.assertEqual(TTLPolicy(mode="fixed", fixed=2.0).ttl("f", 99)[0], 2.0)
        self.assertEqual(TTLPolicy(mode="zero").ttl("f", 99)[0], 0.0)


if __name__ == "__main__":
    unittest.main()
