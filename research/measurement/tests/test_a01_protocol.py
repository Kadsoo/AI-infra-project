import sys
import unittest
from pathlib import Path


MEASUREMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEASUREMENT_ROOT))

from harness import a01_causal

sys.path.insert(0, str(MEASUREMENT_ROOT / "harness"))
import analyze_a01_causal


class A01ProtocolTests(unittest.TestCase):
    def test_formal_specs_use_only_the_locked_interleaved_matrix(self):
        specs = a01_causal.formal_specs()

        self.assertEqual(
            [(item["block"], item["concurrency"], item["seed"]) for item in specs],
            [
                (1, 1, 1010),
                (1, 4, 1040),
                (1, 8, 1080),
                (2, 4, 1041),
                (2, 8, 1081),
                (2, 1, 1011),
                (3, 8, 1082),
                (3, 1, 1012),
                (3, 4, 1042),
            ],
        )

    def test_overhead_specs_are_counterbalanced_and_matched(self):
        specs = a01_causal.overhead_specs()

        self.assertEqual(len(specs), 12)
        for concurrency, seed_a, seed_b in [(1, 3101, 3102), (4, 3401, 3402), (8, 3801, 3802)]:
            subset = [
                (item["seed"], item["trace_enabled"])
                for item in specs
                if item["concurrency"] == concurrency
            ]
            self.assertEqual(
                subset,
                [
                    (seed_a, False),
                    (seed_a, True),
                    (seed_b, True),
                    (seed_b, False),
                ],
            )

    def test_pooled_control_stays_on_the_same_three_concurrencies(self):
        specs = a01_causal.pooled_control_specs()

        self.assertEqual(
            [(item["concurrency"], item["seed"]) for item in specs],
            [(1, 4101), (4, 4401), (8, 4801)],
        )

    def test_trace_component_decomposition_is_additive_to_first_content(self):
        trace = {
            "timestamps": {
                "t_handler_enter_ns": 100,
                "t_stream_runtime_enter_ns": 110,
                "t_prepare_submit_ns": 120,
                "t_prepare_worker_start_ns": 140,
                "t_prepare_worker_done_ns": 150,
                "t_prepare_resume_ns": 155,
                "t_first_step_submit_ns": 160,
                "t_first_step_worker_start_ns": 180,
                "t_first_forward_begin_ns": 185,
                "t_first_forward_end_ns": 205,
                "t_first_content_yield_ns": 215,
                "t_server_done_ns": 300,
            },
            "steps": [
                {
                    "t_submit_ns": 160,
                    "t_worker_start_ns": 180,
                    "t_forward_begin_ns": 185,
                    "t_forward_end_ns": 205,
                },
                {
                    "t_submit_ns": 230,
                    "t_worker_start_ns": 240,
                    "t_forward_begin_ns": 245,
                    "t_forward_end_ns": 270,
                },
            ],
        }

        result = a01_causal.derive_trace_components(trace)

        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["server_first_content_s"], 115e-9)
        self.assertAlmostEqual(
            sum(result["ttfc_components_s"].values()),
            result["server_first_content_s"],
        )
        self.assertAlmostEqual(
            result["ttfc_components_s"]["first_step_executor_wait_s"],
            20e-9,
        )
        self.assertAlmostEqual(result["all_steps_s"]["executor_wait_sum_s"], 30e-9)
        self.assertAlmostEqual(result["all_steps_s"]["forward_wall_sum_s"], 45e-9)

    def test_trace_component_decomposition_rejects_missing_required_timestamps(self):
        result = a01_causal.derive_trace_components({"timestamps": {}, "steps": []})

        self.assertFalse(result["valid"])
        self.assertIn("t_handler_enter_ns", result["reason"])

    def test_overhead_gate_passes_small_counterbalanced_differences(self):
        rows = []
        for concurrency in (1, 4, 8):
            for seed in (concurrency * 1000 + 1, concurrency * 1000 + 2):
                rows.append(
                    {
                        "concurrency": concurrency,
                        "seed": seed,
                        "trace_enabled": False,
                        "throughput_rps": 100.0 / concurrency,
                        "ttft_p95": float(concurrency),
                        "latency_p95": 2.0 * concurrency,
                    }
                )
                rows.append(
                    {
                        "concurrency": concurrency,
                        "seed": seed,
                        "trace_enabled": True,
                        "throughput_rps": 99.0 / concurrency,
                        "ttft_p95": 1.02 * concurrency,
                        "latency_p95": 2.02 * concurrency,
                    }
                )

        gate = a01_causal.evaluate_overhead_gate(rows)

        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(gate["failed_checks"], [])

    def test_overhead_gate_fails_a_large_individual_difference(self):
        rows = []
        for concurrency in (1, 4, 8):
            for seed in (concurrency * 1000 + 1, concurrency * 1000 + 2):
                rows.append(
                    {
                        "concurrency": concurrency,
                        "seed": seed,
                        "trace_enabled": False,
                        "throughput_rps": 100.0 / concurrency,
                        "ttft_p95": float(concurrency),
                        "latency_p95": 2.0 * concurrency,
                    }
                )
                rows.append(
                    {
                        "concurrency": concurrency,
                        "seed": seed,
                        "trace_enabled": True,
                        "throughput_rps": (
                            70.0 / concurrency
                            if concurrency == 4 and seed == 4001
                            else 99.0 / concurrency
                        ),
                        "ttft_p95": 1.02 * concurrency,
                        "latency_p95": 2.02 * concurrency,
                    }
                )

        gate = a01_causal.evaluate_overhead_gate(rows)

        self.assertEqual(gate["status"], "FAIL")
        self.assertTrue(gate["failed_checks"])

    def test_pooled_control_requires_every_server_component_to_stay_stable(self):
        component_names = (
            "server_first_content_s",
            "runtime_pre_executor_s",
            "executor_queue_ttfc_s",
            "model_execution_ttfc_s",
            "other_ttfc_s",
        )

        def make_record(concurrency, e2e, server_multiplier):
            return {
                "concurrency": concurrency,
                "end_to_end_ttft_s": e2e,
                **{
                    name: concurrency * server_multiplier
                    for name in component_names
                },
            }

        primary = [make_record(1, 1.0, 1.0), make_record(8, 2.0, 1.0)]
        pooled = [make_record(1, 1.0, 1.0), make_record(8, 3.0, 1.05)]

        stable = analyze_a01_causal._pooled_control_evidence(primary, pooled)
        self.assertTrue(stable["client_http_artifact_condition"])
        self.assertTrue(stable["all_server_components_within_10pct"])

        pooled[1]["executor_queue_ttfc_s"] = 8 * 1.30
        unstable = analyze_a01_causal._pooled_control_evidence(primary, pooled)
        self.assertFalse(unstable["client_http_artifact_condition"])
        self.assertFalse(unstable["all_server_components_within_10pct"])


if __name__ == "__main__":
    unittest.main()
