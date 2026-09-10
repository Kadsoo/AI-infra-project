import sys
import time
import unittest
from pathlib import Path

MEASUREMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEASUREMENT_ROOT))

from harness.minimal_trace import (
    MinimalRequestTrace,
    MinimalTraceStore,
    CounterStore,
    new_minimal_trace,
    _should_sample,
    L1_HANDLER_ENTER,
    L1_EXECUTOR_SUBMIT,
    L1_EXECUTOR_START,
    L1_FIRST_FORWARD_BEGIN,
    L1_FIRST_CONTENT_YIELD,
    L1_SERVER_DONE,
)


class TimestampsMonotonicTests(unittest.TestCase):
    def test_timestamps_monotonic(self):
        trace = MinimalRequestTrace("run-1", "req-0001-abc123")
        # mark in order with increasing time
        for idx in range(6):
            time.sleep(0.001)
            trace.mark(idx)
        ts = trace.ts
        for i in range(1, len(ts)):
            self.assertGreater(ts[i], ts[i-1], f"ts[{i}] should be > ts[{i-1}]")
            self.assertGreater(ts[i], 0)

    def test_timestamps_non_negative(self):
        trace = MinimalRequestTrace("run-1", "req-1")
        for idx in range(6):
            trace.mark(idx, at_ns=100 + idx*10)
        for v in trace.ts:
            self.assertGreaterEqual(v, 0)


class EventOrderingTests(unittest.TestCase):
    def test_event_ordering_preserved(self):
        trace = MinimalRequestTrace("run-2", "req-2")
        # simulate realistic order: handler -> submit -> start -> forward -> yield -> done
        trace.mark(L1_HANDLER_ENTER, at_ns=1000)
        trace.mark(L1_EXECUTOR_SUBMIT, at_ns=1100)
        trace.mark(L1_EXECUTOR_START, at_ns=1150)
        trace.mark(L1_FIRST_FORWARD_BEGIN, at_ns=1200)
        trace.mark(L1_FIRST_CONTENT_YIELD, at_ns=1300)
        trace.mark(L1_SERVER_DONE, at_ns=2000)
        self.assertTrue(trace.is_valid())
        d = trace.to_dict()
        self.assertEqual(d["timestamps"]["t_handler_enter_ns"], 1000)
        self.assertEqual(d["timestamps"]["t_server_done_ns"], 2000)

    def test_invalid_if_out_of_order(self):
        trace = MinimalRequestTrace("run-2", "req-2")
        trace.mark(L1_HANDLER_ENTER, at_ns=2000)
        trace.mark(L1_EXECUTOR_SUBMIT, at_ns=1000)  # backwards
        self.assertFalse(trace.is_valid())


class BufferOverflowTests(unittest.TestCase):
    def test_buffer_has_fixed_upper_bound(self):
        store = MinimalTraceStore(capacity=5)
        for i in range(10):
            t = MinimalRequestTrace("run-overflow", f"req-{i:04d}-abc123")
            for idx in range(6):
                t.mark(idx, at_ns=1000 + i*100 + idx)
            store.add(t)
        # capacity 5 + ring fallback: size should be 5, not 10
        self.assertEqual(store.size("run-overflow"), 5)
        # total counts still tracked
        self.assertEqual(store.total(), 10)

    def test_no_frequent_realloc(self):
        # preallocated ts list should remain length 6 after many marks
        trace = MinimalRequestTrace("run", "req-abc123")
        self.assertEqual(len(trace.ts), 6)
        for _ in range(100):
            trace.mark(L1_HANDLER_ENTER)
        self.assertEqual(len(trace.ts), 6)

    def test_minimal_memory_no_gpu_impact(self):
        # Minimal trace should not hold any torch tensor reference
        trace = MinimalRequestTrace("run", "req-1")
        for idx in range(6):
            trace.mark(idx, at_ns=1000+idx)
        d = trace.to_dict()
        # ensure no tensor or GPU field
        self.assertNotIn("gpu", str(d).lower())
        self.assertEqual(d["event_count"], 6)


class DisabledTracingNoOpTests(unittest.TestCase):
    def test_disabled_returns_none(self):
        trace = new_minimal_trace(enabled=False, level=1, run_id="run", request_id="req-1")
        self.assertIsNone(trace)
        trace = new_minimal_trace(enabled=True, level=0, run_id="run", request_id="req-1")
        self.assertIsNone(trace)

    def test_disabled_truly_no_perf_counter(self):
        # when disabled, mark should never be called, so no perf_counter overhead
        # we test that creating disabled trace does not allocate ts
        trace = new_minimal_trace(enabled=False, level=1, run_id="run", request_id="req-1")
        self.assertIsNone(trace)
        # also L1c should be None (aggregate only)
        trace_c = new_minimal_trace(enabled=True, level=12, run_id="run", request_id="req-1")
        self.assertIsNone(trace_c)


class SamplingDeterministicTests(unittest.TestCase):
    def test_sampling_deterministic(self):
        req_id = "req-0001-1a2b3c"
        first = _should_sample(req_id, 0.2)
        second = _should_sample(req_id, 0.2)
        self.assertEqual(first, second)
        # same ratio, same id -> same decision across many calls
        for _ in range(10):
            self.assertEqual(_should_sample(req_id, 0.2), first)

    def test_sampling_ratio_respected(self):
        # generate 100 deterministic ids, check ~20% sampled at 0.2
        sampled = sum(1 for i in range(100) if _should_sample(f"req-{i:04d}-{i*12345 % 0xFFFFFF:06x}", 0.2))
        # allow 10-30% due to hash distribution
        self.assertGreaterEqual(sampled, 10)
        self.assertLessEqual(sampled, 35)

    def test_new_minimal_trace_sampling(self):
        # 20% ratio: some requests should be None, some not
        results = []
        for i in range(50):
            rid = f"req-{i:04d}-{i*99991 % 0xFFFFFF:06x}"
            t = new_minimal_trace(enabled=True, level=1, run_id="run", request_id=rid, sample_ratio=0.2)
            results.append(t is not None)
        # deterministic: rerun same ids gives same pattern
        results2 = []
        for i in range(50):
            rid = f"req-{i:04d}-{i*99991 % 0xFFFFFF:06x}"
            t = new_minimal_trace(enabled=True, level=1, run_id="run", request_id=rid, sample_ratio=0.2)
            results2.append(t is not None)
        self.assertEqual(results, results2)
        # not all sampled and not none sampled
        self.assertGreater(sum(results), 0)
        self.assertLess(sum(results), 50)


class SerializationAfterRunTests(unittest.TestCase):
    def test_serialization_happens_after_run(self):
        store = MinimalTraceStore(capacity=100)
        traces = []
        for i in range(5):
            t = MinimalRequestTrace("run-ser", f"req-{i:04d}-abc123")
            for idx in range(6):
                t.mark(idx, at_ns=1000 + i*100 + idx)
            traces.append(t)
            store.add(t)
        # traces should be retrievable only after run via take_run
        self.assertEqual(store.size("run-ser"), 5)
        taken = store.take_run("run-ser")
        self.assertEqual(len(taken), 5)
        # second take should be empty (pop behavior)
        self.assertEqual(store.take_run("run-ser"), [])
        # check that to_dict was called at add time, not during hot path serialization
        for rec in taken:
            self.assertIn("ts", rec)
            self.assertEqual(len(rec["ts"]), 6)
            self.assertIn("timestamps", rec)

    def test_json_serializable(self):
        import json
        t = MinimalRequestTrace("run-json", "req-abc123")
        for idx in range(6):
            t.mark(idx, at_ns=1000+idx*10)
        d = t.to_dict()
        # should be json dumpable without error (no tensors, no non-serializable)
        s = json.dumps(d)
        self.assertIn("run-json", s)
        j = json.loads(s)
        self.assertEqual(j["ts"], d["ts"])


class OutputCompletenessTests(unittest.TestCase):
    def test_output_completeness(self):
        store = MinimalTraceStore(capacity=100)
        run_id = "run-complete"
        expected_ids = []
        for i in range(38):
            rid = f"req-{i:04d}-{i*12345 % 0xFFFFFF:06x}"
            expected_ids.append(rid)
            t = MinimalRequestTrace(run_id, rid)
            for idx in range(6):
                t.mark(idx, at_ns=1000000 + i*10000 + idx*100)
            store.add(t)
        taken = store.take_run(run_id)
        self.assertEqual(len(taken), 38)
        observed_ids = sorted([r["request_id"] for r in taken])
        self.assertEqual(sorted(expected_ids), observed_ids)
        # every trace should have 6 monotonic timestamps
        for rec in taken:
            ts = rec["ts"]
            self.assertEqual(len(ts), 6)
            for i in range(1, len(ts)):
                self.assertGreater(ts[i], ts[i-1])

    def test_counter_store_completeness(self):
        store = CounterStore()
        run_id = "run-counter"
        for _ in range(10):
            store.inc(run_id)
        data = store.take_run(run_id)
        self.assertEqual(data["count"], 10)
        self.assertEqual(store.take_run(run_id)["count"], 0)


class MinimalTraceLevelTests(unittest.TestCase):
    def test_l1a_has_4_events(self):
        t = MinimalRequestTrace("run", "req-1", level=10)
        self.assertEqual(len(t.ts), 4)

    def test_level_switch(self):
        # L0 should be None, L1 should be trace
        off = new_minimal_trace(True, 0, "run", "req-1")
        on = new_minimal_trace(True, 1, "run", "req-1")
        self.assertIsNone(off)
        self.assertIsNotNone(on)


if __name__ == "__main__":
    unittest.main()
