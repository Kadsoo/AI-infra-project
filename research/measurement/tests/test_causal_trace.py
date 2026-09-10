import sys
import tempfile
import unittest
from pathlib import Path


MEASUREMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEASUREMENT_ROOT))

from harness import causal_trace
from harness import harness as benchmark_harness


class _FakeStreamingResponse:
    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def aiter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"hello"}}]}'
        yield "data: [DONE]"


class _FakeClient:
    def __init__(self):
        self.calls = []

    def stream(self, method, url, json, timeout, headers=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "json": json,
                "timeout": timeout,
                "headers": headers,
            }
        )
        return _FakeStreamingResponse()


class RequestTraceTests(unittest.TestCase):
    def test_request_trace_keeps_correlated_timestamps_and_step_aggregates(self):
        trace = causal_trace.RequestTrace(run_id="run-1", request_id="req-1")
        trace.mark("t_handler_enter_ns", 100)
        trace.mark("t_stream_runtime_enter_ns", 120)
        trace.mark("t_prepare_submit_ns", 140)
        trace.mark("t_prepare_worker_start_ns", 170)
        trace.mark("t_prepare_worker_done_ns", 190)
        trace.mark("t_prepare_resume_ns", 195)
        trace.mark("t_first_step_submit_ns", 200)
        trace.mark("t_first_step_worker_start_ns", 260)
        trace.mark("t_first_forward_begin_ns", 270)
        trace.mark("t_first_forward_end_ns", 330)
        trace.mark("t_first_content_yield_ns", 350)
        trace.mark("t_server_done_ns", 700)
        trace.add_step(
            {
                "index": 0,
                "t_submit_ns": 200,
                "t_worker_start_ns": 260,
                "t_forward_begin_ns": 270,
                "t_forward_end_ns": 330,
                "t_resume_ns": 340,
            }
        )
        trace.add_step(
            {
                "index": 1,
                "t_submit_ns": 400,
                "t_worker_start_ns": 430,
                "t_forward_begin_ns": 440,
                "t_forward_end_ns": 500,
                "t_resume_ns": 510,
            }
        )

        payload = trace.to_dict()

        self.assertEqual(payload["run_id"], "run-1")
        self.assertEqual(payload["request_id"], "req-1")
        self.assertEqual(payload["timestamps"]["t_first_content_yield_ns"], 350)
        self.assertEqual(payload["steps"][1]["t_forward_end_ns"], 500)
        self.assertEqual(payload["step_aggregates"]["executor_wait_ns"]["count"], 2)
        self.assertEqual(payload["step_aggregates"]["executor_wait_ns"]["sum"], 90)
        self.assertEqual(payload["step_aggregates"]["forward_wall_ns"]["sum"], 120)
        self.assertEqual(payload["step_aggregates"]["worker_to_forward_ns"]["max"], 10)

    def test_trace_factory_returns_none_when_tracing_is_off(self):
        trace = causal_trace.new_request_trace(
            enabled=False,
            run_id="run-off",
            request_id="req-off",
        )

        self.assertIsNone(trace)

    def test_trace_store_returns_only_the_requested_run_once(self):
        store = causal_trace.TraceStore()
        first = causal_trace.RequestTrace(run_id="run-a", request_id="req-1")
        second = causal_trace.RequestTrace(run_id="run-b", request_id="req-2")
        third = causal_trace.RequestTrace(run_id="run-a", request_id="req-3")
        store.add(first)
        store.add(second)
        store.add(third)

        first_take = store.take_run("run-a")
        second_take = store.take_run("run-a")
        remaining = store.take_run("run-b")

        self.assertEqual(
            [record["request_id"] for record in first_take],
            ["req-1", "req-3"],
        )
        self.assertEqual(second_take, [])
        self.assertEqual([record["request_id"] for record in remaining], ["req-2"])


class ClientTraceHeaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_request_sends_stable_trace_headers_and_client_timestamps(self):
        client = _FakeClient()
        record = benchmark_harness.RequestRecord(
            request_id="req-client-1",
            workload_type="synthetic",
            arrival_time=1.0,
            dispatch_time=0.0,
            queue_time=0.0,
            prompt_text="hello",
        )

        result = await benchmark_harness.execute_request(
            client=client,
            base_url="http://example.test",
            rec=record,
            max_tokens=4,
            temperature=0.0,
            stream=True,
            timeout=5.0,
            trace_run_id="run-client-1",
        )

        self.assertEqual(client.calls[0]["headers"]["X-Stage3-Run-Id"], "run-client-1")
        self.assertEqual(
            client.calls[0]["headers"]["X-Stage3-Request-Id"],
            "req-client-1",
        )
        self.assertLessEqual(
            result.client_send_perf_ns,
            result.client_headers_perf_ns,
        )
        self.assertLessEqual(
            result.client_headers_perf_ns,
            result.client_first_content_perf_ns,
        )
        self.assertLessEqual(
            result.client_first_content_perf_ns,
            result.client_done_perf_ns,
        )


class RequestRecordPersistenceTests(unittest.TestCase):
    def test_save_raw_preserves_client_trace_fields(self):
        record = benchmark_harness.RequestRecord(
            request_id="req-save-1",
            workload_type="synthetic",
            arrival_time=1.0,
            dispatch_time=2.0,
            queue_time=1.0,
            trace_run_id="run-save-1",
            client_slot_acquired_time=1.5,
            client_slot_acquired_perf_ns=150,
            client_send_perf_ns=200,
            client_headers_perf_ns=220,
            client_first_content_perf_ns=260,
            client_done_perf_ns=500,
            success=True,
        )
        result = benchmark_harness.RunResult(
            run_id="run-save-1",
            config={},
            environment_hash="test",
            start_time=1.0,
            end_time=2.0,
            requests=[record],
            system_samples=[],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            request_path, _, _ = benchmark_harness.save_raw(
                result,
                Path(temp_dir),
            )
            rows = request_path.read_text(encoding="utf-8")

        self.assertIn("client_send_perf_ns", rows)
        self.assertIn("run-save-1", rows)
        self.assertIn("260", rows)

    def test_save_raw_preserves_cpu_and_server_process_controls(self):
        sample = benchmark_harness.SystemSample(
            timestamp=1.0,
            cpu_percent=20.0,
            ram_percent=40.0,
            ram_used_gb=8.0,
            gpu_util=None,
            gpu_mem_used_mb=None,
            gpu_mem_total_mb=None,
            gpu_mem_percent=None,
            cpu_freq_current_mhz=3200.0,
            server_process_cpu_percent=75.0,
            server_process_rss_mb=512.0,
            server_process_threads=12,
        )
        result = benchmark_harness.RunResult(
            run_id="run-system-1",
            config={},
            environment_hash="test",
            start_time=1.0,
            end_time=2.0,
            requests=[],
            system_samples=[sample],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            _, system_path, _ = benchmark_harness.save_raw(
                result,
                Path(temp_dir),
            )
            rows = system_path.read_text(encoding="utf-8")

        self.assertIn("cpu_freq_current_mhz", rows)
        self.assertIn("server_process_cpu_percent", rows)
        self.assertIn("3200.0", rows)


if __name__ == "__main__":
    unittest.main()
