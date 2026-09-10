"""Low-overhead request-local data structures for A_01 causal tracing."""

from __future__ import annotations

import statistics
import threading
import time
from typing import Any, Dict, List, Optional


def _percentile(values: List[int], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summary(values: List[int]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "sum": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": statistics.mean(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": max(values),
    }


class RequestTrace:
    """Collects a request's monotonic timestamps without doing I/O."""

    def __init__(self, run_id: str, request_id: str):
        self.run_id = run_id
        self.request_id = request_id
        self.timestamps: Dict[str, int] = {}
        self.steps: List[Dict[str, Any]] = []

    def mark(self, name: str, at_ns: Optional[int] = None) -> int:
        value = time.perf_counter_ns() if at_ns is None else at_ns
        self.timestamps[name] = value
        return value

    def add_step(self, step: Dict[str, Any]) -> None:
        self.steps.append(dict(step))

    def _step_aggregates(self) -> Dict[str, Dict[str, Optional[float]]]:
        executor_wait: List[int] = []
        worker_to_forward: List[int] = []
        forward_wall: List[int] = []
        for step in self.steps:
            submit = step.get("t_submit_ns")
            worker_start = step.get("t_worker_start_ns")
            forward_begin = step.get("t_forward_begin_ns")
            forward_end = step.get("t_forward_end_ns")
            if isinstance(submit, int) and isinstance(worker_start, int):
                executor_wait.append(worker_start - submit)
            if isinstance(worker_start, int) and isinstance(forward_begin, int):
                worker_to_forward.append(forward_begin - worker_start)
            if isinstance(forward_begin, int) and isinstance(forward_end, int):
                forward_wall.append(forward_end - forward_begin)
        return {
            "executor_wait_ns": _summary(executor_wait),
            "worker_to_forward_ns": _summary(worker_to_forward),
            "forward_wall_ns": _summary(forward_wall),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "request_id": self.request_id,
            "clock_source": "perf_counter_ns",
            "timestamps": dict(self.timestamps),
            "steps": [dict(step) for step in self.steps],
            "step_aggregates": self._step_aggregates(),
        }


def new_request_trace(
    enabled: bool,
    run_id: str,
    request_id: str,
) -> Optional[RequestTrace]:
    if not enabled:
        return None
    return RequestTrace(run_id=run_id, request_id=request_id)


class TraceStore:
    """Stores completed traces by run and serializes only post-run retrieval."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: Dict[str, List[Dict[str, Any]]] = {}

    def add(self, trace: RequestTrace) -> None:
        record = trace.to_dict()
        with self._lock:
            self._records.setdefault(trace.run_id, []).append(record)

    def take_run(self, run_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return self._records.pop(run_id, [])
