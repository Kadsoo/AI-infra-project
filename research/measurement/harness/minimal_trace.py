"""Minimal low-overhead tracing for Stage 3M — L1 / L1a / L1b / L1c.

Design goals:
- Hot path: read monotonic timestamp -> write preallocated memory -> return
- No JSON / file / log / string formatting / heavy lock in hot path
- Fixed-size preallocated buffer, ring fallback, no GPU impact, low GC

Levels:
- L0 OFF  : disabled -> None (true no-op)
- L1      : 6 integer timestamps (handler_enter, executor_submit, executor_start,
            first_forward_begin, first_content_yield, server_done)
- L1a     : 4 timestamps (receive, executor_submit/start, first_token, completion)
- L1b     : sampled L1 (deterministic 10-20% of requests)
- L1c     : aggregate counters only (no per-request timeline)
"""

from __future__ import annotations

import statistics
import threading
import time
from typing import Any, Dict, List, Optional


# Event indices for L1 (6 points)
L1_HANDLER_ENTER = 0
L1_EXECUTOR_SUBMIT = 1
L1_EXECUTOR_START = 2
L1_FIRST_FORWARD_BEGIN = 3
L1_FIRST_CONTENT_YIELD = 4
L1_SERVER_DONE = 5
L1_EVENT_COUNT = 6

L1A_EVENT_COUNT = 4

# Sampling helper — deterministic, no randomness
def _should_sample(request_id: str, ratio: float) -> bool:
    """Deterministic sampling: hash hex suffix of request_id mod 100."""
    try:
        # request_id like "req-0003-1a2b3c"
        suffix = request_id.rsplit("-", 1)[-1]
        # suffix is hex 6 chars
        val = int(suffix, 16)
        return (val % 100) < int(ratio * 100)
    except Exception:
        # fallback: hash
        return (hash(request_id) % 100) < int(ratio * 100)


class MinimalRequestTrace:
    """Compact per-request trace: 6 monotonic integer timestamps, __slots__, preallocated list."""

    __slots__ = ("run_id", "request_id", "ts", "level")

    def __init__(self, run_id: str, request_id: str, level: int = 1):
        self.run_id = run_id
        self.request_id = request_id
        self.level = level
        if level == 1:
            self.ts: List[int] = [0] * L1_EVENT_COUNT
        elif level == 10:  # L1a code
            self.ts = [0] * L1A_EVENT_COUNT
        else:
            self.ts = [0] * L1_EVENT_COUNT

    def mark(self, idx: int, at_ns: Optional[int] = None) -> int:
        """Hot path: monotonic read -> write preallocated slot -> return. No string key."""
        value = time.perf_counter_ns() if at_ns is None else at_ns
        # bounds check omitted for speed; caller guarantees idx valid
        self.ts[idx] = value
        return value

    def mark_by_name(self, name: str, at_ns: Optional[int] = None) -> int:
        """Compatibility helper for named events (maps to index). Avoid in L1 hot path."""
        mapping = {
            "t_handler_enter_ns": 0,
            "t_prepare_submit_ns": 1,
            "t_prepare_worker_start_ns": 2,
            "t_first_forward_begin_ns": 3,
            "t_first_content_yield_ns": 4,
            "t_server_done_ns": 5,
        }
        idx = mapping.get(name, 0)
        return self.mark(idx, at_ns)

    def to_dict(self) -> Dict[str, Any]:
        # Minimal serialization: no sorting, no aggregates, flat
        return {
            "run_id": self.run_id,
            "request_id": self.request_id,
            "clock_source": "perf_counter_ns",
            "level": self.level,
            "ts": list(self.ts),
            # keep timestamps dict for compatibility with derive_trace_components
            "timestamps": {
                "t_handler_enter_ns": self.ts[0] if len(self.ts) > 0 else 0,
                "t_prepare_submit_ns": self.ts[1] if len(self.ts) > 1 else 0,
                "t_prepare_worker_start_ns": self.ts[2] if len(self.ts) > 2 else 0,
                "t_first_forward_begin_ns": self.ts[3] if len(self.ts) > 3 else 0,
                "t_first_content_yield_ns": self.ts[4] if len(self.ts) > 4 else 0,
                "t_server_done_ns": self.ts[5] if len(self.ts) > 5 else 0,
            },
            "event_count": len(self.ts),
        }

    def is_valid(self) -> bool:
        # Check monotonic and non-zero
        if not self.ts or any(v == 0 for v in self.ts):
            return False
        for i in range(1, len(self.ts)):
            if self.ts[i] < self.ts[i-1]:
                return False
        return True


class MinimalTraceStore:
    """Bounded store: per-run list with fixed capacity, ring fallback, short lock."""

    def __init__(self, capacity: int = 50000):
        self._lock = threading.Lock()
        self._records: Dict[str, List[Dict[str, Any]]] = {}
        self._capacity = capacity
        self._total = 0

    def add(self, trace: MinimalRequestTrace) -> None:
        # Build record outside lock (no sorting)
        record = trace.to_dict()
        with self._lock:
            bucket = self._records.setdefault(trace.run_id, [])
            if len(bucket) < self._capacity:
                bucket.append(record)
            else:
                # ring overwrite — bounded, no realloc storm
                bucket[self._total % self._capacity] = record
            self._total += 1

    def take_run(self, run_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return self._records.pop(run_id, [])

    def size(self, run_id: str) -> int:
        with self._lock:
            return len(self._records.get(run_id, []))

    def total(self) -> int:
        with self._lock:
            return self._total


# Aggregate-only store for L1c
class CounterStore:
    """L1c: only counters / histograms, no per-request timeline."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[str, Dict[str, Any]] = {}

    def inc(self, run_id: str) -> None:
        with self._lock:
            rec = self._counters.setdefault(run_id, {"count": 0, "first_content_deltas": []})
            rec["count"] += 1

    def observe(self, run_id: str, delta_ns: int) -> None:
        with self._lock:
            rec = self._counters.setdefault(run_id, {"count": 0, "first_content_deltas": []})
            rec["first_content_deltas"].append(delta_ns)

    def take_run(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            return self._counters.pop(run_id, {"count": 0, "first_content_deltas": []})


# Factory

def new_minimal_trace(
    enabled: bool,
    level: int,
    run_id: str,
    request_id: str,
    sample_ratio: float = 1.0,
) -> Optional[MinimalRequestTrace]:
    """Create trace if enabled and level >0. For L1b sampling, check ratio."""
    if not enabled or level == 0:
        return None
    if level == 1 and sample_ratio < 1.0:
        if not _should_sample(request_id, sample_ratio):
            return None
    # level 1, 10 (L1a), 11 (L1b), 12 (L1c) all use MinimalRequestTrace except L1c
    if level == 12:  # L1c aggregate only — no per-request trace
        return None
    # map L1b effective level to 1
    effective = 1 if level in (1, 11) else level
    if effective == 10:
        return MinimalRequestTrace(run_id, request_id, level=10)
    return MinimalRequestTrace(run_id, request_id, level=1)


# Helper to derive minimal components for compatibility with overhead gate
def derive_minimal_components(trace_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Derive simple executor_wait and model_to_token from minimal ts."""
    ts = trace_dict.get("ts", [])
    if len(ts) < 6:
        return {"valid": False, "reason": "minimal trace ts too short"}
    # check monotonic
    for i in range(1, len(ts)):
        if ts[i] < ts[i-1] or ts[i] == 0:
            return {"valid": False, "reason": "non-monotonic or zero timestamp"}
    executor_wait = ts[2] - ts[1]
    worker_to_forward = 0  # not separately recorded in minimal
    first_forward_wall = 0  # not directly, but ts4-ts3 includes it + model_to_token
    model_to_token = ts[4] - ts[3]
    server_first_content = ts[4] - ts[0]
    server_total = ts[5] - ts[0]
    if executor_wait < 0 or model_to_token < 0 or server_first_content < 0:
        return {"valid": False, "reason": "negative interval"}
    return {
        "valid": True,
        "executor_wait_s": executor_wait / 1e9,
        "model_to_token_s": model_to_token / 1e9,
        "server_first_content_s": server_first_content / 1e9,
        "server_total_s": server_total / 1e9,
    }
