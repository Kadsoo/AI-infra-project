"""Frozen A_01 causal experiment matrix and timestamp helpers."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List


CONCURRENCIES = (1, 4, 8)


def formal_specs() -> List[Dict[str, int]]:
    return [
        {"block": 1, "concurrency": 1, "seed": 1010},
        {"block": 1, "concurrency": 4, "seed": 1040},
        {"block": 1, "concurrency": 8, "seed": 1080},
        {"block": 2, "concurrency": 4, "seed": 1041},
        {"block": 2, "concurrency": 8, "seed": 1081},
        {"block": 2, "concurrency": 1, "seed": 1011},
        {"block": 3, "concurrency": 8, "seed": 1082},
        {"block": 3, "concurrency": 1, "seed": 1012},
        {"block": 3, "concurrency": 4, "seed": 1042},
    ]


def overhead_specs() -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    for concurrency, seed_a, seed_b in (
        (1, 3101, 3102),
        (4, 3401, 3402),
        (8, 3801, 3802),
    ):
        specs.extend(
            [
                {
                    "concurrency": concurrency,
                    "seed": seed_a,
                    "trace_enabled": False,
                },
                {
                    "concurrency": concurrency,
                    "seed": seed_a,
                    "trace_enabled": True,
                },
                {
                    "concurrency": concurrency,
                    "seed": seed_b,
                    "trace_enabled": True,
                },
                {
                    "concurrency": concurrency,
                    "seed": seed_b,
                    "trace_enabled": False,
                },
            ]
        )
    return specs


def pooled_control_specs() -> List[Dict[str, int]]:
    return [
        {"concurrency": 1, "seed": 4101},
        {"concurrency": 4, "seed": 4401},
        {"concurrency": 8, "seed": 4801},
    ]


_REQUIRED_TIMESTAMPS = (
    "t_handler_enter_ns",
    "t_stream_runtime_enter_ns",
    "t_prepare_submit_ns",
    "t_prepare_worker_start_ns",
    "t_prepare_worker_done_ns",
    "t_prepare_resume_ns",
    "t_first_step_submit_ns",
    "t_first_step_worker_start_ns",
    "t_first_forward_begin_ns",
    "t_first_forward_end_ns",
    "t_first_content_yield_ns",
    "t_server_done_ns",
)


def _seconds(delta_ns: int) -> float:
    return delta_ns / 1_000_000_000


def _step_sums(steps: List[Dict[str, Any]]) -> Dict[str, float]:
    executor_wait = 0
    worker_to_forward = 0
    forward_wall = 0
    valid_steps = 0
    for step in steps:
        submit = step.get("t_submit_ns")
        worker_start = step.get("t_worker_start_ns")
        forward_begin = step.get("t_forward_begin_ns")
        forward_end = step.get("t_forward_end_ns")
        if not all(
            isinstance(value, int)
            for value in (submit, worker_start, forward_begin, forward_end)
        ):
            continue
        executor_wait += worker_start - submit
        worker_to_forward += forward_begin - worker_start
        forward_wall += forward_end - forward_begin
        valid_steps += 1
    return {
        "count": valid_steps,
        "executor_wait_sum_s": _seconds(executor_wait),
        "worker_to_forward_sum_s": _seconds(worker_to_forward),
        "forward_wall_sum_s": _seconds(forward_wall),
    }


def derive_trace_components(trace: Dict[str, Any]) -> Dict[str, Any]:
    timestamps = trace.get("timestamps", {})
    missing = [
        name
        for name in _REQUIRED_TIMESTAMPS
        if not isinstance(timestamps.get(name), int)
    ]
    if missing:
        return {
            "valid": False,
            "reason": "missing required timestamps: " + ", ".join(missing),
        }

    t = timestamps
    component_ns = {
        "handler_to_runtime_s": t["t_stream_runtime_enter_ns"] - t["t_handler_enter_ns"],
        "runtime_before_prepare_submit_s": t["t_prepare_submit_ns"] - t["t_stream_runtime_enter_ns"],
        "prepare_executor_wait_s": t["t_prepare_worker_start_ns"] - t["t_prepare_submit_ns"],
        "prepare_execution_s": t["t_prepare_worker_done_ns"] - t["t_prepare_worker_start_ns"],
        "prepare_resume_delay_s": t["t_prepare_resume_ns"] - t["t_prepare_worker_done_ns"],
        "runtime_between_prepare_and_first_step_s": t["t_first_step_submit_ns"] - t["t_prepare_resume_ns"],
        "first_step_executor_wait_s": t["t_first_step_worker_start_ns"] - t["t_first_step_submit_ns"],
        "worker_to_first_forward_s": t["t_first_forward_begin_ns"] - t["t_first_step_worker_start_ns"],
        "first_forward_wall_s": t["t_first_forward_end_ns"] - t["t_first_forward_begin_ns"],
        "model_to_first_content_s": t["t_first_content_yield_ns"] - t["t_first_forward_end_ns"],
    }
    server_first_content_ns = (
        t["t_first_content_yield_ns"] - t["t_handler_enter_ns"]
    )
    server_total_ns = t["t_server_done_ns"] - t["t_handler_enter_ns"]
    if server_first_content_ns < 0 or server_total_ns < 0 or any(
        value < 0 for value in component_ns.values()
    ):
        return {
            "valid": False,
            "reason": "negative server timestamp interval",
        }
    if sum(component_ns.values()) != server_first_content_ns:
        return {
            "valid": False,
            "reason": "non-additive time-to-first-content decomposition",
        }

    steps = trace.get("steps", [])
    all_steps = _step_sums(steps)
    if all_steps["count"] == 0:
        return {
            "valid": False,
            "reason": "no complete model-step record",
        }
    if any(
        value < 0
        for key, value in all_steps.items()
        if key != "count"
    ):
        return {
            "valid": False,
            "reason": "negative all-step interval",
        }

    return {
        "valid": True,
        "reason": "",
        "ttfc_components_s": {
            name: _seconds(value) for name, value in component_ns.items()
        },
        "server_first_content_s": _seconds(server_first_content_ns),
        "server_total_s": _seconds(server_total_ns),
        "decode_completion_s": _seconds(
            t["t_server_done_ns"] - t["t_first_content_yield_ns"]
        ),
        "all_steps_s": all_steps,
    }


def evaluate_overhead_gate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply the locked OFF/ON overhead rule to run-level summary rows."""

    metrics = ("throughput_rps", "ttft_p95", "latency_p95")
    grouped: Dict[int, Dict[int, Dict[bool, Dict[str, Any]]]] = {}
    for row in rows:
        concurrency = row.get("concurrency")
        seed = row.get("seed")
        trace_enabled = row.get("trace_enabled")
        if (
            concurrency not in CONCURRENCIES
            or not isinstance(seed, int)
            or not isinstance(trace_enabled, bool)
        ):
            continue
        grouped.setdefault(concurrency, {}).setdefault(seed, {})[trace_enabled] = row

    details: Dict[str, Any] = {}
    failed_checks: List[str] = []
    for concurrency in CONCURRENCIES:
        pairs = grouped.get(concurrency, {})
        pair_deltas: Dict[str, List[float]] = {metric: [] for metric in metrics}
        missing_pairs: List[int] = []
        for seed, modes in sorted(pairs.items()):
            if False not in modes or True not in modes:
                missing_pairs.append(seed)
                continue
            for metric in metrics:
                off = modes[False].get(metric)
                on = modes[True].get(metric)
                if not isinstance(off, (int, float)) or not isinstance(
                    on, (int, float)
                ) or off <= 0:
                    failed_checks.append(
                        f"c{concurrency} seed{seed} invalid {metric}"
                    )
                    continue
                pair_deltas[metric].append(on / off - 1.0)
        if missing_pairs:
            failed_checks.append(
                f"c{concurrency} missing OFF/ON pairs for seeds {missing_pairs}"
            )
        metric_checks = {}
        for metric, deltas in pair_deltas.items():
            median_delta = statistics.median(deltas) if deltas else None
            individual_limit_ok = bool(deltas) and all(
                abs(delta) <= 0.10 for delta in deltas
            )
            median_limit_ok = (
                median_delta is not None and abs(median_delta) <= 0.05
            )
            metric_checks[metric] = {
                "pair_relative_deltas": deltas,
                "median_relative_delta": median_delta,
                "median_limit_ok": median_limit_ok,
                "individual_limit_ok": individual_limit_ok,
            }
            if not median_limit_ok:
                failed_checks.append(
                    f"c{concurrency} {metric} median paired difference exceeds 5%"
                )
            if not individual_limit_ok:
                failed_checks.append(
                    f"c{concurrency} {metric} individual paired difference exceeds 10%"
                )
        details[str(concurrency)] = metric_checks

    multiplier_checks = {}
    mode_medians: Dict[bool, Dict[int, float]] = {False: {}, True: {}}
    for trace_enabled in (False, True):
        for concurrency in CONCURRENCIES:
            values = [
                modes[trace_enabled]["ttft_p95"]
                for modes in grouped.get(concurrency, {}).values()
                if trace_enabled in modes
                and isinstance(modes[trace_enabled].get("ttft_p95"), (int, float))
                and modes[trace_enabled]["ttft_p95"] > 0
            ]
            if values:
                mode_medians[trace_enabled][concurrency] = statistics.median(values)
    for label, low, high in (("c4_over_c1", 1, 4), ("c8_over_c4", 4, 8)):
        if low not in mode_medians[False] or high not in mode_medians[False]:
            failed_checks.append(f"missing OFF TTFT multiplier inputs for {label}")
            continue
        if low not in mode_medians[True] or high not in mode_medians[True]:
            failed_checks.append(f"missing ON TTFT multiplier inputs for {label}")
            continue
        off_multiplier = (
            mode_medians[False][high] / mode_medians[False][low]
        )
        on_multiplier = mode_medians[True][high] / mode_medians[True][low]
        relative_change = on_multiplier / off_multiplier - 1.0
        multiplier_checks[label] = {
            "off_multiplier": off_multiplier,
            "on_multiplier": on_multiplier,
            "relative_change": relative_change,
            "limit_ok": abs(relative_change) <= 0.10,
        }
        if abs(relative_change) > 0.10:
            failed_checks.append(
                f"{label} ON/OFF TTFT multiplier difference exceeds 10%"
            )

    return {
        "status": "PASS" if not failed_checks else "FAIL",
        "by_concurrency": details,
        "ttft_multiplier_checks": multiplier_checks,
        "failed_checks": failed_checks,
    }
