"""Process only the locked A_01 causal-localization artifacts."""

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

MEASUREMENT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MEASUREMENT_ROOT.parent.parent

import sys

sys.path.insert(0, str(MEASUREMENT_ROOT))

from harness.a01_causal import derive_trace_components, evaluate_overhead_gate


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _stats(values):
    numeric = [value for value in values if isinstance(value, (int, float))]
    if not numeric:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "variance": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(numeric),
        "mean": statistics.mean(numeric),
        "median": statistics.median(numeric),
        "p50": _percentile(numeric, 50),
        "p95": _percentile(numeric, 95),
        "p99": _percentile(numeric, 99),
        "variance": statistics.variance(numeric) if len(numeric) > 1 else 0.0,
        "std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
        "min": min(numeric),
        "max": max(numeric),
    }


def _phase_manifests(output_root: Path, phase: str):
    manifests = []
    for path in sorted((output_root / "raw").glob("*_manifest.json")):
        manifest = _read_json(path)
        if manifest.get("phase") == phase:
            manifests.append(manifest)
    return manifests


def _overhead_rows(manifests):
    rows = []
    for manifest in manifests:
        processed = _read_json(Path(manifest["artifacts"]["processed_json"]))
        config = manifest["config"]
        rows.append(
            {
                "run_id": manifest["run_id"],
                "concurrency": config["concurrency"],
                "seed": config["seed"],
                "trace_enabled": config["trace_enabled"],
                "throughput_rps": processed["throughput_rps"],
                "ttft_p95": processed["ttft"]["p95"],
                "latency_p95": processed["latency"]["p95"],
            }
        )
    return rows


def analyze_overhead(output_root: Path):
    manifests = _phase_manifests(output_root, "overhead")
    rows = _overhead_rows(manifests)
    gate = evaluate_overhead_gate(rows)
    gate["run_count"] = len(rows)
    gate["source_run_ids"] = [row["run_id"] for row in rows]
    _write_json(output_root / "processed" / "overhead_gate.json", gate)
    return gate


def _trace_records(manifests):
    records = []
    invalid = []
    for manifest in manifests:
        config = manifest["config"]
        trace_integrity = manifest.get("trace_integrity", {})
        if not trace_integrity.get("complete"):
            invalid.append(
                {
                    "run_id": manifest["run_id"],
                    "reason": "trace integrity was not complete",
                }
            )
            continue
        raw = _read_json(Path(manifest["artifacts"]["raw_json"]))
        traces = _read_json(Path(manifest["artifacts"]["server_traces_json"]))
        trace_by_id = {trace.get("request_id"): trace for trace in traces}
        for request in raw["requests"]:
            if not request.get("success"):
                invalid.append(
                    {
                        "run_id": manifest["run_id"],
                        "request_id": request.get("request_id"),
                        "reason": "request did not succeed",
                    }
                )
                continue
            trace = trace_by_id.get(request["request_id"])
            if trace is None:
                invalid.append(
                    {
                        "run_id": manifest["run_id"],
                        "request_id": request["request_id"],
                        "reason": "missing correlated server trace",
                    }
                )
                continue
            derived = derive_trace_components(trace)
            if not derived.get("valid"):
                invalid.append(
                    {
                        "run_id": manifest["run_id"],
                        "request_id": request["request_id"],
                        "reason": derived.get("reason"),
                    }
                )
                continue
            client_send = request.get("client_send_perf_ns")
            client_headers = request.get("client_headers_perf_ns")
            client_first = request.get("client_first_content_perf_ns")
            client_done = request.get("client_done_perf_ns")
            client_headers_s = None
            if isinstance(client_send, int) and isinstance(client_headers, int):
                client_headers_s = (client_headers - client_send) / 1_000_000_000
            client_ttft_s = request.get("ttft")
            if isinstance(client_send, int) and isinstance(client_first, int):
                client_ttft_s = (client_first - client_send) / 1_000_000_000
            client_total_s = request.get("total_latency")
            if isinstance(client_send, int) and isinstance(client_done, int):
                client_total_s = (client_done - client_send) / 1_000_000_000
            components = derived["ttfc_components_s"]
            record = {
                "run_id": manifest["run_id"],
                "block": config.get("block"),
                "concurrency": config["concurrency"],
                "seed": config["seed"],
                "request_id": request["request_id"],
                "client_semaphore_wait_s": request.get("queue_time"),
                "client_headers_s": client_headers_s,
                "end_to_end_ttft_s": client_ttft_s,
                "client_total_s": client_total_s,
                "server_first_content_s": derived["server_first_content_s"],
                "server_total_s": derived["server_total_s"],
                "decode_completion_s": derived["decode_completion_s"],
                **components,
                **derived["all_steps_s"],
            }
            record["runtime_pre_executor_s"] = (
                record["handler_to_runtime_s"]
                + record["runtime_before_prepare_submit_s"]
                + record["prepare_resume_delay_s"]
                + record["runtime_between_prepare_and_first_step_s"]
            )
            record["executor_queue_ttfc_s"] = (
                record["prepare_executor_wait_s"]
                + record["first_step_executor_wait_s"]
            )
            record["model_execution_ttfc_s"] = (
                record["worker_to_first_forward_s"]
                + record["first_forward_wall_s"]
            )
            record["other_ttfc_s"] = (
                record["prepare_execution_s"]
                + record["model_to_first_content_s"]
            )
            records.append(record)
    return records, invalid


def _group_by(records, key):
    grouped = defaultdict(list)
    for record in records:
        grouped[record[key]].append(record)
    return grouped


def _component_summary(records):
    metric_names = (
        "client_semaphore_wait_s",
        "client_headers_s",
        "end_to_end_ttft_s",
        "client_total_s",
        "server_first_content_s",
        "server_total_s",
        "decode_completion_s",
        "handler_to_runtime_s",
        "runtime_before_prepare_submit_s",
        "prepare_executor_wait_s",
        "prepare_execution_s",
        "prepare_resume_delay_s",
        "runtime_between_prepare_and_first_step_s",
        "first_step_executor_wait_s",
        "worker_to_first_forward_s",
        "first_forward_wall_s",
        "model_to_first_content_s",
        "runtime_pre_executor_s",
        "executor_queue_ttfc_s",
        "model_execution_ttfc_s",
        "other_ttfc_s",
        "executor_wait_sum_s",
        "worker_to_forward_sum_s",
        "forward_wall_sum_s",
    )
    return {
        name: _stats([record.get(name) for record in records])
        for name in metric_names
    }


def _hierarchical_bootstrap_diff(c1_records, c8_records, metric, repetitions=10000):
    by_c1 = list(_group_by(c1_records, "run_id").values())
    by_c8 = list(_group_by(c8_records, "run_id").values())
    if not by_c1 or not by_c8:
        return {"mean_difference": None, "ci95": [None, None]}
    if any(not group for group in by_c1 + by_c8):
        return {"mean_difference": None, "ci95": [None, None]}
    rng = random.Random(20260828 + sum(map(ord, metric)))

    def sample_mean(groups):
        selected_values = []
        for _ in range(len(groups)):
            group = groups[rng.randrange(len(groups))]
            values = [
                value[metric]
                for value in group
                if isinstance(value.get(metric), (int, float))
            ]
            if not values:
                continue
            selected_values.extend(
                values[rng.randrange(len(values))] for _ in range(len(values))
            )
        return statistics.mean(selected_values) if selected_values else math.nan

    samples = []
    for _ in range(repetitions):
        c8_mean = sample_mean(by_c8)
        c1_mean = sample_mean(by_c1)
        if not math.isnan(c8_mean) and not math.isnan(c1_mean):
            samples.append(c8_mean - c1_mean)
    observed = (
        statistics.mean(record[metric] for record in c8_records)
        - statistics.mean(record[metric] for record in c1_records)
    )
    return {
        "mean_difference": observed,
        "ci95": [_percentile(samples, 2.5), _percentile(samples, 97.5)],
    }


def _run_level_metrics(manifests):
    by_block = defaultdict(dict)
    by_concurrency = defaultdict(list)
    for manifest in manifests:
        processed = _read_json(Path(manifest["artifacts"]["processed_json"]))
        config = manifest["config"]
        row = {
            "run_id": manifest["run_id"],
            "block": config.get("block"),
            "concurrency": config["concurrency"],
            "throughput_rps": processed["throughput_rps"],
            "ttft_p95": processed["ttft"]["p95"],
            "ttft_p99": processed["ttft"]["p99"],
            "latency_p95": processed["latency"]["p95"],
            "latency_p99": processed["latency"]["p99"],
            "success": processed["success"],
            "total_requests": processed["total_requests"],
            "system": processed.get("system", {}),
        }
        by_concurrency[row["concurrency"]].append(row)
        by_block[row["block"]][row["concurrency"]] = row
    return by_concurrency, by_block


def _knee_reproduction(manifests):
    by_concurrency, by_block = _run_level_metrics(manifests)
    medians = {}
    for concurrency in (1, 4, 8):
        rows = by_concurrency.get(concurrency, [])
        medians[concurrency] = {
            "ttft_p95": statistics.median([row["ttft_p95"] for row in rows])
            if rows
            else None,
            "throughput_rps": statistics.median(
                [row["throughput_rps"] for row in rows]
            )
            if rows
            else None,
        }
    failures = []
    if any(
        row["success"] != 38 or row["total_requests"] != 38
        for rows in by_concurrency.values()
        for row in rows
    ):
        failures.append("one or more formal runs were not 38/38 successful")
    if not all(medians[concurrency]["ttft_p95"] for concurrency in (1, 4, 8)):
        failures.append("missing run-level p95 TTFT")
    else:
        c4_c1 = medians[4]["ttft_p95"] / medians[1]["ttft_p95"]
        c8_c4 = medians[8]["ttft_p95"] / medians[4]["ttft_p95"]
        c8_c4_thr = medians[8]["throughput_rps"] / medians[4]["throughput_rps"]
        if c4_c1 < 2.0:
            failures.append("median c4/c1 p95 TTFT ratio is below 2.0")
        if c8_c4 < 1.25:
            failures.append("median c8/c4 p95 TTFT ratio is below 1.25")
        if c8_c4_thr > 1.10:
            failures.append("median c8/c4 throughput ratio exceeds 1.10")
    direction_blocks = 0
    for block, rows in by_block.items():
        if all(concurrency in rows for concurrency in (1, 4, 8)):
            if (
                rows[4]["ttft_p95"] > rows[1]["ttft_p95"]
                and rows[8]["ttft_p95"] > rows[4]["ttft_p95"]
            ):
                direction_blocks += 1
    if direction_blocks < 2:
        failures.append("fewer than two blocks show monotonic c1<c4<c8 p95 TTFT")
    return {
        "reproduced": not failures,
        "failures": failures,
        "run_level_medians": medians,
        "direction_blocks": direction_blocks,
        "by_concurrency": by_concurrency,
        "by_block": by_block,
    }


def _component_evidence(records):
    by_concurrency = _group_by(records, "concurrency")
    c1 = by_concurrency.get(1, [])
    c4 = by_concurrency.get(4, [])
    c8 = by_concurrency.get(8, [])
    if not c1 or not c4 or not c8:
        return {}, "missing one or more concurrency groups"
    denominator = (
        statistics.mean(record["server_first_content_s"] for record in c8)
        - statistics.mean(record["server_first_content_s"] for record in c1)
    )
    if denominator <= 0:
        return {}, "server time-to-first-content mean did not increase from c1 to c8"

    components = {
        "runtime": "runtime_pre_executor_s",
        "executor_queue": "executor_queue_ttfc_s",
        "model_execution": "model_execution_ttfc_s",
        "other": "other_ttfc_s",
    }
    by_block = _group_by(records, "block")
    evidence = {}
    for label, metric in components.items():
        c1_stats = _stats([record[metric] for record in c1])
        c4_stats = _stats([record[metric] for record in c4])
        c8_stats = _stats([record[metric] for record in c8])
        mean_difference = c8_stats["mean"] - c1_stats["mean"]
        share = mean_difference / denominator
        c4_c1 = (
            c4_stats["p95"] / c1_stats["p95"]
            if c1_stats["p95"] and c1_stats["p95"] > 0
            else None
        )
        c8_c4 = (
            c8_stats["p95"] / c4_stats["p95"]
            if c4_stats["p95"] and c4_stats["p95"] > 0
            else None
        )
        bootstrap = _hierarchical_bootstrap_diff(c1, c8, metric)
        stable_blocks = 0
        block_shares = {}
        for block, block_rows in by_block.items():
            per_c = _group_by(block_rows, "concurrency")
            if 1 not in per_c or 8 not in per_c:
                continue
            block_denominator = (
                statistics.mean(item["server_first_content_s"] for item in per_c[8])
                - statistics.mean(item["server_first_content_s"] for item in per_c[1])
            )
            if block_denominator <= 0:
                block_shares[str(block)] = None
                continue
            block_share = (
                statistics.mean(item[metric] for item in per_c[8])
                - statistics.mean(item[metric] for item in per_c[1])
            ) / block_denominator
            block_shares[str(block)] = block_share
            if block_share >= 0.25:
                stable_blocks += 1
        ci_low = bootstrap["ci95"][0]
        alignment = (
            c4_c1 is not None
            and c8_c4 is not None
            and c4_c1 >= 1.5
            and c8_c4 >= 1.25
        )
        bootstrap_positive = ci_low is not None and ci_low > 0
        evidence[label] = {
            "metric": metric,
            "c1": c1_stats,
            "c4": c4_stats,
            "c8": c8_stats,
            "mean_difference_c8_minus_c1": mean_difference,
            "component_share": share,
            "p95_ratio_c4_over_c1": c4_c1,
            "p95_ratio_c8_over_c4": c8_c4,
            "bootstrap": bootstrap,
            "block_shares": block_shares,
            "stable_blocks_at_least_25pct": stable_blocks,
            "alignment": alignment,
            "bootstrap_positive": bootstrap_positive,
            "localized": (
                share >= 0.50
                and alignment
                and bootstrap_positive
                and stable_blocks >= 2
            ),
            "multi_stage_eligible": (
                share >= 0.25
                and alignment
                and bootstrap_positive
                and stable_blocks >= 2
            ),
        }
    return evidence, ""


def analyze_formal(output_root: Path):
    manifests = _phase_manifests(output_root, "formal")
    records, invalid = _trace_records(manifests)
    knee = _knee_reproduction(manifests)
    component_evidence, component_problem = _component_evidence(records)
    invalid_fraction = len(invalid) / max(1, len(records) + len(invalid))
    classification = "INCONCLUSIVE"
    rationale = []
    if invalid_fraction > 0.05:
        rationale.append("trace integrity/timestamp invalid fraction exceeds 5%")
    elif not knee["reproduced"]:
        rationale.append("the precommitted knee-reproduction rule failed")
    elif component_problem:
        rationale.append(component_problem)
    else:
        localized = [
            label
            for label, evidence in component_evidence.items()
            if evidence["localized"]
        ]
        multi = [
            label
            for label, evidence in component_evidence.items()
            if evidence["multi_stage_eligible"]
        ]
        if len(multi) >= 2 and sum(
            component_evidence[label]["component_share"] for label in multi
        ) >= 0.70:
            classification = "MULTI-STAGE"
            rationale.append("at least two components meet the multi-stage rule")
        elif "runtime" in localized:
            classification = "LOCALIZED — RUNTIME"
        elif "executor_queue" in localized:
            classification = "LOCALIZED — EXECUTOR QUEUE"
        elif "model_execution" in localized:
            classification = "LOCALIZED — MODEL EXECUTION"
        else:
            classification = "ORDINARY QUEUEING"
            rationale.append("knee reproduced but no component met localization criteria")
    by_concurrency = _group_by(records, "concurrency")
    component_stats = {
        str(concurrency): _component_summary(rows)
        for concurrency, rows in sorted(by_concurrency.items())
    }
    payload = {
        "classification_pre_pooled_control": classification,
        "rationale": rationale,
        "formal_run_count": len(manifests),
        "record_count": len(records),
        "invalid_trace_records": invalid,
        "invalid_fraction": invalid_fraction,
        "knee": knee,
        "component_evidence": component_evidence,
        "component_stats_by_concurrency": component_stats,
        "resource_summaries_by_run": [
            {
                "run_id": row["run_id"],
                "concurrency": row["concurrency"],
                "system": row["system"],
            }
            for rows in knee["by_concurrency"].values()
            for row in rows
        ],
        "not_available": {
            "scheduler_queue_depth": "NOT AVAILABLE: no exposed scheduler queue",
            "batch_state": "NOT AVAILABLE: no continuous batching",
            "gpu_model_execution": "NOT APPLICABLE: locked server device is CPU",
        },
    }
    _write_json(output_root / "processed" / "formal_analysis.json", payload)
    return payload


def _knee_ratio_from_records(records, metric):
    by_concurrency = _group_by(records, "concurrency")
    if 1 not in by_concurrency or 8 not in by_concurrency:
        return None
    c1 = _stats([record.get(metric) for record in by_concurrency[1]])["p95"]
    c8 = _stats([record.get(metric) for record in by_concurrency[8]])["p95"]
    if c1 is None or c1 <= 0 or c8 is None:
        return None
    return c8 / c1


_POOLED_SERVER_METRICS = (
    "server_first_content_s",
    "runtime_pre_executor_s",
    "executor_queue_ttfc_s",
    "model_execution_ttfc_s",
    "other_ttfc_s",
)


def _pooled_control_evidence(primary_records, pooled_records):
    """Compare the primary client's knee against the pooled-client control.

    The precommitted control is only a client/HTTP explanation when its
    end-to-end effect changes materially while every measured server-side
    time-to-first-content component remains stable.  Checking just aggregate
    server time could hide an internal shift that cancels in the aggregate.
    """
    primary_e2e = _knee_ratio_from_records(primary_records, "end_to_end_ttft_s")
    pooled_e2e = _knee_ratio_from_records(pooled_records, "end_to_end_ttft_s")
    e2e_relative_change = None
    if primary_e2e and pooled_e2e:
        e2e_relative_change = pooled_e2e / primary_e2e - 1.0

    component_ratios = {}
    component_relative_changes = {}
    for metric in _POOLED_SERVER_METRICS:
        primary_ratio = _knee_ratio_from_records(primary_records, metric)
        pooled_ratio = _knee_ratio_from_records(pooled_records, metric)
        component_ratios[metric] = {
            "primary_c8_over_c1_p95_ratio": primary_ratio,
            "pooled_c8_over_c1_p95_ratio": pooled_ratio,
        }
        component_relative_changes[metric] = (
            pooled_ratio / primary_ratio - 1.0
            if primary_ratio and pooled_ratio
            else None
        )
    all_server_components_within_10pct = bool(component_relative_changes) and all(
        change is not None and abs(change) <= 0.10
        for change in component_relative_changes.values()
    )
    client_http_artifact = (
        e2e_relative_change is not None
        and abs(e2e_relative_change) > 0.30
        and all_server_components_within_10pct
    )
    return {
        "primary_e2e_c8_over_c1_p95_ratio": primary_e2e,
        "pooled_e2e_c8_over_c1_p95_ratio": pooled_e2e,
        "e2e_ratio_relative_change": e2e_relative_change,
        "server_component_ratios": component_ratios,
        "server_component_ratio_relative_changes": component_relative_changes,
        "all_server_components_within_10pct": all_server_components_within_10pct,
        "client_http_artifact_condition": client_http_artifact,
    }


def analyze_pooled_control(output_root: Path, formal):
    manifests = _phase_manifests(output_root, "pooled")
    records, invalid = _trace_records(manifests)
    primary_records, _ = _trace_records(_phase_manifests(output_root, "formal"))
    by_concurrency = _group_by(records, "concurrency")
    control_complete = (
        len(manifests) == 3
        and sorted(manifest["config"]["concurrency"] for manifest in manifests)
        == [1, 4, 8]
        and not invalid
        and all(len(by_concurrency.get(concurrency, [])) == 38 for concurrency in (1, 4, 8))
    )
    evidence = _pooled_control_evidence(primary_records, records)
    payload = {
        "pooled_run_count": len(manifests),
        "record_count": len(records),
        "invalid_trace_records": invalid,
        "control_complete": control_complete,
        **evidence,
        "formal_classification_before_control": formal.get(
            "classification_pre_pooled_control"
        ),
    }
    _write_json(output_root / "processed" / "pooled_control_analysis.json", payload)
    return payload


def _fmt(value, digits=4):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _write_reports(output_root: Path, overhead, formal, pooled):
    classification = formal["classification_pre_pooled_control"]
    disposition = "INCONCLUSIVE"
    stage3b = "No"
    alternatives = [
        "Cross-process client-to-server ingress was not subtracted; client headers time is reported only client-side.",
        "No continuous batching/scheduler queue depth exists in this runtime, so those states remain unavailable.",
        "GPU model execution is not applicable because the locked server ran on CPU.",
    ]
    if overhead.get("status") != "PASS":
        classification = "DESIGN INVALID"
        disposition = "INCONCLUSIVE"
        alternatives.insert(0, "Tracing overhead gate failed.")
    elif pooled is None or not pooled.get("control_complete"):
        classification = "INCONCLUSIVE"
        disposition = "NEEDS ONE CONTROL"
        alternatives.insert(
            0,
            "The precommitted pooled-client control was incomplete or had invalid traces.",
        )
    elif pooled and pooled.get("client_http_artifact_condition"):
        classification = "INCONCLUSIVE"
        disposition = "INCONCLUSIVE"
        alternatives.insert(
            0,
            "Pooled-client control materially changed the end-to-end knee while server time stayed stable.",
        )
    elif classification.startswith("LOCALIZED") or classification == "MULTI-STAGE":
        disposition = "PROCEED"
        stage3b = "Yes, only for the measured runtime boundary and only with the existing controls preserved."
    elif classification == "ORDINARY QUEUEING":
        disposition = "ORDINARY QUEUEING / DROP"
    else:
        disposition = "INCONCLUSIVE"

    knee = formal.get("knee", {})
    medians = knee.get("run_level_medians", {})
    c1 = medians.get(1, medians.get("1", {}))
    c4 = medians.get(4, medians.get("4", {}))
    c8 = medians.get(8, medians.get("8", {}))
    effect_lines = [
        f"- c1 p95 TTFT: {_fmt(c1.get('ttft_p95'))} s",
        f"- c4 p95 TTFT: {_fmt(c4.get('ttft_p95'))} s",
        f"- c8 p95 TTFT: {_fmt(c8.get('ttft_p95'))} s",
    ]
    evidence = formal.get("component_evidence", {})
    component_lines = []
    for label, item in evidence.items():
        component_lines.append(
            f"- {label}: share={_fmt(item.get('component_share'))}, "
            f"p95 c4/c1={_fmt(item.get('p95_ratio_c4_over_c1'))}, "
            f"p95 c8/c4={_fmt(item.get('p95_ratio_c8_over_c4'))}, "
            f"stable blocks={item.get('stable_blocks_at_least_25pct')}"
        )
    result_text = "\n".join(
        [
            "# A_01 causal-localization result",
            "",
            f"## Final classification",
            "",
            classification,
            "",
            "## Knee reproduction",
            "",
            f"Reproduced: {knee.get('reproduced')}",
            *effect_lines,
            "",
            "## Component evidence",
            "",
            *(component_lines or ["No valid component evidence was available."]),
            "",
            "## Instrumentation gate",
            "",
            f"Status: {overhead.get('status')}",
            f"Failed checks: {overhead.get('failed_checks')}",
            "",
            "## Alternative explanations and boundaries",
            "",
            *alternatives,
            "",
            "## Decision",
            "",
            f"A_01 disposition: {disposition}",
            f"Re-enter Stage 3B causal validation: {stage3b}",
            "",
        ]
    )
    summary_text = "\n".join(
        [
            "# A_01 causal summary",
            "",
            f"1. Knee reproduced: {knee.get('reproduced')}.",
            f"2. Localization: {classification}.",
            f"3. Effect size: c4/c1 p95 TTFT={_fmt((c4.get('ttft_p95') / c1.get('ttft_p95')) if c1.get('ttft_p95') else None)}; c8/c4={_fmt((c8.get('ttft_p95') / c4.get('ttft_p95')) if c4.get('ttft_p95') else None)}.",
            f"4. Stability: {knee.get('direction_blocks')} of 3 blocks showed monotonic p95 TTFT growth.",
            f"5. Strongest alternative explanation: {alternatives[0]}",
            f"6. Concrete runtime boundary localized: {classification.startswith('LOCALIZED') or classification == 'MULTI-STAGE'}.",
            f"7. A_01 disposition: {disposition}.",
            f"8. Return to Stage 3B causal validation: {stage3b}",
            "",
        ]
    )
    (output_root / "result.md").write_text(result_text, encoding="utf-8")
    (output_root / "causal_summary.md").write_text(summary_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "research" / "stage3" / "A_01_causal"),
    )
    parser.add_argument(
        "--phase",
        choices=("overhead", "formal", "pooled", "all"),
        default="all",
    )
    args = parser.parse_args()
    output_root = Path(args.out).resolve()
    overhead = analyze_overhead(output_root)
    if args.phase == "overhead":
        print(json.dumps(overhead, ensure_ascii=False, indent=2))
        return
    formal = analyze_formal(output_root)
    if args.phase == "formal":
        _write_reports(output_root, overhead, formal, None)
        print(json.dumps(formal, ensure_ascii=False, indent=2))
        return
    pooled = analyze_pooled_control(output_root, formal)
    _write_reports(output_root, overhead, formal, pooled)
    print(
        json.dumps(
            {"overhead": overhead, "formal": formal, "pooled": pooled},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
