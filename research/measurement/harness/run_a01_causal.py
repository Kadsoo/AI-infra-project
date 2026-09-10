"""Execute only the locked A_01 causal-localization measurements."""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psutil

MEASUREMENT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MEASUREMENT_ROOT.parent.parent
sys.path.insert(0, str(MEASUREMENT_ROOT))

from harness.a01_causal import formal_specs, overhead_specs, pooled_control_specs
from harness.harness import compute_processed, run_benchmark, save_raw
from workloads.generator import generate, describe_workload


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _source_hashes() -> dict:
    files = {
        "hf_server.py": MEASUREMENT_ROOT / "harness" / "hf_server.py",
        "harness.py": MEASUREMENT_ROOT / "harness" / "harness.py",
        "causal_trace.py": MEASUREMENT_ROOT / "harness" / "causal_trace.py",
        "a01_causal.py": MEASUREMENT_ROOT / "harness" / "a01_causal.py",
        "run_a01_causal.py": Path(__file__),
        "generator.py": MEASUREMENT_ROOT / "workloads" / "generator.py",
        "environment.md": MEASUREMENT_ROOT / "environment.md",
        "LOCKED_CAUSAL_PLAN.md": PROJECT_ROOT
        / "research"
        / "stage3"
        / "A_01_causal"
        / "LOCKED_CAUSAL_PLAN.md",
    }
    return {name: _sha256(path) for name, path in files.items()}


def _client_runtime() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "psutil": psutil.__version__,
        "httpx": httpx.__version__,
    }


async def _health(control: httpx.AsyncClient, base_url: str) -> dict:
    response = await control.get(base_url.rstrip("/") + "/health", timeout=10)
    response.raise_for_status()
    return response.json()


async def _set_trace_mode(
    control: httpx.AsyncClient,
    base_url: str,
    enabled: bool,
) -> None:
    response = await control.post(
        base_url.rstrip("/") + "/stage3/trace/mode",
        json={"enabled": enabled},
        timeout=10,
    )
    response.raise_for_status()
    if response.json().get("trace_enabled") is not enabled:
        raise RuntimeError("server trace mode did not acknowledge requested value")


async def _take_traces(
    control: httpx.AsyncClient,
    base_url: str,
    run_id: str,
) -> list:
    response = await control.get(
        base_url.rstrip("/") + f"/stage3/trace/{run_id}",
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("run_id") != run_id:
        raise RuntimeError("trace endpoint returned a mismatched run id")
    return payload.get("traces", [])


def _specs_for_phase(phase: str) -> list:
    if phase == "overhead":
        return overhead_specs()
    if phase == "formal":
        return formal_specs()
    if phase == "pooled":
        return pooled_control_specs()
    raise ValueError(f"unsupported phase: {phase}")


def _locked_config(
    run_id: str,
    phase: str,
    spec: dict,
    server_pid: int,
    trace_enabled: bool,
    client_mode: str,
    environment_hash: str,
) -> dict:
    return {
        "name": f"a01_{phase}_c{spec['concurrency']}_seed{spec['seed']}",
        "run_id": run_id,
        "phase": phase,
        "block": spec.get("block"),
        "workload_type": "synthetic",
        "request_count": 40,
        "input_tokens": 512,
        "output_tokens": 64,
        "concurrency": spec["concurrency"],
        "arrival_rate": 0,
        "arrival_distribution": "closed",
        "arrival_semantics": "finite_client_gated_burst",
        "prefix_reuse_fraction": 0,
        "seed": spec["seed"],
        "warmup_requests": 2,
        "stream": True,
        "timeout": 180,
        "sampler_interval": 0.3,
        "trace_enabled": trace_enabled,
        "client_mode": client_mode,
        "server_pid": server_pid,
        "environment_hash": environment_hash[:12],
    }


async def _run_one(
    control: httpx.AsyncClient,
    base_url: str,
    output_root: Path,
    session_id: str,
    phase: str,
    sequence: int,
    spec: dict,
    source_hashes: dict,
) -> dict:
    trace_enabled = bool(spec.get("trace_enabled", True))
    client_mode = "pooled" if phase == "pooled" else "per_request"
    await _set_trace_mode(control, base_url, trace_enabled)
    health_before = await _health(control, base_url)
    if not health_before.get("loaded") or health_before.get("mock"):
        raise RuntimeError("server must be loaded real inference, not mock mode")
    if health_before.get("device") != "cpu":
        raise RuntimeError("locked A_01 session requires CPU inference")
    server_pid = health_before.get("pid")
    if not isinstance(server_pid, int):
        raise RuntimeError("server health did not report an integer PID")

    run_id = (
        f"{session_id}_{phase}_{sequence:02d}_c{spec['concurrency']}_"
        f"seed{spec['seed']}_{'on' if trace_enabled else 'off'}"
    )
    config = _locked_config(
        run_id=run_id,
        phase=phase,
        spec=spec,
        server_pid=server_pid,
        trace_enabled=trace_enabled,
        client_mode=client_mode,
        environment_hash=source_hashes["environment.md"],
    )
    workload = generate(
        workload_type="synthetic",
        n=config["request_count"],
        input_tokens=config["input_tokens"],
        output_tokens=config["output_tokens"],
        arrival_rate=0,
        arrival_distribution="closed",
        prefix_reuse_fraction=0,
        seed=config["seed"],
    )
    started_at = _now()
    result = await run_benchmark(
        base_url=base_url,
        workload=workload,
        config=config,
        concurrency=config["concurrency"],
        warmup_requests=config["warmup_requests"],
        stream=True,
        sampler_interval=config["sampler_interval"],
        timeout=config["timeout"],
        arrival_distribution="closed",
        client_mode=client_mode,
    )
    health_after = await _health(control, base_url)
    if health_after.get("pid") != server_pid:
        raise RuntimeError("server PID changed during a locked run")
    traces = await _take_traces(control, base_url, run_id) if trace_enabled else []

    raw_dir = output_root / "raw"
    processed_dir = output_root / "processed"
    req_path, sys_path, result_path = save_raw(result, raw_dir)
    processed = compute_processed(result)
    _write_json(processed_dir / f"{run_id}_processed.json", processed)
    _write_json(raw_dir / f"{run_id}_server_traces.json", traces)

    expected_ids = {
        request.request_id
        for request in result.requests
        if request.success
    }
    observed_ids = {
        trace.get("request_id")
        for trace in traces
    }
    trace_integrity = {
        "enabled": trace_enabled,
        "expected_successful_request_count": len(expected_ids),
        "observed_trace_count": len(traces),
        "missing_request_ids": sorted(expected_ids - observed_ids),
        "unexpected_request_ids": sorted(observed_ids - expected_ids),
        "complete": (
            not trace_enabled
            or (
                len(expected_ids) == len(traces)
                and expected_ids == observed_ids
            )
        ),
    }
    manifest = {
        "run_id": run_id,
        "session_id": session_id,
        "phase": phase,
        "sequence": sequence,
        "started_at_utc": started_at,
        "finished_at_utc": _now(),
        "command": sys.argv,
        "base_url": base_url,
        "config": config,
        "workload": describe_workload(workload),
        "source_hashes": source_hashes,
        "client_runtime": _client_runtime(),
        "server_health_before": health_before,
        "server_health_after": health_after,
        "trace_integrity": trace_integrity,
        "artifacts": {
            "requests_csv": str(req_path),
            "system_csv": str(sys_path),
            "raw_json": str(result_path),
            "processed_json": str(processed_dir / f"{run_id}_processed.json"),
            "server_traces_json": str(raw_dir / f"{run_id}_server_traces.json"),
        },
    }
    _write_json(raw_dir / f"{run_id}_manifest.json", manifest)
    _append_jsonl(
        output_root / "logs" / f"{session_id}_events.jsonl",
        {
            "event": "run_complete",
            "at_utc": _now(),
            "run_id": run_id,
            "phase": phase,
            "trace_integrity": trace_integrity,
        },
    )
    if trace_enabled and not trace_integrity["complete"]:
        raise RuntimeError(f"trace integrity failure for {run_id}")
    return manifest


async def _main_async(args) -> None:
    output_root = Path(args.out).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_hashes = _source_hashes()
    session_id = args.session_id or f"a01-{int(time.time())}"
    if args.phase == "formal":
        gate_path = output_root / "processed" / "overhead_gate.json"
        if not gate_path.exists():
            raise RuntimeError("formal phase requires processed/overhead_gate.json")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") != "PASS":
            raise RuntimeError("formal phase is blocked because overhead gate did not pass")

    async with httpx.AsyncClient() as control:
        initial_health = await _health(control, args.base_url)
        session_manifest = {
            "session_id": session_id,
            "phase": args.phase,
            "started_at_utc": _now(),
            "base_url": args.base_url,
            "command": sys.argv,
            "source_hashes": source_hashes,
            "client_runtime": _client_runtime(),
            "server_health_initial": initial_health,
        }
        _write_json(
            output_root / "logs" / f"{session_id}_{args.phase}_session.json",
            session_manifest,
        )
        manifests = []
        for sequence, spec in enumerate(_specs_for_phase(args.phase), start=1):
            manifest = await _run_one(
                control=control,
                base_url=args.base_url,
                output_root=output_root,
                session_id=session_id,
                phase=args.phase,
                sequence=sequence,
                spec=spec,
                source_hashes=source_hashes,
            )
            manifests.append(manifest)
            print(
                f"[a01] {manifest['run_id']} "
                f"{manifest['trace_integrity']['expected_successful_request_count']} "
                f"successful requests"
            )
        _write_json(
            output_root / "logs" / f"{session_id}_{args.phase}_run_index.json",
            {"session_id": session_id, "phase": args.phase, "runs": manifests},
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("overhead", "formal", "pooled"),
    )
    parser.add_argument(
        "--out",
        default=str(
            PROJECT_ROOT / "research" / "stage3" / "A_01_causal"
        ),
    )
    parser.add_argument("--session-id")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
