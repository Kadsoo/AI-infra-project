"""Measurement Repair Gate Runner — L0 (OFF) vs L1 (Minimal) paired randomized.

Implements Stage 3M gate for c=1, c=4 with interleaved OFF/L1, matched seeds.
Supports both mock and real server; fixed workload synthetic 512/64.
"""
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
import statistics

import httpx
import psutil

MEASUREMENT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MEASUREMENT_ROOT.parent.parent
sys.path.insert(0, str(MEASUREMENT_ROOT))

from harness.harness import compute_processed, run_benchmark, save_raw
from workloads.generator import generate, describe_workload

def _now():
    return datetime.now(timezone.utc).isoformat()

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        json.dump(payload, h, indent=2, ensure_ascii=False)

def _append_jsonl(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as h:
        h.write(json.dumps(payload, ensure_ascii=False) + "\n")

def _source_hashes():
    files = {
        "hf_server.py": MEASUREMENT_ROOT / "harness" / "hf_server.py",
        "harness.py": MEASUREMENT_ROOT / "harness" / "harness.py",
        "causal_trace.py": MEASUREMENT_ROOT / "harness" / "causal_trace.py",
        "minimal_trace.py": MEASUREMENT_ROOT / "harness" / "minimal_trace.py",
        "run_measurement_repair_gate.py": Path(__file__),
        "generator.py": MEASUREMENT_ROOT / "workloads" / "generator.py",
    }
    return {n: _sha256(p) for n,p in files.items()}

def _client_runtime():
    return {"python": sys.version, "platform": platform.platform(), "pid": os.getpid(), "psutil": psutil.__version__, "httpx": httpx.__version__}

async def _health(client, base_url):
    r = await client.get(base_url.rstrip("/") + "/health", timeout=10)
    r.raise_for_status()
    return r.json()

async def _set_level(client, base_url, level, sample_ratio=1.0):
    r = await client.post(base_url.rstrip("/") + "/stage3/trace/level", json={"level": level, "sample_ratio": sample_ratio}, timeout=10)
    r.raise_for_status()
    j = r.json()
    if j.get("trace_level") != level:
        raise RuntimeError(f"level not ack: wanted {level} got {j}")

async def _take_traces(client, base_url, run_id):
    r = await client.get(base_url.rstrip("/") + f"/stage3/trace/{run_id}", timeout=30)
    r.raise_for_status()
    p = r.json()
    return p.get("traces", p.get("counters", []))

def gate_specs(concurrencies="1,4", seeds_per_conc=2, request_count=40):
    # Use seeds 3101-3104 for c1, 3401-3404 for c4
    mapping = {1: [3101,3102,3103,3104], 4: [3401,3402,3403,3404], 8: [3801,3802]}
    specs=[]
    concs=[int(c) for c in concurrencies.split(",")]
    for conc in concs:
        seeds=mapping.get(conc, [3101,3102])[:seeds_per_conc]
        for seed in seeds:
            # each seed has OFF->L1 and L1->OFF? No, each seed is one pair; order alternates
            # We'll create pairs: even index OFF->L1, odd L1->OFF
            idx=seeds.index(seed)
            if idx %2==0:
                order=[0,1]  # OFF then L1
            else:
                order=[1,0]
            for level in order:
                specs.append({"concurrency":conc, "seed":seed, "level":level, "order": order})
    return specs

def interleave_specs(specs):
    # Already interleaved by conc? Reorder to interleave conc: e.g., c1 pair, c4 pair alternating
    # Simple: sort by seed then level order already gives OFF->L1 adjacent
    # For global interleaving, we want: c1 seed3101 OFF/L1, c1 seed3102 L1/OFF, c4 seed3401 OFF/L1, etc.
    # Current gate_specs already in that order if concs=[1,4] iterating conc outer then seed inner
    # That gives all c1 first then c4. To interleave, we can zip.
    # Let's produce: take specs grouped by conc, then round-robin
    from collections import defaultdict
    by_conc=defaultdict(list)
    for s in specs:
        by_conc[s["concurrency"]].append(s)
    # Now interleave groups
    interleaved=[]
    max_len=max(len(v) for v in by_conc.values())
    for i in range(max_len):
        for conc in sorted(by_conc.keys()):
            if i < len(by_conc[conc]):
                interleaved.append(by_conc[conc][i])
    return interleaved

async def _run_one(client, base_url, output_root, session_id, seq, spec, source_hashes, request_count):
    level = spec["level"]
    conc = spec["concurrency"]
    seed = spec["seed"]
    level_name = "off" if level==0 else "l1"
    # set level
    await _set_level(client, base_url, level)
    health_before = await _health(client, base_url)
    if not health_before.get("loaded"):
        raise RuntimeError("server not loaded")
    server_pid = health_before.get("pid")
    run_id = f"repair-{session_id}_{seq:02d}_c{conc}_seed{seed}_{level_name}"
    config = {
        "name": f"repair_c{conc}_seed{seed}_{level_name}",
        "run_id": run_id,
        "phase": "repair_gate",
        "concurrency": conc,
        "seed": seed,
        "level": level,
        "level_name": level_name,
        "request_count": request_count,
        "input_tokens": 512,
        "output_tokens": 64,
        "arrival_distribution": "closed",
        "warmup_requests": 2,
        "stream": True,
        "timeout": 180,
        "sampler_interval": 0.3,
        "client_mode": "per_request",
        "server_pid": server_pid,
        "environment_hash": source_hashes["generator.py"][:12],
    }
    workload = generate(workload_type="synthetic", n=request_count, input_tokens=512, output_tokens=64, arrival_rate=0, arrival_distribution="closed", prefix_reuse_fraction=0, seed=seed)
    started=_now()
    result = await run_benchmark(base_url=base_url, workload=workload, config=config, concurrency=conc, warmup_requests=2, stream=True, sampler_interval=0.3, timeout=180, arrival_distribution="closed", client_mode="per_request")
    health_after = await _health(client, base_url)
    if health_after.get("pid") != server_pid:
        raise RuntimeError("PID changed")
    traces = await _take_traces(client, base_url, run_id) if level!=0 else []
    raw_dir = output_root / "raw"
    processed_dir = output_root / "processed"
    req_path, sys_path, raw_path = save_raw(result, raw_dir)
    processed = compute_processed(result)
    _write_json(processed_dir / f"{run_id}_processed.json", processed)
    _write_json(raw_dir / f"{run_id}_server_traces.json", traces)
    # host state
    manifest = {
        "run_id": run_id,
        "session_id": session_id,
        "seq": seq,
        "started_at_utc": started,
        "finished_at_utc": _now(),
        "level": level,
        "level_name": level_name,
        "concurrency": conc,
        "seed": seed,
        "config": config,
        "workload": describe_workload(workload),
        "source_hashes": source_hashes,
        "client_runtime": _client_runtime(),
        "server_health_before": health_before,
        "server_health_after": health_after,
        "artifacts": {"requests_csv": str(req_path), "system_csv": str(sys_path), "raw_json": str(raw_path)},
        "trace_count": len(traces),
    }
    _write_json(raw_dir / f"{run_id}_manifest.json", manifest)
    _append_jsonl(output_root / "logs" / f"{session_id}_events.jsonl", {"event":"run_complete","run_id":run_id,"level":level_name,"concurrency":conc,"seed":seed})
    return manifest, processed

def evaluate_gate(processed_list):
    # processed_list: list of dicts with config and aggregates
    metrics = ("throughput_rps", "ttft_p95", "latency_p95")  # ttft_p95 field name mapping
    # Map to processed keys: throughput_rps, ttft p95, latency p95
    # processed has throughput_rps, ttft: {p95}, latency:{p95}
    grouped = {}
    for p in processed_list:
        cfg = p["config"]
        conc = cfg["concurrency"]
        seed = cfg["seed"]
        level = cfg["level"]
        is_off = (level==0)
        # extract metrics
        thr = p["throughput_rps"]
        ttft = p["ttft"]["p95"] if p.get("ttft") else None
        lat = p["latency"]["p95"] if p.get("latency") else None
        grouped.setdefault(conc, {}).setdefault(seed, {})[is_off] = {"throughput_rps":thr, "ttft_p95":ttft, "latency_p95":lat, "raw":p}
    details={}
    failed=[]
    for conc in sorted(grouped):
        pairs=grouped[conc]
        deltas={m:[] for m in metrics}
        missing=[]
        for seed, modes in sorted(pairs.items()):
            if True not in modes or False not in modes:
                missing.append(seed)
                continue
            for m in metrics:
                off=modes[True].get(m)
                on=modes[False].get(m)
                if off is None or on is None or off<=0 or on<=0:
                    failed.append(f"c{conc} seed{seed} invalid {m}")
                    continue
                deltas[m].append(on/off-1.0)
        if missing:
            failed.append(f"c{conc} missing pairs seeds {missing}")
        checks={}
        for m, ds in deltas.items():
            median = statistics.median(ds) if ds else None
            indiv_ok = bool(ds) and all(abs(d)<=0.10 for d in ds)
            median_ok = median is not None and abs(median) <=0.05
            checks[m] = {"pair_relative_deltas": ds, "median_relative_delta": median, "median_limit_ok": median_ok, "individual_limit_ok": indiv_ok}
            if not median_ok:
                failed.append(f"c{conc} {m} median paired diff exceeds 5% median={median}")
            if not indiv_ok:
                failed.append(f"c{conc} {m} individual paired diff exceeds 10% deltas={ds}")
        details[str(conc)]=checks
    # multiplier
    mult_checks={}
    mode_medians={True:{}, False:{}}
    for is_off in (True, False):  # True means OFF? Wait our key is True=OFF (level 0)
        for conc in grouped:
            vals=[]
            for seed, modes in grouped[conc].items():
                if is_off in modes:
                    v=modes[is_off].get("ttft_p95")
                    if isinstance(v,(int,float)) and v>0:
                        vals.append(v)
            if vals:
                mode_medians[is_off][conc]=statistics.median(vals)
    # Actually is_off True = OFF, False = ON (since True maps to OFF? confusion)
    # Our grouped key: True = OFF (level 0), False = ON (level 1) because we used is_off flag.
    # So OFF medians are mode_medians[True], ON is mode_medians[False]
    for label, low, high in (("c4_over_c1",1,4),):
        if low not in mode_medians[True] or high not in mode_medians[True]:
            if 1 in mode_medians[True] and 4 in mode_medians[True]:
                pass
            else:
                failed.append(f"missing OFF multiplier for {label}")
                continue
        if low not in mode_medians[False] or high not in mode_medians[False]:
            failed.append(f"missing ON multiplier for {label}")
            continue
        off_mult=mode_medians[True][high]/mode_medians[True][low]
        on_mult=mode_medians[False][high]/mode_medians[False][low]
        rel=on_mult/off_mult-1
        mult_checks[label]={"off_multiplier":off_mult,"on_multiplier":on_mult,"relative_change":rel,"limit_ok":abs(rel)<=0.10}
        if abs(rel)>0.10:
            failed.append(f"{label} ON/OFF multiplier diff exceeds 10% {rel}")
    return {"status":"PASS" if not failed else "FAIL","by_concurrency":details,"ttft_multiplier_checks":mult_checks,"failed_checks":failed}

async def _main_async(args):
    output_root=Path(args.out).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    source_hashes=_source_hashes()
    session_id=args.session_id or f"repair-{int(time.time())}"
    # health check
    async with httpx.AsyncClient() as ctrl:
        h=await _health(ctrl, args.base_url)
        print(f"[gate] server health {h['pid']} loaded={h['loaded']} mock={h['mock']} level={h.get('trace_level')}")
        if not h.get("loaded"):
            raise RuntimeError("need loaded server")
        if h.get("mock"):
            print("[gate] WARNING: running on mock server (for pipeline validation, not formal gate)")
        # build specs
        specs_uninterleaved=gate_specs(args.concurrency, seeds_per_conc=args.seeds_per_conc, request_count=args.request_count)
        specs=interleave_specs(specs_uninterleaved)
        print(f"[gate] running {len(specs)} runs: {specs}")
        manifests=[]
        processed_list=[]
        for seq, spec in enumerate(specs, start=1):
            print(f"[gate] {seq}/{len(specs)} c{spec['concurrency']} seed{spec['seed']} level {spec['level']} ({'off' if spec['level']==0 else 'l1'})")
            man, proc = await _run_one(ctrl, args.base_url, output_root, session_id, seq, spec, source_hashes, request_count=args.request_count)
            manifests.append(man)
            processed_list.append(proc)
            # small gap between runs to let host settle
            await asyncio.sleep(1.0)
        gate=evaluate_gate(processed_list)
        _write_json(output_root / "processed" / "overhead_gate.json", gate)
        _write_json(output_root / "logs" / f"{session_id}_run_index.json", {"session_id":session_id,"runs":manifests,"gate":gate})
        print(f"[gate] result {gate['status']}")
        if gate["failed_checks"]:
            print(" failed:", gate["failed_checks"])
        return gate

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--out", default=str(MEASUREMENT_ROOT.parent / "measurement_repair"))
    p.add_argument("--concurrency", default="1,4")
    p.add_argument("--seeds-per-conc", type=int, default=2)
    p.add_argument("--request-count", type=int, default=40)
    p.add_argument("--session-id")
    args=p.parse_args()
    asyncio.run(_main_async(args))

if __name__=="__main__":
    main()
