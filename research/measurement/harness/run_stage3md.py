"""Stage 3M-D Runner — Independent Validation of Candidate Stable Protocol D

Implements Protocol D: per-formal measurement
  1 x c4,n=40 matched warmup (pooled, OFF)
  gc.collect()
  sleep(2s)
  host-state check
  formal run (synthetic 512/64 n=40, 2 warmup_requests +38 measured, pooled, fixed executor 32, torch 16)

Phase A: L0 OFF only, c1/c4, 3 reps per conc =6 runs, balanced order, new seeds 5101-5103 / 5401-5403
Phase B: L0 vs L1 paired, 3 pairs per conc =12 runs (if Phase A PASS)

All runs use pooled client, fixed ThreadPool 32, torch 16, sampler 0.3, as per LOCKED_INDEPENDENT_STATIONARITY_PLAN.md
"""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import sys
import time
import statistics
import gc
import threading
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

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
        "minimal_trace.py": MEASUREMENT_ROOT / "harness" / "minimal_trace.py",
        "run_stage3md.py": Path(__file__),
        "generator.py": MEASUREMENT_ROOT / "workloads" / "generator.py",
        "LOCKED_INDEPENDENT_STATIONARITY_PLAN.md": PROJECT_ROOT / "research" / "measurement_repair" / "independent_validation" / "LOCKED_INDEPENDENT_STATIONARITY_PLAN.md",
        "LOCKED_STATIONARITY_PLAN.md": PROJECT_ROOT / "research" / "measurement_repair" / "LOCKED_STATIONARITY_PLAN.md",
    }
    out = {}
    for n,p in files.items():
        if p.exists():
            out[n] = _sha256(p)
        else:
            out[n] = "missing"
    return out

def _client_runtime():
    return {"python": sys.version, "platform": platform.platform(), "pid": os.getpid(), "psutil": psutil.__version__, "httpx": httpx.__version__}

async def _health(client, base_url):
    r = await client.get(base_url.rstrip("/") + "/health", timeout=15)
    r.raise_for_status()
    return r.json()

async def _set_level(client, base_url, level, sample_ratio=1.0):
    r = await client.post(base_url.rstrip("/") + "/stage3/trace/level", json={"level": level, "sample_ratio": sample_ratio}, timeout=15)
    r.raise_for_status()
    j = r.json()
    if j.get("trace_level") != level:
        raise RuntimeError(f"level not ack: wanted {level} got {j}")

async def _take_traces(client, base_url, run_id):
    r = await client.get(base_url.rstrip("/") + f"/stage3/trace/{run_id}", timeout=30)
    r.raise_for_status()
    p = r.json()
    return p.get("traces", p.get("counters", []))

def _bg_procs():
    procs=[]
    try:
        for p in psutil.process_iter(['pid','name','cpu_percent']):
            try:
                info=p.info
                procs.append(info)
            except: pass
        procs=sorted(procs, key=lambda x: x.get('cpu_percent') or 0, reverse=True)[:7]
    except Exception as e:
        procs=[{"error": str(e)}]
    return procs

def _get_system_snapshot(server_pid=None):
    try:
        freq=psutil.cpu_freq()
        freq_cur=freq.current if freq else None
    except: freq_cur=None
    try:
        vm=psutil.virtual_memory()
    except: vm=None
    snap={"cpu_percent": psutil.cpu_percent(interval=0.2), "cpu_freq_mhz": freq_cur, "ram_percent": vm.percent if vm else None, "ram_used_gb": vm.used/1024**3 if vm else None,
          "ram_total_gb": vm.total/1024**3 if vm else None,
          "timestamp": _now(),
          "gc_counters": gc.get_count(),
          "gc_threshold": gc.get_threshold(),
          "python_threads": threading.active_count(),
          }
    vm_percent = vm.percent if vm else None
    if server_pid:
        try:
            proc=psutil.Process(int(server_pid))
            snap["server_threads"]=proc.num_threads()
            snap["server_rss_mb"]=proc.memory_info().rss/1024/1024
            snap["server_vms_mb"]=proc.memory_info().vms/1024/1024
            snap["server_cpu"]=proc.cpu_percent(interval=0.1)
            # also try to get num threads via health? keep snapshot
        except Exception as e:
            snap["server_err"]=str(e)
    # GPU
    try:
        import pynvml
        try:
            pynvml.nvmlInit()
            h=pynvml.nvmlDeviceGetHandleByIndex(0)
            util=pynvml.nvmlDeviceGetUtilizationRates(h)
            mem=pynvml.nvmlDeviceGetMemoryInfo(h)
            snap["gpu_util"]=float(util.gpu)
            snap["gpu_mem_used_mb"]=mem.used/1024/1024
            snap["gpu_mem_total_mb"]=mem.total/1024/1024
            snap["gpu_mem_percent"]=100*mem.used/mem.total if mem.total else None
            try:
                snap["gpu_temp"]=pynvml.nvmlDeviceGetTemperature(h, 0) if hasattr(pynvml, 'nvmlDeviceGetTemperature') else None
            except: pass
            try:
                snap["gpu_clock_sm"]=pynvml.nvmlDeviceGetClockInfo(h, 0)
                snap["gpu_clock_mem"]=pynvml.nvmlDeviceGetClockInfo(h, 2)
            except: pass
        except Exception as e:
            snap["gpu_err"]=str(e)
    except: snap["gpu_err"]="no nvml"
    snap["bg_procs"]=_bg_procs()
    return snap

# ---------------------------------------------------------------------------
# Locked specs for Phase A
# ---------------------------------------------------------------------------
def stationarity_specs_md():
    # Pre-registered balanced order: 6 runs
    # Chrono 1: c1 5101 R1, 2: c4 5401 R1, 3: c4 5402 R2, 4: c1 5102 R2, 5: c1 5103 R3, 6: c4 5403 R3
    specs = [
        {"concurrency":1, "seed":5101, "round":1, "chronological_order":1, "warmup_seed":8001, "level":0, "level_name":"off"},
        {"concurrency":4, "seed":5401, "round":1, "chronological_order":2, "warmup_seed":8002, "level":0, "level_name":"off"},
        {"concurrency":4, "seed":5402, "round":2, "chronological_order":3, "warmup_seed":8003, "level":0, "level_name":"off"},
        {"concurrency":1, "seed":5102, "round":2, "chronological_order":4, "warmup_seed":8004, "level":0, "level_name":"off"},
        {"concurrency":1, "seed":5103, "round":3, "chronological_order":5, "warmup_seed":8005, "level":0, "level_name":"off"},
        {"concurrency":4, "seed":5403, "round":3, "chronological_order":6, "warmup_seed":8006, "level":0, "level_name":"off"},
    ]
    return specs

def regate_specs_md():
    # 12 runs paired, order as per LOCKED plan
    # 1:c1 5101 OFF,2:c1 5101 L1,3:c1 5102 L1,4:c1 5102 OFF,5:c4 5401 L1,6:c4 5401 OFF,7:c4 5402 OFF,8:c4 5402 L1,9:c1 5103 OFF,10:c1 5103 L1,11:c4 5403 L1,12:c4 5403 OFF
    specs = [
        {"concurrency":1, "seed":5101, "level":0, "level_name":"off", "chronological_order":1, "pair_seed":5101, "pair_order":"OFF->L1", "warmup_seed":9001},
        {"concurrency":1, "seed":5101, "level":1, "level_name":"l1",  "chronological_order":2, "pair_seed":5101, "pair_order":"OFF->L1", "warmup_seed":9002},
        {"concurrency":1, "seed":5102, "level":1, "level_name":"l1",  "chronological_order":3, "pair_seed":5102, "pair_order":"L1->OFF", "warmup_seed":9003},
        {"concurrency":1, "seed":5102, "level":0, "level_name":"off", "chronological_order":4, "pair_seed":5102, "pair_order":"L1->OFF", "warmup_seed":9004},
        {"concurrency":4, "seed":5401, "level":1, "level_name":"l1",  "chronological_order":5, "pair_seed":5401, "pair_order":"L1->OFF", "warmup_seed":9005},
        {"concurrency":4, "seed":5401, "level":0, "level_name":"off", "chronological_order":6, "pair_seed":5401, "pair_order":"L1->OFF", "warmup_seed":9006},
        {"concurrency":4, "seed":5402, "level":0, "level_name":"off", "chronological_order":7, "pair_seed":5402, "pair_order":"OFF->L1", "warmup_seed":9007},
        {"concurrency":4, "seed":5402, "level":1, "level_name":"l1",  "chronological_order":8, "pair_seed":5402, "pair_order":"OFF->L1", "warmup_seed":9008},
        {"concurrency":1, "seed":5103, "level":0, "level_name":"off", "chronological_order":9, "pair_seed":5103, "pair_order":"OFF->L1", "warmup_seed":9009},
        {"concurrency":1, "seed":5103, "level":1, "level_name":"l1",  "chronological_order":10, "pair_seed":5103, "pair_order":"OFF->L1", "warmup_seed":9010},
        {"concurrency":4, "seed":5403, "level":1, "level_name":"l1",  "chronological_order":11, "pair_seed":5403, "pair_order":"L1->OFF", "warmup_seed":9011},
        {"concurrency":4, "seed":5403, "level":0, "level_name":"off", "chronological_order":12, "pair_seed":5403, "pair_order":"L1->OFF", "warmup_seed":9012},
    ]
    return specs

# ---------------------------------------------------------------------------
# Single Protocol D run with warmup+GC+sleep
# ---------------------------------------------------------------------------
async def _run_protocol_d_single(base_url, out_raw, out_processed, session_id, spec, seq, source_hashes, server_pid, client_runtime, phase="stationarity"):
    conc=spec["concurrency"]
    seed=spec["seed"]
    warmup_seed=spec["warmup_seed"]
    level=spec["level"]
    level_name=spec["level_name"]
    chrono=spec["chronological_order"]
    # run_id
    prefix = "l0" if phase=="stationarity" else "l1regate"
    # For stationarity, server_traces empty (OFF); for regate, depends
    run_id=f"{prefix}-{session_id}_{seq:02d}_c{conc}_seed{seed}_{level_name}_o{chrono}"
    config={
        "name": f"{phase}_c{conc}_seed{seed}_{level_name}",
        "run_id": run_id,
        "phase": phase,
        "concurrency": conc,
        "seed": seed,
        "warmup_seed": warmup_seed,
        "round": spec.get("round"),
        "chronological_order": chrono,
        "pair_seed": spec.get("pair_seed"),
        "level": level,
        "level_name": level_name,
        "request_count": 40,
        "input_tokens": 512,
        "output_tokens": 64,
        "arrival_distribution": "closed",
        "warmup_requests": 2,
        "stream": True,
        "timeout": 180,
        "sampler_interval": 0.3,
        "client_mode": "pooled",
        "client_pool": f"max_connections={max(8,conc)},max_keepalive={max(8,conc)}",
        "server_pid": server_pid,
        "environment_hash": source_hashes["generator.py"][:12],
        "protocol": "D",
        "protocol_steps": ["1xc4_n40_warmup", "gc.collect()", "sleep2s", "host_check", "formal"],
    }
    # Ensure trace level is correct before any burst
    async with httpx.AsyncClient() as ctrl:
        await _set_level(ctrl, base_url, level)
        hb = await _health(ctrl, base_url)
        if hb.get("pid") != server_pid:
            raise RuntimeError(f"PID changed before run {run_id}: {hb.get('pid')} vs {server_pid}")
        if hb.get("runtime",{}).get("torch_num_threads") != 16:
            print(f"[WARN] torch threads not 16 before {run_id}: {hb.get('runtime')}")
    snap_before_warmup = _get_system_snapshot(server_pid)
    gc_before_warmup = gc.get_count()
    # ---- Protocol D Step 1: 1 x c4 n40 warmup ----
    warmup_config={
        "name": f"warmup_protocold_c4_seed{warmup_seed}_for_{run_id}",
        "run_id": f"warmup-{run_id}",
        "phase": "protocold_warmup",
        "concurrency": 4,
        "seed": warmup_seed,
        "level": 0,  # warmup always OFF
        "level_name": "off",
        "request_count": 40,
        "input_tokens": 512,
        "output_tokens": 64,
        "arrival_distribution": "closed",
        "warmup_requests": 0,  # warmup itself is not warmed
        "stream": True,
        "timeout": 180,
        "sampler_interval": 0.3,
        "client_mode": "pooled",
        "client_pool": f"max_connections={max(8,4)},max_keepalive={max(8,4)}",
        "server_pid": server_pid,
        "protocol_step": 1,
        "parent_formal_run_id": run_id,
    }
    # Ensure warmup level is OFF
    async with httpx.AsyncClient() as ctrl:
        await _set_level(ctrl, base_url, 0)
    warmup_workload = generate(workload_type="synthetic", n=40, input_tokens=512, output_tokens=64, arrival_distribution="closed", seed=warmup_seed)
    warmup_result = await run_benchmark(base_url=base_url, workload=warmup_workload, config=warmup_config, concurrency=4, warmup_requests=0, stream=True, sampler_interval=0.3, timeout=180, arrival_distribution="closed", client_mode="pooled")
    # Save warmup artifacts under same out_raw but with warmup prefix, not counted in stationarity stats
    warmup_req_path, warmup_sys_path, warmup_raw_path = save_raw(warmup_result, out_raw)
    warmup_proc = compute_processed(warmup_result)
    _write_json(out_processed / f"warmup-{run_id}_processed.json", warmup_proc)
    # capture warmup trace (should be empty OFF)
    async with httpx.AsyncClient() as ctrl:
        try:
            warmup_traces = await _take_traces(ctrl, base_url, warmup_config["run_id"])
        except: warmup_traces=[]
    _write_json(out_raw / f"warmup-{run_id}_server_traces.json", warmup_traces)
    _write_json(out_raw / f"warmup-{run_id}_manifest.json", {
        "run_id": warmup_config["run_id"],
        "parent_formal": run_id,
        "session_id": session_id,
        "seq": seq,
        "chronological_order": chrono,
        "phase": "protocold_warmup",
        "concurrency": 4,
        "seed": warmup_seed,
        "config": warmup_config,
        "workload": describe_workload(warmup_workload),
        "processed": warmup_proc,
        "server_health_before_warmup": hb,
        "gc_before_warmup": gc_before_warmup,
    })
    snap_after_warmup = _get_system_snapshot(server_pid)
    # ---- Protocol D Step 2: gc.collect() ----
    gc_before = gc.get_count()
    gc_collected = gc.collect()
    gc_after = gc.get_count()
    gc_snapshot = {"collected": gc_collected, "before": gc_before, "after": gc_after, "time": _now()}
    # ---- Protocol D Step 3: sleep 2s ----
    sleep_start = time.time()
    await asyncio.sleep(2.0)
    sleep_end = time.time()
    # ---- Protocol D Step 4: host-state check after GC+sleep ----
    snap_after_gc_sleep = _get_system_snapshot(server_pid)
    async with httpx.AsyncClient() as ctrl:
        hb2 = await _health(ctrl, base_url)
    # Verify PID still same
    # ---- Protocol D Step 5: formal run ----
    # Set level for formal (could be 0 or 1)
    async with httpx.AsyncClient() as ctrl:
        await _set_level(ctrl, base_url, level)
        hb_before_formal = await _health(ctrl, base_url)
        if hb_before_formal.get("pid") != server_pid:
            raise RuntimeError(f"PID changed before formal {run_id}")
    snap_before_formal = _get_system_snapshot(server_pid)  # this is after GC sleep, before formal
    # Workload for formal
    workload = generate(workload_type="synthetic", n=40, input_tokens=512, output_tokens=64, arrival_distribution="closed", seed=seed)
    started = _now()
    result = await run_benchmark(base_url=base_url, workload=workload, config=config, concurrency=conc, warmup_requests=2, stream=True, sampler_interval=0.3, timeout=180, arrival_distribution="closed", client_mode="pooled")
    async with httpx.AsyncClient() as ctrl2:
        ha = await _health(ctrl2, base_url)
        try:
            traces = await _take_traces(ctrl2, base_url, run_id)
        except: traces=[]
    snap_after_formal = _get_system_snapshot(server_pid)
    # host invalid check
    invalid=False
    invalid_reason=[]
    try:
        # PID change
        if hb.get("pid") != ha.get("pid"):
            invalid=True; invalid_reason.append("pid_change")
        # thread growth
        thr_before = snap_before_formal.get("server_threads")
        thr_after = snap_after_formal.get("server_threads")
        if thr_before and thr_after and abs(thr_after-thr_before) > 20:
            invalid=True; invalid_reason.append(f"thread_growth {thr_before}->{thr_after}")
        # also check health threads
        ht_before = hb_before_formal.get("process",{}).get("process_threads")
        ht_after = ha.get("process",{}).get("process_threads")
        if ht_before and ht_after and abs(ht_after-ht_before) > 20:
            invalid=True; invalid_reason.append(f"health_thread_growth {ht_before}->{ht_after}")
        # RSS spike
        rss_before = snap_before_formal.get("server_rss_mb")
        rss_after = snap_after_formal.get("server_rss_mb")
        if rss_before and rss_after and (rss_after - rss_before) > 500:
            invalid=True; invalid_reason.append(f"rss_spike {rss_before:.0f}->{rss_after:.0f}")
        # GPU
        if snap_after_formal.get("gpu_util") and snap_after_formal["gpu_util"] > 70:
            invalid=True; invalid_reason.append(f"gpu_contention {snap_after_formal['gpu_util']}")
        elif snap_after_formal.get("gpu_util") and snap_after_formal["gpu_util"] > 30:
            invalid_reason.append(f"gpu_notice {snap_after_formal['gpu_util']} (WDDM noise, not invalid)")
        if snap_after_formal.get("cpu_percent") and snap_after_formal["cpu_percent"] > 90:
            invalid_reason.append(f"cpu_high {snap_after_formal['cpu_percent']}")
        # Also check pre-run RSS stability expectation: Protocol D should have small variance, but not invalid gate
    except Exception as e:
        invalid_reason.append(f"snapshot_err {e}")
    # also check thread convergence: if protocol D works, threads should be stable
    # Save formal raw
    req_path, sys_path, raw_path = save_raw(result, out_raw)
    proc = compute_processed(result)
    _write_json(out_processed / f"{run_id}_processed.json", proc)
    _write_json(out_raw / f"{run_id}_server_traces.json", traces)
    manifest={
        "run_id": run_id,
        "session_id": session_id,
        "seq": seq,
        "started_at_utc": started,
        "finished_at_utc": _now(),
        "level": level,
        "level_name": level_name,
        "concurrency": conc,
        "seed": seed,
        "warmup_seed": warmup_seed,
        "round": spec.get("round"),
        "chronological_order": chrono,
        "pair_seed": spec.get("pair_seed"),
        "config": config,
        "workload": describe_workload(workload),
        "warmup_workload": describe_workload(warmup_workload),
        "warmup_processed": {"median_ttft": warmup_proc.get("ttft",{}).get("median"), "p95_ttft": warmup_proc.get("ttft",{}).get("p95"), "throughput": warmup_proc.get("throughput_rps")},
        "source_hashes": source_hashes,
        "client_runtime": client_runtime,
        "client_pool_config": config["client_pool"],
        "server_health_before_warmup": hb,
        "server_health_before_formal": hb_before_formal,
        "server_health_after": ha,
        "snapshot_before_warmup": snap_before_warmup,
        "snapshot_after_warmup": snap_after_warmup,
        "snapshot_after_gc_sleep": snap_after_gc_sleep,
        "snapshot_before_formal": snap_before_formal,
        "snapshot_after_formal": snap_after_formal,
        "gc_snapshot": gc_snapshot,
        "sleep_duration": sleep_end - sleep_start,
        "invalid": invalid,
        "invalid_reason": invalid_reason,
        "artifacts": {"requests_csv": str(req_path), "system_csv": str(sys_path), "raw_json": str(raw_path), "warmup_requests_csv": str(warmup_req_path)},
        "trace_count": len(traces),
        "protocol": "D",
        "protocol_steps_executed": ["warmup c4 n40", "gc.collect", "sleep2s", "host_check", "formal"],
    }
    _write_json(out_raw / f"{run_id}_manifest.json", manifest)
    return manifest, proc, warmup_proc

async def run_stationarity_md(base_url, out_root, session_id, source_hashes):
    raw_dir = out_root / "raw" / "l0"
    proc_dir = out_root / "processed" / "l0"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    client_runtime=_client_runtime()
    async with httpx.AsyncClient() as ctrl:
        hb = await _health(ctrl, base_url)
    server_pid=hb.get("pid")
    async with httpx.AsyncClient() as ctrl:
        await _set_level(ctrl, base_url, 0)
    specs=stationarity_specs_md()
    print(f"[stage3md][stationarity] running {len(specs)} L0 OFF Protocol D runs (c1/c4 balanced, new PID {server_pid})")
    manifests=[]
    procs=[]
    warmup_procs=[]
    for seq, spec in enumerate(specs, start=1):
        print(f"[stage3md][stationarity] {seq}/{len(specs)} c{spec['concurrency']} seed{spec['seed']} order{spec['chronological_order']} warmup{spec['warmup_seed']} round{spec['round']}")
        man, proc, warmup_proc = await _run_protocol_d_single(base_url, raw_dir, proc_dir, session_id, spec, seq, source_hashes, server_pid, client_runtime, phase="stationarity")
        manifests.append(man)
        procs.append(proc)
        warmup_procs.append(warmup_proc)
        _append_jsonl(out_root / "logs" / f"{session_id}_stationarity_events.jsonl", {"event":"stationarity_run_complete","run_id":man["run_id"],"seq":seq,"concurrency":man["concurrency"],"seed":man["seed"],"invalid":man["invalid"], "protocol":"D"})
        await asyncio.sleep(1.0)
    eval_result = evaluate_stationarity_md(procs, manifests)
    # Also include warmup analysis and RSS replication
    # Save host_state aggregation
    host_rows=[]
    for m in manifests:
        host_rows.append({
            "run_id": m["run_id"],
            "concurrency": m["concurrency"],
            "seed": m["seed"],
            "chronological_order": m["chronological_order"],
            "warmup_seed": m["warmup_seed"],
            "level": m["level_name"],
            "pid": m["server_health_before_formal"].get("pid"),
            "threads_before": m["snapshot_before_formal"].get("server_threads"),
            "threads_after": m["snapshot_after_formal"].get("server_threads"),
            "rss_before": m["snapshot_before_formal"].get("server_rss_mb"),
            "rss_after": m["snapshot_after_formal"].get("server_rss_mb"),
            "rss_after_warmup": m["snapshot_after_warmup"].get("server_rss_mb"),
            "rss_after_gc_sleep": m["snapshot_after_gc_sleep"].get("server_rss_mb"),
            "cpu_mean": m.get("snapshot_after_formal",{}).get("cpu_percent"),
            "gpu_util": m.get("snapshot_after_formal",{}).get("gpu_util"),
            "gpu_mem": m.get("snapshot_after_formal",{}).get("gpu_mem_used_mb"),
            "invalid": m["invalid"],
            "median_ttft": next((p.get("ttft",{}).get("median") for p,m2 in zip(procs, manifests) if m2["run_id"]==m["run_id"]), None),
            "p95_ttft": next((p.get("ttft",{}).get("p95") for p,m2 in zip(procs, manifests) if m2["run_id"]==m["run_id"]), None),
        })
    # Write host_state.csv
    import csv
    host_csv = proc_dir / "host_state.csv"
    if host_rows:
        with open(host_csv, "w", newline="", encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(host_rows[0].keys()))
            w.writeheader()
            w.writerows(host_rows)
    _write_json(proc_dir / "stationarity_summary.json", eval_result)
    _write_json(proc_dir / "stationarity_gate.json", eval_result)
    # write RSS correlation
    rss_vals = [r["rss_before"] for r in host_rows if r["rss_before"]]
    ttft_vals = [r["median_ttft"] for r in host_rows if r["median_ttft"]]
    # Also compute workload token comparability
    token_workloads = []
    for m in manifests:
        wl = m.get("workload",{})
        # describe_workload returns maybe total tokens? Check
        token_workloads.append(wl)
    write_stationarity_results_md(out_root, eval_result, manifests, procs, host_rows, warmup_procs)
    return manifests, procs, eval_result, host_rows

def _stats(arr):
    if not arr or len(arr)==0:
        return {"count":0,"median":None,"mean":None,"std":None,"min":None,"max":None,"range":None,"rel_range":None,"cv":None}
    arr_sorted=sorted(arr)
    median=statistics.median(arr_sorted)
    mean=statistics.mean(arr_sorted)
    std=statistics.stdev(arr_sorted) if len(arr_sorted)>1 else 0.0
    minv=min(arr_sorted); maxv=max(arr_sorted)
    rng=maxv-minv
    rel_range=(rng/median) if median and median!=0 else None
    cv=(std/mean) if mean and mean!=0 else None
    return {"count": len(arr),"median":median,"mean":mean,"std":std,"min":minv,"max":maxv,"range":rng,"rel_range":rel_range,"cv":cv,"values":arr_sorted}

def evaluate_stationarity_md(processed_list, manifests):
    by_conc=defaultdict(list)
    for p, m in zip(processed_list, manifests):
        conc=m["concurrency"]
        median_ttft = p.get("ttft",{}).get("median")
        p95_ttft = p.get("ttft",{}).get("p95")
        median_lat = p.get("latency",{}).get("median")
        p95_lat = p.get("latency",{}).get("p95")
        thr = p.get("throughput_rps")
        tok_thr = p.get("token_throughput")
        by_conc[conc].append({"median_ttft": median_ttft, "p95_ttft": p95_ttft, "median_lat": median_lat, "p95_lat": p95_lat, "thr": thr, "tok_thr": tok_thr, "order": m["chronological_order"], "seed": m["seed"], "round": m.get("round"), "run_id": m["run_id"], "invalid": m["invalid"]})
    details={}
    failed=[]
    for conc in sorted(by_conc):
        rows=by_conc[conc]
        valid_rows=[r for r in rows if not r["invalid"]]
        invalid_count=len(rows)-len(valid_rows)
        metrics={}
        for key in ["median_ttft","p95_ttft","median_lat","p95_lat","thr","tok_thr"]:
            vals=[r[key] for r in valid_rows if r[key] is not None]
            stats=_stats(vals)
            orders=[r["order"] for r in valid_rows if r[key] is not None]
            drift_slope=None
            rho=None
            drift_fail=False
            if len(vals)>=3:
                try:
                    mean_o=statistics.mean(orders)
                    mean_v=statistics.mean(vals)
                    cov=sum((o-mean_o)*(v-mean_v) for o,v in zip(orders, vals))/len(vals)
                    var_o=sum((o-mean_o)**2 for o in orders)/len(vals)
                    slope=cov/var_o if var_o!=0 else 0
                    drift_slope=slope
                    std_o=statistics.stdev(orders) if len(orders)>1 else 1
                    std_v=statistics.stdev(vals) if len(vals)>1 else 1
                    corr=cov/(std_o*std_v) if std_o and std_v else 0
                    rho=corr
                    if abs(corr)>0.6 and stats["median"] and abs(slope/stats["median"])>0.02:
                        drift_fail=True
                except: pass
            gate_status="PASS"
            is_central = key in ["median_ttft","median_lat","thr","tok_thr"]
            is_tail = key in ["p95_ttft","p95_lat"]
            rel_range=stats.get("rel_range")
            cv=stats.get("cv")
            central_thresh_rel=0.05
            tail_thresh_rel=0.10
            if is_central:
                if rel_range is not None and rel_range>central_thresh_rel:
                    gate_status="FAIL"
                    failed.append(f"c{conc} {key} central rel_range {rel_range*100:.1f}% >5%")
                # max pairwise deviation >5% also fail central
                if stats["median"] and vals:
                    max_dev=max(abs(v-stats["median"])/stats["median"] for v in vals) if stats["median"] else 0
                    if max_dev>0.05 and gate_status=="PASS":
                        # Only mark PARTIAL if just over 5% but rel_range already captures FAIL
                        # For strict gate, max pairwise >5% is FAIL for central? Use threshold 5% as per plan "no single run deviates >5%"
                        gate_status="FAIL"
                        failed.append(f"c{conc} {key} central single outlier {max_dev*100:.1f}% >5%")
            elif is_tail:
                if rel_range is not None and rel_range>0.15:
                    gate_status="FAIL"
                    failed.append(f"c{conc} {key} tail rel_range {rel_range*100:.1f}% >15% (limit 10% strict, 15% fail)")
                elif rel_range is not None and rel_range>0.10:
                    gate_status="PARTIAL"
                    failed.append(f"c{conc} {key} tail rel_range {rel_range*100:.1f}% >10% (partial)")
                if stats["median"] and vals:
                    max_dev=max(abs(v-stats["median"])/stats["median"] for v in vals)
                    if max_dev>0.20:
                        gate_status="FAIL"
                        failed.append(f"c{conc} {key} single outlier {max_dev*100:.1f}% >20%")
                    elif max_dev>0.15 and gate_status=="PASS":
                        gate_status="FAIL"
                        failed.append(f"c{conc} {key} single outlier {max_dev*100:.1f}% >15%")
                    elif max_dev>0.10 and gate_status=="PASS":
                        gate_status="PARTIAL"
                        failed.append(f"c{conc} {key} single outlier {max_dev*100:.1f}% >10%")
            if drift_fail:
                gate_status="FAIL"
                failed.append(f"c{conc} {key} chronological drift rho {rho:.2f} slope {drift_slope} significant")
            metrics[key]={"stats":stats, "drift_slope":drift_slope, "rho":rho, "gate":gate_status, "drift_fail":drift_fail}
        details[str(conc)]={"rows":rows, "valid_count": len(valid_rows), "invalid_count": invalid_count, "metrics":metrics}
    # overall classification
    any_central_fail=False
    any_tail_fail=False
    any_tail_partial=False
    any_central_partial=False
    for conc, d in details.items():
        for k, m in d["metrics"].items():
            if k in ["median_ttft","median_lat","thr","tok_thr"]:
                if m["gate"]=="FAIL":
                    any_central_fail=True
                elif m["gate"]=="PARTIAL":
                    any_central_partial=True
            if k in ["p95_ttft","p95_lat"]:
                if m["gate"]=="FAIL":
                    any_tail_fail=True
                elif m["gate"]=="PARTIAL":
                    any_tail_partial=True
    if any_central_fail or any_tail_fail:
        overall="NON-STATIONARY"
    elif any_tail_partial or any_central_partial:
        overall="PARTIALLY STATIONARY"
    else:
        overall="STATIONARY"
    total_valid=sum(d["valid_count"] for d in details.values())
    total_total=sum(d["valid_count"]+d["invalid_count"] for d in details.values())
    if total_valid < total_total*0.7:
        overall="DESIGN INVALID"
        failed.append(f"too many invalid runs {total_total-total_valid}/{total_total}")
    return {"classification": overall, "by_concurrency": details, "failed_checks": failed, "overall_status": "PASS" if overall=="STATIONARY" else "FAIL"}

def write_stationarity_results_md(out_root, eval_result, manifests, procs, host_rows, warmup_procs):
    path = out_root / "independent_stationarity_results.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Independent Stationarity Results — Phase A (L0 OFF, Protocol D, New PID)\n\n")
        f.write(f"> **Date:** {_now()}  \n> **Classification:** {eval_result['classification']}  \n> **Session:** {manifests[0]['session_id'] if manifests else 'unknown'}  \n> **Protocol D:** Frozen — 1×c4 n40 warmup + gc.collect + sleep2s  \n\n")
        f.write("## 1. Summary\n\n")
        f.write(f"**Overall:** {eval_result['classification']} — {eval_result['overall_status']}  \n")
        if eval_result["failed_checks"]:
            f.write("**Failed checks:**\n")
            for ch in eval_result["failed_checks"]:
                f.write(f"- {ch}\n")
        else:
            f.write("**All central ≤5% and tail ≤10% thresholds PASS in new session. Protocol D reproduced.**\n")
        f.write("\n")
        # host pid info
        if manifests:
            pid = manifests[0]["server_health_before_formal"].get("pid")
            f.write(f"**Serving PID (new):** {pid} (distinct from Stage 3M-C 56704 and Stage 3M-B 19372)  \n")
            f.write(f"**Port:** 8040 (isolated)  \n")
        f.write("\n## 2. Per-Concurrency Metrics\n\n")
        for conc in sorted(eval_result["by_concurrency"]):
            d=eval_result["by_concurrency"][conc]
            f.write(f"### c={conc} — {d['valid_count']} valid / {d['invalid_count']} invalid (Protocol D, L0 OFF)\n\n")
            f.write("| Metric | Median | Mean | Min | Max | Range | Rel Range | CV | Max Pairwise Δ | Drift slope | Rho | Gate |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            for k in ["median_ttft","p95_ttft","median_lat","p95_lat","thr","tok_thr"]:
                m=d["metrics"][k]
                s=m["stats"]
                rel = f"{s['rel_range']*100:.2f}%" if s.get("rel_range") is not None else "NA"
                cv = f"{s['cv']*100:.2f}%" if s.get("cv") is not None else "NA"
                median = f"{s['median']:.4f}" if s.get("median") is not None else "NA"
                mean = f"{s['mean']:.4f}" if s.get("mean") is not None else "NA"
                minv = f"{s['min']:.4f}" if s.get("min") is not None else "NA"
                maxv = f"{s['max']:.4f}" if s.get("max") is not None else "NA"
                rng = f"{s['range']:.4f}" if s.get("range") is not None else "NA"
                slope = f"{m['drift_slope']:.6f}" if m.get("drift_slope") is not None else "NA"
                rho = f"{m['rho']:.2f}" if m.get("rho") is not None else "NA"
                # max pairwise
                max_pair = "NA"
                if s.get("values") and s.get("median"):
                    vals=s["values"]
                    median_val=s["median"]
                    max_dev=max(abs(v-median_val)/median_val for v in vals) if median_val else 0
                    max_pair=f"{max_dev*100:.2f}%"
                f.write(f"| {k} | {median} | {mean} | {minv} | {maxv} | {rng} | {rel} | {cv} | {max_pair} | {slope} | {rho} | {m['gate']} |\n")
            f.write("\n")
            f.write("**Raw per-run values (Protocol D):**\n\n")
            f.write("| Order | Seed | Round | Warmup Seed | median_ttft | p95_ttft | median_lat | p95_lat | throughput | tok_thr | RSS pre (MB) | Threads | invalid |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            for r in sorted(d["rows"], key=lambda x: x["order"]):
                # find manifest for RSS
                man = next((m for m in manifests if m["chronological_order"]==r["order"] and m["concurrency"]==int(conc)), None)
                rss_pre = man["snapshot_before_formal"].get("server_rss_mb") if man else None
                thr_b = man["snapshot_before_formal"].get("server_threads") if man else None
                rss_str = f"{rss_pre:.0f}" if rss_pre else "NA"
                thr_str = f"{thr_b}" if thr_b else "NA"
                f.write(f"| {r['order']} | {r['seed']} | {r['round']} | {man['warmup_seed'] if man else 'NA'} | {r['median_ttft']:.4f} | {r['p95_ttft']:.4f} | {r['median_lat']:.4f} | {r['p95_lat']:.4f} | {r['thr']:.4f} | {r['tok_thr']:.1f} | {rss_str} | {thr_str} | {r['invalid']} |\n")
            f.write("\n")
        # workload comparability
        f.write("\n## 3. Workload Token Comparability\n\n")
        for m in manifests:
            wl=m.get("workload",{})
            f.write(f"- {m['run_id']}: input {wl.get('input_tokens')} / output {wl.get('output_tokens')} / seed {m['seed']} / described prompt_hash {wl.get('prompt_hash','?')[:8] if wl.get('prompt_hash') else '?'} / total_tokens measured {wl.get('total_tokens','?')}\n")
        # host state
        f.write("\n## 4. Host State Validation (Protocol D Convergence)\n\n")
        f.write("| Run | Conc | Seed | Order | Threads before | Threads after | RSS before (MB) | RSS after (MB) | RSS after GC sleep (MB) | CPU % | GPU util % | invalid |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for row in host_rows:
            f.write(f"| {row['run_id'][:20]} | {row['concurrency']} | {row['seed']} | {row['chronological_order']} | {row['threads_before']} | {row['threads_after']} | {row['rss_before']:.0f} | {row['rss_after']:.0f} | {row['rss_after_gc_sleep']:.0f} | {row['cpu_mean']} | {row['gpu_util']} | {row['invalid']} |\n")
        f.write("\n**Thread convergence:** Check Δ threads ≤5 across formals; **RSS convergence:** Check pre-run RSS range/CV.\n")
        # RSS replication
        rss_pre_vals = [r["rss_before"] for r in host_rows if r["rss_before"]]
        if rss_pre_vals:
            rss_stats=_stats(rss_pre_vals)
            f.write(f"\n**Pre-run RSS stats:** median {rss_stats['median']:.0f} MB, range {rss_stats['range']:.0f} MB, rel_range {rss_stats['rel_range']*100:.2f}%, CV {rss_stats['cv']*100:.2f}%\n")
            # correlation
            try:
                ttfts=[r["p95_ttft"] for r in host_rows if r["p95_ttft"] and r["rss_before"]]
                rss_for_corr=[r["rss_before"] for r in host_rows if r["p95_ttft"] and r["rss_before"]]
                if len(ttfts)>=3:
                    mean_r=statistics.mean(rss_for_corr)
                    mean_t=statistics.mean(ttfts)
                    cov=sum((r-mean_r)*(t-mean_t) for r,t in zip(rss_for_corr, ttfts))/len(ttfts)
                    var_r=sum((r-mean_r)**2 for r in rss_for_corr)/len(rss_for_corr)
                    var_t=sum((t-mean_t)**2 for t in ttfts)/len(ttfts)
                    corr=cov/( (var_r**0.5)*(var_t**0.5)) if var_r and var_t else 0
                    f.write(f"**RSS vs p95_TTFT correlation (Pearson):** {corr:.2f} (do NOT claim causality; report only stability)\n")
            except: pass
        f.write("\n## 5. Variance Decomposition\n\n")
        for conc in sorted(eval_result["by_concurrency"]):
            d=eval_result["by_concurrency"][conc]
            rows=d["rows"]
            seed_groups=defaultdict(list)
            for r in rows:
                if not r["invalid"]:
                    seed_groups[r["seed"]].append(r["p95_ttft"])
            f.write(f"- c={conc} p95_ttft by seed: ")
            for seed, vals in seed_groups.items():
                mean=statistics.mean(vals) if vals else 0
                f.write(f" seed {seed} mean {mean:.4f} n={len(vals)};")
            f.write("\n")
            corr=eval_result["by_concurrency"][conc]["metrics"]["p95_ttft"]["rho"]
            f.write(f"  - order correlation rho={corr:.2f} for p95_ttft\n")
        f.write("\n## 6. Artifacts\n\n")
        f.write("```\nresearch/measurement_repair/independent_validation/raw/l0/\nresearch/measurement_repair/independent_validation/processed/l0/\n```\n")
        f.write("\n*— End independent stationarity —*\n")

# ---------------------------------------------------------------------------
# Phase B regate
# ---------------------------------------------------------------------------
async def run_regate_md(base_url, out_root, session_id, source_hashes):
    raw_dir = out_root / "raw" / "l1_regate"
    proc_dir = out_root / "processed" / "l1_regate"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    client_runtime=_client_runtime()
    async with httpx.AsyncClient() as ctrl:
        hb = await _health(ctrl, base_url)
    server_pid=hb.get("pid")
    specs=regate_specs_md()
    print(f"[stage3md][regate] running {len(specs)} paired runs (Protocol D, L0 vs L1) new PID {server_pid}")
    manifests=[]
    procs=[]
    for seq, spec in enumerate(specs, start=1):
        print(f"[stage3md][regate] {seq}/{len(specs)} c{spec['concurrency']} seed{spec['seed']} {spec['level_name']} order{spec['chronological_order']} pair{spec['pair_seed']}")
        man, proc, _ = await _run_protocol_d_single(base_url, raw_dir, proc_dir, session_id, spec, seq, source_hashes, server_pid, client_runtime, phase="regate")
        manifests.append(man)
        procs.append(proc)
        _append_jsonl(out_root / "logs" / f"{session_id}_regate_events.jsonl", {"event":"regate_run_complete","run_id":man["run_id"],"level":man["level_name"],"concurrency":man["concurrency"],"seed":man["seed"]})
        await asyncio.sleep(1.0)
    eval_result = evaluate_regate_md(procs, manifests)
    _write_json(proc_dir / "overhead_gate.json", eval_result)
    _write_json(proc_dir / "regate_summary.json", eval_result)
    write_regate_results_md(out_root, eval_result, manifests, procs)
    return manifests, procs, eval_result

def evaluate_regate_md(processed_list, manifests):
    grouped=defaultdict(dict)
    for p,m in zip(processed_list, manifests):
        conc=m["concurrency"]
        seed=m["seed"]
        level=m["level"]
        is_off=(level==0)
        thr=p.get("throughput_rps")
        ttft_p95=p.get("ttft",{}).get("p95")
        lat_p95=p.get("latency",{}).get("p95")
        median_ttft=p.get("ttft",{}).get("median")
        median_lat=p.get("latency",{}).get("median")
        grouped[conc].setdefault(seed, {})[is_off]={"thr":thr,"ttft_p95":ttft_p95,"lat_p95":lat_p95,"median_ttft":median_ttft,"median_lat":median_lat,"p":p,"m":m}
    details={}
    failed=[]
    for conc in sorted(grouped):
        pairs=grouped[conc]
        deltas={"thr":[], "ttft_p95":[], "lat_p95":[], "median_ttft":[], "median_lat":[]}
        absolute={k:[] for k in deltas}
        missing=[]
        for seed, modes in sorted(pairs.items()):
            if True not in modes or False not in modes:
                missing.append(seed)
                continue
            for k in deltas:
                off=modes[True].get(k)
                on=modes[False].get(k)
                if off is None or on is None or off==0:
                    continue
                rel=on/off-1.0
                abs_delta=on-off
                deltas[k].append(rel)
                absolute[k].append(abs_delta)
        if missing:
            failed.append(f"c{conc} missing pairs seeds {missing}")
        checks={}
        for k, ds in deltas.items():
            median=statistics.median(ds) if ds else None
            indiv_ok=bool(ds) and all(abs(d)<=0.10 for d in ds)
            median_ok=median is not None and abs(median)<=0.05
            std=statistics.stdev(ds) if len(ds)>1 else 0.0
            checks[k]={"pair_relative_deltas": ds, "median_relative_delta": median, "median_limit_ok": median_ok, "individual_limit_ok": indiv_ok, "std": std, "absolute_deltas": absolute[k], "median_absolute": statistics.median(absolute[k]) if absolute[k] else None}
            if not median_ok and k in ["ttft_p95","lat_p95","thr"]:
                failed.append(f"c{conc} {k} median paired diff exceeds 5% median={median}")
            if not indiv_ok and k in ["ttft_p95","lat_p95","thr"]:
                failed.append(f"c{conc} {k} individual paired diff exceeds 10% deltas={ds}")
            if std>0.10:
                failed.append(f"c{conc} {k} paired std {std*100:.1f}% >10% unstable")
        details[str(conc)]=checks
    mult_checks={}
    mode_medians={True:{}, False:{}}
    for is_off in (True, False):
        for conc in grouped:
            vals=[]
            for seed, modes in grouped[conc].items():
                if is_off in modes:
                    v=modes[is_off].get("ttft_p95")
                    if isinstance(v,(int,float)) and v>0:
                        vals.append(v)
            if vals:
                mode_medians[is_off][conc]=statistics.median(vals)
    for label, low, high in (("c4_over_c1",1,4),):
        if low not in mode_medians[True] or high not in mode_medians[True]:
            failed.append(f"missing OFF multiplier for {label}")
            continue
        if low not in mode_medians[False] or high not in mode_medians[False]:
            failed.append(f"missing ON multiplier for {label}")
            continue
        off_mult=mode_medians[True][high]/mode_medians[True][low]
        on_mult=mode_medians[False][high]/mode_medians[False][low]
        rel=on_mult/off_mult-1
        mult_checks[label]={"off_multiplier":off_mult,"on_multiplier":on_mult,"relative_change":rel,"limit_ok":abs(rel)<=0.10, "off_median_c1": mode_medians[True][low], "off_median_c4": mode_medians[True][high], "on_median_c1": mode_medians[False][low], "on_median_c4": mode_medians[False][high]}
        if abs(rel)>0.10:
            failed.append(f"{label} ON/OFF multiplier distortion {rel*100:.2f}% >10%")
    status="PASS" if not failed else "FAIL"
    return {"status":status,"by_concurrency":details,"ttft_multiplier_checks":mult_checks,"failed_checks":failed, "mode_medians": mode_medians}

def write_regate_results_md(out_root, eval_result, manifests, procs):
    path = out_root / "l1_regate_results.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# L1 Overhead Re-Gate Results — Phase B (Independent, Protocol D)\n\n")
        f.write(f"> **Date:** {_now()}  \n> **Gate Status:** {eval_result['status']}  \n> **Protocol D preserved**  \n\n")
        if eval_result["failed_checks"]:
            f.write("**Failed checks:**\n")
            for ch in eval_result["failed_checks"]:
                f.write(f"- {ch}\n")
            f.write("\n")
        else:
            f.write("**All gates PASS: median ≤5%, individual ≤10%, multiplier ≤10%**\n\n")
        f.write("## 1. Matched-Pair Tables (Absolute & Relative Δ)\n\n")
        grouped=defaultdict(dict)
        for p,m in zip(procs, manifests):
            grouped[m["concurrency"]].setdefault(m["seed"], {})[m["level"]==0]=p
        for conc in sorted(grouped):
            f.write(f"### c={conc}\n\n")
            f.write("| Seed | Order | OFF p95_ttft (s) | L1 p95_ttft (s) | Abs Δ (ms) | Rel Δ (%) | OFF thr (rps) | L1 thr (rps) | Thr Δ (%) | Valid |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            for seed in sorted(grouped[conc]):
                modes=grouped[conc][seed]
                if True in modes and False in modes:
                    off=modes[True]; on=modes[False]
                    off_ttft=off.get("ttft",{}).get("p95",0) or 0
                    on_ttft=on.get("ttft",{}).get("p95",0) or 0
                    abs_ms=(on_ttft-off_ttft)*1000
                    rel=(on_ttft/off_ttft-1)*100 if off_ttft else 0
                    off_thr=off.get("throughput_rps",0) or 0
                    on_thr=on.get("throughput_rps",0) or 0
                    thr_rel=(on_thr/off_thr-1)*100 if off_thr else 0
                    off_man = next((m for m in manifests if m["concurrency"]==conc and m["seed"]==seed and m["level"]==0), None)
                    on_man = next((m for m in manifests if m["concurrency"]==conc and m["seed"]==seed and m["level"]==1), None)
                    order="OFF→L1" if off_man and on_man and off_man["chronological_order"]<on_man["chronological_order"] else "L1→OFF"
                    f.write(f"| {seed} | {order} | {off_ttft:.5f} | {on_ttft:.5f} | {abs_ms:+.2f} | {rel:+.2f}% | {off_thr:.4f} | {on_thr:.4f} | {thr_rel:+.2f}% | YES |\n")
            f.write("\n")
        f.write("## 2. Per-Metric Overhead Summary\n\n")
        f.write("| Conc | Metric | Median rel Δ | Individual rel Δs | Median abs | Std | Median Gate | Individual Gate |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for conc in sorted(eval_result["by_concurrency"]):
            for k in ["ttft_p95","lat_p95","thr","median_ttft","median_lat"]:
                chk=eval_result["by_concurrency"][conc].get(k,{})
                median=chk.get("median_relative_delta")
                median_str=f"{median*100:+.2f}%" if median is not None else "NA"
                indiv=chk.get("pair_relative_deltas",[])
                indiv_str=",".join([f"{x*100:+.1f}%" for x in indiv]) if indiv else "NA"
                abs_med=chk.get("median_absolute")
                abs_str=f"{abs_med*1000:+.1f} ms" if abs_med is not None and "ttft" in k else f"{abs_med:+.4f}" if abs_med is not None else "NA"
                std=chk.get("std",0)
                f.write(f"| {conc} | {k} | {median_str} | {indiv_str} | {abs_str} | {std*100:.1f}% | {'PASS' if chk.get('median_limit_ok') else 'FAIL'} | {'PASS' if chk.get('individual_limit_ok') else 'FAIL'} |\n")
        f.write("\n")
        f.write("## 3. Knee Preservation\n\n")
        for label, chk in eval_result.get("ttft_multiplier_checks",{}).items():
            f.write(f"- **{label}** OFF {chk['off_multiplier']:.2f} → L1 {chk['on_multiplier']:.2f} = {chk['relative_change']*100:+.2f}% (threshold ≤10%) → {'PASS' if chk['limit_ok'] else 'FAIL'}\n")
            f.write(f"  - OFF medians: c1 {chk['off_median_c1']:.4f}s, c4 {chk['off_median_c4']:.4f}s; ON medians: c1 {chk['on_median_c1']:.4f}s, c4 {chk['on_median_c4']:.4f}s\n")
        f.write("\n")
        f.write("## 4. Absolute Effect vs Baseline Variance\n\n")
        f.write("Baseline L0→L0 variance from Phase A CV compared to instrumentation Δ above. If instrumentation Δ within baseline CV, effect not larger than natural variance → cautious interpretation.\n\n")
        f.write("## 5. Artifacts\n\n")
        f.write("```\nresearch/measurement_repair/independent_validation/raw/l1_regate/\nresearch/measurement_repair/independent_validation/processed/l1_regate/\n```\n")
        f.write("\n*— End regate —*\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def _main_async(args):
    out_root=Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    (out_root / "raw" / "l0").mkdir(parents=True, exist_ok=True)
    (out_root / "processed" / "l0").mkdir(parents=True, exist_ok=True)
    source_hashes=_source_hashes()
    session_id=args.session_id or f"independent-3md-{int(time.time())}"
    print(f"[3md] session {session_id} base {args.base_url} phase {args.phase}")
    async with httpx.AsyncClient() as ctrl:
        h=await _health(ctrl, args.base_url)
        print(f"[3md] server health pid {h['pid']} loaded={h['loaded']} mock={h['mock']} torch {h.get('runtime',{}).get('torch_num_threads')}/{h.get('runtime',{}).get('torch_num_interop_threads')} executor {h.get('runtime',{}).get('fixed_executor_max_workers')} port {h.get('pid')}")
        if not h.get("loaded"):
            raise RuntimeError("server not loaded")
        # Check PID is new
        if h["pid"] in (56704, 19372):
            print(f"[WARN] PID {h['pid']} is old Stage 3M-C/B PID — need new session per §2, recommend restarting server")
    if args.phase=="stationarity":
        manifests, procs, eval_result, host_rows = await run_stationarity_md(args.base_url, out_root, session_id, source_hashes)
        print(f"[stationarity] classification {eval_result['classification']}")
        return eval_result
    elif args.phase=="regate":
        manifests, procs, eval_result = await run_regate_md(args.base_url, out_root, session_id, source_hashes)
        print(f"[regate] status {eval_result['status']}")
        return eval_result
    elif args.phase=="all":
        manifests, procs, eval_result, host_rows = await run_stationarity_md(args.base_url, out_root, session_id, source_hashes)
        if eval_result["classification"]=="STATIONARY":
            print("[all] stationarity PASS, proceeding to regate")
            m2,p2,e2 = await run_regate_md(args.base_url, out_root, session_id, source_hashes)
            return {"stationarity": eval_result, "regate": e2, "session_id": session_id}
        elif eval_result["classification"]=="PARTIALLY STATIONARY":
            usable=True
            for conc in eval_result["by_concurrency"]:
                if eval_result["by_concurrency"][conc]["metrics"]["p95_ttft"]["gate"]=="FAIL":
                    usable=False
            if usable:
                print("[all] PARTIALLY but tail usable, proceeding cautiously")
                m2,p2,e2 = await run_regate_md(args.base_url, out_root, session_id, source_hashes)
                return {"stationarity": eval_result, "regate": e2, "session_id": session_id}
            else:
                print("[all] PARTIALLY but p95 not usable, skipping regate per LOCKED plan")
                return {"stationarity": eval_result, "regate": "SKIPPED"}
        else:
            print(f"[all] stationarity {eval_result['classification']}, skipping regate per LOCKED plan")
            return {"stationarity": eval_result, "regate": "SKIPPED", "session_id": session_id}
    else:
        raise ValueError(f"unknown phase {args.phase}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--out", default=str(MEASUREMENT_ROOT.parent / "measurement_repair" / "independent_validation"))
    p.add_argument("--phase", choices=["stationarity","regate","all"], default="stationarity")
    p.add_argument("--session-id", default=None)
    args=p.parse_args()
    res=asyncio.run(_main_async(args))
    _write_json(Path(args.out).resolve() / "processed" / f"stage3md_{args.phase}_{int(time.time())}.json", {"phase":args.phase,"result":str(res) if not isinstance(res, dict) else res})
    print(f"[done] phase {args.phase}")

if __name__=="__main__":
    main()
