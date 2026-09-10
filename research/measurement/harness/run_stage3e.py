"""Stage 3E Runner — Research-Grade Linux Migration Gate (L0 only)

Implements Stage 3E §4-§7:
  - L0 OFF only, c=1/c=4, 5 reps per concurrency =10 total, interleaved balanced
  - Simple, uniform, reasonable warmup: 1 x c=4 n=40 pooled OFF before session + per-run 2 warmup_requests
  - No Protocol D GC/sleep, no complex stabilization
  - Reuses: workload generator, pooled client, harness, result schema, host-state logging
  - Pooled client max_connections=max_keepalive=max(8, concurrency), fixed executor 32, torch 16
  - Metrics: median_ttft, p95_ttft, throughput, median_lat, p95_lat plus CV, relative range, absolute range, drift

This runner is OS-agnostic: it runs on Windows (as fallback baseline) and will run identically on Linux/WSL2 when available.
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
        "run_stage3e.py": Path(__file__),
        "generator.py": MEASUREMENT_ROOT / "workloads" / "generator.py",
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

def _get_snapshot(server_pid=None):
    try:
        freq=psutil.cpu_freq()
        freq_cur=freq.current if freq else None
    except: freq_cur=None
    try: vm=psutil.virtual_memory()
    except: vm=None
    snap={"cpu_percent": psutil.cpu_percent(interval=0.2), "cpu_freq_mhz": freq_cur, "ram_percent": vm.percent if vm else None, "ram_used_gb": vm.used/1024**3 if vm else None, "ram_total_gb": vm.total/1024**3 if vm else None, "timestamp": _now()}
    if server_pid:
        try:
            proc=psutil.Process(int(server_pid))
            snap["server_threads"]=proc.num_threads()
            snap["server_rss_mb"]=proc.memory_info().rss/1024/1024
            snap["server_cpu"]=proc.cpu_percent(interval=0.1)
        except Exception as e:
            snap["server_err"]=str(e)
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
            try: snap["gpu_temp"]=pynvml.nvmlDeviceGetTemperature(h, 0)
            except: pass
            try:
                snap["gpu_clock_sm"]=pynvml.nvmlDeviceGetClockInfo(h, 0)
                snap["gpu_clock_mem"]=pynvml.nvmlDeviceGetClockInfo(h, 2)
            except: pass
        except Exception as e:
            snap["gpu_err"]=str(e)
    except: snap["gpu_err"]="no nvml"
    return snap

def stationarity_specs_3e():
    # 5 per conc =10 total, seeds per §4: c1 6101-6105, c4 6401-6405
    # Interleaved balanced: c1,c4,c4,c1,c1,c4,c1,c4,c4,c1  (alternating but not clustered)
    # Pair with rounds 1..3 plus extra
    order = [
        (1, 6101, 1),
        (4, 6401, 1),
        (4, 6402, 2),
        (1, 6102, 2),
        (1, 6103, 3),
        (4, 6403, 3),
        (1, 6104, 4),
        (4, 6404, 4),
        (4, 6405, 5),
        (1, 6105, 5),
    ]
    specs=[]
    for idx, (conc, seed, rnd) in enumerate(order, start=1):
        specs.append({"concurrency": conc, "seed": seed, "round": rnd, "chronological_order": idx, "level":0, "level_name":"off"})
    return specs

async def _run_single(base_url, out_raw, out_processed, session_id, spec, seq, source_hashes, server_pid, client_runtime):
    conc=spec["concurrency"]
    seed=spec["seed"]
    level=spec["level"]
    level_name=spec["level_name"]
    run_id=f"3e-{session_id}_{seq:02d}_c{conc}_seed{seed}_off_o{spec['chronological_order']}"
    config={
        "name": f"3e_c{conc}_seed{seed}",
        "run_id": run_id,
        "phase": "3e_stationarity_L0",
        "concurrency": conc,
        "seed": seed,
        "round": spec["round"],
        "chronological_order": spec["chronological_order"],
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
        "warmup_protocol": "simple_session_1xc4n40_plus_per_run_2",
    }
    async with httpx.AsyncClient() as ctrl:
        await _set_level(ctrl, base_url, level)
        hb = await _health(ctrl, base_url)
        if hb.get("pid") != server_pid:
            raise RuntimeError(f"PID changed before {run_id}: {hb.get('pid')} vs {server_pid}")
    snap_before=_get_snapshot(server_pid)
    workload = generate(workload_type="synthetic", n=40, input_tokens=512, output_tokens=64, arrival_distribution="closed", seed=seed)
    started=_now()
    result = await run_benchmark(base_url=base_url, workload=workload, config=config, concurrency=conc, warmup_requests=2, stream=True, sampler_interval=0.3, timeout=180, arrival_distribution="closed", client_mode="pooled")
    async with httpx.AsyncClient() as ctrl2:
        ha = await _health(ctrl2, base_url)
        try:
            r = await ctrl2.get(base_url.rstrip("/")+f"/stage3/trace/{run_id}", timeout=30)
            r.raise_for_status()
            traces=r.json().get("traces",[])
        except: traces=[]
    snap_after=_get_snapshot(server_pid)
    invalid=False
    reasons=[]
    if hb.get("pid")!=ha.get("pid"):
        invalid=True; reasons.append("pid_change")
    thr_b=snap_before.get("server_threads"); thr_a=snap_after.get("server_threads")
    if thr_b and thr_a and abs(thr_a-thr_b)>20:
        invalid=True; reasons.append(f"thread_growth {thr_b}->{thr_a}")
    rss_b=snap_before.get("server_rss_mb"); rss_a=snap_after.get("server_rss_mb")
    if rss_b and rss_a and (rss_a-rss_b)>500:
        invalid=True; reasons.append(f"rss_spike {rss_b:.0f}->{rss_a:.0f}")
    if snap_after.get("gpu_util") and snap_after["gpu_util"]>70:
        invalid=True; reasons.append(f"gpu_contention {snap_after['gpu_util']}")
    elif snap_after.get("gpu_util") and snap_after["gpu_util"]>30:
        reasons.append(f"gpu_notice {snap_after['gpu_util']} (WDDM noise)")
    req_path, sys_path, raw_path = save_raw(result, out_raw)
    proc=compute_processed(result)
    _write_json(out_processed / f"{run_id}_processed.json", proc)
    _write_json(out_raw / f"{run_id}_server_traces.json", traces)
    manifest={
        "run_id": run_id, "session_id": session_id, "seq": seq, "started_at_utc": started, "finished_at_utc": _now(),
        "level": level, "level_name": level_name, "concurrency": conc, "seed": seed, "round": spec["round"], "chronological_order": spec["chronological_order"],
        "config": config, "workload": describe_workload(workload), "source_hashes": source_hashes, "client_runtime": client_runtime,
        "server_health_before": hb, "server_health_after": ha, "snapshot_before": snap_before, "snapshot_after": snap_after,
        "invalid": invalid, "invalid_reason": reasons, "artifacts": {"requests_csv": str(req_path), "system_csv": str(sys_path), "raw_json": str(raw_path)}, "trace_count": len(traces)
    }
    _write_json(out_raw / f"{run_id}_manifest.json", manifest)
    return manifest, proc

def _stats(arr):
    if not arr: return {"count":0,"median":None,"mean":None,"std":None,"min":None,"max":None,"range":None,"rel_range":None,"cv":None}
    arr_sorted=sorted(arr)
    median=statistics.median(arr_sorted)
    mean=statistics.mean(arr_sorted)
    std=statistics.stdev(arr_sorted) if len(arr_sorted)>1 else 0.0
    minv=min(arr_sorted); maxv=max(arr_sorted)
    rng=maxv-minv
    rel_range=(rng/median) if median else None
    cv=(std/mean) if mean else None
    return {"count": len(arr),"median":median,"mean":mean,"std":std,"min":minv,"max":maxv,"range":rng,"rel_range":rel_range,"cv":cv,"values":arr_sorted}

def evaluate_stationarity_3e(processed_list, manifests):
    from collections import defaultdict
    by_conc=defaultdict(list)
    for p,m in zip(processed_list, manifests):
        conc=m["concurrency"]
        by_conc[conc].append({"median_ttft": p.get("ttft",{}).get("median"), "p95_ttft": p.get("ttft",{}).get("p95"), "median_lat": p.get("latency",{}).get("median"), "p95_lat": p.get("latency",{}).get("p95"), "thr": p.get("throughput_rps"), "tok_thr": p.get("token_throughput"), "order": m["chronological_order"], "seed": m["seed"], "round": m["round"], "run_id": m["run_id"], "invalid": m["invalid"]})
    details={}
    failed=[]
    for conc in sorted(by_conc):
        rows=by_conc[conc]
        valid_rows=[r for r in rows if not r["invalid"] and r["median_ttft"] is not None]
        invalid_count=len(rows)-len(valid_rows)
        metrics={}
        for key in ["median_ttft","p95_ttft","median_lat","p95_lat","thr","tok_thr"]:
            vals=[r[key] for r in valid_rows if r[key] is not None]
            orders=[r["order"] for r in valid_rows if r[key] is not None]
            stats=_stats(vals)
            drift_slope=None; rho=None; drift_fail=False
            if len(vals)>=3:
                try:
                    mean_o=statistics.mean(orders); mean_v=statistics.mean(vals)
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
            gate="PASS"
            is_central=key in ["median_ttft","median_lat","thr","tok_thr"]
            is_tail=key in ["p95_ttft","p95_lat"]
            rel=stats.get("rel_range")
            cv=stats.get("cv")
            if is_central:
                if rel is not None and rel>0.05:
                    gate="FAIL"; failed.append(f"c{conc} {key} central rel_range {rel*100:.1f}% >5%")
                if drift_fail:
                    gate="FAIL"; failed.append(f"c{conc} {key} drift rho {rho:.2f} slope {drift_slope:.6f}")
            elif is_tail:
                if rel is not None and rel>0.15:
                    gate="FAIL"; failed.append(f"c{conc} {key} tail rel_range {rel*100:.1f}% >15% (limit 10%)")
                elif rel is not None and rel>0.10:
                    gate="PARTIAL"; failed.append(f"c{conc} {key} tail rel_range {rel*100:.1f}% >10%")
                if stats.get("median") and vals:
                    max_dev=max(abs(v-stats["median"])/stats["median"] for v in vals) if stats["median"] else 0
                    if max_dev>0.20:
                        gate="FAIL"; failed.append(f"c{conc} {key} outlier {max_dev*100:.1f}% >20%")
                    elif max_dev>0.10 and gate=="PASS":
                        gate="PARTIAL"
                if drift_fail:
                    gate="FAIL"; failed.append(f"c{conc} {key} drift rho {rho:.2f}")
            metrics[key]={"stats":stats,"drift_slope":drift_slope,"rho":rho,"gate":gate,"drift_fail":drift_fail}
        details[str(conc)]={"rows":rows,"valid_count":len(valid_rows),"invalid_count":invalid_count,"metrics":metrics}
    any_central_fail=any(m["gate"]=="FAIL" for d in details.values() for k,m in d["metrics"].items() if k in ["median_ttft","median_lat","thr"])
    any_tail_fail=any(m["gate"]=="FAIL" for d in details.values() for m in d["metrics"].values() if "p95" in [k for k in ["p95_ttft","p95_lat"] if m==m])
    # simplify overall
    has_fail=any(v["metrics"][k]["gate"]=="FAIL" for v in details.values() for k in v["metrics"] if v["metrics"][k]["gate"]=="FAIL")
    has_partial=any(v["metrics"][k]["gate"]=="PARTIAL" for v in details.values() for k in v["metrics"] if v["metrics"][k]["gate"]=="PARTIAL")
    if has_fail:
        overall="NON-STATIONARY"
    elif has_partial:
        overall="PARTIALLY STATIONARY"
    else:
        overall="STATIONARY"
    total_valid=sum(d["valid_count"] for d in details.values())
    total_total=sum(d["valid_count"]+d["invalid_count"] for d in details.values())
    if total_valid < total_total*0.5:
        overall="DESIGN INVALID"
    return {"classification": overall, "by_concurrency": details, "failed_checks": failed, "overall_status": "PASS" if overall=="STATIONARY" else "FAIL"}

async def run_stationarity_3e(base_url, out_root, session_id, source_hashes):
    raw_dir = out_root / "raw"
    proc_dir = out_root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True) if False else None
    client_runtime=_client_runtime()
    async with httpx.AsyncClient() as ctrl:
        hb=await _health(ctrl, base_url)
    server_pid=hb.get("pid")
    async with httpx.AsyncClient() as ctrl:
        await _set_level(ctrl, base_url, 0)
    # --- Simple session warmup: 1 x c=4 n=40 pooled OFF seed 8000 ---
    print(f"[3e] simple session warmup 1xc4 n40 seed 8000 on PID {server_pid}")
    warmup_config={"name":"3e_session_warmup_c4_n40","run_id":f"warmup-3e-{session_id}","phase":"session_warmup","concurrency":4,"seed":8000,"level":0,"level_name":"off","request_count":40,"input_tokens":512,"output_tokens":64,"arrival_distribution":"closed","warmup_requests":0,"stream":True,"timeout":180,"sampler_interval":0.3,"client_mode":"pooled","client_pool":"max_connections=8,max_keepalive=8","server_pid":server_pid}
    async with httpx.AsyncClient() as ctrl:
        await _set_level(ctrl, base_url, 0)
    warmup_workload=generate(workload_type="synthetic", n=40, input_tokens=512, output_tokens=64, arrival_distribution="closed", seed=8000)
    warmup_result=await run_benchmark(base_url=base_url, workload=warmup_workload, config=warmup_config, concurrency=4, warmup_requests=0, stream=True, sampler_interval=0.3, timeout=180, arrival_distribution="closed", client_mode="pooled")
    warmup_proc=compute_processed(warmup_result)
    save_raw(warmup_result, raw_dir)
    _write_json(proc_dir / f"warmup-3e-{session_id}_processed.json", warmup_proc)
    print(f"[3e] warmup done median_ttft {warmup_proc.get('ttft',{}).get('median')} p95 {warmup_proc.get('ttft',{}).get('p95')} thr {warmup_proc.get('throughput_rps')}")
    await asyncio.sleep(1.0)
    specs=stationarity_specs_3e()
    print(f"[3e] running {len(specs)} L0 OFF simple-warmup runs (c1/c4 balanced, PID {server_pid})")
    manifests=[]; procs=[]
    for seq, spec in enumerate(specs, start=1):
        print(f"[3e] {seq}/{len(specs)} c{spec['concurrency']} seed{spec['seed']} order{spec['chronological_order']} round{spec['round']}")
        man, proc = await _run_single(base_url, raw_dir, proc_dir, session_id, spec, seq, source_hashes, server_pid, client_runtime)
        manifests.append(man); procs.append(proc)
        _append_jsonl(out_root / f"logs_3e_{session_id}.jsonl", {"event":"3e_run_complete","run_id":man["run_id"],"seq":seq,"concurrency":man["concurrency"],"seed":man["seed"],"invalid":man["invalid"]})
        await asyncio.sleep(0.8)
    eval_result=evaluate_stationarity_3e(procs, manifests)
    _write_json(proc_dir / "stationarity_summary.json", eval_result)
    _write_json(proc_dir / "stationarity_gate.json", eval_result)
    # also write host_state.csv
    import csv
    host_rows=[]
    for m, p in zip(manifests, procs):
        host_rows.append({"run_id":m["run_id"],"concurrency":m["concurrency"],"seed":m["seed"],"chronological_order":m["chronological_order"],"round":m["round"],"threads_before":m["snapshot_before"].get("server_threads"),"threads_after":m["snapshot_after"].get("server_threads"),"rss_before":m["snapshot_before"].get("server_rss_mb"),"rss_after":m["snapshot_after"].get("server_rss_mb"),"cpu":m["snapshot_after"].get("cpu_percent"),"gpu_util":m["snapshot_after"].get("gpu_util"),"invalid":m["invalid"],"median_ttft":p.get("ttft",{}).get("median"),"p95_ttft":p.get("ttft",{}).get("p95")})
    host_csv=proc_dir / "host_state.csv"
    if host_rows:
        with open(host_csv,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(host_rows[0].keys())); w.writeheader(); w.writerows(host_rows)
    return manifests, procs, eval_result, host_rows, warmup_proc

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8035")
    ap.add_argument("--out", default="F:/AIinfraResearch/research/environment_migration")
    ap.add_argument("--session-id", default=None)
    args=ap.parse_args()
    out_root=Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    source_hashes=_source_hashes()
    session_id=args.session_id or f"3e-{int(time.time())}"
    print(f"[3e] session {session_id} base {args.base_url} out {out_root}")
    async def _run():
        async with httpx.AsyncClient() as ctrl:
            h=await _health(ctrl, args.base_url)
            print(f"[3e] server health pid {h['pid']} loaded={h['loaded']} mock={h['mock']} torch {h.get('runtime',{}).get('torch_num_threads')}")
            if not h.get("loaded"):
                raise RuntimeError("server not loaded")
        manifests, procs, eval_result, host_rows, warmup_proc = await run_stationarity_3e(args.base_url, out_root, session_id, source_hashes)
        print(f"[3e] classification {eval_result['classification']}")
        for ch in eval_result.get("failed_checks",[]):
            print(f"  FAIL: {ch}")
        return eval_result
    asyncio.run(_run())

if __name__=="__main__":
    main()
