"""
CLI for harness — run_baseline and generic runs
Usage:
  python -m harness.run --base-url http://127.0.0.1:8000 --workload synthetic --n 20 --input-tokens 512 --output-tokens 64 --concurrency 4 --stream
  python -m harness.run --config configs/baseline_short.json
"""
import argparse, json, hashlib, time, asyncio, sys
from pathlib import Path

# Ensure research/measurement is on path
sys.path.insert(0, str(Path(__file__).parent.parent))
from harness.harness import run_benchmark, save_raw, compute_processed
from workloads.generator import generate, describe_workload

def _hash_env():
    p = Path(__file__).parent.parent / "environment.md"
    if p.exists():
        return hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    return "unknown"

async def _run_one(config, base_url, out_dir):
    # workload from config
    wl = generate(
        workload_type=config.get("workload_type","synthetic"),
        n=config.get("request_count",20),
        input_tokens=config.get("input_tokens",512),
        output_tokens=config.get("output_tokens",64),
        arrival_rate=config.get("arrival_rate",0),
        arrival_distribution=config.get("arrival_distribution","closed"),
        prefix_reuse_fraction=config.get("prefix_reuse_fraction",0),
        seed=config.get("seed",0),
    )
    desc = describe_workload(wl)
    print(f"[harness] workload {desc}")
    raw_cfg = dict(config)
    raw_cfg["environment_hash"] = _hash_env()
    raw_cfg["run_id"] = config.get("run_id") or f"{config.get('name','run')}-{int(time.time())}"
    result = await run_benchmark(
        base_url=base_url,
        workload=wl,
        config=raw_cfg,
        concurrency=config.get("concurrency",1),
        warmup_requests=config.get("warmup_requests",0),
        stream=config.get("stream", True),
        sampler_interval=config.get("sampler_interval",0.5),
        timeout=config.get("timeout",120),
        arrival_distribution=config.get("arrival_distribution","closed"),
    )
    # save
    out = Path(out_dir)
    req, sysp, js = save_raw(result, out)
    proc = compute_processed(result)
    proc_path = out / f"{result.run_id}_processed.json"
    with open(proc_path,"w",encoding="utf-8") as f:
        json.dump(proc, f, indent=2)
    print(f"[harness] saved raw {js.name}, processed {proc_path.name}")
    thr = proc['throughput_rps']
    tok = proc['token_throughput']
    p50 = proc['latency']['p50']
    p95 = proc['latency']['p95']
    print(f"[harness] aggregates: throughput {thr if thr is not None else 'N/A'} rps, token {tok if tok is not None else 'N/A'} tok/s, p50 {p50}, p95 {p95}, success {proc['success']}/{proc['total_requests']}")
    return result, proc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--config", type=str, help="JSON config file (overrides other flags)")
    ap.add_argument("--configs-dir", type=str, help="run all JSON in dir")
    ap.add_argument("--workload", default="synthetic", choices=["synthetic","chat","prefix_reuse","long_context","rag","agent"])
    ap.add_argument("--n", type=int, default=20, dest="request_count")
    ap.add_argument("--input-tokens", type=int, default=512)
    ap.add_argument("--output-tokens", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--arrival-rate", type=float, default=0)
    ap.add_argument("--arrival-distribution", default="closed", choices=["closed","poisson","gamma","bursty","uniform"])
    ap.add_argument("--prefix-reuse-fraction", type=float, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1, dest="warmup_requests")
    ap.add_argument("--repeats", type=int, default=1, help="repeat same config N times")
    ap.add_argument("--stream", action="store_true", default=True)
    ap.add_argument("--no-stream", dest="stream", action="store_false")
    ap.add_argument("--out", default="research/measurement/raw")
    ap.add_argument("--name", default="manual")
    args = ap.parse_args()

    base_url = args.base_url
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        # relative to project root
        out_dir = Path("F:/AIinfraResearch") / args.out

    configs = []
    if args.configs_dir:
        for p in Path(args.configs_dir).glob("*.json"):
            with open(p,encoding="utf-8") as f:
                cfg = json.load(f)
                cfg["_src"] = str(p)
                configs.append(cfg)
    elif args.config:
        with open(args.config,encoding="utf-8") as f:
            cfg = json.load(f)
            configs.append(cfg)
    else:
        cfg = {
            "name": args.name,
            "workload_type": args.workload,
            "request_count": args.request_count,
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "concurrency": args.concurrency,
            "arrival_rate": args.arrival_rate,
            "arrival_distribution": args.arrival_distribution,
            "prefix_reuse_fraction": args.prefix_reuse_fraction,
            "seed": args.seed,
            "warmup_requests": args.warmup_requests,
            "stream": args.stream,
            "concurrency_sweep": None,
        }
        configs = [cfg]

    async def _all():
        for cfg in configs:
            for rep in range(args.repeats):
                c = dict(cfg)
                c["repeat"] = rep
                c["run_id"] = f"{c.get('name','run')}-rep{rep}-{int(time.time())}"
                print(f"\n=== run {c['run_id']} ===")
                print(json.dumps({k:v for k,v in c.items() if not k.startswith("_")}, indent=2))
                await _run_one(c, base_url, out_dir)
                await asyncio.sleep(0.5)

    asyncio.run(_all())

if __name__ == "__main__":
    main()
