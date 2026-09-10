"""
Stage 3R-B Workload Sweep Runner
Coarse-to-fine exploration on hf_transformers_naive tiny-gpt2 CPU.
Generates configs, runs via harness, saves raw/processed to sweeps/raw.
"""
import json, time, pathlib, asyncio, sys, hashlib, statistics
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from harness.harness import run_benchmark, save_raw, compute_processed
from workloads.generator import generate, describe_workload

BASE_URL = "http://127.0.0.1:8000"
OUT_DIR = pathlib.Path(r"F:\AIinfraResearch\research\measurement\sweeps\raw")
CONFIG_DIR = pathlib.Path(r"F:\AIinfraResearch\research\measurement\sweeps\configs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def env_hash():
    p = pathlib.Path(r"F:\AIinfraResearch\research\measurement\environment.md")
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else "unknown"

async def run_one(config, repeats=1):
    results=[]
    for rep in range(repeats):
        cfg = dict(config)
        cfg["environment_hash"] = env_hash()
        cfg["run_id"] = f"{cfg['name']}-rep{rep}-{int(time.time())}"
        cfg["repeat"] = rep
        wl = generate(
            workload_type=cfg.get("workload_type","synthetic"),
            n=cfg.get("request_count",20),
            input_tokens=cfg.get("input_tokens",512),
            output_tokens=cfg.get("output_tokens",64),
            arrival_rate=cfg.get("arrival_rate",0),
            arrival_distribution=cfg.get("arrival_distribution","closed"),
            prefix_reuse_fraction=cfg.get("prefix_reuse_fraction",0),
            seed=cfg.get("seed",0),
        )
        print(f"\n=== {cfg['run_id']} ===")
        print(json.dumps({k:v for k,v in cfg.items() if k not in ["environment_hash"]}, indent=2))
        print(f" workload {describe_workload(wl)}")
        result = await run_benchmark(
            base_url=BASE_URL,
            workload=wl,
            config=cfg,
            concurrency=cfg.get("concurrency",1),
            warmup_requests=cfg.get("warmup_requests",1),
            stream=cfg.get("stream", True),
            sampler_interval=cfg.get("sampler_interval",0.3),
            timeout=cfg.get("timeout",120),
            arrival_distribution=cfg.get("arrival_distribution","closed"),
        )
        req_path, sys_path, json_path = save_raw(result, OUT_DIR)
        proc = compute_processed(result)
        proc_path = OUT_DIR / f"{result.run_id}_processed.json"
        with open(proc_path, "w", encoding="utf-8") as f:
            json.dump(proc, f, indent=2)
        print(f" -> throughput {proc['throughput_rps']:.3f} rps token {proc['token_throughput']:.1f} p50 {proc['latency']['p50']:.3f} p95 {proc['latency']['p95']:.3f} TTFT p50 {proc['ttft']['p50']:.4f} CPU {proc['system']['cpu']['mean']:.1f}% succ {proc['success']}/{proc['total_requests']}")
        results.append(proc)
        await asyncio.sleep(0.5)
    return results

def make_configs():
    configs=[]
    # Sweep A: Concurrency coarse (synthetic 512/64 closed)
    for c in [1,2,4,8,16,32]:
        configs.append({
            "name": f"sweepA_conc_c{c:03d}",
            "workload_type": "synthetic",
            "request_count": 40,
            "input_tokens": 512,
            "output_tokens": 64,
            "concurrency": c,
            "arrival_rate": 0,
            "arrival_distribution": "closed",
            "prefix_reuse_fraction": 0,
            "seed": 100+c,
            "warmup_requests": 2,
            "stream": True,
            "timeout": 180,
            "sampler_interval": 0.3,
            "description": f"Concurrency sweep c={c}, synthetic 512/64 closed"
        })
    # Sweep B: Input length
    for inp in [128,256,512,768,1024]:
        configs.append({
            "name": f"sweepB_input_i{inp:04d}",
            "workload_type": "synthetic",
            "request_count": 20,
            "input_tokens": inp,
            "output_tokens": 64,
            "concurrency": 2,
            "arrival_rate": 0,
            "arrival_distribution": "closed",
            "prefix_reuse_fraction": 0,
            "seed": 200+inp,
            "warmup_requests": 2,
            "stream": True,
            "timeout": 120,
            "sampler_interval": 0.3,
            "description": f"Input sweep input={inp}, conc2, synthetic 64 out"
        })
    # Sweep C: Output length
    for out in [16,32,64,128,256]:
        configs.append({
            "name": f"sweepC_output_o{out:03d}",
            "workload_type": "synthetic",
            "request_count": 20,
            "input_tokens": 512,
            "output_tokens": out,
            "concurrency": 2,
            "arrival_rate": 0,
            "arrival_distribution": "closed",
            "prefix_reuse_fraction": 0,
            "seed": 300+out,
            "warmup_requests": 2,
            "stream": True,
            "timeout": 180,
            "sampler_interval": 0.3,
            "description": f"Output sweep out={out}, 512 input conc2"
        })
    # Sweep D: Prefix reuse
    for r in [0.0,0.2,0.5,0.8,1.0]:
        # encode as integer for name
        ri=int(r*10)
        configs.append({
            "name": f"sweepD_reuse_r{ri}",
            "workload_type": "prefix_reuse",
            "request_count": 20,
            "input_tokens": 1024,
            "output_tokens": 64,
            "concurrency": 4,
            "arrival_rate": 0,
            "arrival_distribution": "closed",
            "prefix_reuse_fraction": r,
            "seed": 400+ri,
            "warmup_requests": 2,
            "stream": True,
            "timeout": 120,
            "sampler_interval": 0.3,
            "description": f"Prefix reuse sweep r={r}, 1024/64 conc4"
        })
    # Sweep E: Arrival pattern
    # closed baseline
    configs.append({
        "name": "sweepE_arrival_closed",
        "workload_type": "synthetic",
        "request_count": 40,
        "input_tokens": 512,
        "output_tokens": 64,
        "concurrency": 8,
        "arrival_rate": 0,
        "arrival_distribution": "closed",
        "prefix_reuse_fraction": 0,
        "seed": 500,
        "warmup_requests": 2,
        "stream": True,
        "timeout": 180,
        "sampler_interval": 0.3,
        "description": "Arrival closed conc8 512/64"
    })
    for rate in [2,4,8]:
        for dist in ["poisson","bursty","gamma","uniform"]:
            configs.append({
                "name": f"sweepE_arrival_{dist}_r{rate}",
                "workload_type": "synthetic",
                "request_count": 40,
                "input_tokens": 512,
                "output_tokens": 64,
                "concurrency": 8,
                "arrival_rate": rate,
                "arrival_distribution": dist,
                "prefix_reuse_fraction": 0,
                "seed": 510+rate*10+hash(dist)%10,
                "warmup_requests": 2,
                "stream": True,
                "timeout": 180,
                "sampler_interval": 0.3,
                "description": f"Arrival {dist} rate={rate} conc8"
            })
    # Sweep F: Workload types
    for wt in ["synthetic","chat","prefix_reuse","rag","agent","long_context"]:
        # for long_context use 1024 to avoid truncation beyond
        inp = 1024 if wt!="long_context" else 1024
        # reuse fraction for prefix_reuse
        reuse = 0.8 if wt=="prefix_reuse" else 0
        configs.append({
            "name": f"sweepF_workload_{wt}",
            "workload_type": wt,
            "request_count": 20,
            "input_tokens": inp,
            "output_tokens": 64,
            "concurrency": 2,
            "arrival_rate": 0,
            "arrival_distribution": "closed",
            "prefix_reuse_fraction": reuse,
            "seed": 600+hash(wt)%100,
            "warmup_requests": 2,
            "stream": True,
            "timeout": 120,
            "sampler_interval": 0.3,
            "description": f"Workload type {wt} 1024/64 conc2"
        })
    # Additional: isolate short vs long input-output ratio
    # input/output ratio sweep
    for ratio_name, inp, out in [("r_low","128","32"),("r_bal","512","64"),("r_high","1024","128"),("r_long_in","1024","32"),("r_long_out","128","256")]:
        configs.append({
            "name": f"sweepG_ratio_{ratio_name}",
            "workload_type": "synthetic",
            "request_count": 20,
            "input_tokens": int(inp),
            "output_tokens": int(out),
            "concurrency": 2,
            "arrival_rate": 0,
            "arrival_distribution": "closed",
            "prefix_reuse_fraction": 0,
            "seed": 700+hash(ratio_name)%100,
            "warmup_requests": 2,
            "stream": True,
            "timeout": 180,
            "sampler_interval": 0.3,
            "description": f"Ratio sweep {ratio_name} {inp}/{out}"
        })
    # Isolation / mixed workload: 50% short + 50% long interleaved vs homogeneous
    # We'll do homogeneous short and long already in F/G, but add mixed via manual generation later if needed; for now placeholder synthetic mixed not via generator factory
    # Add concurrency 64 coarse
    configs.append({
        "name": "sweepA_conc_c064",
        "workload_type": "synthetic",
        "request_count": 40,
        "input_tokens": 512,
        "output_tokens": 64,
        "concurrency": 64,
        "arrival_rate": 0,
        "arrival_distribution": "closed",
        "prefix_reuse_fraction": 0,
        "seed": 164,
        "warmup_requests": 2,
        "stream": True,
        "timeout": 180,
        "sampler_interval": 0.3,
        "description": "Concurrency sweep c=64 extension"
    })

    return configs

async def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--filter", type=str, default=None, help="substring filter on name")
    args=ap.parse_args()
    configs=make_configs()
    if args.filter:
        configs=[c for c in configs if args.filter in c["name"]]
        print(f"filtered to {len(configs)} configs with '{args.filter}'")
    else:
        print(f"total {len(configs)} configs")
    # save configs to dir
    for c in configs:
        p=CONFIG_DIR / f"{c['name']}.json"
        with open(p,"w",encoding="utf-8") as f:
            json.dump(c,f,indent=2)
    print(f"saved {len(configs)} configs to {CONFIG_DIR}")
    # run sequentially
    all_results=[]
    for cfg in configs:
        try:
            res=await run_one(cfg, repeats=args.repeats)
            all_results.extend(res)
        except Exception as e:
            print(f"ERROR running {cfg['name']}: {e}")
            import traceback; traceback.print_exc()
    # summary
    print("\n=== SWEEP SUMMARY ===")
    for r in all_results:
        print(f"{r['config']['name']} thr {r['throughput_rps']:.3f} p50 {r['latency']['p50']:.3f} p95 {r['latency']['p95']:.3f} p99 {r['latency']['p99']:.3f} TTFT {r['ttft']['p50']:.3f} CPU {r['system']['cpu']['mean']:.1f}")

if __name__=="__main__":
    asyncio.run(main())
