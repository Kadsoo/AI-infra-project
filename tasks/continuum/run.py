"""One-click entry: python run.py --load swe|bfcl|mixed --policy all|baseline|prog|static|full ..."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from continuum_sim import Sim
from gen_loads import add_arrivals, gen_programs


def summarize(res):
    ps = res["programs"]
    j = [p["jct"] for p in ps]
    q = [p["avg_q"] for p in ps]
    pre = sum(p["prefill_s"] for p in ps)
    hits = sum(p["hits"] for p in ps)
    pins = sum(p["pins"] for p in ps)
    j.sort()
    def pct(a, p):
        return a[min(len(a) - 1, int(p * len(a)))] if a else 0.0
    return {"n": len(ps), "jct_mean": statistics.mean(j) if j else 0,
            "jct_p50": pct(j, 0.5), "jct_p90": pct(j, 0.9),
            "avg_q": statistics.mean(q) if q else 0,
            "prefill_total_s": pre, "pins": pins, "hits": hits,
            "hit_rate": hits / pins if pins else 0.0,
            "pin_token_s": res["pin_token_s"],
            "expired_wasted": res["expired_wasted"],
            "hol_events": len(res["hol"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default="mixed", choices=["swe", "bfcl", "mixed"])
    ap.add_argument("--policy", default="all",
                    choices=["all", "baseline", "prog", "static", "full"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--jps", type=float, default=0.15)
    ap.add_argument("--gpu-cap", type=int, default=150000)
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--token-scale", type=float, default=0.15,
                    help="scale paper tokens to fit sim GPU cap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    kinds = {"swe": ["swe"], "bfcl": ["bfcl"], "mixed": ["mixed"]}[args.load]
    progs = []
    for i, k in enumerate(kinds):
        sub = gen_programs(k, args.n, seed=args.seed + i * 1000,
                           token_scale=args.token_scale,
                           max_ctx=int(args.gpu_cap * 0.5))
        progs.extend(sub)
    for i, p in enumerate(progs):
        p["pid"] = i
    add_arrivals(progs, args.jps, seed=args.seed)

    pols = (["baseline", "prog", "static", "full"] if args.policy == "all"
            else [args.policy])
    table = {}
    full_res = {}
    for pol in pols:
        sim = Sim(progs, gpu_cap_tokens=args.gpu_cap, offload=args.offload,
                  policy=pol)
        res = sim.run()
        table[pol] = summarize(res)
        full_res[pol] = res
    print(f"load={args.load} n={len(progs)} jps={args.jps} cap={args.gpu_cap} "
          f"offload={args.offload} scale={args.token_scale}")
    print(f"{'policy':10s} {'n':>4s} {'JCTmean':>9s} {'JCTp50':>8s} {'JCTp90':>8s} "
          f"{'avgQ':>7s} {'prefill':>9s} {'pins':>5s} {'hit%':>6s} {'HOL':>5s}")
    for pol in pols:
        s = table[pol]
        print(f"{pol:10s} {s['n']:4d} {s['jct_mean']:9.2f} {s['jct_p50']:8.2f} "
              f"{s['jct_p90']:8.2f} {s['avg_q']:7.2f} "
              f"{s['prefill_total_s']:9.1f} {s['pins']:5d} "
              f"{100*s['hit_rate']:5.1f}% {s['hol_events']:5d}")
    if "baseline" in table and "full" in table:
        b, f = table["baseline"]["jct_mean"], table["full"]["jct_mean"]
        print(f"full vs baseline JCT: {b:.2f} -> {f:.2f} "
              f"({b/f:.2f}x)" if f > 0 else "n/a")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"args": vars(args), "summary": table,
                       "hol_full": full_res.get("full", {}).get("hol", [])[:20]},
                      fh, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
