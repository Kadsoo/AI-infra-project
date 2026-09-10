"""RQ-1 E2 driver: mean-rate probe, conditions A-E sweep, colocated baselines, reports.
Spec: research/stage3/RQ_01/experiment_plan.md # E2. Run: python code/e2_run.py
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from e2_sim import SimPD, SimColocated, summarize

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = os.path.join(BASE, "raw", "trace_5000.csv")
SLO_P95 = 4.0
SLO_ATTAIN = 0.95

CONDITIONS = {
    "A": {"cv": 1, "policy": "affine", "n_prefill": 1},
    "B": {"cv": 3, "policy": "affine", "n_prefill": 1},
    "C": {"cv": 3, "policy": "loadbal", "n_prefill": 1},
    "D": {"cv": 2, "policy": "affine", "n_prefill": 1},
    "E": {"cv": 3, "policy": "affine", "n_prefill": 2},
}
SERVICE_SCALES = [0.5, 1.0, 2.0]
TRANSFER_SCALES = [0.7, 1.0, 1.4]
SEEDS = [1, 2, 3, 4, 5]


def load_trace():
    return pd.read_csv(TRACE)


def run_probe(trace, lam_candidates):
    rows = []
    for lam in lam_candidates:
        rng = np.random.default_rng(1)
        s = SimColocated(rng, lam, 1.0, 1.0, trace)
        res = s.run()
        m = summarize(res, len(trace))
        rows.append({"lam": lam, "p95": m["ttft_p95"], "attainment": m["attainment_p95_slo"]})
        print(f"probe lam={lam}: P95={m['ttft_p95']:.3f}s attainment={m['attainment_p95_slo']:.3f}")
    ok = [r for r in rows if r["attainment"] >= SLO_ATTAIN and r["p95"] <= SLO_P95]
    if not ok:
        return rows, None
    return rows, max(ok, key=lambda r: r["lam"])["lam"]


def main():
    trace = load_trace()
    probe_rows, lam = run_probe(trace, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    if lam is None:
        raise SystemExit("no lam satisfies the SLO probe; pick largest-attainment point manually")
    print(f"chosen mean rate lam = {lam} req/s")

    # regenerate the trace at the chosen rate so arrival times match lam exactly
    import subprocess, sys
    subprocess.run([sys.executable, os.path.join(BASE, "code", "trace_gen.py"),
                    "--lam", str(lam), "--out", TRACE], check=True, cwd=BASE)
    trace = load_trace()

    runs = []
    for cond, cfg in CONDITIONS.items():
        for ss in SERVICE_SCALES:
            for ts in TRANSFER_SCALES:
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    s = SimPD(rng, lam, cfg["cv"], cfg["policy"], cfg["n_prefill"], ss, ts, trace)
                    res = s.run()
                    m = summarize(res, len(trace))
                    m["max_pair_share"] = max(s.arrivals_per_pair) / sum(s.arrivals_per_pair)
                    runs.append({"condition": cond, "cv": cfg["cv"], "policy": cfg["policy"],
                                 "n_prefill": cfg["n_prefill"], "service_scale": ss,
                                 "transfer_scale": ts, "seed": seed, **m})
    colo = []
    for cv in [1.0, 2.0, 3.0]:
        for ss in SERVICE_SCALES:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                s = SimColocated(rng, lam, cv, ss, trace)
                res = s.run()
                m = summarize(res, len(trace))
                m["max_pair_share"] = 0.5
                colo.append({"cv": cv, "service_scale": ss, "seed": seed, **m})

    df_pd = pd.DataFrame(runs)
    df_colo = pd.DataFrame(colo)
    os.makedirs(os.path.join(BASE, "raw"), exist_ok=True)
    df_pd.to_csv(os.path.join(BASE, "raw", "e2_pd_runs.csv"), index=False)
    df_colo.to_csv(os.path.join(BASE, "raw", "e2_colocated_runs.csv"), index=False)

    # aggregate per cell: mean/std over seeds
    agg = []
    for (cond, ss), g in df_pd.groupby(["condition", "service_scale"]):
        ts = g["transfer_scale"].iloc[0]
        c = CONDITIONS[cond]
        colo_key = (float(c["cv"]), ss)
        cg = df_colo[(df_colo["cv"] == colo_key[0]) & (df_colo["service_scale"] == colo_key[1])]
        c_p99 = cg["ttft_p99"]
        agg.append({
            "condition": cond, "cv": c["cv"], "policy": c["policy"], "n_prefill": c["n_prefill"],
            "service_scale": ss, "transfer_scale": ts,
            "p99_mean": g["ttft_p99"].mean(), "p99_std": g["ttft_p99"].std(),
            "p95_mean": g["ttft_p95"].mean(), "p50_mean": g["ttft_p50"].mean(),
            "qfrac_p99_mean": g["qfrac_p99"].mean(), "qfrac_p99_std": g["qfrac_p99"].std(),
            "p99_ratio_vs_colo_mean": float((g["ttft_p99"] / c_p99.mean()).mean()),
            "p99_ratio_vs_colo_std": float((g["ttft_p99"] / c_p99.mean()).std()),
            "colo_p99_mean": float(c_p99.mean()),
            "max_pair_share_mean": g["max_pair_share"].mean(),
            "attainment_mean": g["attainment_p95_slo"].mean(),
        })
    df_agg = pd.DataFrame(agg)
    df_agg.to_csv(os.path.join(BASE, "raw", "e2_summary.csv"), index=False)

    config = {"lam": lam, "probe": probe_rows, "slo_p95": SLO_P95, "slo_attain": SLO_ATTAIN,
              "hot_prefix_len": 6144, "transfer_dist": "lognormal mean 8ms x scale, sigma 0.5",
              "arrival_dist": "Gamma renewal CV=1/sqrt(a)", "seeds": SEEDS}
    with open(os.path.join(BASE, "configs", "e2_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    for f in ["code/e2_sim.py", "code/e2_run.py", "code/trace_gen.py", "code/e1_cost_model.py"]:
        p = os.path.join(BASE, f)
        if os.path.exists(p):
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            print(f"{f} sha256: {h}")
    print(df_agg.to_string(index=False))
    print("all done")


if __name__ == "__main__":
    main()