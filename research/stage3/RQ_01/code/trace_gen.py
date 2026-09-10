"""RQ-1 fixed 5,000-request trace generator (Stage 3B Wave-1, CPU-only).

Spec source: research/stage3/RQ_01/experiment_plan.md (E1/E2 Workload sections).
- 5,000 requests: 70% short-class (median ~1020 tokens), 30% long-class (8k-13k, lognormal mean ~9k)
- output length fixed 200 tokens
- 60% of long-class share a common hot prefix (hot flag)
- inter-arrival times at fixed mean rate, CV in {1,2,3} via Gamma renewal:
  Gamma(shape a) inter-arrival has CV = 1/sqrt(a) exactly -> a = 1/CV^2.
  (Implementation-level choice, recorded: the plan suggests ON/OFF two-state Markov
   modulation; a Gamma renewal achieves the exact target CV deterministically.)
"""
import argparse
import hashlib
import json
import os
import numpy as np
import pandas as pd

N_REQUESTS = 5000
DEFAULT_SEED = 20260827
HOT_PREFIX_LEN = 6144  # implementation-level: shared hot prefix length for long class


def gen_trace(n: int, rng: np.random.Generator) -> pd.DataFrame:
    n_short = int(round(0.7 * n))
    n_long = n - n_short
    short = np.clip(rng.lognormal(np.log(1020.0), 0.35, n_short), 256, 3072).astype(int)
    s = 0.2
    mu = np.log(9000.0) - s * s / 2.0
    long_ = np.clip(rng.lognormal(mu, s, n_long), 8000, 13000).astype(int)
    prompt_len = np.concatenate([short, long_])
    cls = np.array(["short"] * n_short + ["long"] * n_long)
    hot = np.zeros(n, dtype=int)
    hot[cls == "long"] = (rng.random(n_long) < 0.6).astype(int)
    perm = rng.permutation(n)
    df = pd.DataFrame({
        "request_id": np.arange(1, n + 1),
        "cls": cls[perm],
        "prompt_len": prompt_len[perm],
        "output_len": np.full(n, 200),
        "hot": hot[perm],
    })
    return df


def arrivals(rng: np.random.Generator, lam: float, cv: float, n: int) -> np.ndarray:
    """Gamma-renewal inter-arrival times with exact CV = 1/sqrt(a)."""
    a = 1.0 / (cv * cv)
    theta = 1.0 / (a * lam)
    gaps = rng.gamma(a, theta, n)
    return np.cumsum(gaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True, help="mean arrival rate (req/s)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default="raw/trace_5000.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = gen_trace(N_REQUESTS, rng)
    # separate RNG streams per CV so the per-CV arrival sequences are independent but fixed
    for cv, col in [(1.0, "arrival_cv1"), (2.0, "arrival_cv2"), (3.0, "arrival_cv3")]:
        r = np.random.default_rng(args.seed + int(cv * 1000))
        df[col] = arrivals(r, args.lam, cv, N_REQUESTS)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    meta = {
        "n": N_REQUESTS, "seed": args.seed, "lam": args.lam, "hot_prefix_len": HOT_PREFIX_LEN,
        "short_frac": float((df.cls == "short").mean()),
        "short_len_median": float(df.loc[df.cls == "short", "prompt_len"].median()),
        "long_len_mean": float(df.loc[df.cls == "long", "prompt_len"].mean()),
        "long_len_p50": float(df.loc[df.cls == "long", "prompt_len"].quantile(0.5)),
        "long_len_p95": float(df.loc[df.cls == "long", "prompt_len"].quantile(0.95)),
        "long_len_p99": float(df.loc[df.cls == "long", "prompt_len"].quantile(0.99)),
        "hot_share_of_long": float(df.loc[df.cls == "long", "hot"].mean()),
        "achieved_cv": {cv: float(df[col].diff().dropna().std() / df[col].diff().dropna().mean())
                        for cv, col in [(1.0, "arrival_cv1"), (2.0, "arrival_cv2"), (3.0, "arrival_cv3")]},
        "all_len_p50": float(df["prompt_len"].quantile(0.5)),
        "all_len_p95": float(df["prompt_len"].quantile(0.95)),
    }
    meta_path = os.path.join(os.path.dirname(args.out) or ".", "trace_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    with open(args.out, "rb") as f:
        print("trace sha256:", hashlib.sha256(f.read()).hexdigest())
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
