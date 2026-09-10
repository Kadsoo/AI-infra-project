"""
E1 trace generator for RQ-4 (frozen experiment_plan.md 1.3).

Fixed 10,000-request traces, one per seeded replication (10 reps, seeds
1001..1010). A trace is stored in UNIT arrival-rate form (exponential mean 1)
plus a uniform class key; each E1 cell rescales the inter-arrival times by the
cell's Poisson rate lambda = load_frac * lambda_sat(p_hot, gamma) so that the
identical inter-arrival sequence is replayed across policies at equal load
(Confounder 5) and across cells (same fixed traces, recorded).

Request fields (all recorded):
  seq         1-based request index
  unit_dt     exponential(1) inter-arrival time (unit rate; scaled per cell)
  u_class     uniform [0,1) draw; hot iff u_class < p_hot (per-cell skew)
  total_tokens  integer Uniform(1920, 2176), mean 2048 (both classes)
  max_tokens  256 (fixed output length control; not part of prefill TTFT)

Reuse structure per frozen 1.3: the hot class shares ONE prefix of length
1024 = L_prefix (hot requests all carry the shared prefix; hot suffix =
total - 1024, mean 1024). Cold prefixes are unique and never reused.

NOTE (recorded in experiment_log.md): the task line "60% hot-prefix reuse
structure" has no counterpart in the frozen workload spec; the frozen spec
(1.3: "hot prefix shared by fraction p_hot of requests; cold prefixes never
reused") is implemented as-is.

Outputs: raw/trace_10000_repNN.csv (NN = 01..10).
"""

import os
import sys
import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
os.makedirs(RAW, exist_ok=True)

N_REQUESTS = 10000
SEEDS = list(range(1001, 1011))          # 10 seeded replications
TOT_LO, TOT_HI = 1920, 2176              # integer tokens
MAX_TOKENS = 256


def gen_trace(seed):
    rng = np.random.default_rng(seed)
    unit_dt = rng.exponential(1.0, N_REQUESTS)
    u_class = rng.uniform(0.0, 1.0, N_REQUESTS)
    total_tokens = rng.integers(TOT_LO, TOT_HI + 1, N_REQUESTS)
    return pd.DataFrame({
        "seq": np.arange(1, N_REQUESTS + 1),
        "unit_dt": unit_dt,
        "u_class": u_class,
        "total_tokens": total_tokens,
        "max_tokens": np.full(N_REQUESTS, MAX_TOKENS, dtype=np.int64),
    })


def main():
    t0 = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print("trace generation start", t0)
    for seed in SEEDS:
        df = gen_trace(seed)
        path = os.path.join(RAW, "trace_10000_rep%02d.csv" % (seed - 1000))
        df.to_csv(path, index=False)
        print("wrote", path)
    print("trace generation done", datetime.datetime.now(datetime.timezone.utc).isoformat())


if __name__ == "__main__":
    main()