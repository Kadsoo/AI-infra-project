"""RQ-1 E2 diagnostic: low-load runs to test reachability of the kill-condition
conjunct 'P99 queue fraction < 15%' (frozen H2 kill branch 1).
Runs conditions A/B/C at lam in {0.5, 1.0} (below the matched-SLO rate 3.0).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
from e2_sim import SimPD, SimColocated, summarize

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
trace = pd.read_csv(os.path.join(BASE, "raw", "trace_5000.csv"))

rows = []
for lam in [0.5, 1.0]:
    for cond, cv, policy in [("A", 1, "affine"), ("B", 3, "affine"), ("C", 3, "loadbal")]:
        for seed in [1, 2, 3]:
            rng = np.random.default_rng(seed)
            s = SimPD(rng, lam, cv, policy, 1, 1.0, 1.0, trace)
            res = s.run()
            m = summarize(res, len(trace))
            m["max_pair_share"] = max(s.arrivals_per_pair) / sum(s.arrivals_per_pair)
            rows.append({"lam": lam, "cond": cond, "cv": cv, "policy": policy, "seed": seed, **m})
            rng = np.random.default_rng(seed)
            c = SimColocated(rng, lam, cv, 1.0, trace)
            cres = c.run()
            cm = summarize(cres, len(trace))
            rows.append({"lam": lam, "cond": "COL", "cv": cv, "policy": "colocated",
                         "seed": seed, "p99": cm["ttft_p99"], "qfrac_p99": cm["qfrac_p99"],
                         "ttft_p50": cm["ttft_p50"], "ttft_p95": cm["ttft_p95"]})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(BASE, "raw", "e2_lowload_diag.csv"), index=False)
print(df.to_string(index=False))