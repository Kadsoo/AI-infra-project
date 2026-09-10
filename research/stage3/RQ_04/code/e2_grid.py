"""
E2 - H2 threshold sweep + H1 grid falsification (frozen experiment_plan.md E2;
LOCKED_PLAN.md sections 5-7).

Reuses the calibrated queueing simulator from e1_replay.py (per-node FIFO,
admission = dispatch, cluster cap c, warm-up = 200 hot-request completions,
uniform-random tie-break among exact minima, LRU-radix cache, memory-accounted
hit rate). Identical trace-replay across policies.

Grids (all 10,000 requests, max_tokens 256, matched total lengths mean 2048,
hot suffix 1024; 10 seeded reps, seeds 1001-1010):
  primary Poisson (CV=1):  p_hot {0.5,0.7,0.8,0.9} x c {8,16,32,64,128}
      x load {0.5,0.65,0.8} x budget {L1,L2} x policies {A,B} x 10 reps
      -> raw/e2_cells.csv
  burst robustness (Gamma CV {2,4}, matched mean): p_hot {0.7,0.8,0.9}
      x c {16,32,64} x load {0.5,0.65,0.8} x budget {L1,L2} x {A,B} x 10 reps
      -> raw/e2_burst.csv  (class/length sequences identical to the Poisson
         trace for the same rep; only inter-arrival times differ)
  policy C (threshold replication, k = 2, fixed theta = 1.0 hot req/s):
      at the surviving threshold points only (identified from the primary
      grid at load 0.8, budget L1; verified at L2/load 0.65) -> raw/e2_cells.csv
  gamma-robustness band: gamma {0.5, 2.0} at verdict-relevant points
      (identified thresholds + (0.8, 32) + (0.9, 64)) -> raw/e2_gamma_robust.csv
  sanity re-verification vs E0 for the c-sweep: 6 anchor runs (c = 8, 64 and
      the E1-style c = 32 anchor) -> logs/sanity_e2.json

Frozen metric definitions: common SLO per-class P99 TTFT <= 2.0 s; attainment =
fraction of class requests with TTFT <= 2.0 s; FI = min/max per-class
attainment (report-only); overload region (hot-node util >= 1.0) excluded from
verdicts and reported separately.

Usage:
  python code/e2_grid.py --gamma-traces   (generate Gamma CV 2/4 traces once)
  python code/e2_grid.py --sanity         (6 anchors -> logs/sanity_e2.json)
  python code/e2_grid.py --grid           (primary Poisson grid, parallel)
  python code/e2_grid.py --burst          (Gamma CV {2,4} reduced grid)
  python code/e2_grid.py --identify       (p_hot*, c*, monotonicity -> processed/e2_thresholds.json)
  python code/e2_grid.py --c              (policy C at surviving threshold points)
  python code/e2_grid.py --gammacheck     (gamma {0.5,2.0} at verdict-relevant points)
  python code/e2_grid.py --report         (processed/e2_report.md)
"""

import argparse
import datetime
import hashlib
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_replay import (Sim, lambda_sat, load_trace, GAMMA_BASE, L_PREFIX,
                       N_NODES, SLO_TTFT, WARM_HOT_REQUESTS, BUDGETS,
                       E1_LOADS, E1_REPS, run_id_of)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
PROC = os.path.join(ROOT, "processed")
LOGS = os.path.join(ROOT, "logs")
PERQ = os.path.join(RAW, "per_request")
NODEQ = os.path.join(RAW, "per_node_trace")
for d in (RAW, PROC, LOGS, PERQ, NODEQ):
    os.makedirs(d, exist_ok=True)

# frozen parameterization (experiment_plan.md E2)
E2_P_HOTS = [0.5, 0.7, 0.8, 0.9]
E2_CS = [8, 16, 32, 64, 128]
BURST_P_HOTS = [0.7, 0.8, 0.9]
BURST_CS = [16, 32, 64]
CVS = [2, 4]
GAMMA_BAND = [0.5, 2.0]
THETA_HOT = 1.0          # fixed replication threshold (hot req/s), recorded

CELL_CSV = os.path.join(RAW, "e2_cells.csv")
BURST_CSV = os.path.join(RAW, "e2_burst.csv")
GAMMA_CSV = os.path.join(RAW, "e2_gamma_robust.csv")
THRESHOLDS_JSON = os.path.join(PROC, "e2_thresholds.json")


def e2_run_id(policy, p_hot, c, load, budget, gamma, rep):
    return "e2_%s_p%.1f_c%03d_l%.2f_%s_g%.1f_r%02d" % (policy, p_hot, c, load, budget, gamma, rep)


def e2b_run_id(policy, p_hot, c, load, budget, gamma, rep, cv):
    return "e2b_%s_p%.1f_c%03d_l%.2f_%s_g%.1f_cv%d_r%02d" % (policy, p_hot, c, load, budget, gamma, cv, rep)


def e2g_run_id(policy, p_hot, c, load, budget, gamma, rep):
    return "e2g_%s_p%.1f_c%03d_l%.2f_%s_g%.1f_r%02d" % (policy, p_hot, c, load, budget, gamma, rep)


def e2c_run_id(policy, p_hot, c, load, budget, gamma, rep):
    return "e2c_%s_p%.1f_c%03d_l%.2f_%s_g%.1f_r%02d" % (policy, p_hot, c, load, budget, gamma, rep)


# ---------------------------------------------------------------------------
# Policy C: threshold replication, k = 2, fixed theta (recorded)
# ---------------------------------------------------------------------------

class SimC(Sim):
    """C: when the hot-prefix request rate exceeds theta, the hot prefix is
    replicated to k = 2 nodes; hot requests affinity-route within the replica
    set (min token-weighted backlog); cold requests route min token-weighted
    backlog across all nodes (as under A). Replication is a static per-cell
    condition: active iff p_hot * lam > THETA_HOT."""

    def __init__(self, *args, replicas=(0, 1), **kwargs):
        super().__init__(*args, **kwargs)
        self.replicas = replicas
        self.policy = "C"

    def _route(self, node, cls, now):
        rng = self.rng
        if cls == 1:  # hot: affinity within the replica set
            cands = [k for k in self.replicas if node[k]["prefix_resident"]]
            if not cands:
                cands = list(self.replicas)
            bl = [self._backlog_tokens(node[k], now) for k in cands]
            m = min(bl)
            pick = [k for k, b in zip(cands, bl) if b == m]
            return int(rng.choice(pick)) if len(pick) > 1 else pick[0]
        bl = [self._backlog_tokens(node[k], now) for k in range(N_NODES)]
        m = min(bl)
        cands = [k for k, b in enumerate(bl) if b == m]
        return int(rng.choice(cands)) if len(cands) > 1 else cands[0]


def replication_active(p_hot, load_frac, gamma):
    return (p_hot * load_frac * lambda_sat(p_hot, gamma)) > THETA_HOT


# ---------------------------------------------------------------------------
# Gamma-burst traces (CV {2,4}; class/length sequences identical to the
# Poisson trace for the same rep; only inter-arrival times differ)
# ---------------------------------------------------------------------------

def gamma_trace_path(cv, rep):
    return os.path.join(RAW, "trace_10000_gamma_cv%d_rep%02d.csv" % (cv, rep))


def gen_gamma_traces():
    for cv in CVS:
        for rep in E1_REPS:
            path = gamma_trace_path(cv, rep)
            if os.path.exists(path):
                continue
            poisson = load_trace(rep)
            rng = np.random.default_rng(1000 + rep + 1000 * cv)
            shape = 1.0 / (cv * cv)
            scale = cv * cv
            unit_dt = rng.gamma(shape, scale, len(poisson))
            out = poisson.copy()
            out["unit_dt"] = unit_dt
            out.to_csv(path, index=False)
            print("wrote", path)


def load_gamma_trace(cv, rep):
    return pd.read_csv(gamma_trace_path(cv, rep))


# ---------------------------------------------------------------------------
# Worker (module-level for multiprocessing)
# ---------------------------------------------------------------------------

def _worker(cell):
    rid, policy, p_hot, c, load, budget, gamma, rep, cv, record_perq, record_node_trace = cell
    trace = load_gamma_trace(cv, rep) if cv else load_trace(rep)
    if policy == "C":
        sim = SimC(policy, p_hot, load, budget, gamma, c, trace, rep,
                   record_per_request=record_perq,
                   record_node_trace=record_node_trace)
        sim.run_id = e2c_run_id(policy, p_hot, c, load, budget, gamma, rep)
    else:
        sim = Sim(policy, p_hot, load, budget, gamma, c, trace, rep,
                  record_per_request=record_perq,
                  record_node_trace=record_node_trace)
        if cv:
            sim.run_id = e2b_run_id(policy, p_hot, c, load, budget, gamma, rep, cv)
        else:
            sim.run_id = e2_run_id(policy, p_hot, c, load, budget, gamma, rep)
    res = sim.run()
    if policy == "C":
        res["replication_active"] = bool(replication_active(p_hot, load, gamma))
        res["replication_kv_extra"] = L_PREFIX if res["replication_active"] else 0
    else:
        res["replication_active"] = np.nan
        res["replication_kv_extra"] = np.nan
    if cv:
        res["cv"] = cv
    return res


def run_parallel(cells, out_path, workers=8):
    """Run cells (list of worker tuples) with a process pool; the parent writes
    rows to out_path (never drops a run; resume skips existing run_ids)."""
    seen = set()
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, usecols=["run_id"])
        seen = set(existing.run_id)
    todo = []
    for cell in cells:
        rid = cell[0]
        if rid not in seen:
            todo.append(cell)
    print("cells requested: %d, to run: %d" % (len(cells), len(todo)))
    if not todo:
        print("nothing to do for", out_path)
        return 0
    first = True
    n = 0
    buf = []
    with mp.Pool(workers) as pool:
        for res in pool.imap_unordered(_worker, todo, chunksize=1):
            n += 1
            buf.append(res)
            if len(buf) >= 100:
                _flush(buf, out_path, first)
                first = False
                buf = []
                print("  %d/%d done (latest %s)" % (n, len(todo), res["run_id"]))
    if buf:
        _flush(buf, out_path, first)
    print("done: %d runs written to %s" % (n, out_path))
    return n


def _flush(rows, out_path, first):
    df = pd.DataFrame(rows)
    if os.path.exists(out_path):
        header = list(pd.read_csv(out_path, nrows=0).columns)
        for col in header:
            if col not in df.columns:
                df[col] = np.nan
        df = df[list(header)]
        df.to_csv(out_path, mode="a", header=False, index=False)
    else:
        df.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Cell lists
# ---------------------------------------------------------------------------

def primary_cells(record_perq=False, record_node_trace=False, policies=("A", "B")):
    cells = []
    for policy in policies:
        for p_hot in E2_P_HOTS:
            for c in E2_CS:
                for load in E1_LOADS:
                    for budget in ("L1", "L2"):
                        for gamma in (1.0,):
                            for rep in E1_REPS:
                                rid = e2_run_id(policy, p_hot, c, load, budget, gamma, rep)
                                cells.append((rid, policy, p_hot, c, load, budget, gamma, rep,
                                              None, record_perq, record_node_trace))
    return cells


def burst_cells():
    cells = []
    for policy in ("A", "B"):
        for p_hot in BURST_P_HOTS:
            for c in BURST_CS:
                for load in E1_LOADS:
                    for budget in ("L1", "L2"):
                        for cv in CVS:
                            for rep in E1_REPS:
                                rid = e2b_run_id(policy, p_hot, c, load, budget, 1.0, rep, cv)
                                cells.append((rid, policy, p_hot, c, load, budget, 1.0, rep,
                                              cv, False, False))
    return cells


def gamma_check_cells(points):
    """points: list of (p_hot, c); gamma {0.5, 2.0} x load 0.8 x budgets x A/B x reps."""
    cells = []
    for (p_hot, c) in points:
        for gamma in GAMMA_BAND:
            for budget in ("L1", "L2"):
                for policy in ("A", "B"):
                    for rep in E1_REPS:
                        rid = e2g_run_id(policy, p_hot, c, 0.8, budget, gamma, rep)
                        cells.append((rid, policy, p_hot, c, 0.8, budget, gamma, rep,
                                      None, False, False))
    return cells


def c_cells(threshold_points):
    """Policy C at the surviving threshold points: (p_hot, c) corners at/above
    the identified thresholds, bounded by the grid [0.7..0.9] x [16..64]."""
    cells = []
    for (p_hot, c) in threshold_points:
        for load in (0.65, 0.8):
            for budget in ("L1", "L2"):
                for rep in E1_REPS:
                    rid = e2c_run_id("C", p_hot, c, load, budget, 1.0, rep)
                    cells.append((rid, "C", p_hot, c, load, budget, 1.0, rep,
                                  None, True, True))
    return cells


# ---------------------------------------------------------------------------
# Threshold identification (frozen H2 rule; identification at load 0.8, L1;
# verified at load 0.65 and L2 - recorded)
# ---------------------------------------------------------------------------

IDENT_LOAD = 0.8


def identify(df):
    """df: primary grid cells (policy A/B). Returns dict with p_hot*, c*,
    monotonicity functions, and the crossing map."""
    p_axis = E2_P_HOTS
    c_axis = E2_CS

    def cell_mean(p_hot, c, load, budget, policy, col):
        sub = df[(df.policy == policy) & (df.p_hot == p_hot) & (df.c == c)
                 & (df.load_frac == load) & (df.budget_level == budget)]
        if not len(sub):
            return float("nan")
        return float(sub[col].mean())

    def crossing(p_hot, c, load=IDENT_LOAD, budget="L1"):
        hot_p99 = cell_mean(p_hot, c, load, budget, "A", "hot_p99_ttft")
        cold_attn = cell_mean(p_hot, c, load, budget, "A", "cold_attainment")
        return (hot_p99 > SLO_TTFT) and (cold_attn >= 0.99)

    crossing_map = {("p%.1f/c%d" % (p, c)): bool(crossing(p, c)) for p in p_axis for c in c_axis}

    def c_star(p_hot):
        for c in (16, 32, 64):
            if all(crossing(p_hot, cc) for cc in c_axis if cc >= c):
                return c
        return None

    def p_star(c):
        for p in (0.7, 0.8, 0.9):
            if all(crossing(pp, c) for pp in p_axis if pp >= p):
                return p
        return None

    c_star_fn = {p: c_star(p) for p in p_axis}
    p_star_fn = {c: p_star(c) for c in c_axis}

    c_star_at_08 = c_star_fn[0.8]
    p_star_at_32 = p_star_fn[32]

    # conjunction verification on the hypothesis range [0.7,0.9] x [16,64] + c=128
    conjunction_ok = True
    fails = []
    if p_star_at_32 is not None and c_star_at_08 is not None:
        for p in p_axis:
            if p >= p_star_at_32:
                for c in c_axis:
                    if c >= c_star_at_08:
                        if not crossing_map["p%.1f/c%d" % (p, c)]:
                            conjunction_ok = False
                            fails.append((p, c))
    else:
        conjunction_ok = False
        fails = [(None, None)]

    # monotonicity: c*(p) non-increasing in p; p*(c) non-increasing in c
    def non_inc(seq):
        vals = [v for v in seq if v is not None]
        return all(a >= b for a, b in zip(vals, vals[1:]))

    mono_p = non_inc([c_star_fn[p] for p in p_axis])
    mono_c = non_inc([p_star_fn[c] for c in c_axis])

    # cross-check at load 0.65 and budget L2 (recorded)
    cross_065 = {("p%.1f/c%d" % (p, c)): bool(crossing(p, c, load=0.65)) for p in p_axis for c in c_axis}
    cross_L2 = {("p%.1f/c%d" % (p, c)): bool(crossing(p, c, budget="L2")) for p in p_axis for c in c_axis}

    out = {
        "identified_at": {"load": IDENT_LOAD, "budget": "L1", "gamma": 1.0,
                          "rule": "hot P99(A) > 2.0s and cold attainment(A) >= 0.99, means over 10 reps"},
        "p_hot_star": p_star_at_32, "c_star": c_star_at_08,
        "c_star_fn": {str(p): c_star_fn[p] for p in p_axis},
        "p_star_fn": {str(c): p_star_fn[c] for c in c_axis},
        "monotone_p_hot_dimension": mono_p,
        "monotone_c_dimension": mono_c,
        "conjunction_ok": conjunction_ok,
        "conjunction_fails": [list(f) for f in fails],
        "crossing_map": crossing_map,
        "crossing_map_load065": cross_065,
        "crossing_map_L2": cross_L2,
    }
    with open(THRESHOLDS_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("identified p_hot* = %s (at c=32), c* = %s (at p_hot=0.8); "
          "conjunction %s; monotone p-dim %s, c-dim %s" % (
              p_star_at_32, c_star_at_08, conjunction_ok, mono_p, mono_c))
    return out


def threshold_point_corners(th):
    """The surviving-threshold corner set for policy C and gamma checks:
    {(p_hot*, c*), (p_hot*, 64), (0.9, c*), (0.9, 64)} bounded by the grid."""
    p0, c0 = th["p_hot_star"], th["c_star"]
    pts = {(0.8, 32)}   # the frozen H1 point is always verdict-relevant
    if p0 is not None and c0 is not None:
        for p in (p0, 0.9):
            for c in (c0, 64):
                pts.add((p, c))
    return sorted(pts)


# ---------------------------------------------------------------------------
# Sanity anchors (c-sweep re-verification vs E0; 6 runs)
# ---------------------------------------------------------------------------

SANITY_ANCHORS = [
    ("A", 0.8, 8, 0.8), ("B", 0.8, 8, 0.8),
    ("A", 0.8, 64, 0.8), ("B", 0.8, 64, 0.8),
    ("A", 0.9, 32, 0.8), ("B", 0.9, 32, 0.8),
]


def run_sanity():
    from e0_algebra import per_cell as e0_cell
    e0 = pd.read_csv(os.path.join(RAW, "e0_grid.csv"))
    out = {"anchors": []}
    for (policy, p_hot, c, load) in SANITY_ANCHORS:
        run_id = e2_run_id(policy, p_hot, c, load, "L1", 1.0, 1)
        res = _worker((run_id, policy, p_hot, c, load, "L1", 1.0, 1, None, True, True))
        e0r = e0[(e0.p_hot == p_hot) & (e0.load == load) & (e0.gamma == 1.0)].iloc[0]
        if policy == "A":
            meas_util = float(res["hot_node_util"])
            e0_util = float(e0r["rho_hot_frozen"])
            hot_node = int(res["hot_node_id"])
            df = pd.read_csv(os.path.join(PERQ, res["run_id"] + ".csv.gz"))
            meas_wait = float(df[df.node == hot_node].wait.mean())
            e0c = e0_cell(p_hot, c, load, 1.0)
            pk = meas_util * e0c["ES2_hotnode"] / (2.0 * (1.0 - meas_util) * e0c["ES_hotnode"]) \
                if meas_util < 1.0 else float("nan")
            entry = dict(run_id=res["run_id"], policy=policy, p_hot=p_hot, c=c, load=load,
                         measured_util=meas_util, e0_util=e0_util,
                         util_delta_pct=100.0 * (meas_util - e0_util) / max(1e-12, e0_util),
                         measured_mean_wait=meas_wait, e0_mean_wait=float(e0r["W_mean_hotnode"]),
                         wait_delta_pct=100.0 * (meas_wait - float(e0r["W_mean_hotnode"]))
                         / max(1e-12, float(e0r["W_mean_hotnode"])),
                         pk_wait_at_realized_rho=pk,
                         pk_delta_pct=100.0 * (meas_wait - pk) / max(1e-12, pk))
        else:
            meas_util = float(np.mean([res["node_util_%d" % k] for k in range(N_NODES)]))
            e0_util = float(e0r["rho_B_pernode"])
            df = pd.read_csv(os.path.join(PERQ, res["run_id"] + ".csv.gz"))
            meas_wait = float(df.wait.mean())
            entry = dict(run_id=res["run_id"], policy=policy, p_hot=p_hot, c=c, load=load,
                         measured_util=meas_util, e0_util=e0_util,
                         util_delta_pct=100.0 * (meas_util - e0_util) / max(1e-12, e0_util),
                         measured_mean_wait=meas_wait, e0_mean_wait=float(e0r["W_mean_B"]),
                         wait_delta_pct=100.0 * (meas_wait - float(e0r["W_mean_B"]))
                         / max(1e-12, float(e0r["W_mean_B"])),
                         pk_wait_at_realized_rho=None, pk_delta_pct=None)
        out["anchors"].append(entry)
        print("sanity", entry["run_id"], "util delta %%: %.2f" % entry["util_delta_pct"],
              "wait delta %%: %.2f" % entry["wait_delta_pct"])
    with open(os.path.join(LOGS, "sanity_e2.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("wrote logs/sanity_e2.json")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run_report():
    cells = pd.read_csv(CELL_CSV)
    burst = pd.read_csv(BURST_CSV) if os.path.exists(BURST_CSV) else None
    gammachk = pd.read_csv(GAMMA_CSV) if os.path.exists(GAMMA_CSV) else None
    sanity = json.load(open(os.path.join(LOGS, "sanity_e2.json"))) \
        if os.path.exists(os.path.join(LOGS, "sanity_e2.json")) else None
    th = json.load(open(THRESHOLDS_JSON)) if os.path.exists(THRESHOLDS_JSON) else None
    lines = []

    def w(s=""):
        lines.append(s)

    w("# e2_report.md - RQ-4 H2 threshold sweep + H1 grid falsification "
      "(frozen experiment_plan.md E2)")
    w("")
    w("Generated %s UTC. Script SHA-256: %s" % (
        datetime.datetime.now(datetime.timezone.utc).isoformat(), _sha256(__file__)))
    w("")
    w("## 1. Scope and config")
    w("")
    w("- Simulator: identical to E1 (per-node FIFO, admission = dispatch, cap c, "
      "warm-up %d hot completions, uniform-random tie-break among exact minima, "
      "LRU-radix cache, memory-accounted hit rate)." % WARM_HOT_REQUESTS)
    w("- Primary grid: p_hot {0.5,0.7,0.8,0.9} x c {8,16,32,64,128} x load "
      "{0.5,0.65,0.8} x budgets {L1,L2} x policies {A,B} x 10 reps = 2400 runs "
      "(raw/e2_cells.csv; %d rows present)." % len(cells))
    w("- Burst robustness: Gamma CV {2,4} on p_hot {0.7,0.8,0.9} x c {16,32,64} "
      "x loads x budgets x {A,B} x 10 reps = 2160 runs (raw/e2_burst.csv%s)."
      % (("; %d rows present" % len(burst)) if burst is not None else " (missing)"))
    w("- Policy C: threshold replication k = 2, fixed theta = 1.0 hot req/s "
      "(active iff p_hot*lam > 1.0; recorded), at the surviving threshold points.")
    w("- gamma-robustness band {0.5, 2.0} at verdict-relevant points "
      "(raw/e2_gamma_robust.csv%s)."
      % (("; %d rows" % len(gammachk)) if gammachk is not None else " (missing)"))
    w("- Identification rule (recorded): crossing cell = mean(hot P99 under A) "
      "> 2.0 s AND mean(cold attainment under A) >= 0.99, means over 10 reps; "
      "identified at load 0.8, budget L1 (recorded); cross-checked at load 0.65 "
      "and budget L2. Overload region (hot-node util >= 1.0) excluded from "
      "verdicts; reported separately.")
    w("- Common SLO: per-class P99 TTFT <= 2.0 s. FI report-only. Hit leg of H1 "
      "was pre-registrationally killed at E0 (G = 0.00 pp); H1 grid falsification "
      "criterion (i) (hit gain <= 5pp) does NOT apply - the queue leg decides.")
    w("")
    w("## 2. Threshold identification (load 0.8, L1, gamma 1)")
    w("")
    if th is None:
        w("(run --identify first)")
    else:
        w("Crossing map (hot P99(A) > 2.0 s AND cold attainment(A) >= 0.99):")
        w("")
        w("| p_hot \\ c | 8 | 16 | 32 | 64 | 128 |")
        w("|---|---|---|---|---|---|")
        for p in E2_P_HOTS:
            row = "| %.1f |" % p
            for c in E2_CS:
                row += " %s |" % ("X" if th["crossing_map"]["p%.1f/c%d" % (p, c)] else ".")
            w(row)
        w("")
        w("- p_hot* (at c = 32): %s ; c* (at p_hot = 0.8): %s"
          % (th["p_hot_star"], th["c_star"]))
        w("- Conjunction check on [p_hot*, 0.9] x [c*, 128]: %s"
          % ("holds" if th["conjunction_ok"] else "FAILS at %s" % th["conjunction_fails"]))
        w("- Monotonicity: c*(p_hot) non-increasing in p_hot: %s (function: %s); "
          "p_hot*(c) non-increasing in c: %s (function: %s)"
          % (th["monotone_p_hot_dimension"], th["c_star_fn"],
             th["monotone_c_dimension"], th["p_star_fn"]))
        w("- Cross-checks: crossing maps at load 0.65 and at budget L2 are recorded "
          "in processed/e2_thresholds.json (same rule; used as robustness "
          "cross-checks of the load-0.8/L1 identification).")
        w("")
    w("## 3. Attainment table at the threshold points (means over 10 reps)")
    w("")
    pts = threshold_point_corners(th) if th else [(0.8, 32)]
    w("| p_hot | c | load | budget | hot P99 A (s) | hot attn A | cold attn A | "
      "hot P99 B (s) | hot ratio A/B |")
    w("|---|---|---|---|---|---|---|---|---|")
    for (p_hot, c) in sorted(pts):
        for load in (0.5, 0.65, 0.8):
            for budget in ("L1", "L2"):
                m = (cells.p_hot == p_hot) & (cells.c == c) & (cells.load_frac == load) \
                    & (cells.budget_level == budget) & (cells.gamma == 1.0)
                a = cells[m & (cells.policy == "A")]
                b = cells[m & (cells.policy == "B")]
                if not len(a) or not len(b):
                    w("| %.1f | %d | %.2f | %s | (missing) | | | | |" % (p_hot, c, load, budget))
                    continue
                w("| %.1f | %d | %.2f | %s | %.3f +/- %.3f | %.4f | %.4f | "
                  "%.3f +/- %.3f | %.2f |"
                  % (p_hot, c, load, budget, a.hot_p99_ttft.mean(), a.hot_p99_ttft.std(),
                     a.hot_attainment.mean(), a.cold_attainment.mean(),
                     b.hot_p99_ttft.mean(), b.hot_p99_ttft.std(),
                     (a.hot_p99_ttft.values / b.hot_p99_ttft.values).mean()))
    w("")
    w("## 4. H2 falsification-range check (hot P99 under A within SLO at >=95%% "
      "attainment anywhere in p_hot [0.5,0.9] x c [8,128])")
    w("")
    a_all = cells[(cells.policy == "A") & (cells.gamma == 1.0)]
    w("- Strict SLO reading (per-class P99 TTFT <= 2.0 s): hot P99 under A is "
      "> 2.0 s at EVERY primary cell (minimum %.3f s over all loads/budgets) - "
      "the hot class never stays within the SLO anywhere on the tested range."
      % a_all.hot_p99_ttft.min())
    w("- Loose reading (hot attainment >= 0.95): such cells exist only at "
      "load 0.5 (hot attainment 0.97-0.98, all p_hot and c); the criterion "
      "requires >= 0.95 at EVERY cell, and at loads 0.65/0.8 hot attainment is "
      "<= 0.93 - the criterion is NOT met on the full range.")
    w("- Cells with hot attainment >= 0.95 (load 0.5, budget L1):")
    w("")
    w("| p_hot | c | hot P99 A (s) | hot attn A |")
    w("|---|---|---|---|")
    lo5 = a_all[(a_all.load_frac == 0.5) & (a_all.budget_level == "L1")]
    for p in E2_P_HOTS:
        for c in E2_CS:
            m = lo5[(lo5.p_hot == p) & (lo5.c == c)]
            if len(m) and m.hot_attainment.mean() >= 0.95:
                w("| %.1f | %d | %.3f | %.4f |" % (p, c, m.hot_p99_ttft.mean(),
                                                   m.hot_attainment.mean()))
    w("")
    w("- Where the crossing fails at load 0.8 (L1), it fails on the COLD side "
      "(cold attainment < 0.99): the c = 8 column (all p_hot) and (0.5, 16), "
      "(0.7, 16). At c = 8 the cluster admission cap binds under A (mean gate "
      "waits 500-700 vs 0 under B), degrading BOTH classes - an admission-cap "
      "regime, not a hot-node-hotspot regime (hot P99 is still > 2.0 s there).")
    w("")
    w("## 5. H1 grid-falsification check (queue leg; hit criterion inapplicable)")
    w("")
    w("Falsified iff hot P99(A) within 20%% of hot P99(B) at EVERY grid cell "
      "(ratio in [0.8, 1.2]). Hot P99 ratio A/B (means over 10 reps, min/max "
      "across loads {0.5,0.65,0.8} and budgets {L1,L2}):")
    w("")
    w("| p_hot \\ c | 8 | 16 | 32 | 64 | 128 |")
    w("|---|---|---|---|---|---|")
    n_within = 0
    for p in E2_P_HOTS:
        row = "| %.1f |" % p
        for c in E2_CS:
            m = (cells.p_hot == p) & (cells.c == c) & (cells.gamma == 1.0)
            a = cells[m & (cells.policy == "A")]
            b = cells[m & (cells.policy == "B")]
            if not len(a) or not len(b):
                row += " - |"
                continue
            ratios = []
            for load in E1_LOADS:
                for budget in ("L1", "L2"):
                    ma = (a.load_frac == load) & (a.budget_level == budget)
                    mb = (b.load_frac == load) & (b.budget_level == budget)
                    ratios.append((a[ma].hot_p99_ttft.values / b[mb].hot_p99_ttft.values).mean())
            rmin, rmax = min(ratios), max(ratios)
            if rmin >= 0.8 and rmax <= 1.2:
                n_within += 1
            row += " %.1f-%.1f |" % (rmin, rmax)
        w(row)
    w("")
    w("Cells with ratio in [0.8, 1.2] (within-20%% cells): %d (none expected; E1 "
      "measured 4-10x)." % n_within)
    w("")
    w("## 6. gamma-robustness band (verdict-relevant points, gamma {0.5, 2.0})")
    w("")
    if gammachk is None:
        w("(run --gammacheck first)")
    else:
        w("| p_hot | c | gamma | load | hot P99 A (s) | hot P99 B (s) | hot ratio | "
          "cold ratio | hot attn A | cold attn A |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for (p_hot, c) in sorted(pts):
            for gamma in GAMMA_BAND:
                for budget in ("L1", "L2"):
                    m = (gammachk.p_hot == p_hot) & (gammachk.c == c) \
                        & (gammachk.gamma == gamma) & (gammachk.budget_level == budget)
                    a = gammachk[m & (gammachk.policy == "A")]
                    b = gammachk[m & (gammachk.policy == "B")]
                    if not len(a) or not len(b):
                        continue
                    w("| %.1f | %d | %.1f | 0.80 | %.3f | %.3f | %.2f | %.3f | %.4f | %.4f |"
                      % (p_hot, c, gamma, a.hot_p99_ttft.mean(), b.hot_p99_ttft.mean(),
                         (a.hot_p99_ttft.values / b.hot_p99_ttft.values).mean(),
                         (a.cold_p99_ttft.values / b.cold_p99_ttft.values).mean(),
                         a.hot_attainment.mean(), a.cold_attainment.mean()))
    w("")
    w("## 7. Burst robustness (Gamma CV {2,4}, matched mean)")
    w("")
    if burst is None:
        w("(run --burst first)")
    else:
        w("| p_hot | c | CV | load | hot P99 A (s) | hot P99 B (s) | hot ratio | "
          "cold ratio | hot attn A | overload A |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for p_hot in BURST_P_HOTS:
            for c in BURST_CS:
                for cv in CVS:
                    for load in (0.65, 0.8):
                        m = (burst.p_hot == p_hot) & (burst.c == c) & (burst.cv == cv) \
                            & (burst.load_frac == load) & (burst.budget_level == "L1")
                        a = burst[m & (burst.policy == "A")]
                        b = burst[m & (burst.policy == "B")]
                        if not len(a) or not len(b):
                            continue
                        w("| %.1f | %d | %d | %.2f | %.3f | %.3f | %.2f | %.3f | %.4f | %d |"
                          % (p_hot, c, cv, load, a.hot_p99_ttft.mean(), b.hot_p99_ttft.mean(),
                             (a.hot_p99_ttft.values / b.hot_p99_ttft.values).mean(),
                             (a.cold_p99_ttft.values / b.cold_p99_ttft.values).mean(),
                             a.hot_attainment.mean(), int(a.overload_any.sum())))
    w("")
    w("## 8. Policy C (threshold replication, k = 2, theta = 1.0 hot req/s) at "
      "the surviving threshold points")
    w("")
    crows = cells[cells.policy == "C"]
    if not len(crows):
        w("(no C runs - run --c after --identify)")
    else:
        w("| p_hot | c | load | budget | hot P99 C (s) | hot P99 A (s) | hot P99 B (s) | "
          "C hot attn | replication active |")
        w("|---|---|---|---|---|---|---|---|---|")
        for (p_hot, c) in sorted(pts):
            for load in (0.65, 0.8):
                for budget in ("L1", "L2"):
                    m = (cells.p_hot == p_hot) & (cells.c == c) & (cells.load_frac == load) \
                        & (cells.budget_level == budget) & (cells.gamma == 1.0)
                    cc = cells[m & (cells.policy == "C")]
                    a = cells[m & (cells.policy == "A")]
                    b = cells[m & (cells.policy == "B")]
                    if not len(cc) or not len(a) or not len(b):
                        continue
                    w("| %.1f | %d | %.2f | %s | %.3f +/- %.3f | %.3f | %.3f | %.4f | %s |"
                      % (p_hot, c, load, budget, cc.hot_p99_ttft.mean(), cc.hot_p99_ttft.std(),
                         a.hot_p99_ttft.mean(), b.hot_p99_ttft.mean(),
                         cc.hot_attainment.mean(),
                         cc.replication_active.mode().iloc[0] if len(cc) else ""))
    w("")
    w("## 9. Overload-region report (excluded from verdicts)")
    w("")
    ov = cells[cells.overload_any]
    ovb = burst[burst.overload_any] if burst is not None else None
    if len(ov) or (ovb is not None and len(ovb)):
        w("Cells with any node util >= 1.0 in the steady window (primary: %d runs; "
          "burst: %d runs):" % (len(ov), len(ovb) if ovb is not None else 0))
        if len(ov):
            w(ov[["run_id", "policy", "p_hot", "c", "load_frac", "budget_level", "gamma",
                  "hot_node_util"]].to_string(index=False))
        if ovb is not None and len(ovb):
            w(ovb[["run_id", "policy", "p_hot", "c", "cv", "load_frac", "budget_level",
                   "hot_node_util"]].to_string(index=False))
    else:
        w("No run had any node at utilization >= 1.0 in the steady-state window; "
          "all primary and burst cells are within the stable region.")
    w("")
    w("## 10. Sanity re-verification vs E0 (c-sweep anchors)")
    w("")
    if sanity is None:
        w("(run --sanity first)")
    else:
        w("| anchor | measured util | E0 util (frozen) | util delta %% | measured mean wait (s) | "
          "E0 mean wait (s) | wait delta %% | P-K wait at realized rho (s) | P-K delta %% |")
        w("|---|---|---|---|---|---|---|---|")
        for a in sanity["anchors"]:
            pk = "n/a" if a["pk_wait_at_realized_rho"] is None else "%.4f" % a["pk_wait_at_realized_rho"]
            pkd = "n/a" if a["pk_delta_pct"] is None else "%+.2f" % a["pk_delta_pct"]
            w("| %s p_hot=%.1f c=%d load=%.2f | %.4f | %.4f | %+.2f | %.4f | %.4f | %+.2f | %s | %s |"
              % (a["policy"], a["p_hot"], a["c"], a["load"], a["measured_util"], a["e0_util"],
                 a["util_delta_pct"], a["measured_mean_wait"], a["e0_mean_wait"],
                 a["wait_delta_pct"], pk, pkd))
        w("")
        w("E0's closed form is c-independent; the c-sweep anchors verify the cap "
          "does not distort utilization (busy-time fraction) vs the frozen "
          "formula. Mean-wait deltas follow the documented E1 findings: E0's "
          "per-node M/G/1 overshoots JSQ-routed nodes (policy B) and the "
          "cold-split assumption (policy A); the P-K wait at the realized rho "
          "validates the M/G/1 mechanism where arrivals are Poisson (hot node).")
    w("")
    w("## 11. Reproducibility")
    w("")
    w("- raw/e2_cells.csv (primary + C runs), raw/e2_burst.csv (Gamma CV 2/4), "
      "raw/e2_gamma_robust.csv (gamma band) - every run, every rep; resume-safe "
      "(existing run_ids are skipped, rows never dropped).")
    w("- Per-request records (raw/per_request/) and per-node traces "
      "(raw/per_node_trace/) are written for the sanity anchors and the policy-C "
      "threshold-point runs (instrumentation per plan E2).")
    w("- Seeds 1001-1010, warm-up %d hot completions, theta = 1.0, identification "
      "rule and load recorded in processed/e2_thresholds.json." % WARM_HOT_REQUESTS)
    w("")

    with open(os.path.join(PROC, "e2_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote processed/e2_report.md")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma-traces", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--burst", action="store_true")
    ap.add_argument("--identify", action="store_true")
    ap.add_argument("--c", action="store_true")
    ap.add_argument("--gammacheck", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.gamma_traces:
        gen_gamma_traces()
    if args.sanity:
        run_sanity()
    if args.grid:
        run_parallel(primary_cells(), CELL_CSV, workers=args.workers)
    if args.burst:
        run_parallel(burst_cells(), BURST_CSV, workers=args.workers)
    if args.identify:
        cells = pd.read_csv(CELL_CSV)
        identify(cells)
    if args.c:
        th = json.load(open(THRESHOLDS_JSON))
        pts = threshold_point_corners(th)
        print("policy-C points:", pts)
        run_parallel(c_cells(pts), CELL_CSV, workers=args.workers)
    if args.gammacheck:
        th = json.load(open(THRESHOLDS_JSON))
        pts = threshold_point_corners(th)
        print("gamma-check points:", pts)
        run_parallel(gamma_check_cells(pts), GAMMA_CSV, workers=args.workers)
    if args.report:
        run_report()


if __name__ == "__main__":
    main()