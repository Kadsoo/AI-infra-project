"""
E1 - H1 point kill test: offline trace replay through the queueing model
(frozen experiment_plan.md section E1, 1.2/1.4; LOCKED_PLAN.md sections 4-7).

Replays the fixed 10,000-request traces (raw/trace_10000_repNN.csv) through
per-node FIFO queues. Admission = dispatch (no global FIFO; the cluster-wide
admission cap c bounds in-flight = in-service + queued; arrivals beyond the cap
wait at the gate in FIFO order and are dispatched immediately upon admission).
Service T_prefill = gamma * tokens_not_cached (gamma_base = 1/2048 s/token).

Policies (only the routing decision differs):
  A  affinity-only: greedy longest-prefix to the node holding the longest
     matching cached prefix; tiebreak minimum token-weighted backlog.
  B  load-only: minimum token-weighted backlog, no prefix preference.
  B' request-count least-loaded (sensitivity cell, p_hot = 0.8).

Cache model (frozen 1.2): per-node LRU radix-style prefix cache, memory
accounted (equal total KV budget across policies). The hot prefix becomes
resident on a node once a hot request is served there and is never evicted
(leaf-first LRU evicts unique suffix leaves first; both budget levels exceed
the 4 x 1024 tokens B needs for its prefix copies). Cold prefixes are unique
=> always miss. Memory-accounted hit rate = cached tokens / input tokens.

Warm-up (frozen schedule, recorded): hot prefix filled by WARM_HOT_REQUESTS
hot-request completions. Policy A routes warm-up hot requests to a single node
(node 0) - the deterministic realization of the frozen single-hot-node steady
state (recorded in experiment_log.md / e1_report.md); policies B/B' route
warm-up hot requests by their own metric (min-backlog spreads them, so the hot
prefix accumulates on every node, per the frozen cache model). Metrics are
computed on the steady-state window only: arrivals with t >= t_warm, where
t_warm = completion time of the WARM_HOT_REQUESTS-th hot request.

Per run, all per-request records (steady window) are written to
raw/per_request/<run_id>.csv.gz and one aggregate row is appended to
raw/e1_cells.csv (never drop a run; duplicate run_ids are replaced in place).

Usage:
  python code/e1_replay.py --single --policy A --p_hot 0.8 --load 0.65 \
      --budget L1 --gamma 1.0 --rep 1
  python code/e1_replay.py --all
  python code/e1_replay.py --sanity
  python code/e1_replay.py --report
"""

import argparse
import datetime
import gzip
import hashlib
import heapq
import json
import os
import zlib
from collections import deque

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
PROC = os.path.join(ROOT, "processed")
LOGS = os.path.join(ROOT, "logs")
PERQ = os.path.join(RAW, "per_request")
NODEQ = os.path.join(RAW, "per_node_trace")
for d in (RAW, PROC, LOGS, PERQ, NODEQ):
    os.makedirs(d, exist_ok=True)

N_NODES = 4
L_PREFIX = 1024
GAMMA_BASE = 1.0 / 2048.0          # s per token (2048-token prefill ~ 1.0 s)
SLO_TTFT = 2.0                     # common absolute SLO (s)
WARM_HOT_REQUESTS = 200            # frozen warm-up schedule (recorded)
MAX_C = 32                         # E1 admission cap (frozen)

BUDGETS = {"L1": 32768, "L2": 16384}   # total KV tokens (see e0_report.md 6)
BUDGET_PER_NODE = {k: v // N_NODES for k, v in BUDGETS.items()}

E1_P_HOTS = [0.8, 0.9]
E1_LOADS = [0.5, 0.65, 0.8]
E1_GAMMAS = [0.5, 1.0, 2.0]
E1_BUDGETS = ["L1", "L2"]
E1_REPS = list(range(1, 11))

CELL_CSV = os.path.join(RAW, "e1_cells.csv")

P99_QUANTILE = 99
P95_QUANTILE = 95


def lambda_sat(p_hot, gamma):
    """Affinity-configuration saturation rate (E0): 1/(p_hot*1024*gamma*GAMMA_BASE) req/s."""
    return 1.0 / (p_hot * L_PREFIX * gamma * GAMMA_BASE)


def run_id_of(policy, p_hot, load, budget, gamma, rep):
    return "e1_%s_p%.1f_l%.2f_%s_g%.1f_r%02d" % (policy, p_hot, load, budget, gamma, rep)


def load_trace(rep):
    return pd.read_csv(os.path.join(RAW, "trace_10000_rep%02d.csv" % rep))


class Sim:
    def __init__(self, policy, p_hot, load_frac, budget_level, gamma, c, trace, rep,
                 record_per_request=True, record_node_trace=False):
        self.policy = policy
        self.p_hot = p_hot
        self.load_frac = load_frac
        self.budget_level = budget_level
        self.gamma = gamma
        self.c = c
        self.trace = trace
        self.rep = rep
        self.lam = load_frac * lambda_sat(p_hot, gamma)
        self.budget_node = BUDGET_PER_NODE[budget_level]
        self.record_per_request = record_per_request
        self.record_node_trace = record_node_trace
        self.run_id = run_id_of(policy, p_hot, load_frac, budget_level, gamma, rep)
        # deterministic per-run tie-break RNG (recorded)
        self.tie_seed = zlib.crc32(self.run_id.encode("utf-8")) & 0xFFFFFFFF
        self.rng = np.random.default_rng(self.tie_seed)

    # ---- routing ---------------------------------------------------------
    def _backlog_tokens(self, node, now):
        b = node["queued_tokens"]
        if node["busy_until"] > now:
            b += (node["busy_until"] - now) / (self.gamma * GAMMA_BASE)
        return b

    def _route(self, node, cls, now):
        """Routing decision. Ties at the exact minimum are broken UNIFORMLY AT
        RANDOM (seeded per run, deterministic) so that load-only routing
        realizes the balanced equilibrium assumed by E0; deterministic index
        tie-breaks would bias low-index nodes (documented in e1_report.md)."""
        rng = self.rng
        if self.policy == "A":
            best_len, best = -1, []
            for k in range(N_NODES):
                m = L_PREFIX if (node[k]["prefix_resident"] and cls == 1) else 0
                if m > best_len:
                    best_len, best = m, [k]
                elif m == best_len:
                    best.append(k)
            if not best:
                best = list(range(N_NODES))
            bl = [self._backlog_tokens(node[k], now) for k in best]
            m = min(bl)
            cands = [k for k, b in zip(best, bl) if b == m]
            return int(rng.choice(cands)) if len(cands) > 1 else cands[0]
        if self.policy == "B":
            bl = [self._backlog_tokens(node[k], now) for k in range(N_NODES)]
            m = min(bl)
            cands = [k for k, b in enumerate(bl) if b == m]
            return int(rng.choice(cands)) if len(cands) > 1 else cands[0]
        if self.policy == "Bp":
            cnt = [len(node[k]["queue"]) + (1 if node[k]["busy_until"] > now else 0)
                   for k in range(N_NODES)]
            m = min(cnt)
            cands = [k for k, c in enumerate(cnt) if c == m]
            return int(rng.choice(cands)) if len(cands) > 1 else cands[0]
        raise ValueError("unknown policy")

    # ---- main loop -------------------------------------------------------
    def run(self):
        tr = self.trace
        n = len(tr)
        cls = np.where(tr["u_class"].values < self.p_hot, 1, 0)      # 1 = hot
        tokens = tr["total_tokens"].values.astype(np.float64)
        unit_dt = tr["unit_dt"].values
        # Poisson arrivals at rate lam; identical inter-arrival sequence per cell
        t_arr = np.concatenate([[0.0], np.cumsum(unit_dt[1:])]) / self.lam

        node = [dict(queue=deque(), busy_until=0.0, serving=None, prefix_resident=False,
                     queued_tokens=0.0, busy_intervals=[], leaves=deque(),
                     leaf_tokens=0.0, evictions=0, hot_served=0)
                for _ in range(N_NODES)]

        events = []          # heap of (completion_time, node_id)
        gate = deque()       # arrival indices waiting at the admission gate
        inflight = 0
        i = 0
        t = 0.0
        hot_completions = 0
        t_warm = None
        gate_waits = 0

        rows = []
        node_trace = [] if self.record_node_trace else None

        def admit(idx, now):
            nonlocal inflight, gate_waits, hot_completions
            t_a = t_arr[idx]
            # Frozen warm-up schedule, deterministic realization (recorded):
            # while fewer than WARM_HOT_REQUESTS hot requests have completed,
            # policy A routes hot requests to node 0 (the single hot node of the
            # frozen steady state); policies B/B' route by their own metric, so
            # the hot prefix accumulates on every node (frozen cache model).
            if self.policy == "A" and cls[idx] == 1 and hot_completions < WARM_HOT_REQUESTS:
                k = 0
            else:
                k = self._route(node, cls[idx], now)
            if os.environ.get("E1_DEBUG_ROUTE") and idx < 30:
                print("DBG", self.policy, idx, "cls", int(cls[idx]), "node", k,
                      "t", round(t_a, 4), "backlogs",
                      [round(self._backlog_tokens(node[j], now), 2) for j in range(4)],
                      "prefix", [bool(node[j]["prefix_resident"]) for j in range(4)])
            cached = L_PREFIX if (cls[idx] == 1 and node[k]["prefix_resident"]) else 0
            svc_tokens = tokens[idx] - cached
            svc = svc_tokens * self.gamma * GAMMA_BASE
            req = {"seq": int(tr["seq"].values[idx]), "cls": int(cls[idx]),
                   "t_arr": t_a, "svc_tokens": float(svc_tokens), "cached": cached}
            # A node is idle iff it has no in-service request (serving is None).
            # busy_until == now can still mean a completion event for this node is
            # pending (simultaneous completions); admitting onto such a node would
            # push a duplicate completion event (recorded fix, E2).
            if node[k]["serving"] is None:
                node[k]["busy_until"] = now + svc
                node[k]["serving"] = req
                node[k]["busy_intervals"].append([now, now + svc])
                req["wait"] = 0.0
                heapq.heappush(events, (now + svc, k))
            else:
                req["wait"] = None      # resolved at service start
                node[k]["queue"].append(req)
                node[k]["queued_tokens"] += svc_tokens
            inflight += 1

        while i < n or gate or events:
            nxt_arr = t_arr[i] if i < n else float("inf")
            nxt_comp = events[0][0] if events else float("inf")
            if nxt_arr < nxt_comp:
                t = nxt_arr
                if self.record_node_trace:
                    for k in range(N_NODES):
                        node_trace.append((t, k, len(node[k]["queue"])
                                           + (1 if node[k]["busy_until"] > t else 0),
                                           self._backlog_tokens(node[k], t)))
                if inflight < self.c:
                    admit(i, t)
                else:
                    gate.append(i)
                    gate_waits += 1
                i += 1
            else:
                t = nxt_comp
                _, k = heapq.heappop(events)
                req = node[k]["serving"]
                inflight -= 1
                is_hot = req["cls"] == 1
                if is_hot:
                    node[k]["hot_served"] += 1
                    node[k]["prefix_resident"] = True
                    hot_completions += 1
                    if hot_completions == WARM_HOT_REQUESTS:
                        t_warm = t
                # cache accounting: prefix retained (leaf-first LRU cannot evict it);
                # unique leaves retained under budget, LRU order, leaf-first
                leaf_add = req["svc_tokens"]
                node[k]["leaves"].append((t, leaf_add))
                node[k]["leaf_tokens"] += leaf_add
                while node[k]["leaf_tokens"] > self.budget_node - (
                        L_PREFIX if node[k]["prefix_resident"] else 0):
                    _, rem = node[k]["leaves"].popleft()
                    node[k]["leaf_tokens"] -= rem
                    node[k]["evictions"] += 1
                # record per-request outcome
                rows.append((req["seq"], req["cls"], req["t_arr"], k, req["wait"],
                             req["svc_tokens"], req["cached"], t - req["t_arr"]))
                # serve next in node queue
                if node[k]["queue"]:
                    nq = node[k]["queue"].popleft()
                    node[k]["queued_tokens"] -= nq["svc_tokens"]
                    nq["wait"] = t - nq["t_arr"]
                    node[k]["serving"] = nq
                    svc = nq["svc_tokens"] * self.gamma * GAMMA_BASE
                    node[k]["busy_until"] = t + svc
                    node[k]["busy_intervals"].append([t, t + svc])
                    heapq.heappush(events, (t + svc, k))
                else:
                    node[k]["serving"] = None
                # drain the gate FIFO while slots free (admission = dispatch)
                while gate and inflight < self.c:
                    admit(gate.popleft(), t)
                if self.record_node_trace:
                    for kk in range(N_NODES):
                        node_trace.append((t, kk, len(node[kk]["queue"])
                                           + (1 if node[kk]["busy_until"] > t else 0),
                                           self._backlog_tokens(node[kk], t)))

        if t_warm is None:
            raise RuntimeError("warm-up not reached (%d hot requests) in run %s"
                               % (hot_completions, self.run_id))
        t_end = t
        return self._aggregate(rows, node, t_warm, t_end, node_trace, gate_waits)

    # ---- aggregation -----------------------------------------------------
    def _aggregate(self, rows, node, t_warm, t_end, node_trace, gate_waits):
        cols = ["seq", "cls", "t_arr", "node", "wait", "svc_tokens", "cached", "ttft"]
        df = pd.DataFrame(rows, columns=cols)
        win = df[df.t_arr >= t_warm].copy()
        n_win = len(win)

        def pct(x, q):
            return float(np.percentile(x, q)) if len(x) else float("nan")

        hot = win[win.cls == 1]
        cold = win[win.cls == 0]
        input_tokens = float(win.svc_tokens.sum() + win.cached.sum())
        hit_rate = float(win.cached.sum() / input_tokens) if n_win else float("nan")

        res = {
            "run_id": self.run_id, "policy": self.policy, "p_hot": self.p_hot,
            "load_frac": self.load_frac, "lam": self.lam,
            "budget_level": self.budget_level, "gamma": self.gamma, "c": self.c,
            "rep": self.rep, "trace_seed": 1000 + self.rep,
            "warm_hot_count": WARM_HOT_REQUESTS, "t_warm": t_warm,
            "gate_waits": gate_waits,
            "n_steady": n_win,
            "hit_rate": hit_rate,
            "agg_mean_ttft": float(win.ttft.mean()) if n_win else float("nan"),
            "hot_p99_ttft": pct(hot.ttft, P99_QUANTILE),
            "hot_mean_ttft": float(hot.ttft.mean()) if len(hot) else float("nan"),
            "cold_p99_ttft": pct(cold.ttft, P99_QUANTILE),
            "cold_mean_ttft": float(cold.ttft.mean()) if len(cold) else float("nan"),
            "hot_p95_ttft": pct(hot.ttft, P95_QUANTILE),
            "cold_p95_ttft": pct(cold.ttft, P95_QUANTILE),
            "hot_p99_wait": pct(hot.wait, P99_QUANTILE),
            "cold_p99_wait": pct(cold.wait, P99_QUANTILE),
        }
        a_hot = float((hot.ttft <= SLO_TTFT).mean()) if len(hot) else float("nan")
        a_cold = float((cold.ttft <= SLO_TTFT).mean()) if len(cold) else float("nan")
        res["hot_attainment"] = a_hot
        res["cold_attainment"] = a_cold
        res["fi"] = (min(a_hot, a_cold) / max(a_hot, a_cold)) \
            if (a_hot == a_hot and a_cold == a_cold) else float("nan")

        util = []
        for k in range(N_NODES):
            ov = 0.0
            for s, e in node[k]["busy_intervals"]:
                ov += max(0.0, min(e, t_end) - max(s, t_warm))
            util.append(ov / max(1e-12, t_end - t_warm))
        for k in range(N_NODES):
            res["node_util_%d" % k] = util[k]
        hot_node = int(np.argmax(util))
        res["hot_node_id"] = hot_node
        res["hot_node_util"] = util[hot_node]
        res["overload_any"] = bool(max(util) >= 1.0)

        for k in range(N_NODES):
            res["cache_tokens_%d" % k] = (L_PREFIX if node[k]["prefix_resident"] else 0) \
                + node[k]["leaf_tokens"]
            res["evictions_%d" % k] = node[k]["evictions"]

        if self.record_per_request:
            win.to_csv(gzip.open(os.path.join(PERQ, self.run_id + ".csv.gz"), "wt"),
                       index=False, float_format="%.6f")
        if node_trace is not None:
            pd.DataFrame(node_trace, columns=["t", "node", "qlen", "backlog_tokens"]).to_csv(
                gzip.open(os.path.join(NODEQ, self.run_id + ".csv.gz"), "wt"),
                index=False, float_format="%.6f")
        return res


# ---------------------------------------------------------------------------
# Cell bookkeeping (raw/e1_cells.csv; never drop a run)
# ---------------------------------------------------------------------------

def read_cells():
    if not os.path.exists(CELL_CSV):
        return pd.DataFrame()
    return pd.read_csv(CELL_CSV)


def cell_seen(run_id):
    cells = read_cells()
    return len(cells) and run_id in set(cells.run_id)


def append_cell(row, sanity_flag=False):
    row = dict(row)
    row["sanity_flag"] = bool(sanity_flag)
    row["ts_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cells = read_cells()
    if len(cells):
        cells = cells[cells.run_id != row["run_id"]]
    new = pd.DataFrame([row])
    pd.concat([cells, new], ignore_index=True).to_csv(CELL_CSV, index=False)


def run_cell(policy, p_hot, load_frac, budget, gamma, rep, c=MAX_C,
             record_node_trace=False, sanity_flag=False):
    run_id = run_id_of(policy, p_hot, load_frac, budget, gamma, rep)
    trace = load_trace(rep)
    sim = Sim(policy, p_hot, load_frac, budget, gamma, c, trace, rep,
              record_node_trace=record_node_trace)
    res = sim.run()
    assert res["run_id"] == run_id
    append_cell(res, sanity_flag=sanity_flag)
    return res


# ---------------------------------------------------------------------------
# Full grid
# ---------------------------------------------------------------------------

def run_all():
    t0 = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print("E1 full grid start", t0)
    n_runs = 0
    for policy in ("A", "B"):
        for p_hot in E1_P_HOTS:
            for load in E1_LOADS:
                for budget in E1_BUDGETS:
                    for gamma in E1_GAMMAS:
                        for rep in E1_REPS:
                            if cell_seen(run_id_of(policy, p_hot, load, budget, gamma, rep)):
                                continue
                            run_cell(policy, p_hot, load, budget, gamma, rep)
                            n_runs += 1
                            if n_runs % 60 == 0:
                                print("  %d runs done (latest %s)"
                                      % (n_runs, run_id_of(policy, p_hot, load, budget, gamma, rep)))
    # B' sensitivity cell (request-count least-loaded), p_hot = 0.8, gamma = 1
    for load in E1_LOADS:
        for budget in E1_BUDGETS:
            for rep in E1_REPS:
                if cell_seen(run_id_of("Bp", 0.8, load, budget, 1.0, rep)):
                    continue
                run_cell("Bp", 0.8, load, budget, 1.0, rep)
                n_runs += 1
    print("E1 full grid done", datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "runs executed:", n_runs)


# ---------------------------------------------------------------------------
# Sanity: 3 anchor points (E0-vs-E1) + monotone-vs-load
# ---------------------------------------------------------------------------

SANITY_ANCHORS = [
    dict(policy="A", p_hot=0.9, load=0.8, budget="L1", gamma=1.0),
    dict(policy="B", p_hot=0.9, load=0.8, budget="L1", gamma=1.0),
    dict(policy="A", p_hot=0.8, load=0.65, budget="L1", gamma=1.0),
    dict(policy="B", p_hot=0.8, load=0.65, budget="L1", gamma=1.0),
    dict(policy="A", p_hot=0.8, load=0.8, budget="L1", gamma=2.0),
    dict(policy="B", p_hot=0.8, load=0.8, budget="L1", gamma=2.0),
]


def _mean_wait_at_node(run_id, node_id):
    df = pd.read_csv(gzip.open(os.path.join(PERQ, run_id + ".csv.gz"), "rt"))
    if node_id is None:
        return float(df.wait.mean())
    return float(df[df.node == node_id].wait.mean())


def run_sanity():
    from e0_algebra import per_cell as e0_cell
    e0 = pd.read_csv(os.path.join(RAW, "e0_grid.csv"))
    e0 = e0[e0.c == MAX_C].set_index(["p_hot", "load", "gamma"])
    out = {"anchors": [], "monotone": {}}
    for a in SANITY_ANCHORS:
        run_id = run_id_of(a["policy"], a["p_hot"], a["load"], a["budget"], a["gamma"], 1)
        res = run_cell(a["policy"], a["p_hot"], a["load"], a["budget"], a["gamma"], 1,
                       record_node_trace=True, sanity_flag=True)
        e0r = e0.loc[(a["p_hot"], a["load"], a["gamma"])]
        if a["policy"] == "A":
            meas_util = float(res["hot_node_util"])
            e0_util_frozen = float(e0r["rho_hot_frozen"])
            e0_util_full = float(e0r["rho_hot_full"])
            hot_node = int(res["hot_node_id"])
            meas_wait = _mean_wait_at_node(run_id, hot_node)
            e0_wait = float(e0r["W_mean_hotnode"])
            # derived check: P-K mean wait at the REALIZED hot-node util and the
            # E0 hot-node mixture moments (validates the M/G/1 mechanism itself)
            c = e0_cell(a["p_hot"], MAX_C, a["load"], a["gamma"])
            pk_realized = meas_util * c["ES2_hotnode"] / (
                2.0 * (1.0 - meas_util) * c["ES_hotnode"]) if meas_util < 1.0 else float("nan")
            entry = dict(anchor=a, run_id=run_id, measured_util=meas_util,
                         e0_util_frozen=e0_util_frozen, e0_util_full=e0_util_full,
                         util_delta_pct_frozen=100.0 * (meas_util - e0_util_frozen)
                         / max(1e-12, e0_util_frozen),
                         util_delta_pct_full=100.0 * (meas_util - e0_util_full)
                         / max(1e-12, e0_util_full),
                         measured_mean_wait=meas_wait, e0_mean_wait=e0_wait,
                         wait_delta_pct=100.0 * (meas_wait - e0_wait) / max(1e-12, e0_wait),
                         pk_wait_at_realized_rho=pk_realized,
                         pk_delta_pct=100.0 * (meas_wait - pk_realized) / max(1e-12, pk_realized))
        else:
            meas_util = float(np.mean([res["node_util_%d" % k] for k in range(N_NODES)]))
            e0_util = float(e0r["rho_B_pernode"])
            meas_wait = _mean_wait_at_node(run_id, None)
            e0_wait = float(e0r["W_mean_B"])
            entry = dict(anchor=a, run_id=run_id, measured_util=meas_util,
                         e0_util_frozen=e0_util, e0_util_full=None,
                         util_delta_pct_frozen=100.0 * (meas_util - e0_util)
                         / max(1e-12, e0_util),
                         util_delta_pct_full=None,
                         measured_mean_wait=meas_wait, e0_mean_wait=e0_wait,
                         wait_delta_pct=100.0 * (meas_wait - e0_wait) / max(1e-12, e0_wait),
                         pk_wait_at_realized_rho=None, pk_delta_pct=None)
        out["anchors"].append(entry)
        print("sanity anchor", a, "util delta (frozen) %%: %.2f"
              % entry["util_delta_pct_frozen"],
              "wait delta %%: %.2f" % entry["wait_delta_pct"])
    for policy in ("A", "B"):
        seq = []
        for load in E1_LOADS:
            res = run_cell(policy, 0.8, load, "L1", 1.0, 1, sanity_flag=True)
            seq.append({"load": load, "agg_mean": res["agg_mean_ttft"],
                        "hot_p99": res["hot_p99_ttft"]})
        ok_mean = seq[0]["agg_mean"] < seq[1]["agg_mean"] < seq[2]["agg_mean"]
        ok_p99 = seq[0]["hot_p99"] < seq[1]["hot_p99"] < seq[2]["hot_p99"]
        out["monotone"][policy] = {"sequence": seq, "monotone_ok": bool(ok_mean and ok_p99)}
        print("monotone", policy, "ok:", ok_mean and ok_p99)
    with open(os.path.join(LOGS, "sanity_e1.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("sanity checks written to logs/sanity_e1.json")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_ms(x):
    return "%.4f +/- %.4f" % (x.mean(), x.std())


def run_report():
    cells = read_cells()
    if not len(cells):
        raise SystemExit("no cells yet - run --all first")
    sanity_path = os.path.join(LOGS, "sanity_e1.json")
    sanity = json.load(open(sanity_path)) if os.path.exists(sanity_path) else None
    lines = []

    def w(s=""):
        lines.append(s)

    w("# e1_report.md - RQ-4 H1 point kill test, offline trace replay "
      "(frozen experiment_plan.md E1)")
    w("")
    w("Generated %s UTC. Script SHA-256: %s" % (
        datetime.datetime.now(datetime.timezone.utc).isoformat(), _sha256(__file__)))
    w("")
    w("## 1. Scope")
    w("")
    w("- 10,000-request Poisson traces (10 seeded replications, seeds 1001-1010, "
      "recorded in raw/trace_10000_repNN.csv); inter-arrival sequences scaled per "
      "cell to lambda = load_frac x lambda_sat(p_hot, gamma) (E0 anchoring).")
    w("- Per-node FIFO queues, admission = dispatch, cluster admission cap c = 32; "
      "gate FIFO for arrivals beyond the cap (recorded per run as gate_waits; "
      "binds only near/over overload).")
    w("- Warm-up (frozen schedule, recorded): hot prefix filled by %d hot-request "
      "completions; steady-state window = arrivals with t >= t_warm. Under policy A "
      "all warm-up hot requests are routed to node 0 (deterministic realization of "
      "the frozen single-hot-node steady state); under B/B' they follow the policy "
      "metric (hot prefix accumulates on every node, per the frozen cache model)."
      % WARM_HOT_REQUESTS)
    w("- Task-line note: '60%% hot-prefix reuse structure' has no counterpart in the "
      "frozen workload spec; the frozen 1.3 structure (hot prefix shared by all hot "
      "requests, cold never reused) is implemented.")
    w("- E0 hit leg: G = 0.00 pp < 10 pp at both budget levels => pre-registrationally "
      "killed (processed/e0_report.md 6); H1 rests on the queue leg; the E1 "
      "falsification criterion (i) (hit gain <= 5pp) does NOT apply. Hit rates below "
      "are reported for completeness, not as verdict inputs.")
    w("- P99 = 99th percentile with linear interpolation between order statistics "
      "(numpy default), pooled over the steady-state window per run.")
    w("")
    w("## 2. Aggregates over 10 replications (mean +/- std)")
    w("")
    for p_hot in E1_P_HOTS:
        for load in E1_LOADS:
            for budget in E1_BUDGETS:
                for gamma in E1_GAMMAS:
                    w("### p_hot = %.1f, load = %.2f, budget = %s, gamma = %.1f"
                      % (p_hot, load, budget, gamma))
                    w("")
                    m = (cells.p_hot == p_hot) & (cells.load_frac == load) \
                        & (cells.budget_level == budget) & (cells.gamma == gamma)
                    a = cells[m & (cells.policy == "A")]
                    b = cells[m & (cells.policy == "B")]
                    if not len(a) or not len(b):
                        w("(missing runs)")
                        w("")
                        continue
                    hit_gain = (a.hit_rate.values - b.hit_rate.values).mean()
                    w("| metric | A | B | ratio A/B |")
                    w("|---|---|---|---|")
                    w("| hit rate | %s | %s | gain %.2f pp |" % (_fmt_ms(a.hit_rate),
                                                                  _fmt_ms(b.hit_rate), hit_gain))
                    w("| aggregate mean TTFT (s) | %s | %s | %.3f |" %
                      (_fmt_ms(a.agg_mean_ttft), _fmt_ms(b.agg_mean_ttft),
                       (a.agg_mean_ttft.values / b.agg_mean_ttft.values).mean()))
                    w("| hot P99 TTFT (s) | %s | %s | %.3f |" %
                      (_fmt_ms(a.hot_p99_ttft), _fmt_ms(b.hot_p99_ttft),
                       (a.hot_p99_ttft.values / b.hot_p99_ttft.values).mean()))
                    w("| cold P99 TTFT (s) | %s | %s | %.3f |" %
                      (_fmt_ms(a.cold_p99_ttft), _fmt_ms(b.cold_p99_ttft),
                       (a.cold_p99_ttft.values / b.cold_p99_ttft.values).mean()))
                    w("| hot P99 queue wait (s) | %s | %s | |" %
                      (_fmt_ms(a.hot_p99_wait), _fmt_ms(b.hot_p99_wait)))
                    w("| cold P99 queue wait (s) | %s | %s | |" %
                      (_fmt_ms(a.cold_p99_wait), _fmt_ms(b.cold_p99_wait)))
                    w("| hot mean TTFT (s) | %s | %s | |" %
                      (_fmt_ms(a.hot_mean_ttft), _fmt_ms(b.hot_mean_ttft)))
                    w("| cold mean TTFT (s) | %s | %s | |" %
                      (_fmt_ms(a.cold_mean_ttft), _fmt_ms(b.cold_mean_ttft)))
                    w("| hot attainment (TTFT<=2.0s) | %s | %s | |" %
                      (_fmt_ms(a.hot_attainment), _fmt_ms(b.hot_attainment)))
                    w("| cold attainment | %s | %s | |" %
                      (_fmt_ms(a.cold_attainment), _fmt_ms(b.cold_attainment)))
                    w("| FI (report-only) | %s | %s | |" % (_fmt_ms(a.fi), _fmt_ms(b.fi)))
                    w("| hot-node util (measured) | %s | - | |" % _fmt_ms(a.hot_node_util))
                    w("| overload runs (any node util>=1.0) | %d | %d | |" %
                      (int(a.overload_any.sum()), int(b.overload_any.sum())))
                    w("")
    w("## 3. B' sensitivity cell (request-count least-loaded), p_hot = 0.8, gamma = 1.0")
    w("")
    w("| load | budget | B' hot P99 TTFT (s) | B hot P99 TTFT (s) | B' agg mean (s) | B agg mean (s) |")
    w("|---|---|---|---|---|---|")
    for load in E1_LOADS:
        for budget in E1_BUDGETS:
            bp = cells[(cells.policy == "Bp") & (cells.p_hot == 0.8) &
                       (cells.load_frac == load) & (cells.budget_level == budget)]
            b = cells[(cells.policy == "B") & (cells.p_hot == 0.8) &
                      (cells.load_frac == load) & (cells.budget_level == budget) &
                      (cells.gamma == 1.0)]
            if not len(bp) or not len(b):
                w("| %.2f | %s | (missing) | | | |" % (load, budget))
                continue
            w("| %.2f | %s | %s | %s | %s | %s |"
              % (load, budget, _fmt_ms(bp.hot_p99_ttft), _fmt_ms(b.hot_p99_ttft),
                 _fmt_ms(bp.agg_mean_ttft), _fmt_ms(b.agg_mean_ttft)))
    w("")
    w("## 4. Frozen thresholds (for the analyst; no verdict is drawn here)")
    w("")
    w("- H1 success (p_hot in {0.8, 0.9}, c = 32, stable region, equal load, equal "
      "budget): hit-rate gain >= G (hit leg pre-registrationally killed: G = 0.00 pp), "
      "aggregate mean TTFT(A) <= 0.8 x B, hot P99(A) >= 2 x hot P99(B), "
      "cold P99(A) <= 1.2 x cold P99(B) (LOCKED_PLAN 5).")
    w("- H1 kill (queue leg): hot P99(A) within 20%% of hot P99(B) at both test points "
      "(LOCKED_PLAN 6; criterion (i), hit gain <= 5pp, does not apply).")
    w("- Overload region (hot-node util >= 1.0) excluded from verdicts, reported "
      "separately (frozen).")
    w("- Simulator passes are 'queue-concentration passes; tail claim untested until "
      "E4' (frozen revision R1).")
    w("")
    w("## 5. Sanity checks vs E0 (3 anchor points) and monotonicity")
    w("")
    if sanity is None:
        w("(no sanity log found - run --sanity first)")
    else:
        w("| anchor | measured util | E0 util (frozen) | util delta %% | measured mean wait (s) | "
          "E0 mean wait (s) | wait delta %% | P-K wait at realized rho (s) | P-K delta %% |")
        w("|---|---|---|---|---|---|---|---|")
        for a in sanity["anchors"]:
            pk = "n/a" if a["pk_wait_at_realized_rho"] is None \
                else "%.4f" % a["pk_wait_at_realized_rho"]
            pkd = "n/a" if a["pk_delta_pct"] is None else "%+.2f" % a["pk_delta_pct"]
            w("| %s p_hot=%.1f load=%.2f gamma=%.1f | %.4f | %.4f | %+.2f | %.4f | %.4f | %+.2f | %s | %s |"
              % (a["anchor"]["policy"], a["anchor"]["p_hot"], a["anchor"]["load"],
                 a["anchor"]["gamma"], a["measured_util"], a["e0_util_frozen"],
                 a["util_delta_pct_frozen"], a["measured_mean_wait"], a["e0_mean_wait"],
                 a["wait_delta_pct"], pk, pkd))
        w("")
        w("| policy | monotone vs load (mean TTFT and hot P99 increasing in "
          "0.5 -> 0.65 -> 0.8) |")
        w("|---|---|")
        for policy in ("A", "B"):
            w("| %s | %s |" % (policy, sanity["monotone"][policy]["monotone_ok"]))
        w("")
        w("Frozen sanity rule: utilization and mean wait should match E0 within a "
          "few %% at the 3 anchor points; deltas are reported as measured. Notes:")
        w("- Policy A: measured hot-node util sits between E0's frozen formula "
          "(rho_hot = load) and the full-utilization variant (which assumed a 25%% "
          "cold split onto the hot node); the realized cold share is ~4%%. The "
          "P-K wait at the realized rho (E0 moments) validates the M/G/1 mechanism "
          "itself (column 8-9).")
        w("- Policy B: min-backlog routing keeps per-node utils at E0's rho_B (column "
          "5), but the per-node M/G/1 Poisson-arrival assumption does not hold for "
          "load-balanced nodes (arrivals are routed away from busy servers), so "
          "measured mean waits are far below E0's W_mean_B. This is a documented "
          "approximation of the frozen algebra, not a simulator defect; the "
          "affinity hot node (Poisson arrivals) is the cell where the algebra's "
          "wait predictions apply.")
    w("")
    w("## 6. Overload-region report (excluded from verdicts, reported separately)")
    w("")
    ov = cells[cells.overload_any]
    if len(ov):
        w("| run_id | policy | p_hot | load | budget | gamma | hot_node_util |")
        w("|---|---|---|---|---|---|---|")
        for _, r in ov.iterrows():
            w("| %s | %s | %.1f | %.2f | %s | %.1f | %.4f |" %
              (r.run_id, r.policy, r.p_hot, r.load_frac, r.budget_level, r.gamma,
               r.hot_node_util))
    else:
        w("No run had any node at utilization >= 1.0 in the steady-state window; "
          "all E1 cells are within the stable region.")
    w("")
    w("## 7. Reproducibility")
    w("")
    w("- raw/e1_cells.csv: every run, every replication (never dropped; duplicate "
      "run_ids replaced in place).")
    w("- raw/per_request/<run_id>.csv.gz: per-request records (arrival time, node, "
      "queue wait, prefill tokens, hit/miss, TTFT) for the steady-state window.")
    w("- raw/per_node_trace/<run_id>.csv.gz: per-node queue-length/backlog traces at "
      "the 6 sanity anchor runs.")
    w("- Seeds, warm-up count, lambda, gate-wait counts, and t_warm are recorded per "
      "run in raw/e1_cells.csv.")
    w("")

    with open(os.path.join(PROC, "e1_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote processed/e1_report.md")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--policy", choices=["A", "B", "Bp"], default="A")
    ap.add_argument("--p_hot", type=float, default=0.8)
    ap.add_argument("--load", type=float, default=0.65)
    ap.add_argument("--budget", choices=["L1", "L2"], default="L1")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--rep", type=int, default=1)
    args = ap.parse_args()

    if args.all:
        run_all()
    elif args.sanity:
        run_sanity()
    elif args.report:
        run_report()
    elif args.single:
        res = run_cell(args.policy, args.p_hot, args.load, args.budget, args.gamma, args.rep)
        keep = {k: v for k, v in res.items() if k in (
            "run_id", "policy", "p_hot", "load_frac", "lam", "budget_level", "gamma",
            "c", "rep", "t_warm", "gate_waits", "n_steady", "hit_rate",
            "agg_mean_ttft", "hot_p99_ttft", "cold_p99_ttft", "hot_p99_wait",
            "cold_p99_wait", "hot_attainment", "cold_attainment", "fi",
            "hot_node_id", "hot_node_util", "overload_any")}
        print(json.dumps(keep, indent=1, default=str))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()