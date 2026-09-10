"""
E0 - gate-0 algebra for RQ-4 (frozen LOCKED_PLAN.md / experiment_plan.md section E0).

Closed-form M/G/1 concentration analysis over the grid
    p_hot in {0.5, 0.7, 0.8, 0.9}, c in {8, 16, 32, 64},
    load in {0.5, 0.65, 0.8} x affinity saturation, gamma in {0.5, 1, 2},
plus the memory-accounted LRU-radix hit-gain G derivation at two budget levels.

CPU only. No simulator, no GPU. Pure derivation.

Outputs
  raw/e0_grid.csv          full per-cell algebra table
  raw/e0_budget.csv        budget-level derivation + G
  processed/e0_results.csv per (p_hot, c, load, gamma): hot-node util,
                           predicted ratios, SLO crossing flags, G
  processed/e0_report.md   derivation, stable-region map, pre-registered
                           E1/E2 grid points with expected values
"""

import os
import sys
import json
import hashlib
import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
PROC = os.path.join(ROOT, "processed")
LOGS = os.path.join(ROOT, "logs")
for d in (RAW, PROC, LOGS):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Frozen constants (experiment_plan.md 1.2/1.3, E0; LOCKED_PLAN.md sections 1-7)
# ---------------------------------------------------------------------------
N_NODES = 4
L_PREFIX = 1024                      # hot shared prefix length (tokens)
TOT_MEAN = 2048.0                    # equal mean total prompt length
TOT_LO, TOT_HI = 1920.0, 2176.0      # Uniform(1920, 2176) total length
GAMMA_BASE = 1.0 / 2048.0            # s/token: full 2048-token prefill ~ 1.0 s
SLO_TTFT = 2.0                       # common absolute SLO: per-class P99 TTFT <= 2.0 s
WARM_HOT_REQUESTS = 200              # fixed warm-up schedule (recorded, used by E1)

P_HOTS = [0.5, 0.7, 0.8, 0.9]
CS = [8, 16, 32, 64]
LOADS = [0.5, 0.65, 0.8]
GAMMAS = [0.5, 1.0, 2.0]

# hot suffix = total - 1024  ->  Uniform(896, 1152), mean 1024
S_HOT_LO, S_HOT_HI = TOT_LO - L_PREFIX, TOT_HI - L_PREFIX
S_COLD_LO, S_COLD_HI = TOT_LO, TOT_HI

# ---------------------------------------------------------------------------
# Uniform service-time moments (token units, then scaled by gamma)
# ---------------------------------------------------------------------------

def u_mean(a, b):
    return 0.5 * (a + b)

def u_m2(a, b):
    return (a * a + a * b + b * b) / 3.0

ES_HOT = u_mean(S_HOT_LO, S_HOT_HI)          # 1024
ES2_HOT = u_m2(S_HOT_LO, S_HOT_HI)           # ~1.054e6
ES_COLD = u_mean(S_COLD_LO, S_COLD_HI)       # 2048
ES2_COLD = u_m2(S_COLD_LO, S_COLD_HI)        # ~4.200e6

P99_S_HOT = (S_HOT_LO + 0.99 * (S_HOT_HI - S_HOT_LO)) * GAMMA_BASE   # 1149.44 * gamma_base
P99_S_COLD = (S_COLD_LO + 0.99 * (S_COLD_HI - S_COLD_LO)) * GAMMA_BASE  # 2173.44 * gamma_base

# ---------------------------------------------------------------------------
# Numerical Laplace inversion of the M/G/1 waiting-time CDF (Abate-Whitt Euler)
# ---------------------------------------------------------------------------
# For an M/G/1 FIFO queue:  W*(s) = (1-rho)*s / (s - lam + lam*S~(s))   (P-K)
# and F_W(t) = invL[ W*(s)/s ]. Inversion parameters validated against the
# exact M/M/1 closed form (see _validate_inversion below).

def _invert_laplace(F, t, n=45, m=17, A=19.5):
    """Abate-Whitt Euler summation inversion of the Laplace transform F(s).

    f(t) = (e^{A/2}/t) * [ b_0/2 + sum_{k>=1} (-1)^k b_k ],  b_k = Re F(A/2t + i k pi/t)
    (b_0 carries the trapezoidal half-weight; see Abate & Whitt 1995, eq. (2.4)-(2.5)).
    """
    from math import comb
    s0 = A / (2.0 * t)
    terms = [0.5 * np.real(F(complex(s0, 0.0)))]
    for k in range(1, n + m + 1):
        terms.append((-1.0) ** k * np.real(F(complex(s0, k * np.pi / t))))
    partials = np.cumsum(np.asarray(terms))
    euler = 0.0
    for k in range(m + 1):
        euler += comb(m, k) * (2.0 ** (-m)) * partials[n + k]
    return np.exp(A / 2.0) * euler / t


def _validate_inversion():
    """Check the Euler inversion against the exact M/M/1 tail P(W>t)=rho*exp(-mu(1-rho)t)."""
    worst = 0.0
    for rho in (0.3, 0.5, 0.7, 0.85, 0.9):
        lam, mu = rho, 1.0
        F = lambda s: (1.0 - rho) * s / (s - lam + lam * mu / (s + mu)) / s
        for t99_frac in (0.95, 0.99):
            t_exact = -np.log((1.0 - t99_frac) / rho) / (mu * (1.0 - rho))
            f_est = _invert_laplace(F, t_exact, n=120, m=25)
            err = abs(f_est - t99_frac) / t99_frac
            worst = max(worst, err)
    return worst


def mg1_wait_cdf(rho, lam, S_LST, t, n=120, m=25, A=19.5):
    """P(W <= t) for M/G/1 with load rho, arrival rate lam, service LST S_LST."""

    def F(s):
        return (1.0 - rho) * s / (s - lam + lam * S_LST(s)) / s

    return _invert_laplace(F, t, n=n, m=m, A=A)


def uni_lst(a, b, gamma):
    """Laplace-Stieltjes transform of Uniform(a,b)*gamma*GAMMA_BASE service time (seconds)."""
    c = gamma * GAMMA_BASE

    def S(s):
        sa, sb = c * a * s, c * b * s
        if abs(s) < 1e-9:
            return 1.0
        return (np.exp(-sa) - np.exp(-sb)) / (c * s * (b - a))

    return S


def mix_lst(p_hot_node, a_h, b_h, a_c, b_c, gamma):
    Sh, Sc = uni_lst(a_h, b_h, gamma), uni_lst(a_c, b_c, gamma)

    def S(s):
        return p_hot_node * Sh(s) + (1.0 - p_hot_node) * Sc(s)

    return S


def mg1_quantile(rho, lam, S_LST, q, lo=0.0, hi=500.0, n=120, m=25, A=19.5, tol=1e-4):
    """Quantile of the M/G/1 waiting-time CDF by bisection on the inversion.

    hi starts at 500 s: for all cells (rho <= 0.9, service ~0.5-1 s) the P99 wait
    is far below 500 s, and the Euler inversion is numerically clean only up to
    ~1e4 s (aliasing noise appears at larger t). Doubles hi if the CDF has not
    yet reached q. Cells with rho >= 1.0 must NOT call this (unstable queue)."""
    while _invert_laplace(lambda s: (1.0 - rho) * s / (s - lam + lam * S_LST(s)) / s,
                          hi, n=n, m=m, A=A) < q:
        hi *= 2.0
        if hi > 1e7:
            raise RuntimeError("quantile bracket failed at rho=%.3f lam=%.4f" % (rho, lam))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fmid = _invert_laplace(lambda s: (1.0 - rho) * s / (s - lam + lam * S_LST(s)) / s,
                               mid, n=n, m=m, A=A)
        if fmid < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Budget derivation (frozen cache model, experiment_plan.md 1.2)
# ---------------------------------------------------------------------------

def derive_budgets():
    """
    Two concrete total-KV budget levels.

    L1 (high): total 32,768 KV tokens = 8,192 tokens/node.
        Derivation: "4x2048-token-class concurrency" - per node, 4 concurrent
        2048-token request classes = 4*2048 = 8,192; x 4 nodes = 32,768.
        (Equivalently 4 nodes x 2048-token class x 4 = 32,768.) Consistent with
        the corpus envelope (Mooncake LRU: 30% hit @ 1K, 50% @ 50K, 51% @ Inf;
        SGLang production hits 52.4-74.1%): 8K/node sits mid-curve.
    L2 (low): total 16,384 KV tokens = 4,096/node = 50% smaller (hypothesis.md IV-6).
    """
    per_node = 4 * 2048
    l1 = per_node * N_NODES
    l2 = l1 // 2
    return l1, l2, {
        "L1": {
            "total_kv_tokens": l1,
            "per_node_kv_tokens": per_node,
            "derivation": ("4x2048-token-class concurrency: per node 4 concurrent "
                           "2048-token classes = 4*2048 = 8192 tokens; x4 nodes = 32768. "
                           "Within Mooncake LRU envelope 30%@1K -> 50%@50K -> 51%@Inf "
                           "(8K/node mid-curve) and SGLang production hit range."),
        },
        "L2": {
            "total_kv_tokens": l2,
            "per_node_kv_tokens": per_node // 2,
            "derivation": "50% smaller than L1 (hypothesis.md IV-6).",
        },
    }


def derive_g(l1, l2):
    """
    Memory-accounted hit-gain G from the LRU radix cache model (frozen 1.2):

    Reusable content in the workload = exactly ONE prefix (the hot prefix, 1024
    tokens). Cold requests carry unique, disjoint prefixes => never reused
    (Mooncake: >50% of blocks never reused).

    Policy A: hot prefix resident on the single hot node (1024 tokens).
      Every hot request is routed to that node => hits 1024 of its ~2048 tokens.
      Token hit rate = p_hot * 1024 / 2048 = p_hot/2.
    Policy B: load-only serves hot traffic on every node in steady state
      (min-backlog spreads hot arrivals over all 4 nodes), so the hot prefix
      accumulates on all 4 nodes (4096 tokens total).
      Every hot request hits wherever it lands => token hit rate = p_hot/2.

    LRU leaf-first eviction cannot evict the shared prefix while suffix leaves
    exist (leaves are evicted first, and they regenerate with every hot
    request); the prefix is therefore permanent on any node that ever served a
    hot request. Neither budget (L1=32768, L2=16384) can evict it (needs only
    <= 4096 tokens). The extra copies B holds buy nothing: there is no other
    reusable content to spend memory on.

    => G = p_hot/2 - p_hot/2 = 0.00 pp at BOTH budget levels, for every p_hot.
    """
    g = 0.0
    derivation = (
        "G = 0.00 pp at both budget levels, for all p_hot. Under the frozen cache "
        "model both policies cache ALL reusable content in steady state: A keeps "
        "the hot prefix on the hot node, B accumulates it on every node that "
        "serves hot traffic (all 4 nodes in steady state under min-backlog). "
        "Steady-state token hit rate = p_hot/2 under both policies. LRU leaf-first "
        "never evicts the shared prefix (suffix leaves are evicted first and "
        "regenerate), and both budget levels (L1=32768, L2=16384 total KV tokens) "
        "exceed the 4096 tokens B needs for its 4 prefix copies, so equal budgets "
        "do not create a differential. B's extra copies consume memory that buys "
        "no hits (no other reusable content exists). Hence G < 10pp => the "
        "pre-registered E0 kill of the hit leg fires (LOCKED_PLAN 6 / plan E0)."
    )
    return g, derivation


# ---------------------------------------------------------------------------
# Per-cell algebra
# ---------------------------------------------------------------------------

def affinity_saturation(p_hot, gamma):
    """Frozen definition (E0 formula (i)): hot-node util = p_hot*lam*1024*gamma*GAMMA_BASE.
    Saturation = the total offered rate at which that utilization reaches 1.0
    (req/s)."""
    return 1.0 / (p_hot * L_PREFIX * gamma * GAMMA_BASE)


def per_cell(p_hot, c, load, gamma):
    lam_sat = affinity_saturation(p_hot, gamma)
    lam = load * lam_sat

    # utilizations (gamma-invariant after anchoring)
    rho_hot_frozen = p_hot * lam * L_PREFIX * gamma * GAMMA_BASE    # = load (frozen def (i))
    rho_B = lam * (p_hot * ES_HOT + (1 - p_hot) * ES_COLD) * gamma * GAMMA_BASE / N_NODES
    cold_share_hot = 1.0 / N_NODES                           # mean-field min-backlog split
    rho_hot_full = rho_hot_frozen + cold_share_hot * (1 - p_hot) * lam * ES_COLD * gamma * GAMMA_BASE
    rho_cold = cold_share_hot * (1 - p_hot) * lam * ES_COLD * gamma * GAMMA_BASE

    # arrival-rate mixing probabilities (FIFO node sees class mixture)
    p_h_mix_hot = p_hot / (p_hot + cold_share_hot * (1 - p_hot))
    p_h_mix_B = p_hot

    # service-time moments in SECONDS (gamma * GAMMA_BASE scaling)
    ES_hotnode = (p_h_mix_hot * ES_HOT + (1 - p_h_mix_hot) * ES_COLD) * gamma * GAMMA_BASE
    ES2_hotnode = (p_h_mix_hot * ES2_HOT + (1 - p_h_mix_hot) * ES2_COLD) * (gamma * GAMMA_BASE) ** 2
    ES_B = (p_h_mix_B * ES_HOT + (1 - p_h_mix_B) * ES_COLD) * gamma * GAMMA_BASE
    ES2_B = (p_h_mix_B * ES2_HOT + (1 - p_h_mix_B) * ES2_COLD) * (gamma * GAMMA_BASE) ** 2
    ES_cold = ES_COLD * gamma * GAMMA_BASE
    ES2_cold = ES2_COLD * (gamma * GAMMA_BASE) ** 2

    def pk_mean(rho, es2, es):
        return rho * es2 / (2.0 * (1.0 - rho) * es)

    W_hotnode = pk_mean(rho_hot_full, ES2_hotnode, ES_hotnode)
    W_B = pk_mean(rho_B, ES2_B, ES_B)
    W_cold = pk_mean(rho_cold, ES2_cold, ES_cold)

    stable_hot = rho_hot_full < 1.0

    if stable_hot:
        # exact M/G/1 P99/P95 waiting times (numerical inversion)
        lam_hotnode = lam * (p_hot + cold_share_hot * (1 - p_hot))
        lam_B = lam / N_NODES            # each load-only node sees 1/4 of all arrivals
        lam_cold = lam * cold_share_hot * (1 - p_hot)
        S_hotnode = mix_lst(p_h_mix_hot, S_HOT_LO, S_HOT_HI, S_COLD_LO, S_COLD_HI, gamma)
        S_B = mix_lst(p_h_mix_B, S_HOT_LO, S_HOT_HI, S_COLD_LO, S_COLD_HI, gamma)
        S_cold = uni_lst(S_COLD_LO, S_COLD_HI, gamma)

        p99w_hotnode = mg1_quantile(rho_hot_full, lam_hotnode, S_hotnode, 0.99)
        p95w_hotnode = mg1_quantile(rho_hot_full, lam_hotnode, S_hotnode, 0.95)
        p99w_B = mg1_quantile(rho_B, lam_B, S_B, 0.99)
        p95w_B = mg1_quantile(rho_B, lam_B, S_B, 0.95)
        p99w_cold = mg1_quantile(rho_cold, lam_cold, S_cold, 0.99)

        # cold-class wait under A = mixed traffic: 75% cold-node FIFO, 25% hot-node FIFO
        # (mean-field min-backlog split of cold arrivals). Primary variant (cold never
        # lands on the hot node) is reported separately.
        def cold_mix_cdf(t):
            return 0.75 * mg1_wait_cdf(rho_cold, lam_cold, S_cold, t, n=120, m=25) \
                + 0.25 * mg1_wait_cdf(rho_hot_full, lam_hotnode, S_hotnode, t, n=120, m=25)

        p99w_coldmix_A = mg1_quantile_mix(cold_mix_cdf, 0.99)

        # TTFT P99 predictions (wait P99 + service P99, independent approximation)
        p99_ttft_hotA = p99w_hotnode + P99_S_HOT * gamma
        p99_ttft_hotB = p99w_B + P99_S_HOT * gamma
        p99_ttft_coldA_primary = p99w_cold + P99_S_COLD * gamma
        p99_ttft_coldA_mix = p99w_coldmix_A + P99_S_COLD * gamma
        p99_ttft_coldB = p99w_B + P99_S_COLD * gamma

        hot_ratio_p99TTFT = p99_ttft_hotA / p99_ttft_hotB
        cold_ratio_prim = p99_ttft_coldA_primary / p99_ttft_coldB
        cold_ratio_mix = p99_ttft_coldA_mix / p99_ttft_coldB
        slo_flag_hotA = bool(p99_ttft_hotA > SLO_TTFT)
        slo_flag_coldA_primary = bool(p99_ttft_coldA_primary <= SLO_TTFT)
        slo_flag_coldA_mix = bool(p99_ttft_coldA_mix <= SLO_TTFT)
        slo_crossed_primary = slo_flag_hotA and slo_flag_coldA_primary
        slo_crossed_mix = slo_flag_hotA and slo_flag_coldA_mix
    else:
        # overloaded hot node (rho >= 1.0): M/G/1 predictions undefined; cell is
        # reported separately and excluded from all verdicts (frozen rule)
        lam_hotnode = lam_B = lam_cold = np.nan
        p99w_hotnode = p99w_B = p99w_cold = np.nan
        p95w_hotnode = p95w_B = np.nan
        p99w_coldmix_A = np.nan
        p99_ttft_hotA = p99_ttft_hotB = np.nan
        p99_ttft_coldA_primary = p99_ttft_coldA_mix = p99_ttft_coldB = np.nan
        hot_ratio_p99TTFT = cold_ratio_prim = cold_ratio_mix = np.nan
        slo_flag_hotA = False
        slo_flag_coldA_primary = False
        slo_flag_coldA_mix = False
        slo_crossed_primary = False
        slo_crossed_mix = False

    # aggregate mean TTFT predictions
    mean_sv_hot = ES_HOT * gamma * GAMMA_BASE
    mean_sv_cold = ES_COLD * gamma * GAMMA_BASE
    agg_A_primary = p_hot * (W_hotnode + mean_sv_hot) + (1 - p_hot) * (W_cold + mean_sv_cold)
    agg_A_mix = p_hot * (W_hotnode + mean_sv_hot) + (1 - p_hot) * (
        0.75 * (W_cold + mean_sv_cold) + 0.25 * (W_hotnode + mean_sv_cold))
    agg_B = W_B + (p_hot * mean_sv_hot + (1 - p_hot) * mean_sv_cold)

    return dict(
        p_hot=p_hot, c=c, load=load, gamma=gamma,
        lam_sat=lam_sat, lam=lam,
        rho_hot_frozen=rho_hot_frozen, rho_B_pernode=rho_B,
        rho_hot_full=rho_hot_full, rho_cold_node=rho_cold,
        p_hot_mix_hotnode=p_h_mix_hot, p_hot_mix_B=p_h_mix_B,
        ES_hotnode=ES_hotnode, ES2_hotnode=ES2_hotnode,
        ES_B=ES_B, ES2_B=ES2_B,
        W_mean_hotnode=W_hotnode, W_mean_B=W_B, W_mean_cold=W_cold,
        p99w_hotnode=p99w_hotnode, p95w_hotnode=p95w_hotnode,
        p99w_B=p99w_B, p95w_B=p95w_B, p99w_cold=p99w_cold,
        p99w_coldmix_A=p99w_coldmix_A,
        p99_ttft_hotA=p99_ttft_hotA, p99_ttft_hotB=p99_ttft_hotB,
        p99_ttft_coldA_primary=p99_ttft_coldA_primary,
        p99_ttft_coldA_mix=p99_ttft_coldA_mix,
        p99_ttft_coldB=p99_ttft_coldB,
        hot_ratio_meanW=W_hotnode / W_B,
        hot_ratio_p99TTFT=hot_ratio_p99TTFT,
        cold_ratio_p99TTFT_primary=cold_ratio_prim,
        cold_ratio_p99TTFT_mix=cold_ratio_mix,
        agg_mean_ttft_A_primary=agg_A_primary,
        agg_mean_ttft_A_mix=agg_A_mix,
        agg_mean_ttft_B=agg_B,
        agg_mean_ratio_primary=agg_A_primary / agg_B,
        agg_mean_ratio_mix=agg_A_mix / agg_B,
        slo_flag_hotA=slo_flag_hotA,
        slo_flag_coldA_primary=slo_flag_coldA_primary,
        slo_flag_coldA_mix=slo_flag_coldA_mix,
        slo_crossed_primary=slo_crossed_primary,
        slo_crossed_mix=slo_crossed_mix,
        stable_frozen=(rho_hot_frozen < 1.0),
        stable_full=stable_hot,
    )


def mg1_quantile_mix(cdf, q, lo=0.0, hi=500.0, tol=1e-4):
    while cdf(hi) < q:
        hi *= 2.0
        if hi > 1e7:
            raise RuntimeError("mixture quantile bracket failed")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = datetime.datetime.now(datetime.timezone.utc).isoformat()

    print("E0: validating Laplace inversion against exact M/M/1 ...")
    inv_err = _validate_inversion()
    print("  worst relative error vs exact M/M/1 quantiles: %.4g" % inv_err)
    if inv_err > 0.01:
        raise RuntimeError("inversion validation failed (>1% error) - do not trust P99")

    l1, l2, budget_meta = derive_budgets()
    g, g_derivation = derive_g(l1, l2)

    rows = []
    for p_hot in P_HOTS:
        for c in CS:
            for load in LOADS:
                for gamma in GAMMAS:
                    rows.append(per_cell(p_hot, c, load, gamma))
    grid = pd.DataFrame(rows)

    # processed/e0_results.csv (required artifact): per (p_hot, c, load, gamma)
    res = grid[[
        "p_hot", "c", "load", "gamma",
        "rho_hot_frozen", "rho_hot_full", "rho_B_pernode",
        "hot_ratio_meanW", "hot_ratio_p99TTFT",
        "cold_ratio_p99TTFT_primary", "cold_ratio_p99TTFT_mix",
        "agg_mean_ratio_primary", "agg_mean_ratio_mix",
        "slo_flag_hotA", "slo_flag_coldA_primary", "slo_flag_coldA_mix",
        "slo_crossed_primary", "slo_crossed_mix",
        "stable_frozen", "stable_full",
    ]].copy()
    res["G_pp_L1"] = g
    res["G_pp_L2"] = g
    res.to_csv(os.path.join(PROC, "e0_results.csv"), index=False)

    grid.to_csv(os.path.join(RAW, "e0_grid.csv"), index=False)

    budget_rows = pd.DataFrame([
        {"budget_level": "L1", "total_kv_tokens": l1, "per_node_kv_tokens": l1 // N_NODES,
         "G_pp": g, "derivation": budget_meta["L1"]["derivation"]},
        {"budget_level": "L2", "total_kv_tokens": l2, "per_node_kv_tokens": l2 // N_NODES,
         "G_pp": g, "derivation": budget_meta["L2"]["derivation"]},
    ])
    budget_rows.to_csv(os.path.join(RAW, "e0_budget.csv"), index=False)

    _write_report(grid, budget_rows, g, g_derivation, inv_err, l1, l2)

    print("wrote raw/e0_grid.csv (%d cells), raw/e0_budget.csv, "
          "processed/e0_results.csv, processed/e0_report.md" % len(grid))
    print("E0 done at", datetime.datetime.now(datetime.timezone.utc).isoformat())


def _write_report(grid, budget_rows, g, g_derivation, inv_err, l1, l2):
    gamma_lo, gamma_hi, gamma_mid = 0.5, 2.0, 1.0
    md = []
    md.append("# e0_report.md - RQ-4 gate-0 algebra (frozen LOCKED_PLAN.md, plan section E0)")
    md.append("")
    md.append("Derivation run at %s UTC on %s (CPU only). Script SHA-256: %s"
              % (datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 os.uname().sysname if hasattr(os, "uname") else "Windows",
                 _sha256(__file__)))
    md.append("")
    md.append("## 1. Model (frozen)")
    md.append("")
    md.append("- N = 4 homogeneous nodes, each one FIFO prefill server; admission = dispatch, "
              "no global FIFO (plan 1.2).")
    md.append("- Service T_prefill = gamma * tokens_not_cached, gamma_base = 1/2048 s/token "
              "(2048-token prefill ~ 1.0 s).")
    md.append("- Hot class: shared 1024-token prefix + unique suffix, total Uniform(1920,2176) "
              "(suffix Uniform(896,1152), mean 1024). Cold class: unique prefixes, same total "
              "length distribution, always miss.")
    md.append("- Policies: A affinity (longest-prefix, tiebreak min token-weighted backlog); "
              "B load-only (min token-weighted backlog); B' request-count least-loaded (E1 cell).")
    md.append("- Load anchoring (frozen): loads are fractions of the AFFINITY configuration's own "
              "saturation; overload region (hot-node util >= 1.0, frozen definition) excluded "
              "from verdicts.")
    md.append("- Hot-node utilization (frozen formula (i)) = (p_hot * suffix_hot work) / "
              "(per-node total work) * load-only utilization; equivalently "
              "rho_hot = p_hot * lam * 1024 * gamma = load fraction by construction.")
    md.append("")
    md.append("## 2. Affinity saturation (per-policy load anchoring base)")
    md.append("")
    md.append("lambda_sat(p_hot, gamma) = 1 / (p_hot * 1024 * gamma)  [req/s]  (hot-node util = 1.0).")
    md.append("")
    md.append("| p_hot | gamma=0.5 | gamma=1.0 | gamma=2.0 |")
    md.append("|---|---|---|---|")
    for p in P_HOTS:
        md.append("| %s | %.5f | %.5f | %.5f |" % (
            p, affinity_saturation(p, gamma_lo),
            affinity_saturation(p, gamma_mid), affinity_saturation(p, gamma_hi)))
    md.append("")
    md.append("E1 load levels {0.5, 0.65, 0.8} x lambda_sat. In the closed form the resulting "
              "hot-node util (frozen def) = load exactly; utilizations and P99 ratios are "
              "gamma-scale-invariant under the linear service model (E1 re-checks at "
              "gamma in {0.5, 1, 2}).")
    md.append("")
    md.append("## 3. Stable-region map (gamma = 1; frozen definition; utilizations are gamma-invariant)")
    md.append("")
    md.append("rho_hot (frozen def) = load for every cell. rho_hot_full additionally includes the "
              "cold-class work the hot node serves under mean-field min-backlog routing "
              "(cold share 1/4 per node). Cells where rho_hot_full >= 1.0 are marked; the frozen "
              "overload definition uses rho_hot_frozen (>= 1.0 never occurs at these loads).")
    md.append("")
    md.append("| p_hot | load | rho_hot (frozen) | rho_hot_full | rho_B per-node | stable(frozen) | stable(full) |")
    md.append("|---|---|---|---|---|---|---|")
    for p in P_HOTS:
        for load in LOADS:
            r = grid[(grid.p_hot == p) & (grid.load == load) & (grid.gamma == gamma_mid)].iloc[0]
            md.append("| %.1f | %.2f | %.3f | %.3f | %.3f | %s | %s |" % (
                p, load, r.rho_hot_frozen, r.rho_hot_full, r.rho_B_pernode,
                r.stable_frozen, r.stable_full))
    md.append("")
    md.append("## 4. Predicted ratios at the pre-registered E1 points (c = 32)")
    md.append("")
    md.append("Expected values per E1 cell (c = 32; budget L1/L2 do not enter the algebra). "
              "cold_*_primary = cold traffic served only on cold nodes; cold_*_mix = 25% of cold "
              "arrivals land on the hot node (mean-field split). E1 measures the realized split.")
    md.append("")
    md.append("| p_hot | load | gamma | hot P99 TTFT ratio | cold P99 ratio (prim) | "
              "cold P99 ratio (mix) | agg mean ratio (prim) | agg mean ratio (mix) | "
              "hot P99 TTFT A (s) | cold P99 TTFT A (s) | hot P99 TTFT B (s) | "
              "hot util (full) | stable |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in (0.8, 0.9):
        for load in LOADS:
            for gamma in GAMMAS:
                r = grid[(grid.p_hot == p) & (grid.load == load) & (grid.gamma == gamma)
                         & (grid.c == 32)].iloc[0]
                md.append("| %.1f | %.2f | %.1f | %.2f | %.3f | %.3f | %.3f | %.3f | "
                          "%.2f | %.2f | %.2f | %.3f | %s |" % (
                    p, load, gamma, r.hot_ratio_p99TTFT, r.cold_ratio_p99TTFT_primary,
                    r.cold_ratio_p99TTFT_mix, r.agg_mean_ratio_primary, r.agg_mean_ratio_mix,
                    r.p99_ttft_hotA, r.p99_ttft_coldA_primary, r.p99_ttft_hotB,
                    r.rho_hot_full, r.stable_full))
    md.append("")
    md.append("## 5. SLO crossing flags (common SLO: per-class P99 TTFT <= 2.0 s)")
    md.append("")
    md.append("Cells whose hot node is overloaded (rho_hot_full >= 1.0) are marked "
              "'overload' - M/G/1 predictions are undefined there and the cell is "
              "excluded from verdicts (frozen rule); reported separately.")
    md.append("")
    md.append("| p_hot | load | gamma | hot P99 A > 2.0s | cold P99 A <= 2.0s (prim) | "
              "cold P99 A <= 2.0s (mix) | SLO crossed (prim) | SLO crossed (mix) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for p in P_HOTS:
        for load in LOADS:
            for gamma in GAMMAS:
                r = grid[(grid.p_hot == p) & (grid.load == load) & (grid.gamma == gamma)
                         & (grid.c == 32)].iloc[0]
                if not r.stable_full:
                    flags = ["overload"] * 5
                else:
                    flags = [r.slo_flag_hotA, r.slo_flag_coldA_primary,
                             r.slo_flag_coldA_mix, r.slo_crossed_primary, r.slo_crossed_mix]
                md.append("| %.1f | %.2f | %.1f | %s | %s | %s | %s | %s |" % (
                    p, load, gamma, flags[0], flags[1], flags[2], flags[3], flags[4]))
    md.append("")
    md.append("## 6. Budget levels and memory-accounted hit-gain G")
    md.append("")
    md.append("| level | total KV tokens | per-node | G (pp) | derivation |")
    md.append("|---|---|---|---|---|")
    for _, row in budget_rows.iterrows():
        md.append("| %s | %d | %d | %.2f | %s |" % (
            row.budget_level, row.total_kv_tokens, row.per_node_kv_tokens,
            row.G_pp, row.derivation))
    md.append("")
    md.append("### G derivation")
    md.append("")
    md.append(g_derivation)
    md.append("")
    md.append("**Pre-registered E0 outcome:** G = 0.00 pp < 10 pp at both budget levels => "
              "the hit-rate leg of H1 is killed pre-registrationally (LOCKED_PLAN 6, plan E0 "
              "success criterion); H1 rests on the queue leg alone (hot P99 ratio >= 2, "
              "cold P99 ratio <= 1.2, aggregate mean TTFT <= 0.8x). The E1 falsification "
              "criterion (i) (hit gain <= 5pp kills H1) does NOT apply.")
    md.append("")
    md.append("## 7. Pre-registered E1 / E2 grid points")
    md.append("")
    md.append("- E1 (this wave): p_hot in {0.8, 0.9}, c = 32, loads {0.5, 0.65, 0.8} x "
              "affinity saturation, budgets {L1, L2}, gamma {0.5, 1, 2}, 10 reps, "
              "policies A, B (+ B' sensitivity at p_hot = 0.8). Expected values: see table 4.")
    md.append("- E2 (next wave, pre-registered): full grid p_hot in {0.5, 0.7, 0.8, 0.9} x "
              "c in {8, 16, 32, 64} x loads {0.5, 0.65, 0.8} x gamma {0.5, 1, 2}, both budget "
              "levels; expected values = the per-cell predictions in raw/e0_grid.csv "
              "(hot P99 TTFT ratio, cold P99 TTFT ratio, aggregate mean ratio, SLO flags).")
    md.append("")
    md.append("## 8. Inversion validation")
    md.append("")
    md.append("The M/G/1 P99 waiting times use numerical Laplace inversion (Abate-Whitt Euler) "
              "of the exact Pollaczek-Khinchin wait transform with Uniform service. "
              "Worst relative error vs the exact M/M/1 closed-form quantiles: %.4g (threshold 1%%)." % inv_err)
    md.append("")
    md.append("## 9. Verification of the frozen sanity rule")
    md.append("")
    md.append("E0 predictions are checked against the E1 simulator at 3 anchor points "
              "(utilization and mean wait within a few %), per plan E0 baseline / E1 sanity; "
              "see processed/e1_report.md for the measured-vs-predicted comparison.")
    md.append("")

    with open(os.path.join(PROC, "e0_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
