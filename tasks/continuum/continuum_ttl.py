"""Continuum TTL core (paper v7, Sec 4.1-4.2). Shared contract for sim + real vLLM hook."""
from __future__ import annotations

import math
from collections import defaultdict, deque


K_COLD = 100          # cold-start threshold (paper: K=100)
HIST_WINDOW = 1000    # bounded per-tool history (our reconstruction choice; paper says online records)
T_DEFAULT_FALLBACK = 2.0  # safety if T*eta+PR <= 1 under Exp(1) derivation


class ToolHistory:
    """Per-tool durations S[f] + global S, bounded windows."""

    def __init__(self, window: int = HIST_WINDOW):
        self.per_tool: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.global_hist: deque = deque(maxlen=window * 4)

    def add(self, tool: str, duration_s: float) -> None:
        if duration_s < 0:
            return
        self.per_tool[tool].append(float(duration_s))
        self.global_hist.append(float(duration_s))

    def get(self, tool: str):
        """Return (samples, is_global_fallback, is_cold)."""
        s = list(self.per_tool.get(tool, []))
        if len(s) > K_COLD:
            return s, False, False
        g = list(self.global_hist)
        if len(g) > K_COLD:
            return g, True, False
        return g, True, True


def ecdf(tau: float, samples: list[float]) -> float:
    if not samples:
        return 0.0
    return sum(1.0 for t in samples if t <= tau) / len(samples)


def t_default(T: float, eta: float, prefill_reload: float) -> float:
    """Optimal tau under ToolDuration~Exp(1), fully memoryful eta=1 (paper 4.2).

    maximize (1-exp(-tau)) * B - tau, B = T*eta + PR  ->  tau* = ln(B) if B > 1 else 0.
    """
    B = T * eta + prefill_reload
    if B <= 1.0:
        return 0.0
    return math.log(B)


def calc_ttl(tool: str, hist: ToolHistory, T: float, eta: float,
             prefill_reload: float) -> tuple[float, dict]:
    """Eq.(2): tau* = argmax P(tau,f)*(T*eta + PR) - tau over unique durations + 0."""
    samples, is_global, is_cold = hist.get(tool)
    B = T * eta + prefill_reload
    if is_cold:
        tau = t_default(T, 1.0, prefill_reload)
        return tau, {"mode": "cold", "tau": tau, "B": B}
    cands = sorted(set(samples) | {0.0})
    best_tau, best_val = 0.0, 0.0  # tau=0 -> value 0
    for tau in cands:
        v = ecdf(tau, samples) * B - tau
        if v > best_val + 1e-12:
            best_val, best_tau = v, tau
    return best_tau, {"mode": "global" if is_global else "per_tool",
                      "n": len(samples), "B": B, "value": best_val}


def compute_eta(finished_programs: list[int]) -> float:
    """eta = -Corr(k, N-k) over k=1..N-1 for finished programs with N>=2.

    Fixed N -> Corr(k,-k) = -1 -> eta=1. Geometric-ish -> ~0.
    Returns 1.0 when undefined (too few samples / zero variance).
    """
    xs, ys = [], []
    for N in finished_programs:
        if N < 2:
            continue
        for k in range(1, N):
            xs.append(float(k))
            ys.append(float(N - k))
    n = len(xs)
    if n < 2:
        return 1.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    vx = sum((x - mx) ** 2 for x in xs) / n
    vy = sum((y - my) ** 2 for y in ys) / n
    if vx <= 0 or vy <= 0:
        return 1.0
    corr = cov / math.sqrt(vx * vy)
    corr = max(-1.0, min(1.0, corr))
    return max(-1.0, min(1.0, -corr))


def prefill_reload_cost(ctx_tokens: int, offload: bool = False,
                        bw_gbps: float = 20.0, a: float = 6.0e-10,
                        b: float = 1.0e-5) -> float:
    """Offline-profile surrogate (paper 5.2): quadratic prefill + optional reload.

    Calibrated so 70k ctx ~ 3.6s prefill (V100-class, cf. A0 2.8k ctx -> ~5s
    prefill incl. queueing), 10k ctx ~ 0.16s. Reload = bytes/bandwidth.
    """
    prefill = a * ctx_tokens ** 2 + b * ctx_tokens
    if not offload:
        return prefill
    kv_bytes = ctx_tokens * 131072  # ~128KB/token Llama-8B-class (GQA)
    reload = kv_bytes / (bw_gbps * 1e9)
    return min(prefill, reload) if offload else prefill
