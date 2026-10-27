"""Load reconstruction (paper v7 Sec 3.1/6.1). Original 100+100 GPT-5 traces not open;
we rebuild BFCL-like / SWE-like / mixed with matching (turns, tool ms, tokens/program).

Paper stats (mean, std): SWE turns (10.9,2.1) tool (925,3550)ms tok (70126,19732);
BFCL turns (6.3,2.3) tool (1923,2133)ms tok (93256,68687). Tool tail: slowest 10%
fetch_url = 52.5% delay, cd = 94.1% -> lognormal mixture with explicit tail frac.
Differences vs paper: synthetic durations/contexts, no real tool execution, BFCL
0.4x token scaling supported, arrivals are Poisson programs (paper replay).
"""
from __future__ import annotations

import math
import random


def _lognorm(rng: random.Random, mean: float, std: float) -> float:
    if mean <= 0:
        return 0.0
    var = std ** 2
    sigma2 = math.log1p(var / mean ** 2)
    mu = math.log(mean) - sigma2 / 2
    return rng.lognormvariate(mu, math.sqrt(sigma2))


def _tool_duration(rng: random.Random, kind: str) -> tuple[str, float]:
    """Return (tool_name, seconds). Mixture: 90% body + 10% Pareto-ish tail."""
    if kind == "bfcl":
        name = rng.choice(["fetch_url", "search", "lookup", "calc"])
        body = _lognorm(rng, 1.2, 1.2)
        tail = rng.paretovariate(1.4) * 2.0  # seconds, heavy tail
        d = body if rng.random() < 0.9 else body + tail
        return name, max(0.05, min(d, 30.0))
    name = rng.choice(["cd", "ls", "git", "pytest", "grep"])
    body = _lognorm(rng, 0.55, 0.9)
    tail = rng.paretovariate(1.25) * 1.5
    d = body if rng.random() < 0.9 else body + tail
    return name, max(0.03, min(d, 25.0))


def gen_programs(kind: str, n: int, seed: int = 0, token_scale: float = 1.0,
               max_ctx: int | None = None) -> list[dict]:
    rng = random.Random(seed)
    progs = []
    for pid in range(n):
        if kind == "bfcl":
            turns = max(2, int(round(rng.gauss(6.3, 2.3))))
            total_tok = max(8000, rng.gauss(93256, 68687)) * token_scale
        elif kind == "swe":
            turns = max(3, int(round(rng.gauss(10.9, 2.1))))
            total_tok = max(8000, rng.gauss(70126, 19732)) * token_scale
        else:  # mixed: alternate + random
            sub = "bfcl" if (pid % 2 == 0) else "swe"
            base = gen_programs(sub, 1, seed + pid * 7919, token_scale,
                                max_ctx)[0]
            base["pid"] = pid
            progs.append(base)
            continue
        if max_ctx is not None:
            # paper 6.1 analogue: scale to fit context (their BFCL 0.4x for 128k)
            total_tok = min(total_tok, max_ctx)
        # growing context: each turn prefills the FULL accumulated prefix
        # (real serving), ending near total_tok; early turns start at 30%.
        ctxs, tools = [], []
        for k in range(turns):
            frac = 0.3 + 0.7 * (k + 1) / turns
            ctxs.append(int(total_tok * frac))
            tools.append(_tool_duration(rng, kind))
        # last turn has no trailing tool
        tools[-1] = (tools[-1][0], 0.0)
        progs.append({"pid": pid, "kind": kind, "turns": turns,
                      "ctx": ctxs, "tools": tools})
    return progs


def add_arrivals(progs: list[dict], jps: float, seed: int = 0) -> list[dict]:
    """Poisson program arrivals (paper replay). Mutates dicts with t_arrive."""
    rng = random.Random(seed + 999)
    t = 0.0
    for p in progs:
        t += rng.expovariate(jps) if jps > 0 else 0.0
        p["t_arrive"] = t
    return progs
