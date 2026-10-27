"""Unit checks for TTL math (paper v7 Eq.2, cold start, eta). Run: python test_ttl.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from continuum_ttl import (ToolHistory, calc_ttl, compute_eta, t_default,
                           prefill_reload_cost)

# 1. cold start -> T_default = ln(B) regime
h = ToolHistory()
tau, info = calc_ttl("fetch_url", h, T=2.0, eta=1.0, prefill_reload=3.0)
assert info["mode"] == "cold" and abs(tau - t_default(2.0, 1.0, 3.0)) < 1e-9, (tau, info)
print("cold start ok:", round(tau, 3), info["mode"])

# 2. per-tool CDF: short tool -> pin covers most mass; value must beat tau=0
for t in [0.4, 0.5, 0.6, 0.5, 0.45, 0.55, 0.5, 0.6, 0.4, 0.5] * 12:
    h.add("cd", t)
tau, info = calc_ttl("cd", h, T=1.0, eta=1.0, prefill_reload=3.0)
assert info["mode"] == "per_tool" and tau > 0, (tau, info)
print("per-tool ok: tau*=%.2f n=%d B=%.2f val=%.2f" % (tau, info["n"], info["B"], info["value"]))

# 3. global fallback between K>=|S[f]| and cold
h2 = ToolHistory()
for t in [1.0, 1.2, 0.9, 1.1] * 30:
    h2.add("other", t)
tau, info = calc_ttl("unseen_tool", h2, T=1.0, eta=1.0, prefill_reload=3.0)
assert info["mode"] == "global", info
print("global fallback ok: tau*=%.2f" % tau)

# 4. eta: fixed-length programs -> 1.0; single sample -> 1.0
assert abs(compute_eta([8] * 20) - 1.0) < 1e-9
assert compute_eta([5]) == 1.0
print("eta fixed-length ok: %.3f" % compute_eta([8] * 20))

# 5. cost calibration sanity: 70k ctx prefill ~ seconds; reload < prefill
pr = prefill_reload_cost(70000, offload=False)
rl = prefill_reload_cost(70000, offload=True)
assert 1.0 < pr < 10.0 and 0 < rl < pr, (pr, rl)
print("prefill/reload ok: prefill=%.2fs reload=%.2fs @70k" % (pr, rl))
print("ALL TTL UNIT CHECKS PASSED")
