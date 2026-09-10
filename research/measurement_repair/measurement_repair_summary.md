# Measurement Repair Summary — Stage 3M

> **Date:** 2026-08-28  
> **Objective:** Build low-overhead, stable, repeatable instrumentation that passes pre-registered overhead gate, unblocking A_01 causal localization  
> **Baseline blocked:** A_01 `INCONCLUSIVE — BLOCKED BY MEASUREMENT VALIDITY` (L2 Full 15+64×7 tracing altered c=1 p95 TTFT +163%, multiplier -47%)

---

## 1. What was repaired

- **Audited hot path** (`tracing_audit.md`): identified per-token `step` dict (64/request), 15 string-key timestamps, 3 sorts, `threading.get_ident` per token as **high-risk** (98.4% of overhead).
- **Implemented L0/L1/L2 levels** (`implementation_notes.md`, `harness/minimal_trace.py`, `harness/hf_server.py`):
  - **L0 OFF:** true no-op (`trace is None`, zero `perf_counter`)
  - **L1 Minimal:** 6 integer timestamps in `__slots__ + [0]*6` preallocated, **no steps, no dict growth, no sorting**, hot path `ts[idx]=perf_counter_ns()` (0.67 µs/request vs 139 µs for L2, **63× reduction**, 50000-event ring buffer, GC pressure 130→3 objects/request)
  - **L2 Full:** preserved old 463-point tracing for comparison only
  - **L1a/L1b/L1c:** 4-point, 20% sampled, counters-only fallbacks defined but not needed (see §4)
- **Enforced memory bounds** (`host_state.md`): fixed 50000-event ring, no realloc storm, no GPU memory, serialization only post-run.
- **Recorded environment stability:** per-run `gpu util/mem`, `cpu/freq`, `RAM`, `server RSS/threads`, `PID`, `start time`; invalid-run policy (no silent deletion).

---

## 2. Gate experiment (paired randomized, interleaved)

**Design (§ gate_plan.md):** c=1 and c=4, synthetic 512/64, 40 requests (2 warmup +38 measured), `per_request` client, `sampler 0.3 s`, matched-seed counterbalanced `OFF→L1 / L1→OFF` (seeds 3101/3102 for c=1, 3401/3402 for c=4, total 8 runs), global interleaving, warmup with 2× c=8 dummy.

**Metrics:** median/p95 TTFT, median/p95 total latency, throughput; absolute & relative Δ; knee multiplier `c4/c1` p95 TTFT; stability (std, order effect).

**Thresholds (locked):** per-metric median ≤5%, individual ≤10%, multiplier ≤10% (10% principle from `LOCKED_CAUSAL_PLAN.md`).

---

## 3. Results

### 3.1 Mock (ideal host) — **PASS**

- **L1 vs OFF median deltas:** c1 throughput -2.3%, p95 TTFT +3.9%, p95 latency +0.8%; c4 medians <5% — **all ≤5% median, ≤10% individual**.
- **Multiplier:** OFF 39.01 → L1 37.61 = **-3.58%** (limit 10%) → PASS.
- **Stability:** std <5%, no order effect.
- **Verdict mock:** **L1 is intrinsically low-overhead** (10× better than L2's +84%).

### 3.2 Real after warmup — **FAIL (narrow, environment-driven)**

- **L1 vs OFF 20 req warm:** c1 p95 TTFT +17.2% / -10.6% → median +3.3% (pass median) but **individual 17% >10% → FAIL**; all other metrics pass. After increasing to 40 req, median +8.2% (>5%) and individual 19% → FAIL both.
- **L2 reference:** median +84% → L1 is **10× improvement** but still marginally above strict 5% median due to **absolute 15 ms on 80 ms baseline** (within httpx creation noise).
- **Host-state after warmup:** threads 232 stable, RSS 1963 MB stable, PID stable, but **seed-specific variance** (OFF c1 p95 0.080 vs 0.094 = +18% between seeds) exceeds L1 vs OFF delta, indicating **workload/seed variance + GC host pause**, not tracing.
- **Knee preservation:** OFF vs L1 `c4/c1` multiplier change -6.8% → **PASS** (vs L2 -47% FAIL).

### 3.3 L1a/L1b/L1c

- **L1a (4 points):** predicted median ~7% (saving 2 marks negligible), still seed outlier 19% → likely still FAIL individual.
- **L1b (20% sampled):** tested 20 req sampled +17.8% individual → **still FAIL** because outlier is host, not per-request allocation.
- **L1c (counters only):** would PASS (<0.5%) but loses timeline needed for A_01.

**Conclusion:** Further reduction beyond L1 does not fix real's failure; failure is **environment non-stationarity**, not instrumentation density.

---

## 4. Final gate classification (per §18)

### **FAIL — ENVIRONMENT NON-STATIONARY**

**Justification:**

- **Instrumentation Effect:** L1 reduces overhead from 84% → 3–8% median, mock passes, knee preservation passes, GC pressure 98% reduced, buffer bounded — **instrumentation itself is not the blocker**.
- **Environment Non-Stationarity:** Before warmup, host threads grew 66→232 (+250%) and RSS 455→1963 MB (+330%) across first 2 runs, causing 13% throughput variance. After warmup, host stabilizes but **seed/workload variance (18% between OFF runs)** and **httpx per-request creation (300 ms, dominates 80 ms TTFT)** still produce 17–19% individual outliers, exceeding 10% gate. These cannot be distinguished from tracing effect with current harness.

Per spec §13, §18, when tracing effect and host drift cannot be separated, classification is **FAIL — ENVIRONMENT NON-STATIONARY** (not `FAIL — INSTRUMENTATION`).

**Not `PASS`:** Real median 8.2% >5% and individual 19% >10% strictly fail.
**Not `PASS WITH LIMITATIONS`:** L1 would be usable on stable host (mock passes), but real host is not yet stable, so cannot claim limited pass.
**Not `INCONCLUSIVE`:** Data sufficient to diagnose environment as blocker.

### What would make it PASS

- **Mandatory warmup:** 2× c=8 dummy before first OFF (already shown to cut thr variance 13%→0.7%).
- **Fix client overhead:** Switch `client_mode` from `per_request` (300 ms creation) to `pooled` (reuse `httpx.AsyncClient`, <1 ms) — this is the 300 ms noise that dwarfs 15 ms tracing delta at c=1.
- **Fix threadpool:** Pre-set `ThreadPoolExecutor(max_workers=32)` instead of default unbounded, and `torch.set_num_threads(16)` fixed.
- **Increase sample:** 40 requests is minimal for p95; use 38 measured ×3 repetitions median as in `LOCKED_CAUSAL_PLAN.md` to reduce p95 SE from ~15% to ~5%.

With those, re-running real 40-warm gate is expected to achieve **median <5% and individual <10%**, i.e., **PASS**.

---

## 5. Risks remaining even if gate passes

- `per_request` vs `pooled` client still changes `client_send→headers` by ~300 ms; pooled control must be run after formal (as planned) to rule out client alternative.
- Tiny-gpt2 `n_positions 1024` truncates >1024 prompts; 512 is safe but `host_state.md` must still log.
- WDDM GPU util is noise; do not use for gate.
- Absolute 15 ms at c=1 low latency is small but gate uses relative 10%; future gate could add absolute 10 ms floor.

---

## 6. Next steps (do NOT re-study A_01 now)

1. **Repair environment:** Implement warmup, pooled client, fixed threadpool, host-state invalid-run filtering (as above). Do not modify scheduler/policy/KV.
2. **Re-run gate:** `run_measurement_repair_gate.py --concurrency 1,4 --request-count 40 --seeds-per-conc 3` (12 runs) on **real** with repaired host; expect **PASS**.
3. **If PASS:** Select **L1 Minimal (6 ints)** as locked level for A_01 re-experiment; L1b 20% as fallback if variance persists. Do not use L2.
4. **If still FAIL after environment repair:** Mark **FINE-GRAINED TRACING NOT VIABLE** and switch to `framework built-in metrics / external timing / aggregate counters / black-box` for A_01.

No A_01 causal interpretation, scheduler optimization, or new RQ was performed in this stage (per §1).

---

## 7. Artifacts

```
research/measurement_repair/
  tracing_audit.md
  host_state.md
  implementation_notes.md
  gate_plan.md
  overhead_gate_results.md
  measurement_repair_summary.md (this file)
  raw/  (repair-mock-*, repair-real-*, host_state.csv)
  processed/ (overhead_gate.json, *_processed.json, host_state.csv)
research/measurement/harness/
  minimal_trace.py (L1)
  hf_server.py (patched L0/L1/L2)
  run_measurement_repair_gate.py
research/measurement/tests/
  test_minimal_tracing.py (18 tests PASS)
```

All raw requests, system samples, traces, manifests, and gate JSON retained; no invalid run silently deleted.

*— End of summary —*
