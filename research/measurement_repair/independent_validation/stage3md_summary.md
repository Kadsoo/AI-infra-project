# Stage 3M-D Summary — Independent Validation of Candidate Stable Protocol D

> **Date:** 2026-08-29T07:45:30Z  
> **Server (new isolated session):** `sshleifer/tiny-gpt2` CPU FP32, `hf_server.py` hash `39cc7cd90b`, `run_stage3md.py` hash `3b2d5ab1`, `minimal_trace.py` not used in Phase A (L0 OFF)  
> **Host:** LAPTOP-1PB54QSI i7-14650HX 24T / 31.78 GB / RTX 4060 Laptop WDDM 596.21, port **8040**, **PID 34892 (new)** (distinct from Stage 3M-C 56704 and Stage 3M-B 19372)  
> **Locked Plan:** `LOCKED_INDEPENDENT_STATIONARITY_PLAN.md` (2026-08-29, hash `983387852875e7bb`, thresholds frozen before data)  
> **Previous Stage:** **CANDIDATE STABLE PROTOCOL** D claimed at 56704: c4 median CV 1.29%, p95 CV 1.55%, central ≤5% tail ≤10% PASS, no drift  
> **This Stage Classification:** **FAIL — NOT REPRODUCED (FAIL L0)**  
> **A_01:** Remains **FROZEN** — no causal attribution performed

---

## 1. What Was Validated (Protocol D Frozen)

Protocol D was **not modified** after observation (per §1). Every formal measurement executed the frozen 5-step sequence:

1. `1 × c4, n=40` matched warmup — synthetic 512/64, closed, pooled, OFF, seed `8000+chrono` (distinct but same token count, matched to formal 40)
2. `gc.collect()` — collected 1.7k–3.1k objects, before `[912,5,3]` → after `[0,0,0]` logged per run
3. `sleep(2s)` — measured 2.001–2.015 s per run
4. `host-state check` — snapshot after GC+sleep, before formal (threads, RSS, CPU, RAM, GPU 13–15%, GC counters, timestamp)
5. `formal run` — synthetic 512/64, 40 req (2 warmup sequential pooled excluded +38 measured), closed, stream True, temp 0, timeout 180, sampler 0.3, pooled `max_connections=max(8,concurrency)`, torch 16/16, executor 32, L0 OFF (Phase A)

All formal runs used identical config except `concurrency` (1 or 4) and `seed` (new). No per-run variation of threads/pool/client/warmup size/sleep/GC.

**Verification that protocol was followed:**
- Warmup bursts: 6/6 executed, c4 n40 pooled, OFF, median_ttft of warmups 0.04–0.06 s, throughput 4.2–4.5 rps, traces empty (OFF), manifests logged `warmup-*_processed.json`
- GC: `gc_snapshot.collected` 1793,2830,3188,3024,2748,2666 (mean 2.7k), `before` vs `after` counters confirm reclamation
- Sleep: 2.001–2.015 s (threshold 2.0 ±0.02)
- Host check: `snapshot_after_gc_sleep` captured per run, threads 183–185, RSS logged
- Formal: 6/6 completed, 0 invalid, 38/40 requests succeed per run (2 warmup excluded), total tokens per run stable (see §3)

---

## 2. Phase A Design Executed (Locked §5)

- **Condition:** L0 OFF only (`level 0`), fully forbids L1 (verified via `_set_level 0` before each burst).
- **Workload:** synthetic 512/64, 40 req (2 warmup +38 measured), closed, stream True, temp 0, timeout 180, same as A_01 and candidate.
- **Concurrencies:** `c=1` and `c=4` only (c=8 not measured; warmup uses c=4 as per protocol).
- **Repetitions:** 3 per concurrency = **6 formal runs** (plus 6 warmup bursts =12 bursts). Meets minimum 3 per conc; exactly 3 pre-registered, no extra runs added after seeing results.
- **Seeds (all new, avoid 4401 and Stage 3M-B 4101/4102/4401/4402):**
  - `c=1`: `5101 (A), 5102 (B), 5103 (C)`
  - `c=4`: `5401 (A), 5402 (B), 5403 (C)`
  - Warmup seeds: `8001..8006` (distinct, same 512/64 token size, logged)
- **Balanced Order (pre-registered, orthogonalizes concurrency vs time):**

  | Chrono | Conc | Seed | Round | Warmup Seed |
  |---|---|---|---|---|
  | 1 | 1 | 5101 | 1 | 8001 |
  | 2 | 4 | 5401 | 1 | 8002 |
  | 3 | 4 | 5402 | 2 | 8003 |
  | 4 | 1 | 5102 | 2 | 8004 |
  | 5 | 1 | 5103 | 3 | 8005 |
  | 6 | 4 | 5403 | 3 | 8006 |

  Global interleaving `c1,c4,c4,c1,c1,c4` matches `Round1 c1→c4, Round2 c4→c1, Round3 c1→c4`. Neither concurrency is clustered; seed order within each conc is chronological A→B→C.

- **Session:** `independent-3md-01`, PID **34892** persistent all 6 formals, verified via `/health pid` per run, pooled client, fixed executor 32, torch 16.

- **Workload token comparability (logged per run `describe_workload`):**

  | Run | Input mean | Range | Output |
  |---|---|---|---|
  | c1 5101 | 511.08 | 510–512 | 64 fixed |
  | c4 5401 | 510.80 | 510–512 | 64 |
  | c4 5402 | 511.08 | 510–512 | 64 |
  | c1 5102 | 511.18 | 510–512 | 64 |
  | c1 5103 | 511.20 | 510–512 | 64 |
  | c4 5403 | 511.03 | 510–512 | 64 |

  Mean input 510.8–511.2, variance <0.4 tokens (<0.08%), total token workload variance <1% — **comparable**, not confounding.

---

## 3. Phase A Results (Independent L0 Stationarity Gate)

Each formal's `processed/l0/*_processed.json` reports median_ttft, p95_ttft, median_lat, p95_lat, throughput, token_throughput plus range, rel_range, SD, CV, max pairwise, drift.

### 3.1 Per-Concurrency Metrics (Locked Thresholds: central ≤5% rel_range + CV ≤3%, tail ≤10% + CV ≤7%, no drift |ρ|<0.6 or slope ≤2% per step)

| Conc | Metric | Median | Rel Range | CV | Max Pairwise | Gate | Absolute Range |
|---|---|---|---|---|---|---|---|
| **c=1 (n=3)** | median_ttft | 0.0486 s | **5.57%** (>5%) | **3.10%** (>3%) | 4.94% | **FAIL** | 2.71 ms |
|  | p95_ttft | 0.0641 s | 6.71% (≤10%) | 3.69% (≤7%) | 5.76% | **PASS** | 4.30 ms |
|  | median_lat | 0.2615 s | 0.74% | 0.37% | 0.41% | PASS | 1.94 ms |
|  | p95_lat | 0.2831 s | 3.24% | 1.74% | 2.67% | PASS | 9.19 ms |
|  | throughput | 3.752 rps | 1.21% | 0.62% | 0.82% | PASS | 0.045 rps |
|  | token_thr | 240.1 tok/s | 1.21% | 0.62% | 0.82% | PASS | 2.91 tok/s |
| **c=4 (n=3)** | median_ttft | 0.0615 s | **20.29%** (>5%) | **10.52%** (>3%) | **17.73%** | **FAIL** | 12.48 ms |
|  | p95_ttft | 0.1307 s | **29.88%** (>15% limit, >10% strict) | **16.07%** (>7%) | **21.29%** (>20%) | **FAIL** | 39.07 ms |
|  | median_lat | 0.8894 s | 2.31% | 1.16% | 1.28% | PASS | 20.57 ms |
|  | p95_lat | 1.0024 s | 3.87% | 2.01% | 2.93% | PASS | 38.83 ms |
|  | throughput | 4.316 rps | 2.02% | 1.09% | 1.73% | PASS | 0.087 rps |
|  | token_thr | 276.2 tok/s | 2.02% | 1.09% | 1.73% | PASS | 5.58 tok/s |

**Failed checks (locked §5.4):**
- c1 median_ttft central rel_range 5.57% >5% (and CV 3.10% >3%, max pairwise 4.94% close to 5% limit) — **FAIL** central
- c4 median_ttft central rel_range 20.3% >5%, CV 10.52% >3%, max pairwise 17.73% >5%, **and** significant chronological drift rho 0.62 slope 0.003 s per order step (2% of median per step threshold =0.00123 s, slope 0.003 >0.00123 and |ρ|=0.62 → drift FAIL)
- c4 p95_ttft tail rel_range 29.9% >15% (strict 10%), CV 16.07% >7%, single outlier 21.29% >20% → **FAIL** tail
- No other metric fails; **throughput and latency are PASS at both concurrencies** (CV 0.6–2%)

**Comparison to Stage 3M-B (without Protocol D) and Candidate (with Protocol D):**

| Stage | c1 median_ttft | c4 median_ttft | c4 p95_ttft | Notes |
|---|---|---|---|---|
| **3M-B (2×c8 n20 warmup, no GC)** | 5.35% rel, CV 2.09% FAIL | **67.7%** rel, CV 22.8% FAIL | 36.4% rel, CV 13.2% FAIL | severe first-token instability |
| **Candidate 3M-C Protocol D at 56704** | ? | **1.29% CV** claimed | **1.55% CV** claimed | PASS, no drift — **candidate** |
| **3M-D Independent Protocol D at 34892** | **5.57%** rel, CV 3.10% FAIL | **20.29%** rel, CV 10.5% FAIL | **29.9%** rel, CV 16.1% FAIL | **Not reproduced** — improvement over 67% to 20% (3.3×) but still >5% and drift |

Protocol D **does reduce** c4 TTFT variance (67.7% →20.3%, factor 3.3), but **not enough** to meet ≤5% central / ≤10% tail. c1 variance unchanged (5.35% →5.57%), essentially same.

**Absolute effects (required alongside %):** c1 median 48.6 ms with 2.71 ms range — absolute 2.7 ms is small but relative 5.57% still exceeds 5% gate; gate correctly fails on relative (low baseline). c4 median 61.5 ms with 12.48 ms range and p95 130.7 ms with 39.1 ms range — both large absolute and large relative, unequivocally non-stationary. Even with absolute floor 10 ms, c4 would still fail.

**Chronological drift:** c1 rho -0.25 (weak, slope -0.00027 ms per step, -0.55% of median per step <2% → no drift). c4 median rho 0.62 with slope 0.003 s per step (4.97% of median per step >2%) → **significant drift** detected despite Protocol D. p95 rho 0.31 weak. This drift suggests residual initialization/allocator effect still correlates with time for c4 even after GC+sleep.

**Max pairwise relative deviation (required):**
- c1 median 4.94% (fail threshold 5% central, just under but rel_range already 5.57% fails)
- c4 median 17.73% (>5% fail)
- c4 p95 21.29% (>10% fail)

---

## 4. TTFT Pre-Eminence (Required Separation)

Prior non-stationarity was concentrated in **first-token path**. This replicates:

- **Throughput** is **stable** at both concurrencies (CV 0.62% c1, 1.09% c4, rel 1.21% and 2.02% PASS)
- **Total latency** is **stable** (median_lat CV 0.37% c1, 1.16% c4; p95_lat CV 1.74% and 2.01% PASS)
- **TTFT** is **unstable** (median CV 3.10% c1 FAIL, 10.52% c4 FAIL; p95 CV 3.69% c1 PASS but median fails, 16.07% c4 FAIL)

Therefore stable throughput **does mask** TTFT failure if not separately gated — protocol correctly isolates.

**Required TTFT-only report:**

- c1 median_ttft CV 3.10%, rel_range 5.57% → **FAIL central**
- c1 p95_ttft CV 3.69%, rel_range 6.71% → PASS tail (but central already fails)
- c4 median_ttft CV 10.52%, rel_range 20.29% → **FAIL central + drift**
- c4 p95_ttft CV 16.07%, rel_range 29.88% → **FAIL tail**

---

## 5. Host State Validation (Per-Run After Warmup+GC+Sleep)

Captured every run before/after formal plus `snapshot_after_gc_sleep` (Protocol D settle point):

| Run (chrono) | Conc | Seed | Threads before | Threads after | RSS before (MB) | RSS after (MB) | RSS after GC sleep (MB) | CPU mean % | GPU util % | GPU mem MB | Temp | Invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 5101 | 185 | 185 | 494.6 | 497.8 | 494.6 | 13.7 | 14 | 1769 | 43°C | False |
| 2 | 4 | 5401 | 185 | 183 | 673.6 | 673.4 | 673.6 | 19.5 | 14 | 1769 | 43°C | False |
| 3 | 4 | 5402 | 183 | 184 | 493.0 | 495.2 | 493.0 | 12.2 | 15 | 1769 | 43°C | False |
| 4 | 1 | 5102 | 184 | 184 | 489.3 | 682.7 | 489.3 | 8.0 | 14 | 1749 | 43°C | False |
| 5 | 1 | 5103 | 184 | 184 | 495.7 | 503.8 | 495.7 | 7.0 | 15 | 1744 | 43°C | False |
| 6 | 4 | 5403 | 183 | 183 | 679.3 | 673.1 | 679.3 | 17.8 | 13 | 1744 | 43°C | False |

**PID:** 34892 stable all 6 formals (no change, new PID confirmed distinct).

**Thread state convergence:** `server_process_threads` 183–185 stable ±2, Δ ≤2 between consecutive formals (threshold >20). No monotonic growth +50. **PASS** — Protocol D does make thread state converge (vs Stage 3M-B's earlier unfixed 66→232 growth, and even fixed 247→248 ±1). Threads are now stable.

**RSS:** Per-run `snapshot_before_formal` (after Protocol D settle):
- **c1:** 489.3, 494.6, 495.7 → median 494.6, range **6.3 MB**, rel_range **1.28%**, CV **0.69%** — **excellent convergence** (small variance).
- **c4:** 493.0, 673.6, 679.3 → median 673.6, range **186.3 MB**, rel_range **27.7%**, CV **17.2%** — **large variance**, same magnitude as before.
- **All 6 combined:** median 495.1, range 190 MB, rel_range 38.4%, CV 17.1% — dominated by c4 spread.

**Does Protocol D still produce small RSS variance?** **Partially:** for c1 yes (1.28% rel), for c4 **no** (27.7% rel). Stage 3M-C's observation of small RSS variance is **not reproduced** for c4. The c4 outlier (493 MB) is at seed 5402, which also had median_ttft 59.9 ms (lowest) and p95 142 ms (highest) — inconsistent pattern.

**GC and sleep:** All 6 runs executed `gc.collect()` with 1.7k–3.1k objects collected, `gc.get_count()` after = `[0,0,0]`, sleep measured 2.001–2.015 s, as locked.

**Other host:** CPU mean 7–19.5%, no sustained >90%, RAM 54% stable, GPU util 13–15% (WDDM noise, threshold 70 not exceeded), GPU mem 1744–1769 MB stable ±1.4%, temp 43°C, clocks stable, no thermal throttling, background Chrome not dominant.

**Invalid runs:** 0/6 (all valid for stationarity stats). No PID change, no thread spike, no RSS spike >500 within single formal run (though run 4's RSS after grew 489→683 =194 MB within run, but recovered next pre-run to 495, so not >500 spike without GC).

---

## 6. RSS Replication and Correlation (No Causality Claim)

- **Stage 3M-C claim:** Protocol D produces small RSS variance and stable pre-run RSS.
- **Independent replication:** **Does not replicate for c4.** c1 pre-run RSS is stable (1.28%), but c4 pre-run RSS shows 186 MB range (27.7%) — similar to Stage 3M-B's warmup RSS Δ185 MB failure, not the claimed stability.
- **RSS vs TTFT correlation:**
  - c1: Pearson 0.97 (high but n=3, small sample, wide CI) — **do not claim allocator causality**
  - c4: Pearson 0.62 (moderate positive)
  - All 6: Pearson 0.49
  - Interpretation: correlation exists but is **not proven causal**; allocator state correlates with first-token jitter but could be confounded by GC timing, thread scheduling, or other runtime init. We report only stability, not causality, per §12.

**Conclusion:** Protocol D does **not** continue to produce small RSS variance for c4 in new PID; the per-run allocator sawtooth (190–560 MB within-run in Stage 3M-B) is reduced for c1 but persists for c4.

---

## 7. Variance Decomposition (Coarse)

- **Seed vs repetition vs run-order:**

  | Conc | Seed 5101 mean p95 | 5102 mean | 5103 mean | Δ max | Order rho |
  |---|---|---|---|---|---|
  | c1 | 0.0641 | 0.0604 | 0.0647 | 4.3 ms (7.1%) | -0.10 |
  | c4 | 0.1029 | 0.1420 | 0.1307 | 39.1 ms (30%) | 0.31 (median rho 0.62) |

  - **Seed variance** for c4 is large (39 ms, 30% relative) — seed composition (prompt hash) contributes but not dominant over repetition variance (same seed not repeated, so repetition variance not separable; but with 3 distinct seeds, seed variance is the observed spread).
  - **Run-order variance:** c4 median rho 0.62 significant, p95 rho 0.31 weak — indicates **chronological drift** for median, not just seed.
  - **Repetition variance** (if same seed repeated) would be >20% based on c4 spread, larger than seed mean differences for c1 (c1 seed Δ 4.3 ms vs c4 Δ 39 ms).

- **Primary contributor:** For c4, **repetition/run-order** variance (~20 ms) exceeds seed mean differences (~30 ms but confounded), and both exceed instrumentation variance (Phase B not run). For c1, seed and order variances are small (4 ms).

---

## 8. Environment vs Instrumentation Estimation

- **Baseline Environment Variance (L0→L0, Phase A):** As above — c1 TTFT CV 3.1–3.7%, rel 5.6–6.7%; c4 TTFT CV 10.5–16.1%, rel 20–29.9%; throughput CV 0.6–1.1% (noise floor). This is the **environment noise floor** without tracing.
- **Instrumentation Effect:** Prior mock L1 median was 3.9% (within c1 noise floor 5.6% and far within c4 noise 20–29%). Therefore any L1 effect of <5% would be **within natural variance** for TTFT and cannot be distinguished. For throughput/latency where baseline CV is 0.6–2%, a 5% effect would be detectable. This separation was pre-registered and is confirmed.

Thus even if Phase B had run, interpretation would have been **cautious: L1 effect not larger than baseline variance for TTFT**.

---

## 9. Final Classification (Stage 3M-D)

### **FAIL — NOT REPRODUCED (FAIL L0)**

**Justification (must not tune until lucky PASS, per §1):**

- **Q1 (Phase A):** In tracing fully closed (L0 OFF) and after implementing **locked Protocol D** (1×c4 n40 warmup + gc.collect + sleep 2s + host check) on a **new PID 34892** and **new seeds 5101–5103/5401–5403** with balanced order, the environment is **still not sufficiently stationary**: same synthetic 512/64, same PID, same concurrency, same protocol shows **central >5% and tail >10% violations**, with c4 TTFT 20–29% variance and single outlier 21% >20% plus significant drift. This is **not client** (pooled <10 ms, throughput stable 1–2%), **not thread pool** (threads stable 183–185 ±2), **not host CPU/RAM/GPU** (all stable), but **runtime initialization / allocator / GC / first-token scheduling** as in Stage 3M-B.
- **Improvement vs Stage 3M-B:** c4 median_ttft rel_range **improved 67.7% →20.3%** (3.3×) and CV 22.8% →10.5% (2.2×), p95 36.4% →29.9% — Protocol D **helps** but does not reach 5%/10% thresholds. c1 unchanged (5.35% →5.57%).
- **Not PASS — INDEPENDENTLY STATIONARY:** Phase A fails, so cannot upgrade to VALIDATED STABLE PROTOCOL.
- **Not PARTIAL PASS?** This is **FAIL**, not partial, because **both** c1 central fails and c4 central+tail fails (partial would require only one conc or only tail). Here both conc fail central, and c4 tail fails.
- **Not DESIGN INVALID:** Design is sound (6 runs, balanced order, thresholds pre-registered, 0 invalid, PID stable, host_state valid, workload comparable, no silent deletion, sufficient data to judge). Protocol was executed correctly (GC/sleep verified). Therefore **FAIL — NOT REPRODUCED** is the correct classification, not invalid.
- **Q2 (Phase B):** Blocked — cannot assess L1 overhead on a non-stationary host per locked §6. No paired OFF/L1 data collected; overhead_gate.json not produced (empty dirs per spec §20).

**Comparison to Candidate at 56704:** Candidate claimed c4 median CV 1.29% and p95 CV 1.55% (PASS). Independent data shows CVs **8× larger** (10.5% and 16.1%) — **no replication**.

---

## 10. Risks Remaining and Next Step

**Remaining measurement risk even if future gate passes:**
- Pooled client median `send→headers` 6–7 ms (not <5 strictly) — residual client jitter ~5–7 ms on 48–61 ms TTFT is 10–12% floor.
- Allocator per-run sawtooth 6–186 MB (c1 vs c4) will always add GC jitter; GC tuning or larger warmup may be needed.
- Tiny-gpt2 `n_positions=1024` truncates >1024 prompts; 512 is safe but not stressing long-context.
- WDDM GPU util noise must be ignored for gate (threshold 70, observed 13–15).
- Absolute 2–12 ms on 48–61 ms TTFT is 5–20% — gate must keep both relative and absolute reporting.

**Minimal next validation experiment (single-variable, do not change 10 params at once, as in Stage 3M-B §7):**

> **Test larger warmup / GC isolation:** Run **c=4, seed 5401–5403, OFF, pooled, fixed 32, torch 16, but with **2 × c4 n40 warmup** (instead of 1×) + `gc.collect()` + `sleep(2)` + host check, and formal. Keep everything else identical (workload, seeds, client, PID reuse). Collect 3 repetitions same concurrency back-to-back. If TTFT variance drops from 20% → <10% (CV <5%), allocator warmup count is the cause. If variance remains >15%, then GC pause is not sole cause and next test should instrument `executor_wait` via L1 minimal trace on *one* OFF run to capture `model_to_token` vs `executor_wait` — but only after allocator count test.

This single-variable test directly addresses c4 RSS variance (27% rel) without touching scheduler/KV/policy.

**Do NOT re-study A_01 now.**

---

## 11. Can A_01 Causal Localization Be Re-Run?

**No — not yet.** Per LOCKED plan §19, only PASS — MEASUREMENT INFRASTRUCTURE VALIDATED (independent L0 PASS **and** L1 PASS) allows A_01 resumption.

**Checklist §21 (10 items):**

1. **New PID used?** **Yes** — PID **34892** (new, distinct from 56704 candidate and 19372 3M-B), port 8040, startup 2026-08-29T07:36Z, hash `39cc7cd90b`.
2. **Protocol D reproduced in new session?** **No** — c4 median CV 10.5% vs claimed 1.29% (8×), p95 16.1% vs 1.55% (10×), variance not reproduced.
3. **c1 median TTFT variance?** **FAIL** — median rel_range **5.57%** (>5%), CV **3.10%** (>3%), max pairwise 4.94% — central FAIL; p95 6.71% PASS tail.
4. **c4 median/p95 TTFT variance?** **FAIL** — median rel_range **20.29%** (>5%), CV 10.52%, **drift rho 0.62 significant**, p95 rel_range **29.88%** (>10%, >15% fail), CV 16.07%, outlier 21.3% >20% — both central and tail FAIL.
5. **Throughput / total latency stable?** **Yes** — throughput CV 0.62% c1, 1.09% c4 (PASS); median_lat CV 0.37% and 1.16% PASS; p95_lat CV 1.74% and 2.01% PASS. Total latency and throughput are stable, **TTFT instability isolated**.
6. **RSS again convergent?** **Partial/No** — c1 pre-run RSS stable 6.3 MB range (1.28% CV PASS), but **c4 pre-run RSS 186 MB range (27.7% CV FAIL)** — Stage 3M-C small RSS variance not reproduced for c4; correlation 0.49–0.97 but do not claim causality.
7. **Phase A PASS/FAIL?** **FAIL — NOT REPRODUCED** (NON-STATIONARY). Not PARTIAL, not DESIGN INVALID.
8. **Phase B (if executed):**
   - L1 overhead median %: **NOT EXECUTED** (skipped per gate)
   - Individual paired maximum %: **NOT EXECUTED**
   - Knee multiplier distortion c4/c1 p95: **NOT EXECUTED**
   - Reason: Phase A FAIL blocks Phase B per §14, §6.1 — overhead cannot be isolated from 20–29% baseline variance.
9. **Measurement infrastructure final state?** **FAIL L0** — Protocol D not independently validated; environment still non-stationary for first-token path. **Not validated.**
10. **Allow A_01 resumption?** **No — keep A_01 frozen.** Must first achieve Phase A STATIONARY PASS (central ≤5% CV ≤3% and tail ≤10% CV ≤7% no drift) on independent PID with Protocol D or successor. Until then, any A_01 causal claim (e.g., c4/c1 multiplier) would be confounded by 20% TTFT variance.

**Even if all PASS, this stage must not do A_01 causal attribution** — per §21, stop after reporting, do not analyze A_01.

---

## 12. Artifacts (Stage 3M-D Independent Validation)

```
research/measurement_repair/independent_validation/
  LOCKED_INDEPENDENT_STATIONARITY_PLAN.md          (locked 2026-08-29 hash 983387852875e7bb)
  environment.md                                   (PID 34892, port 8040, torch 16/16, executor 32, 07:36Z)
  raw/l0/            (6 formal OFF runs +6 warmup bursts: manifests, requests.csv, system.csv, raw.json, server_traces.json)
    l0-independent-3md-01_01_c1_seed5101_off_o1_* (×5 files)
    warmup-l0-independent-3md-01_01_c1_seed5101_off_o1_* (×5 files)
    ... (total 12×5=60 files +6 processed warmups)
  processed/l0/      (6 formal processed json +6 warmup processed json + host_state.csv + stationarity_summary.json + stationarity_gate.json)
  independent_stationarity_results.md               (NON-STATIONARY, per-metric tables, RSS correlation)
  raw/l1_regate/     (empty, Phase B skipped per spec §20)
  processed/l1_regate/ (empty, no overhead_gate.json)
  l1_regate_results.md                              (SKIPPED, locked design preserved)
  stage3md_summary.md                               (this file, FAIL — NOT REPRODUCED)
  logs/server_8040.out.log, .err.log, .pid (PID 34892, torch 16, executor 32, uptime)
  logs/independent-3md-01_stationarity_events.jsonl (6 events)
```

All raw requests, system samples, traces, manifests retained; no invalid run silently deleted (0 invalid). Protocol steps (warmup, GC, sleep, host check) logged per manifest.

*— End Stage 3M-D Summary (Independent Validation FAIL) —*
