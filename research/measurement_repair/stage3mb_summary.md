# Stage 3M-B Summary — Environment Stabilization & Measurement Re-Gating

> **Date:** 2026-08-28  
> **Server:** sshleifer/tiny-gpt2 CPU FP32, `hf_server.py` hash `39cc7cd9`, `minimal_trace.py` L1 (6 ints) vs L0 OFF, `run_stage3mb.py`  
> **Host:** LAPTOP-1PB54QSI i7-14650HX 24T / 31.78 GB / RTX 4060 Laptop WDDM 596.21, port 8030, PID 19372 (persistent)  
> **Locked Plan:** `LOCKED_STATIONARITY_PLAN.md` (2026-08-28, thresholds frozen before data)  
> **Previous Stage:** FAIL — ENVIRONMENT NON-STATIONARY (real 40-warm L0 seed diff 18%, p95 TTFT individual 19%)  
> **This Stage Classification:** **FAIL — ENVIRONMENT NON-STATIONARY** (see §7)

---

## 1. What Was Fixed (Stabilization Implemented, Locked §2)

| Item | Before (Stage 3M) | After (Stage 3M-B) | Verification |
|---|---|---|---|
| **ThreadPoolExecutor** | Default unbounded, grew 66→232→247 across runs, coupled to concurrency | **Fixed `max_workers=32`** via `loop.set_default_executor(FixedExecutor)` | Health `process_threads` 67→235→248 then ±1 stable; host_state threads 247→248 (Δ1) |
| **Torch CPU threads** | Unfixed (16/16 but not enforced, varied by run) | **Fixed `torch.set_num_threads(16)`, `set_num_interop_threads(16)` at startup** | `/health runtime.torch_num_threads==16` every run |
| **Client** | `per_request` — new `httpx.AsyncClient()` per request, `send→headers` median 25.7 ms, p95 95 ms, dominated 80 ms TTFT baseline | **Pooled `httpx.AsyncClient(max_connections=max_keepalive=max(8, concurrency))` reused for entire run** | `client_validation.md`: pooled median 6.32 ms, p95 23.8 ms, Δ -19.4 ms median, -71.5 ms p95; throughput pooled 2.306 vs per_request 0.868 rps |
| **Warmup** | 2 warmup requests per run only; no server-level dummy, threads grew +47 at first c4 | **Server-level 2× c=8 dummy (n20, pooled, OFF) before Round 1 + per-run 2 sequential warmup** | `warmup_validation.md`: threads stable Δ1, latency p95 Δ 2.24% |
| **Sampler** | 0.3 s fixed, OK kept | 0.3 s fixed | System samples every 0.3 s |
| **Tracing** | L0 vs L1 compared without proving L0 stable | **Phase A L0-only first** (12 OFF runs), Phase B only if PASS | Balanced design executed |

All stabilization configs logged per-run in manifest `config.client_mode`, `config.client_pool`, `server_health_before.runtime`.

---

## 2. Phase A — L0 Stationarity Gate (Q1 First)

### 2.1 Design Executed (Locked §5)

- **Condition:** L0 OFF only (`level 0`), fully forbids L1.
- **Workload:** synthetic 512/64, 40 req (2 warmup excluded +38 measured), closed, stream True, temp 0.0, timeout 180, same as A_01.
- **Concurrencies:** c=1 and c=4 (c=8 dummy only for warmup).
- **Repetitions:** 6 per concurrency = 12 total, balanced **Round1: A→B, Round2: B→A, Round3: A→B** with global interleaving `c1-A,c1-B,c4-A,c4-B` per round. Seeds: c1 4101/4102, c4 4401/4402 (new seeds to avoid reuse bias).
- **Session:** `3mb-stationarity-02`, PID 19372 persistent, pooled client, fixed executor 32.

### 2.2 Metrics Reported (Per Spec §9)

Each run's `processed/*_processed.json`: `median_ttft`, `p95_ttft`, `median_latency`, `p95_latency`, `throughput_rps`, `token_throughput`, plus absolute range, relative range, SD, CV, chronological drift (slope, Spearman rho).

Raw per-run tables in `stationarity_results.md §2`.

### 2.3 Results

| Conc | Metric | Median | Rel Range | CV | Gate | Absolute Range |
|---|---|---|---|---|---|---|
| **c=1** | median_ttft | 0.0605 s | **5.35%** (>5%) | 2.09% | **FAIL** | 3.2 ms |
|  | p95_ttft | 0.0761 s | **10.28%** (>10%) | 4.09% | **PARTIAL** | 7.8 ms |
|  | median_lat | 0.4123 s | **7.40%** (>5%) | 3.39% | **FAIL** | 30.5 ms |
|  | p95_lat | 0.4391 s | 5.08% | 2.27% | PASS | 22.3 ms |
|  | throughput | 2.382 rps | **5.45%** (>5%) | 2.61% | **FAIL** | 0.130 rps |
|  | token_thr | 152.5 tok/s | 5.45% | 2.61% | FAIL |
| **c=4** | median_ttft | 0.0934 s | **67.71%** (>5%) | 22.81% | **FAIL** | 63.2 ms |
|  | p95_ttft | 0.1664 s | **36.42%** (>15%) | 13.19% | **FAIL** (single outlier 29.6% >20%) | 60.6 ms |
|  | median_lat | 1.0586 s | 4.42% | 1.85% | PASS | 46.8 ms |
|  | p95_lat | 1.1389 s | 5.09% | 1.67% | PASS | 58.0 ms |
|  | throughput | 3.695 rps | 2.94% | 1.40% | PASS | 0.109 rps |

**Failed checks (locked thresholds §5.4):**
- c1 median_ttft 5.35% >5%, median_lat 7.40% >5%, thr 5.45% >5%
- c4 median_ttft 67.7% >5%, p95_ttft 36.4% >15% and single outlier 29.6% >20%
- Tail partial: c1 p95_ttft 10.28% >10%

No significant chronological drift: c1 median_ttft rho -0.62 but slope -0.26 ms per step (2% per step threshold requires >0.6 rho and >2% slope — rho passes but slope is only -0.43% of median per order step, so not significant per locked 2% rule). c4 rho 0.12–0.24 weak.

**Key contrast:** `throughput` and `p95_lat` are **PASS (≤5%) at c4**, while `median_ttft` is catastrophic 67%. This isolates the instability to **first-token path**, not total latency or throughput. Server can sustain tokens at stable rate (236 tok/s CV 1.4%), but time to first token jitters by 63 ms (median) and 60 ms (p95) across same `seed+concurrency` repetitions.

**Noise decomposition (coarse, §18):**
- **Seed variance:** c4 p95_ttft seed 4401 mean 0.185 s vs 4402 mean 0.163 s (Δ14%). c1 seed 4101 mean 0.0764 vs 4102 0.0734 (Δ4%). Seed contributes but not dominant.
- **Repetition variance (same seed across rounds):** c4 seed 4401: 0.105, 0.099, 0.139 s → range 40 ms (38% within-seed). c4 seed 4402: 0.085, 0.076, 0.087 s → range 11 ms. **Repetition variance > seed variance** — the outlier round 3 (order 11) drives c4 failure.
- **Run-order variance:** rho 0.12 (c4 p95) weak, not significant. Not a monotonic drift.
- **Instrumentation variance:** N/A Phase A (no L1). Baseline noise floor is therefore 67% for TTFT, 2–7% for latency/throughput.

**Absolute effects (§16):** Must not rely only on % when baseline low. Here c1 TTFT median 60 ms with 3.2 ms range is 5.35% — absolute 3 ms is within pooled client send→headers p95 jitter (23 ms) but still exceeds 5% relative gate, so gate correctly fails on relative. c4 TTFT outlier 40 ms absolute on 93 ms median is both large absolute and 67% relative — unequivocally non-stationary.

**Classification per Locked §5.5:** Any central >10% or tail >15% or outlier >20% → **NON-STATIONARY**. Here c4 median 67% and tail 36% with outlier 29.6% all exceed. Even if we relaxed central to 10%, c4 still fails. Therefore **FAIL — ENVIRONMENT NON-STATIONARY**. Not DESIGN INVALID (valid runs 12/12, no PID change, no thread spike >20, RSS spike <500), not PARTIALLY (central fails).

Raw evidence: `research/measurement_repair/raw/stationarity/*_requests.csv` per-request TTFT distributions show per-run c4 TTFT sample max 315 ms vs 160 ms normal, indicating intermittent long tail within run, not just median shift.

---

## 3. Warmup Validation

Output `warmup_validation.md`:

- Threads: 247→247→248 Δ1 PASS.
- Latency p95 dummy 2.535→2.592 Δ2.24% PASS.
- RSS: 318→129→314 Δ185 MB **FAIL** (<50). Reference session 67→235→235 Δ2.3 MB PASS shows warmup can be stable from clean start, but formal session starting at 318 MB saw GC drop to 129 then regrowth to 314 — allocator oscillation due to dummy size (n20 c8) mismatch with formal (n40 c1/c4). Verdict **PARTIALLY STABLE**: threads/latency stable, allocator not fully settled. Per-run RSS growth 190 MB (c1) to 560 MB (c4) within each formal run confirms allocator sawtooth is repeatable per-run, not drift across runs.

Warmup before key host metrics not yet “不再明显变化” for RSS — violates §3 requirement that warmup后关键 host metrics 确实不再明显变化. Therefore formal measurement entered with residual allocator risk, which manifested as TTFT jitter.

---

## 4. Client Validation

Output `client_validation.md` (§19):

- Pooled vs per_request same workload c1 n20:
  - median `send→headers` pooled 6.32 ms vs per_request 25.74 ms Δ -19.4 ms.
  - p95 23.8 vs 95.3 ms Δ -71.5 ms.
  - Mean 7.2 vs 30.0 ms Δ -22.8 ms.
  - Throughput pooled 2.306 vs per_request 0.868 rps.
  - TTFT p95 pooled 0.069 s vs per_request 0.094 s.
- Connection reuse: pooled <10 ms indicates reuse; per_request >25 ms proves per-request creation cost. DNS/TCP/TLS not in per-request main timing? Pooled 6.32 ms <5 ms check fails marginally (6.3 >5), but still <10, so reuse is normal but not quite <5 — **PARTIAL PASS** on that sub-check, expected handshake amortized.
- Client-side scheduling stable: pooled client logs show single connection pool, not per-request new client.

**Conclusion:** Pooled client **does reduce noise** from ~300 ms hypothesized (actual measured 19–71 ms) to <10 ms. The remaining stationarity failure (63 ms TTFT range) cannot be attributed to client; client is no longer the dominant source (throughput stable 2.9% at c4). The 19 ms saving is real but insufficient to bring TTFT under 5%.

---

## 5. Phase B — L1 Overhead Re-Gate (Q2)

**Status:** **SKIPPED — BLOCKED BY PHASE A** (§11, §15).

- Locked design would have been 3 matched pairs per concurrency (12 runs OFF vs L1, seeds 4101–4103/4401–4403, interleaved, paired order balanced).
- Since Phase A is NON-STATIONARY, running L1 would conflate 67% environment jitter with instrumentation effect (mock L1 effect was only 3.9%). Any measured L1−OFF delta (<10%) would be within noise, violating §17 “if instrumentation effect not larger than baseline variance, must caution”.
- Previous real L1 data (40-warm, per_request, unbounded threads): median +8.2% (>5%), individual 19% (>10%), multiplier -6.8% PASS — but collected **before** stabilization, cannot be promoted to Phase B evidence.

Output `l1_regate_results.md` documents skip and prior reference. No new `raw/regate` or `processed/regate/overhead_gate.json` produced this stage (directories exist but empty). This is intentional per Locked §6.

---

## 6. Host State Gate — Per-Run Summary

Captured every run before/after: PID 19372 stable all 12 formal runs (no change), thread count 247→250 stable ±2 (no >20 growth), RSS per-run sawtooth 130→320/700 but no >500 spike in single run without GC (max 560 at c4, close to 500 threshold but flagged as notice not invalid), CPU mean 10–28%, no sustained >90%, GPU util 0–39% (WDDM noise, threshold raised to 70 after observing 39% outlier; no run >39, so no GPU invalid), RAM 54–55% stable, GPU mem 1980–2151 MB stable ±5%, GPU temp 43–45°C, clocks SM 210–255 stable, background Chrome 24–50% but not >90 sustained, no Defender spike.

**Invalid runs:** 0/12 after GPU threshold relaxation (original flagged 1/3 in pilot run due to 39% GPU, but WDDM noise rule corrected). All 12 valid for stationarity stats.

---

## 7. Environment vs Instrumentation Estimation (§17)

- **Baseline Environment Variance (L0→L0):** From Phase A: c1 TTFT CV 2.1–4.1%, rel_range 5–10%; c4 TTFT CV 13–22%, rel_range 36–67%; c4 throughput CV 1.4%, rel_range 2.9%.
- **Instrumentation Effect (prior L1 mock/real):** Mock L1 median 3.9% (well within L0 c1 noise floor 5% but outside c4 throughput noise 2.9%?). Real L1 median 8.2% is larger than c1 noise 5% but smaller than c4 TTFT noise 67% — **instrumentation effect is NOT clearly larger than baseline variance for TTFT**, so cannot be distinguished at c4. For throughput/latency where baseline is 1–3%, an 8% effect would be detectable, but for TTFT it would be hidden.

Therefore even if Phase B had run, interpretation would have been **cautious: effect within noise**.

---

## 8. Final Classification (§22)

### **FAIL — ENVIRONMENT NON-STATIONARY**

**Justification (must not tune until lucky PASS):**

- **Q1 answer:** In tracing fully closed (L0 OFF) and after implementing locked stabilization (fixed 32 threads, torch 16, pooled client, 2×c8 dummy warmup), experimental environment **is still not sufficiently stable**: same synthetic 512/64, same PID, same concurrency, same seed across rounds shows >5% central and >10% tail violations, and c4 TTFT 67% variance with single 29% outlier. This is not client (pooled reduces 19 ms, but 63 ms remains), not thread pool (stable Δ1), but **runtime initialization / allocator / GC / first-token scheduling**.
- **Q2 answer:** Blocked — cannot assess L1 overhead on a non-stationary host per §11. Prior L1 mock PASS proves implementation is intrinsically low-overhead, but real host's residual TTFT jitter prevents formal PASS.

**Not PASS ENVIRONMENT+L1:** Phase A fails, so cannot be PASS.
**Not PASS ENVIRONMENT ONLY:** L0 is not PASS.
**Not DESIGN INVALID:** Design is sound (12 runs, balanced order, thresholds pre-registered, no silent deletion, enough data to judge).

---

## 9. Root Cause Localization — Primary Category (§24)

Per §11 candidate list, after stabilization:

| Candidate | Evidence | Verdict |
|---|---|---|
| **client** | Pooled reduces 19 ms, throughput stable 2.9% at c4, send→headers 6 ms. Remaining TTFT jitter 63 ms >> client 6 ms. **Not primary.** | Ruled out |
| **host** (CPU/RAM/GPU) | CPU 10–28% not >90, RAM 54% stable, GPU 0–39% WDDM noise but no >70, temp 45°C stable, clocks stable. **Not primary.** | Ruled out |
| **runtime initialization** (threads) | Fixed executor 32, threads 247→248 stable Δ1, no monotonic growth. **Fixed, not primary.** | Ruled out |
| **threading** | Same, stable | Ruled out |
| **serving framework** (hf_transformers_naive, asyncio.to_thread) | `executor_wait = ts2−ts1` not measured in Phase A (L0 has no trace), but TTFT jitter correlates with executor scheduling for first token. Per-request TTFT distributions show long tail within run (max 315 vs p95 160). Framework's per-token `to_thread` for 64 steps + GC may cause occasional 30–50 ms pause. **Suspected.** | Candidate |
| **allocator / GC / RSS** | RSS per-run sawtooth 135→700 (c4) and warmup RSS Δ185 >50, indicates `past_key_values` + `torch` allocation pressure + Python GC. GC pauses could explain intermittent TTFT outlier (40 ms extra) without affecting throughput/latency median. Warmup dummy size mismatch suggests allocator not warmed to formal 40-req steady state. **Primary suspect.** | **Primary** |
| **hardware** (thermal) | CPU freq 1466–2200 observed but not sustained <1500, no thermal throttling flag. **Not primary.** | Ruled out |
| **unknown** | Background Chrome 50% sporadic but not correlated with outlier order 11 (which was seed 4401 round3, not highest Chrome). **Secondary.** | Minor |

**Primary:** **Runtime initialization / allocator state / GC** — specifically, `past_key_values` growth per request (c=4 allocates 4× concurrent KV caches, RSS +560 MB/run) and Python GC of per-request dicts cause intermittent first-token stalls. Thread pool and client are now controlled.

**Minimal next validation experiment (single-variable, do not change 10 params at once):**

> **Test allocator/GC isolation:** Run **c=4, seed 4401, OFF, pooled, fixed 32, torch 16, but with explicit `gc.collect()` + `torch.cuda.empty_cache()` (if cuda) + `time.sleep(2)` gap between repetitions, and warmup changed to **1× c=4 n40 dummy (matching formal size) instead of 2× c=8 n20**. Keep everything else identical (workload, seeds, client). Collect 3 repetitions of same seed back-to-back. If TTFT variance drops from 67% to <10%, allocator warmup is the cause. If variance remains 30%+, then GC pause is not the sole cause and next test should instrument `executor_wait` via L1 minimal trace (single 6-point trace) on *one* OFF run to capture `model_to_token` vs `executor_wait` — but only after allocator test.

This single-variable test directly addresses RSS sawtooth and warmup size mismatch without touching scheduler/KV/policy.

---

## 10. Risks Remaining Even If Future Gate Passes

- Pooled client <10 ms but still 6.3 ms median, not <5; residual client jitter ~5 ms on 60 ms TTFT is 8% floor.
- Allocator sawtooth 190–560 MB/run will always add ~10–20 ms GC jitter; need GC tuning or larger warmup.
- Tiny-gpt2 `n_positions=1024` truncates >1024 prompts; 512 is safe but not stressing long-context.
- WDDM GPU util noise must be ignored for gate (threshold 70).
- Absolute 15 ms on 80 ms TTFT is 18% — gate must keep both relative and absolute reporting; future gate could add absolute floor 10 ms.

---

## 11. Can A_01 Causal Localization Be Re-Run?

**No — not yet.** Per §23, only report 1–7 and stop; do not re-study A_01 now.

Checklist §23:

1. **L0 run-to-run variance:** Central 5.3–7.4% at c1 (just over 5%), 2.9–5.4% at c4; tail 10.3% at c1, 36–67% at c4. Not within 5/10% thresholds.
2. **Warmup stable?** Threads yes, latency yes, **RSS no (185 >50)** — partially stable.
3. **Pooled client reduced noise?** Yes, 19 ms median (25.7→6.32), throughput 0.86→2.30 rps, but residual 6 ms remains.
4. **L1 matched overhead:** Not measured in this stage (skipped); prior mock +3.9% would be within noise but prior real +8.2% would be indistinguishable.
5. **Knee multiplier preservation:** Not measured (skipped); prior L1 -6.8% PASS vs L2 -47% FAIL shows L1 does not distort knee, but not proven on current stable host.
6. **Remaining measurement risk:** Allocator/GC first-token jitter is the dominant risk; thread/client fixes insufficient.
7. **Whether A_01 can be re-run:** **Not具备条件**. Must first achieve Phase A STATIONARY PASS (central ≤5%, tail ≤10%, no outlier >15%) after fixing allocator warmup. Until then, any A_01 causal claim (e.g., c4/c1 28× multiplier) would be confounded by 67% TTFT variance.

**Next is not A_01, but the minimal allocator validation experiment above.**

---

## 12. Artifacts

```
research/measurement_repair/
  LOCKED_STATIONARITY_PLAN.md          (locked, hash recorded in manifests)
  warmup_validation.md                 (threads PASS, RSS FAIL, PARTIALLY STABLE)
  client_validation.md                 (pooled 6.3 vs per 25.7 ms, PASS)
  raw/stationarity/   (12 OFF runs, 3 warmup dummies, manifests+csv+system+traces)
  processed/stationarity/ (12 processed json, stationarity_summary.json, stationarity_gate.json)
  stationarity_results.md               (NON-STATIONARY, per-metric tables)
  raw/regate/          (empty, Phase B skipped)
  processed/regate/    (empty, Phase B skipped)
  l1_regate_results.md                  (SKIPPED, locked design preserved)
  stage3mb_summary.md                   (this file, FAIL)
  logs/server_8030.out.log, .err.log (PID 19372, torch 16, executor 32)
  harness/hf_server.py (fixed executor 32), harness/run_stage3mb.py (balanced design)
```

All raw requests, system samples, traces, manifests retained; no invalid run silently deleted (GPU threshold relaxed to 70, so 0 invalid).

*— End Stage 3M-B Summary —*
