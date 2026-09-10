# LOCKED STATIONARITY PLAN — Stage 3M-B

> **Stage:** 3M-B Environment Stabilization & Measurement Re-Gating  
> **Date Locked:** 2026-08-28  
> **Status:** LOCKED — thresholds frozen before any Phase A data collection. No post-hoc relaxation.  
> **Previous classification:** FAIL — ENVIRONMENT NON-STATIONARY (OFF-only seed diff 18%, c1 p95 TTFT individual 19% outlier)  
> **Gate order:** Q1 (L0 stationarity) before Q2 (L1 overhead) — non-invertible

---

## 1. Objective and Non-Goals

### Q1 (Phase A, Answer First)
> In L0 OFF (tracing fully closed), is experimental environment itself sufficiently stationary to reproduce same condition within gate tolerances?

### Q2 (Phase B, Only if Q1 PASS)
> On the *proven-stationary* environment, does L1 Minimal tracing pass the pre-registered overhead gate vs L0 OFF?

**Non-goals (frozen A_01):** No A_01 mechanism attribution, no scheduler/KV modification, no optimization design, no new RQ, no expanded sweep, no threshold shopping.

---

## 2. Fixed Environment Configuration (Locked)

All Phase A and Phase B formal runs use identical config. Any deviation → run INVALID.

| Item | Locked Value | Source / Record |
|---|---|---|
| Model | `sshleifer/tiny-gpt2` CPU FP32 | `hf_server.py` `--model sshleifer/tiny-gpt2 --device cpu` |
| Workload | `synthetic` 512 input / 64 output / no prefix reuse / closed burst | `workloads/generator.py generate(n=40, input 512, output 64, seed)` |
| Request count | 40 total = 2 warmup (sequential, excluded) + 38 measured | `harness.run_benchmark(warmup_requests=2)` |
| Arrival | `closed` (all arrival_offset 0, concurrency via `asyncio.Semaphore`) | generator + harness |
| Stream | `True` (SSE), temperature 0.0, timeout 180 s | harness `execute_request` |
| Sampler | `0.3 s` fixed NVML+psutil | `SystemSampler interval=0.3` |
| Serving PID strategy | Single persistent process per session, PID must not change across all Phase A and Phase B runs | `/health pid` check |
| Framework | `hf_transformers_naive` via `hf_server.py` |  |
| **Torch CPU threads** | `torch.set_num_threads(16)` and `torch.set_num_interop_threads(16)` (or max allowed) at server start, verified via `/health runtime.torch_num_threads==16` | Logged per run |
| **ThreadPoolExecutor** | Fixed `ThreadPoolExecutor(max_workers=32)` set as `loop.set_default_executor(...)` at server start | Startup log `executor_fixed=32`, host_state threads must show stable after warmup |
| **Client** | **Pooled** `httpx.AsyncClient(http2=False, limits=max_connections=max_keepalive=max(8, concurrency))` reused for entire run (warmup+measured). Per-request `httpx.AsyncClient()` is FORBIDDEN. Connection pool config logged per run. | `harness.run_benchmark(client_mode="pooled")` |
| Tracing | Phase A: **L0 OFF only** (`POST /stage3/trace/level {"level":0}`). Phase B: paired L0 vs L1 (`level 1`) only. |  |
| Hardware | `LAPTOP-1PB54QSI` i7-14650HX / 31.78 GB / RTX 4060 Laptop WDDM | `environment.md` |

No per-run variation of threads/pool/client/warmup is allowed.

---

## 3. Warmup Protocol (Locked)

### 3.1 Server-level warmup (once per server session, before Round 1)

1. Server started, `GET /health` confirms `loaded=true`, `device=cpu`, `trace_level=0`, `torch_num_threads=16`, `executor_fixed=32`.
2. Execute **2× dummy workloads c=8** not counted: `synthetic 512/64, n=20, concurrency 8, warmup_requests=0, pooled client`. Each dummy ~30–50 s.
3. After each dummy, sample host state: threads, RSS, CPU freq, GPU mem.
4. Proceed to formal runs only when after second dummy: `threads_after` delta ≤5 and `RSS` delta <50 MB vs prior dummy → threads considered stabilized (threshold per `host_state.md §2.1` where c=8 dummy drove 103→232→stable). If not stable, repeat one more dummy (max 3).

### 3.2 Per-run warmup (each formal run, uniform)

- `warmup_requests=2` sequential pooled requests before concurrent burst, excluded from metrics (existing harness mechanism). Identical for every run.
- 1.0 s gap between runs to let host settle (as in prior gate).

### 3.3 Warmup Validation Output

`warmup_validation.md` must show table: threads_before/after, RSS_before/after, median TTFT of dummy vs first formal OFF run, GPU state, and verdict **STABLE / UNSTABLE**. No formal run with `UNSTABLE` warmup may be counted.

---

## 4. Host State Gate (Per-Run, Mandatory)

Before and after each formal run, capture:

| Metric | Source | Invalid Threshold (Gate) |
|---|---|---|
| serving PID | `/health pid` | Any change → INVALID session |
| thread count (`server_process_threads` + `python_threads`) | `psutil.Process(pid).num_threads()` / `threading.active_count()` | Growth >20 between consecutive OFF runs without warmup, or monotonic +50 across session → INVALID |
| RSS (MB) | `process.memory_info().rss` | Spike >500 MB in single run without GC reclamation → INVALID |
| CPU utilization (%) | `psutil.cpu_percent` mean/max | Sustained >90% unrelated to concurrency → INVALID (background contention) |
| RAM (%) | `psutil.virtual_memory` | >90% → INVALID |
| GPU utilization (%) | `pynvml` | >30% → INVALID (CPU inference should be 0–10% WDDM noise) |
| GPU memory (MB, %) | `pynvml` | Jump >10% across OFF runs → INVALID |
| GPU temperature / clocks | `pynvml` if available | >85°C or clock drop >30% → INVALID |
| background top-5 CPU processes | `psutil.process_iter` | Log name/cpu, flag if Defender/Update/Chrome >40% |
| client process state | harness `client_runtime`, pool stats | Pooled client must show connection reuse (see §9) |

If `thread_count`, `RSS` or other key metric still in monotonic growth after warmup protocol, run is marked `invalid=True`, retained in `raw/` but excluded from stationarity statistics with reason logged.

---

## 5. Phase A — L0 Stationarity Gate

### 5.1 Scope

- **Fully forbids L1 tracing.** All runs `level=0` OFF.
- **Workload locked:** Same as §2 (synthetic 512/64, 40 req, closed).
- **Concurrencies:** `c=1` and `c=4` only. `c=8` is dummy warmup only, not measured.
- **Repetitions:** **6 formal runs per concurrency** = 3 rounds ×2 seeds, total 12 L0 OFF runs. Minimum 3 per condition is satisfied; 6 provides seed×order disentanglement.

### 5.2 Seeds and Run Order (Balanced Design)

| Concurrency | Seeds (new, to avoid reuse bias) | Round 1 | Round 2 | Round 3 |
|---|---|---|---|---|
| c=1 | 4101 (A), 4102 (B) | A→B (4101 then 4102) | B→A (4102 then 4101) | A→B (4101 then 4102) |
| c=4 | 4401 (A), 4402 (B) | A→B (4401 then 4402) | B→A (4402 then 4401) | A→B (4401 then 4402) |

Global interleaving (to orthogonalize concurrency vs time):
```
Warmup dummy c8#1, c8#2
Round1: c1-A, c1-B, c4-A, c4-B
Round2: c1-B, c1-A, c4-B, c4-A
Round3: c1-A, c1-B, c4-A, c4-B
```
Each run logs `seed`, `chronological_order` (1..12), `repetition_id` (round 1/2/3), `concurrency`.

### 5.3 Stationarity Metrics (At Least)

Per run `processed/*_processed.json` aggregates:

- `median_ttft`
- `p95_ttft` (primary)
- `median_latency` (total_latency p50)
- `p95_latency` (total_latency p95) (primary)
- `throughput_rps`
- `token_throughput`
- Absolute range (max-min), relative range ((max-min)/median), SD, CV (SD/mean), chronological drift (linear slope vs order, Spearman correlation).

Primary decision uses **relative range and CV**; absolute Δ also reported per §16.

### 5.4 Phase A Gate Thresholds (Locked, Pre-Registered)

Thresholds intentionally identical to prior overhead gate philosophy (10% principle) but applied to **L0 vs L0** (environment self-variation, not instrumentation):

| Metric Class | Metrics | PASS Threshold | PARTIAL / FAIL Semantics |
|---|---|---|---|
| **Central** | `median_ttft`, `median_latency`, `throughput_rps`, `token_throughput` | **Relative range ≤5%** AND **CV ≤3%** AND no single run deviates >5% from median | ≤5% is PASS central; 5–10% is PARTIAL if tail passes; >10% is FAIL |
| **Tail** | `p95_ttft`, `p95_latency` | **Relative range ≤10%** AND **CV ≤7%** AND no single run deviates >10% from median | ≤10% is PASS tail; 10–15% is PARTIAL; >15% or any single >20% is FAIL |
| **Chron drift** | All primary | Linear slope absolute ≤2% per run-order step AND Spearman |ρ| <0.6 (p>0.05) — no stable drift | Significant monotonic drift → NON-STATIONARY even if range passes |
| **Throughput** | `throughput_rps` | Same as central: ≤5% |  |

**Do not ignore a single 15–20% outlier even if median passes.** If any tail metric shows one repetition 15–20% deviation, classification cannot be STATIONARY.

### 5.5 Phase A Classification (Locked)

| Category | Criteria |
|---|---|
| **STATIONARY — PASS** | All central metrics PASS (≤5%) AND all tail metrics PASS (≤10%) AND no significant chronological drift AND no single >10% outlier (central) or >15% outlier (tail) |
| **PARTIALLY STATIONARY** | Central PASS but tail 10–15% OR one metric borderline, with drift absent. Must explicitly list which metrics are trustworthy (`throughput` and `median_latency` may be usable even if `p95_ttft` is not). |
| **NON-STATIONARY** | Any central >10% or any tail >15% or single outlier >20% or significant drift → same L0 condition cannot be stably reproduced. **Prohibits Phase B.** |
| **DESIGN INVALID** | Missing runs, invalid warmup, PID change, host_state invalid majority, or sample too small to judge. Must re-design, not claim PASS. |

Phase A verdict is recorded in `stationarity_results.md` with per-metric tables.

### 5.6 If Phase A FAIL

Do NOT run Phase B L1. Instead run minimal control experiments (one variable at a time) to locate drift source candidate list:

- thread pool initialization (compare fixed 32 vs default)
- client connection behavior (pooled vs per_request)
- CPU scheduling / frequency (log `cpu_freq` variance)
- GC (enable `gc` stats if needed)
- RSS growth / allocator (track `rss_after - rss_before`)
- GPU clocks/temperature (thermal throttling)
- allocator state
- background load (top-5 processes)
- run order (correlation with order)
- seed-specific workload differences (token count actual variance)

Each candidate is tested via **single-variable** control, not 10 parameters at once.

---

## 6. Phase B — L1 Overhead Re-Gate (Only if Phase A STATIONARY/PARTIAL with TTFT Usable)

### 6.1 Condition

Only enters if Phase A is **STATIONARY — PASS** (or PARTIALLY with `p95_ttft` declared usable). If PARTIALLY where `p95_ttft` is unusable, L1 gate for that metric is NOT AVAILABLE.

### 6.2 Workload and Levels

Same workload as Phase A (`synthetic 512/64, 40 req, pooled, warmup 2`). Two levels:

- **L0 OFF** (`level 0`) — baseline
- **L1 Minimal** (`level 1`, 6 int timestamps, `minimal_trace.py`) — the sole fine-grained candidate (L1a/b/c are fallback only if L1 FAIL, per `gate_plan.md §8`)

Per-run threadpool/torch/client/warmup identical to Phase A.

### 6.3 Paired Design (Matched, Order-Balanced)

3 seeds per concurrency (to match Phase A repetitions), each seed forms a **matched pair** (OFF then L1 or L1 then OFF), next repetition swaps order to balance.

Example for c=1 (seed 4101/4102/4103):

```
Pair A (seed 4101): OFF → L1
Pair B (seed 4102): L1 → OFF
Pair C (seed 4103): OFF → L1
Round2 Pair A: L1 → OFF (swap)
Round2 Pair B: OFF → L1 (swap)
... (or simply next round swaps each pair's order)
```

Locked for this plan: **2 seeds per concurrency ×2 orders = 4 pairs (8 runs) per concurrency**, total 16 runs (c1 8 + c4 8). Seeds: c1 4101/4102/4103/4104? To keep minimal, use **4 seeds per conc**: 4101/4102 for Phase A reuse, plus 4103/4104 new for Phase B to get 4 pairs. Simpler: reuse Phase A seeds 4101/4102/4103? Let's lock explicit:

| Conc | Seed | Pair Order (Round 1) | Pair Order (Round 2 swapped) |
|---|---|---|---|
| c=1 | 4101 | OFF→L1 | L1→OFF |
| c=1 | 4102 | L1→OFF | OFF→L1 |
| c=1 | 4103 | OFF→L1 | (single if 3 seeds) |
| c=4 | 4401 | OFF→L1 | L1→OFF |
| c=4 | 4402 | L1→OFF | OFF→L1 |
| c=4 | 4403 | OFF→L1 | |

For minimal 3 reps per condition: **3 seeds per conc ×2 levels =6 runs per conc =12 runs total** with alternating OFF→L1 / L1→OFF per successive seed, and round order swapped as above. This yields 3 matched pairs per concurrency, 6 pairs total, sufficient to evaluate median and individual gates (prior gate used 2 pairs per conc; 3 is stricter).

Global interleaving: `A_OFF, A_L1, B_L1, B_OFF, E_OFF, E_L1, F_L1, F_OFF, ...` making instrumentation condition orthogonal to order.

Each pair's two runs share same `seed` and adjacent `chronological_order`, differing only by `level`.

### 6.4 Phase B Metrics (Locked to Stage 3M)

- `median_ttft`
- `p95_ttft` (primary for gate)
- `median_latency`
- `p95_latency`
- `throughput_rps`
- `token_throughput`
- `c4/c1 knee multiplier preservation` (using `p95_ttft` primary, `p95_latency` secondary):
  ```
  off_multiplier = median_OFF(c4 p95_ttft) / median_OFF(c1 p95_ttft)
  on_multiplier  = median_L1(c4 p95_ttft)  / median_L1(c1 p95_ttft)
  relative_change = on/off -1
  ```

Also record **absolute Δ**: `L1 - OFF` in ms for TTFT/latency, and in rps for throughput.

### 6.5 Phase B Gate Criteria (Locked, Do Not Relax)

Same philosophy as `gate_plan.md §5` and prior `overhead_gate.json:evaluate_gate`:

- **Median gate:** For each of `throughput`, `p95_ttft`, `p95_latency`, median of paired relative Δ (L1/OFF-1) across pairs per concurrency must be **|median| ≤5%**.
- **Individual gate:** Every single paired relative Δ must satisfy **|Δ| ≤10%**. One >10% → FAIL.
- **Knee preservation:** `|relative_change| ≤10%` for `c4/c1 p95_ttft` (and `p95_latency`). >10% → FAIL (L1 distorts knee, as L2 did -47%).
- **Variance:** Paired Δ std should be <10% and no >3σ outlier mixing +5% and +150%. Std >10% → FAIL — ENVIRONMENT NON-STATIONARY conflated.
- **Absolute floor:** Report but do not use to excuse relative failure. If baseline TTFT is 80 ms, 15 ms is 18%; gate still fails on relative (but absolute is discussed in results).

Any single failure → FAIL.

### 6.6 Environment vs Instrumentation Separation (Mandatory)

After Phase B, separately estimate:

- **Baseline Environment Variance:** Distribution of L0→L0 deltas (from Phase A CV and Phase B's OFF vs OFF across seeds/orders). This is the noise floor.
- **Instrumentation Effect:** Distribution of matched L1→L0 paired deltas.

If instrumentation effect size is **not larger than baseline variance**, instrumentation cannot be claimed as cause; must report **careful interpretation: effect within noise**.

### 6.7 Optional Noise Decomposition (If Data Allows)

At minimum judge which source dominates by comparing variance:

- `seed variance` (same concurrency, different seed, same order position, both OFF)
- `repetition variance` (same seed, different round, OFF)
- `run-order variance` (correlation with chronological order)
- `instrumentation variance` (paired OFF→L1 delta)

No complex model required; report largest.

---

## 7. Repetitions Rule (Locked Before Seeing Results)

- Phase A: 6 OFF runs per concurrency (3 rounds ×2 seeds) =12 runs total. No extra runs added after seeing results; if HOST_INVALID removes a run, that pair is marked INVALID but not silently replaced — report as DESIGN INVALID if majority lost.
- Phase B: 3 matched pairs (6 runs) per concurrency =12 runs total (or 4 pairs =16 if we use 4 seeds). Pre-register here as **3 pairs per conc (6 runs per conc)**. Must decide before data.
- Final locked: **Phase A 12 OFF, Phase B 12 paired (6 L0 +6 L1 per conc? Wait 3 pairs per conc already includes both levels → 6 per conc).** Actually Phase B 12 runs total includes both OFF and L1. For total experiment: Warmup 2 dummies + Phase A 12 + Phase B 12 =26 runs.

---

## 8. Client Validation (Separate Control)

Before or alongside formal runs, validate pooled client:

- Connection reuse: `httpx` pooled client should show `established_connections` reuse; per-request client creates new TCP per call (prove via `client` logs: pooled shows headers at `client_headers_perf_ns - client_send_perf_ns` <5 ms vs per_request 200–400 ms).
- Do not create new client per request inside `execute_request` when pooled mode.
- DNS/TCP/TLS setup not in per-request timing: compare `client_send → client_headers` interval distribution pooled vs per_request.
- Client-side scheduling stable: `client_slot_acquired` vs `dispatch` variance small (<2 ms).
- If possible retain server-side timestamps (`trace.ts[0]` receive) vs client `send` to separate client/network noise vs server execution variance.

Output `client_validation.md` with latency breakdown table and reuse proof.

---

## 9. Warmup Validation (Separate)

Output `warmup_validation.md` showing:

- Table warmup_before vs after: thread count, RSS, median latency of dummy, GPU state.
- Formal first OFF run vs last warmup dummy latency comparison.
- Chart/table proving metric entered stable plateau (threads ±5, RSS ±50 MB).

Only if stable → formal measurement allowed. Document PID and torch threads.

---

## 10. Output Directory Contract (Stage 3M-B)

```
research/measurement_repair/
  LOCKED_STATIONARITY_PLAN.md          ← this file (locked)
  warmup_validation.md
  client_validation.md
  raw/
    stationarity/      (Phase A OFF 12 runs: manifests, requests.csv, system.csv, raw.json, server_traces.json)
    regate/            (Phase B paired 12 runs)
    warmup/            (dummy warmup runs)
  processed/
    stationarity/      (12 processed json)
    regate/            (12 processed json, overhead_gate.json)
    stationarity_summary.json
  stationarity_results.md
  l1_regate_results.md
  stage3mb_summary.md
```

All raw requests, system samples, traces, manifests retained; no silent deletion.

---

## 11. Final Classification (Stage 3M-B)

| Code | Meaning | Next Step |
|---|---|---|
| **PASS — ENVIRONMENT + L1** | L0 STATIONARY PASS and L1 overhead PASS | Measurement infra usable for A_01 re-run (do not auto-start A_01; report 7 items per §23) |
| **PASS — ENVIRONMENT ONLY** | L0 STATIONARY but L1 FAIL | Fine-grained tracing still not usable; consider L1a/b/c or framework metrics |
| **FAIL — ENVIRONMENT NON-STATIONARY** | L0 STATIONARY FAIL even after stabilization | Do NOT restore A_01; diagnose per §11 |
| **DESIGN INVALID** | Cannot reliably judge stationarity | Redesign experiment |

If PASS — ENVIRONMENT+L1, report (§23):

1. L0 run-to-run variance (central 5%, tail 10%)
2. Warmup stable? (threads/RSS/latency)
3. Pooled client reduced noise? (300 ms → <5 ms)
4. L1 matched overhead (median/individual)
5. Knee multiplier preservation (relative_change)
6. Remaining measurement risk
7. Whether A_01 causal localization can be re-run

If FAIL: no tuning until lucky PASS; classify root cause among `client / host / runtime init / threading / framework / hardware / unknown` and propose minimal next validation experiment.

---

## 12. Execution Commands (Frozen)

```bash
# Start server with fixed stabilization
python research/measurement/harness/hf_server.py \
  --model sshleifer/tiny-gpt2 --device cpu --port 8030

# Phase A warmup validation + stationarity (L0 only, pooled, fixed threadpool)
python research/measurement/harness/run_stage3mb.py \
  --phase warmup --base-url http://127.0.0.1:8030 --out research/measurement_repair

python research/measurement/harness/run_stage3mb.py \
  --phase stationarity --base-url http://127.0.0.1:8030 --out research/measurement_repair

# Phase B only if Phase A PASS
python research/measurement/harness/run_stage3mb.py \
  --phase regate --base-url http://127.0.0.1:8030 --out research/measurement_repair
```

---

> **Lock confirmation:** Thresholds above are frozen 2026-08-28 and will not be changed post-hoc based on observed proximity to limits. `LOCKED_STATIONARITY_PLAN.md` hash will be recorded in all Phase A manifests.

*— End of LOCKED PLAN —*
