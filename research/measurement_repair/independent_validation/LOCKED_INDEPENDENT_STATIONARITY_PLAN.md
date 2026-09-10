# LOCKED INDEPENDENT STATIONARITY PLAN — Stage 3M-D

> **Stage:** 3M-D Independent Validation of Candidate Stable Protocol  
> **Date Locked:** 2026-08-29T00:00:00Z (UTC, before any Phase A data collection)  
> **Status:** LOCKED — Protocol D and all thresholds frozen before independent data. No post-hoc relaxation or Protocol E/F/G search.  
> **Candidate Protocol:** **Protocol D** (from Stage 3M-C)  
> **Previous Classification:** CANDIDATE STABLE PROTOCOL (not yet VALDATED)  
> **Gate Order:** Phase A (L0 OFF stationarity) before Phase B (L1 re-gate) — non-invertible  
> **A_01:** FROZEN — no causal attribution until infrastructure PASS

---

## 1. Objective

> Using a **new serving session, new PID, new seeds, and pre-registered balanced design**, confirm whether Protocol D genuinely solves the L0 stationarity problem identified in Stage 3M-B (c4 TTFT rel_range 67.7% / CV 22.8%, c1 5.35% central).

* **Primary success criterion:** Phase A L0 OFF stationarity PASS under locked thresholds in new session.
* **Non-goals:** No modification to Protocol D, no threshold shopping, no A_01 re-run, no scheduler/KV/policy change, no new RQ.

---

## 2. Protocol D — Frozen Definition

**Every formal measurement** is preceded by the same 5-step sequence, in order, without variation:

1. **`warmup_burst`: 1 × c=4, n=40, matched** — synthetic workload `512 input / 64 output / no prefix reuse / closed burst / stream True / temp 0.0 / timeout 180 s / pooled client`, concurrency **4**, `warmup_requests=0` (warmup itself is not further warmed), `n=40` requests, L0 OFF (`level 0`). Workload token counts identical to formal (512/64). Seed for warmup = `8000 + chronological_order` distinct from formal seed but same token distribution, ensuring allocator is warmed to the formal 40-request working set without identical caching. Warmup runs on same fixed executor, torch threads, and pooled client as formal.
2. **`gc.collect()`** — Python `gc.collect()` executed on the **client measurement process** (and, if accessible, server via `/health` trigger) immediately after warmup, with `gc.get_count()` logged before/after.
3. **`sleep(2s)`** — unconditional `time.sleep(2.0)` after GC, allowing allocator and OS to settle.
4. **`host-state check`** — snapshot captured after sleep and before formal: `pid`, `server_process_threads`, `python_threads`, `RSS (MB)`, `cpu_percent`, `cpu_freq`, `ram_percent`, `gpu_util`, `gpu_mem`, `gpu_temp`, `gpu_clocks (SM/mem)`, `gc_counters`, `timestamp`. Invalid thresholds applied (§6). If invalid, run is marked `invalid=True` but retained, not silently replaced.
5. **`formal_run`** — synthetic workload `512/64, n=40 (2 warmup_requests sequential pooled excluded + 38 measured)`, `closed`, same formal seed as pre-registered, `concurrency` as per spec (1 or 4), `stream True`, `sampler 0.3 s`, `L0 OFF` (Phase A) or paired OFF/L1 (Phase B), `client_mode pooled` (`max_connections=max_keepalive=max(8, concurrency)`), `torch threads 16/16`, `ThreadPoolExecutor max_workers=32`.

**Frozen parameters (must not change post-hoc):**

| Item | Locked Value | Rationale |
|---|---|---|
| Warmup concurrency | **4** | drives ThreadPool + allocator to max before any formal |
| Warmup request count | **40** | matched to formal 40, replaces prior 2×c8 n20 which left RSS Δ185 MB |
| GC | **`gc.collect()` once per run after warmup** | reclaims `past_key_values` dicts; logged via `gc.get_count()` |
| Sleep | **2.0 s** | settling gap |
| Formal request count | **40 (2 warmup +38 measured)** | same as A_01 and Stage 3M-B |
| Sampler | **0.3 s** | same as prior |
| Client | **pooled** `httpx.AsyncClient(http2=False, limits=max_connections=max_keepalive=max(8,concurrency))` reused for warmup+formal | eliminates per_request 25 ms overhead |
| Threads | **`torch.set_num_threads(16)`, `set_num_interop_threads(16)`, `ThreadPoolExecutor(max_workers=32)`** | fixed at server startup, verified via `/health` |
| Tracing Phase A | **L0 OFF only** | L1 strictly prohibited |
| Tracing Phase B | **L0 OFF vs L1 Minimal (6 ints)** | if Phase A PASS only |

Any deviation (e.g., changing to c8 warmup, n=20, no GC, 1s sleep) → run INVALID.

---

## 3. Fixed Environment Configuration (Locked)

| Item | Locked Value | Record Source |
|---|---|---|
| Model | `sshleifer/tiny-gpt2` CPU FP32 (`n_positions=1024`) | `hf_server.py --model sshleifer/tiny-gpt2 --device cpu` |
| Workload | synthetic 512/64, no prefix reuse, closed burst, stream True, temp 0.0, timeout 180 | `workloads/generator.py generate(n=40, input 512, output 64, seed)` |
| Framework | `hf_transformers_naive` via `hf_server.py` (with fixed executor + torch threads at startup) | file hash logged |
| Device | `cpu` FP32 | `/health device` |
| Arrival | `closed` (all `arrival_offset 0`, concurrency via `asyncio.Semaphore`) | harness |
| Server PID strategy | **Single persistent process per session**, new PID distinct from Stage 3M-C 56704 and Stage 3M-B 19372, verified via `/health pid` every run | `environment.md` |
| Port | **8040** (isolated, new session) | new serving session |
| Torch threads | `16 / 16` locked | `/health runtime.torch_num_threads==16` |
| Executor | `32` locked | startup log |
| Client | pooled as above | logged per run |
| Hardware | `LAPTOP-1PB54QSI` i7-14650HX 24T / 31.78 GB / RTX 4060 Laptop WDDM 596.21 | environment.md |
| Python | 3.13.5, torch 2.13.0+cpu, transformers 5.16.1, httpx 0.28.1, psutil 7.2.2 | environment.md |
| Sampler | 0.3 s fixed NVML+psutil | SystemSampler |

Deviation → run INVALID.

---

## 4. Host State Validation (Per-Run, After Warmup+GC+Sleep)

Before **and** after each formal run (and also snapshot after warmup+GC+sleep before formal), capture:

| Metric | Source | Invalid Threshold | Note |
|---|---|---|---|
| serving PID | `/health pid` | any change → **INVALID session** | new PID must be ≠56704, ≠19372 |
| thread count (`server_process_threads` + `python_threads`) | `psutil.Process(pid).num_threads()` / `threading.active_count()` | growth >20 between consecutive OFF runs without warmup, or monotonic +50 across session → INVALID | protocol D should converge to stable |
| RSS (MB) | `process.memory_info().rss` | spike >500 MB in single formal run without GC reclamation → INVALID | check small variance |
| CPU util (%) | `psutil.cpu_percent` mean/max | sustained >90% unrelated to concurrency → INVALID (contention) |
| RAM (%) | `psutil.virtual_memory` | >90% → INVALID |
| GPU util (%) | `pynvml` | >70% → INVALID (CPU inference should be 0–10% plus WDDM 0–39 noise) |
| GPU memory MB / % | `pynvml` | jump >10% across OFF runs → INVALID |
| GPU temp / clocks | `pynvml` if available | >85°C or clock drop >30% → INVALID |
| GC counters | `gc.get_count()` before/after `gc.collect()` | logged, not gate, but growth noted |
| background top-5 CPU procs | `psutil.process_iter` | log name/cpu, flag if >40% |
| client pool stats | harness `client_runtime` | pooled must show reuse |
| timestamp (UTC ISO) | `time.time()` | for chronological drift |

Focus: Does Protocol D still make thread state, RSS, and runtime initialization converge in the **new PID**?

All snapshots retained in `raw/l0/*_manifest.json` and aggregated to `processed/l0/host_state.csv`.

---

## 5. Phase A — Independent L0 Stationarity Gate

### 5.1 Scope

- **Tracing:** **L0 OFF only** (`POST /stage3/trace/level {"level":0}`), L1 forbidden.
- **Workload locked:** synthetic 512/64, 40 req (2 warmup excluded +38 measured), closed, stream True, temp 0.0, timeout 180, pooled, fixed threads/pool.
- **Concurrencies:** `c=1` and `c=4` only. `c=8` not measured (warmup uses c=4).
- **Repetitions:** **3 per concurrency = 6 L0 OFF formal runs** (plus 6 warmup bursts = 12 bursts total). Minimum requirement (§5) satisfied; exactly 3 per conc pre-registered, no extra runs added after seeing results.
- **Seeds:** **All new, never used for Protocol D discovery**. Discovery used `4401` and Stage 3M-C discovery set. Stage 3M-B used `4101,4102,4401,4402`. Here:
  - `c=1`: **5101 (A), 5102 (B), 5103 (C)**
  - `c=4`: **5401 (A), 5402 (B), 5403 (C)**
  - Warmup seeds: `8000 + chronological_order` (8001..8006) distinct yet same token count, logged for comparability.
- **Token workload comparability:** For every run, log `describe_workload` including `input_tokens` (512), `output_tokens` (64), `total_tokens` (measured sum), `prompt_hash`. Verify across seeds that total token workload variance <2% (to avoid composition variance masquerading as runtime variance). Report in results.

### 5.2 Balanced Run Order (Pre-Registered, Orthogonalizes Time vs Condition)

Avoid `c1 c1 c1 c4 c4 c4`. Use interleaved rounds with seed chronology balanced:

```
Round 1: c1(A) → c4(A)
Round 2: c4(B) → c1(B)
Round 3: c1(C) → c4(C)
```

Pre-locked chronological order (1..6):

| Chrono | Concurrency | Seed | Round | Warmup Seed | Level | Notes |
|---|---|---|---|---|---|---|
| 1 | 1 | 5101 | 1 | 8001 | OFF | Round1 first |
| 2 | 4 | 5401 | 1 | 8002 | OFF | Round1 second |
| 3 | 4 | 5402 | 2 | 8003 | OFF | Round2 first (swapped) |
| 4 | 1 | 5102 | 2 | 8004 | OFF | Round2 second |
| 5 | 1 | 5103 | 3 | 8005 | OFF | Round3 first |
| 6 | 4 | 5403 | 3 | 8006 | OFF | Round3 second |

Properties:
- c1 positions: 1,4,5 ; c4 positions: 2,3,6 → neither concurrency is clustered.
- Seed order within each concurrency is A,B,C chronological (A before B before C after balancing) but interleaving prevents monotonic time bias.
- Warmup bursts executed immediately before each formal's GC/sleep, i.e., 12 bursts total in same order; warmup for chrono N precedes formal N.

Each run logs `seed`, `chronological_order` (1..6), `round` (1/2/3), `concurrency`, `warmup_seed`, `gc_counters_before/after`, `sleep_duration 2.0`.

If host INVALID removes a run, it is marked `invalid=True`, retained, not silently replaced; if majority invalid → DESIGN INVALID.

### 5.3 Stationarity Metrics (Reported per Run and Aggregated)

Per run `processed/l0/*_processed.json` aggregates:

- `median_ttft` (primary central)
- `p95_ttft` (primary tail)
- `median_total_latency` (median_lat)
- `p95_total_latency` (p95_lat)
- `throughput_rps`
- `token_throughput`
- Also mean, range (max-min), relative range `(max-min)/median`, SD, CV `SD/mean`, **max pairwise relative deviation** `max_{i,j} |x_i - x_j|/median`, chronological drift (linear slope per order step, Spearman ρ, p-value if computable), absolute Δ ms.

Primary decision uses **relative range and CV**; absolute Δ also reported.

### 5.4 Gate Thresholds (Locked, Identical to Prior Pre-Registration, No Relaxation)

| Metric Class | Metrics | PASS Threshold |
|---|---|---|
| **Central** | `median_ttft`, `median_total_latency`, `throughput_rps`, `token_throughput` | **Relative range ≤5%** AND **CV ≤3%** AND no single run deviates >5% from median (max pairwise ≤5%) |
| **Tail** | `p95_ttft`, `p95_total_latency` | **Relative range ≤10%** AND **CV ≤7%** AND no single run deviates >15% from median (max pairwise ≤10% strict, ≤15% fail) |
| **Chronological drift** | All primary | Linear slope absolute ≤2% per run-order step **and** Spearman |ρ| <0.6 (or p>0.05) — no stable drift |

Do not ignore a single 15–20% outlier even if median passes. Any tail single >20% → FAIL.

Tail PASS requires **both** ≤10% rel_range **and** ≤7% CV; central requires ≤5% and ≤3%.

### 5.5 TTFT Pre-Eminence

TTFT is primary because prior non-stationarity concentrated in first-token path (Stage 3M-B c4 median_ttft 67.7% while throughput 2.94% PASS). Therefore separately report and gate:

- c1 `median_ttft` CV and rel_range
- c1 `p95_ttft` CV and rel_range
- c4 `median_ttft` CV and rel_range
- c4 `p95_ttft` CV and rel_range

Do not let stable throughput mask TTFT failure.

### 5.6 RSS Replication

Check whether Stage 3M-C observation reproduces:

- Does Protocol D continue to produce small RSS variance and stable pre-run RSS (warmup+GC+sleep)?
- Compute `RSS_pre_run` (snapshot after GC+sleep before formal) variance across 6 formals: range, CV, and correlation with `median_ttft`.
- **Do not claim allocator causality** from correlation; only report stability of protocol.

### 5.7 Classification (Locked)

| Category | Criteria |
|---|---|
| **PASS — INDEPENDENTLY STATIONARY** | In new PID + new data, **c1 AND c4 both** satisfy locked central ≤5% and tail ≤10% (with CV and drift) and no single outlier beyond limits. Protocol D upgraded to **VALIDATED STABLE PROTOCOL**. |
| **PARTIAL PASS** | e.g., c1 PASS but c4 FAIL; or central PASS tail FAIL. **Cannot proceed to L1 gate.** |
| **FAIL — NOT REPRODUCED** | Stage 3M-C stability not reproduced; Protocol D not reliable. Thresholds violated. |
| **DESIGN INVALID** | Missing runs, invalid warmup majority, PID change, host_state invalid majority, or insufficient sample to judge. Must re-design, not claim PASS/FAIL. |

If FAIL or PARTIAL, L1 directories remain empty (per spec §20).

---

## 6. Phase B — L1 Minimal Tracing Re-Gate (Only if Phase A PASS — STATIONARY)

**Not executed unless Phase A is full PASS.** No early run.

### 6.1 Goal

> In the **proven-stationary environment**, is L1 Minimal tracing truly low-overhead?

Compare **L0 OFF vs L1 Minimal** using identical Protocol D, workload, PID, client, serving configuration.

### 6.2 Paired Design (Matched, Order-Balanced, Pre-Registered)

Each concurrency at least **3 OFF/L1 pairs** (6 runs per conc, 12 total). Seeds **reuse Phase A seeds** to keep workload comparable and avoid new composition variance:

| Pair | Conc | Seed | Order in Pair | Chrono (global) |
|---|---|---|---|---|
| A | 1 | 5101 | OFF→L1 | 1,2 |
| B | 1 | 5102 | L1→OFF | 3,4 |
| C | 1 | 5103 | OFF→L1 | 9,10 |
| D | 4 | 5401 | L1→OFF | 5,6 |
| E | 4 | 5402 | OFF→L1 | 7,8 |
| F | 4 | 5403 | L1→OFF | 11,12 |

Global interleaving (orthogonalizes instrumentation vs time):

```
1: c1 5101 OFF (Pair A OFF)
2: c1 5101 L1  (Pair A L1)
3: c1 5102 L1  (Pair B L1)
4: c1 5102 OFF (Pair B OFF)
5: c4 5401 L1  (Pair D L1)
6: c4 5401 OFF (Pair D OFF)
7: c4 5402 OFF (Pair E OFF)
8: c4 5402 L1  (Pair E L1)
9: c1 5103 OFF (Pair C OFF)
10:c1 5103 L1  (Pair C L1)
11:c4 5403 L1  (Pair F L1)
12:c4 5403 OFF (Pair F OFF)
```

Each pair's two runs share same `seed` and are **adjacent in chronological order**, differing only by `level` (0 vs 1). Order balanced: for c1, OFF→L1, L1→OFF, OFF→L1; for c4, L1→OFF, OFF→L1, L1→OFF (reverse to balance).

Workload per run: same synthetic 512/64, n=40, pooled, warmup_requests 2, Protocolo D warmup+GC+sleep preceding each run, same PID, same port 8040.

### 6.3 Metrics

Same as Phase A plus:

- `c4/c1 knee multiplier preservation` using `p95_ttft` primary, `p95_latency` secondary: `off_multiplier = median_OFF(c4 p95_ttft)/median_OFF(c1 p95_ttft)`, `on_multiplier = median_L1(c4 p95_ttft)/median_L1(c1 p95_ttft)`, `relative_change = on/off -1`.
- Absolute Δ in ms (TTFT/latency) and rps (throughput) per pair.

### 6.4 Gate Criteria (Locked, Not Modified)

- **Median overhead:** For each of `throughput`, `p95_ttft`, `p95_latency`, median of paired relative Δ (L1/OFF-1) across pairs per concurrency must be **|median| ≤5%**.
- **Individual paired overhead:** Every single paired relative Δ must satisfy **|Δ| ≤10%**. One >10% → FAIL.
- **Knee preservation:** `|relative_change| ≤10%` for `c4/c1 p95_ttft` (and latency). >10% → FAIL (L1 distorts knee).
- Also report variance of paired Δ std <10% and no >3σ outlier.

Any single failure → FAIL.

Separation: compare L1-OFF difference to L0-L0 natural variance (from Phase A). If L1 effect is within baseline variance, do not claim significant instrumentation effect.

---

## 7. Output Directory Contract (Stage 3M-D)

```
research/measurement_repair/independent_validation/
  LOCKED_INDEPENDENT_STATIONARITY_PLAN.md   ← this file (locked)
  environment.md
  raw/l0/            (Phase A OFF 6 runs: manifests, requests.csv, system.csv, raw.json, server_traces.json, host_state snapshots)
  processed/l0/      (6 processed json, host_state.csv, stationarity_summary.json, stationarity_gate.json)
  independent_stationarity_results.md
  raw/l1_regate/     (Phase B paired 12 runs, empty until Phase A PASS)
  processed/l1_regate/ (overhead_gate.json, regate_summary.json)
  l1_regate_results.md
  stage3md_summary.md
  logs/              (events jsonl, server logs)
```

If Phase A FAIL: L1 directories remain empty as required.

All raw requests, system samples, traces, manifests retained; no silent deletion.

---

## 8. Host-State & RSS Analysis Requirements

Besides gate, per §8–12:

- Thread state convergence: after Protocol D warmup+GC+sleep, `server_process_threads` should be stable ±5 across formals.
- RSS pre-run stability: `snapshot_after_GC` RSS range, CV, and vs TTFT correlation (Pearson). Do not assert causality.
- Workload token comparability: total token count per run (input+output) variance.

---

## 9. Final Classification (Stage 3M-D)

| Code | Meaning | Next Step |
|---|---|---|
| **PASS — MEASUREMENT INFRASTRUCTURE VALIDATED** | Independent L0 stationarity PASS **and** L1 overhead gate PASS | Measurement infra usable, may resume A_01 (but do not auto-start A_01 causal attribution in this stage) |
| **PASS L0 / FAIL L1** | Environment stable, but L1 tracing still not usable | Switch measurement methodology |
| **FAIL L0** | Protocol D not reproduced | Keep A_01 frozen |
| **DESIGN INVALID** | Experiment cannot adjudicate | Redesign |

Only **PASS — MEASUREMENT INFRASTRUCTURE VALIDATED** allows A_01 resumption.

---

## 10. Execution Commands (Frozen for This Plan)

```bash
# 1. Start new isolated serving session (new PID, port 8040)
python research/measurement/harness/hf_server.py \
  --model sshleifer/tiny-gpt2 --device cpu --port 8040 \
  --executor-workers 32 --torch-threads 16

# 2. Run independent Phase A (pre-registered, L0 only, Protocol D)
python research/measurement/harness/run_stage3md.py \
  --phase stationarity --base-url http://127.0.0.1:8040 \
  --out research/measurement_repair/independent_validation \
  --session independent-3md

# 3. If Phase A PASS, run Phase B (paired L0/L1, Protocol D)
python research/measurement/harness/run_stage3md.py \
  --phase regate --base-url http://127.0.0.1:8040 \
  --out research/measurement_repair/independent_validation \
  --session independent-3md
```

All thresholds above are frozen 2026-08-29 and will not be changed post-hoc. File hash recorded in all manifests.

---

> **Lock confirmation:** This plan is locked before any independent data collection. Running formal data after this commit must not modify concurrency, seeds, repetitions, metrics, run order, exclusion rules, or gate criteria.

*— End of LOCKED PLAN —*
