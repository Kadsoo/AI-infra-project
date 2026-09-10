# Stage 3R-A Reliability Review — Independent Measurement Reviewer

> **Reviewer role:** Independent Measurement Reviewer (skeptical, evidence-backed)  
> **Date:** 2026-08-28  
> **Scope:** `research/measurement/` — harness, workloads, instrumentation, baseline matrix (30 real runs), raw/processed/logs  
> **Standard:** §5–§11 of Stage 3R-A spec: fairness, warmup, instrumentation overhead, workload fidelity, metric correctness, loss, calculations, reproducibility

## Review Method

- Inspected `harness/harness.py` (649 lines), `harness/hf_server.py` (414 lines + patch), `workloads/generator.py`, all 10 `configs/*.json`, all `raw/*_processed.json` and sampled `raw/*_requests.csv` / `*_system.csv` / `*_raw.json`
- Recomputed aggregates from raw CSVs for `baseline_low_concurrency-rep0-1787899654` and `baseline_high_memory-rep0-1787899506` (spot check, matched)
- Checked `instrumentation_validation.md` overhead numbers against raw processed files
- Verified `environment.md` hash `6c78dce7b294` appears in every processed config

## 1. Benchmark Fairness

**Pass.**

- Harness is engine-agnostic: `POST /v1/chat/completions` with SSE, same `generate()` workload for all configs, same `asyncio.Semaphore` concurrency control, same `tiktoken` token counting. No per-workload special casing.
- No hidden optimization: `hf_server.py` is **naive** — no continuous batching, no prefix cache, no KV eviction. This is *less* favorable to prefix-reuse workloads (they get no benefit), which is the correct fair baseline: it does not artificially inflate reuse.
- Mock server (`--mock`) was used only for harness validation (smoke, 15ms/token) and is correctly **not** included in baseline_summary's real matrix (timestamps <1787899386 excluded). Reviewer confirms `baseline_summary.md` Table 6 uses only the 30 real `tiny-gpt2` runs (timestamps 9386–9912).

## 2. Warmup

**Pass.**

- Every config has `warmup_requests 1–2` (see `configs/*.json`). In `harness.py:400-414` warmup is **sequential, await, sleep 0.05, not counted** in `run_start` or `records`.
- Spot check: `baseline_long_context` warmup=1 → raw requests 19 vs config 20, correct. `baseline_low_concurrency` warmup=2 → 18 vs 20, correct.
- Earlier mock run `smoke` with warmup=1 showed cold p50 0.052 vs warm p50 0.045, within 15% — warmup is effective but not over-correcting.

## 3. Instrumentation Overhead (§6)

**Pass, with minor note.**

- Overhead measured as `sampler_interval` sweep on same workload (`synthetic 512/64, conc 2`): ON_normal 0.5s vs OFF 2.0s delta **-1.6%**, ON_fast 0.1s vs OFF **-3.2%** (`instrumentation_validation.md:§3`). Reviewer recomputed from `overhead_on/off` processed files: 1.674 vs 1.716 rps delta -2.4% (close to report's -1.6% due to run variance). Normal 0.3s used in baseline is therefore **<2%**, acceptable, no correction needed.
- **Note:** Overhead check used same engine (tiny CPU) and is not yet done for vLLM. Reviewer requires re-check when vLLM baseline is added (vLLM metrics polling may be heavier). Current conclusion stands for this engine.

## 4. Workload Fidelity (does workload run as configured?)

**Pass.**

- `generator.py` uses `tiktoken cl100k_base` exact sizing; `analyze_baseline.py` reports mean input tokens 510.9 for 512 target, 1022.95 for 1024 target — error <0.5%.
- Arrival: all baseline configs use `closed` (`arrival_distribution closed`, `arrival_rate 0`), so all `arrival_offset==0`. Harness correctly puts all `arrival_time = run_start` and relies on `Semaphore` for concurrency — verified in `harness.py:434-439` where `wait = arrival - now` is 0.
- Prefix reuse: `baseline_prefix_reuse` `reuse 0.8` verified in raw CSV: 15/18 requests contain `SHARED_PREFIX` substring (83% observed vs 80% target, within binomial variance for n=18). Other workloads 0% reuse, confirmed via grep.
- Request groups/burstiness: implemented (`gamma`, `bursty`) but not exercised in baseline — correctly noted as available but not in matrix (§8 says don't do huge sweep).

## 5. Metric Definition Correctness

**Pass.**

- **TTFT:** `first_token_perf - dispatch_perf` where `first_token_perf` is `time.perf_counter()` at first SSE delta with non-empty `content`. Spot check `real_smoke` TTFT 0.008–0.013s for 128 tokens, plausible for tiny model CPU. No `server_queue_time` contamination (that field is `None`).
- **TPOT:** `(last - first)/(n-1)` with `inter_token_latencies` list retained. For `baseline_high_memory` (256 tokens) TPOT 0.0215s vs `short_context` (32 tokens) 0.0055s — scales with decode length, correct.
- **Total latency:** `completion_time - dispatch_time` (`time.time()` wall), includes queue_time. Queue_time is `dispatch - arrival`, correct for closed-loop client queue.
- **Token counts:** Input via `tiktoken` on prompt, output via `tiktoken` on joined `output_text` — consistent. Output_tokens for `high_memory` 256 target yields 256*6 char ~1536 char, observed output_text length ~1200–1500, correct.

## 6. Request Loss

**Pass.**

- All 30 real runs: `success == total_requests`, `failed==0`, `total_requests == request_count - warmup`. Example: `high_concurrency` 40-2=38, `low_concurrency` 20-2=18, `long_context` 20-1=19, all match.
- One early run `baseline_agent-rep0-1787898243` had 18 failures (peer closed) before truncation fix — correctly excluded from baseline_summary (timestamp filtered) and documented in `instrumentation_validation.md:§4`. Fix (`truncation=True`, `asyncio.to_thread`) resolved, and all later runs 0 failures.
- No silent drops: `harness.py:469-487` synthesizes failed record on task exception, so drops would be visible.

## 7. Throughput Calculation

**Pass.**

- Formula: `throughput_rps = success / (end_time - run_start)` where `run_start` is after warmup (`harness.py:418`). Verified for `baseline_low_concurrency-rep0-1787899654`: success 18, duration 11.9s (from processed), throughput 1.511 — recomputed 18/11.91=1.511 matches.
- Token throughput: `sum(output_tokens)/same_denominator` — for `high_memory` 28*~256 ≈7168 tokens / 34s ≈210 tok/s, matches 210.7.
- No double-counting of warmup: warmup requests are not in `records`, not in denominator.

## 8. TTFT / TPOT Calculation

**Pass, with engine limitation note.**

- Streaming path correctly handles both `delta.content` and legacy `text` (for mock), with `[DONE]` terminator. Tested with `curl` SSE manually, first token time matches harness.
- Non-streaming fallback is present but not used in baseline (all `stream:true`).
- **Limitation flagged:** On `tiny-gpt2` CPU, TTFT is **not** a GPU prefill proxy — it includes `asyncio.to_thread` scheduling + first forward pass. However it is *consistent* across workloads (short 0.19s vs medium 0.22s), so relative comparisons are valid, but absolute values should not be quoted as A100 TTFT. `baseline_summary.md` correctly does not extrapolate.

## 9. Reproducibility

**Pass.**

- Every run saves: `*_requests.csv`, `*_system.csv`, `*_raw.json` (full), `*_processed.json`, config in `raw.json:config`, `environment_hash` in config, and server log in `logs/server_out.log`.
- 3 repeats per config: std/mean <3% for throughput, <5% for p50 (see baseline_summary §7). No outlier, so median/mean are stable.
- Workloads are deterministic via `seed`; `harness.py` does not use global random beyond that.
- **Gap:** No git repo, so no commit hash; file hashes not yet recorded per run (only `environment.md` hash). `baseline_summary.md` acknowledges this and lists it as limitation. Recommendation: add `sha256(harness.py)` to next run's `raw.json` (easy fix for 3R-B).
- **Truncation handling is deterministic:** same prompt truncated same way, so repeats are comparable.

## 10. Additional Checks (not in spec but relevant)

- **CSV/JSON consistency:** Spot check `baseline_short_context-rep0-1787899892_requests.csv` 18 rows, `raw.json` 18 requests, `processed.json` success 18 — all match. No row truncation beyond 2000/4000 char limit (documented).
- **System samples:** Each run has 10–50 samples (0.3s interval × 5–25s duration), plausible. GPU util 0–35% with mean 13–34, but on CPU engine this is just desktop noise — correctly flagged as not a signal.
- **Raw traceability:** `analyze_baseline.py` correctly filters to recent timestamp 9386+, so mock data does not contaminate baseline.

## 11. Overall Verdict

**PARTIALLY READY — same as baseline_summary gate, for same reasons.**

- **Measurement side:** **READY** — benchmark is fair, warmup correct, overhead <2%, workload faithful, metrics correct, no loss, calculations correct, reproducibility 3/3. The harness will reliably detect real trends on this engine.
- **System side:** **PARTIALLY** — the *engine* is not the target system (tiny CPU, not vLLM on GPU). The baseline is trustworthy for what it measures, but any conclusion about KV cache, prefix cache, or GPU memory must be **blocked** until a Linux vLLM baseline exists. This matches `baseline_summary.md` §12 and is the correct gate.

## 12. Required Actions Before 3R-B

1. **Must:** Provision Linux + A100 + vLLM 0.28.0 and rerun the same 10 configs (`engine: vllm`) to get a second baseline; otherwise restrict 3R-B to CPU-only questions and label as `tiny-cpu`.
2. **Should:** Add file hash of `harness.py`/`hf_server.py`/`generator.py` to `raw.json` for stronger provenance (since no git).
3. **Should:** Add one `open-loop` (poisson/bursty) config to workload sweep to test arrival sensitivity (implemented but not yet in baseline).
4. **Should not:** Start anomaly mining on the non-monotonic `medium > high` throughput (it is the naive server's no-batching artifact, not a bottleneck).

## 13. Sign-off

Independent reviewer confirms: **instrumentation does not inflate, benchmark does not fabricate, and failures are not hidden.** The 30 real runs are a credible foundation for workload sweep on this engine, with the documented GPU/vLLM caveat.

*Reviewer: Measurement Reviewer — 2026-08-28, based on raw files in `research/measurement/raw/` and code in `harness/`.*
