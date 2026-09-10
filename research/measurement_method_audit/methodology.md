# Methodology — Measurement Definition Audit & Stationarity Gate

## 1. Definitions (do not conflate)

- **User-visible TTFT (client)**: `TTFT_client = client_first_content_perf_ns - client_send_perf_ns` (seconds). Measures `dispatch → headers → SSE first delta`. Includes client queuing, HTTP/SSE, server ingress, runtime dispatch, queue, prefill, forward, serialization. Answers SLO experienced by request issuer. Harness records it per `RequestRecord.ttft` via `client_send_perf_ns` (before `client.stream POST`) and `client_first_content_perf_ns` (at first non-empty SSE `delta.content`). Source: `research/measurement/harness/harness.py: execute_request`.
- **Server-side TTFT (ground truth for science)**: `TTFT_server = t_first_content_yield - t_handler_enter` (or stricter `t_first_token_sampled - t_handler_enter` if isolating compute). Measures `server ingress → first token ready to yield` (before SSE framing). Excludes network/client. Answers serving runtime boundary. Source: `hf_server.py: trace.mark(L1_HANDLER_ENTER)` and `L1_FIRST_CONTENT_YIELD` (minimal_trace 6-pts).
- **Model-execution first-token latency**: `L_exec = t_first_token_sampled - t_first_forward_begin` or `model_to_token = first_content_yield - first_forward_begin` (minimal) — forward-only. Debug slice, not SLO.

Frozen rule: compare only within definition family; cross-family comparison is error. Selected gate metric for paper is **TTFT_server** (SERVER).

## 2. Obtainable timestamps (lowest overhead first)

Priority per §10: built-in > server timestamp > minimal request-level > sampled > aggregate > perturbation.

| # | Timestamp | Clock | Overhead | Availability |
|---|---|---|---|---|
| C1 | `client_send_perf_ns` | `perf_counter_ns` client | ~0 | always (harness) |
| C2 | `client_headers_perf_ns` | `perf_counter_ns` | ~0 | always |
| C3 | `client_first_content_perf_ns` | `perf_counter_ns` | ~0 | always for streaming |
| S1 | `t_handler_enter_ns` (L1 idx0) | `perf_counter_ns` server | 1 int write | L1 handler_enter |
| S2 | `t_prepare_submit_ns` (L1 idx1) | server | 1 int | L1 executor_submit |
| S3 | `t_prepare_worker_start_ns` (L1 idx2) | server | 1 int | L1 executor_start (inside worker) |
| S4 | `t_first_forward_begin_ns` (L1 idx3) | server | 1 int | L1 first forward (inside torch worker) |
| S5 | `t_first_content_yield_ns` (L1 idx4) | server | 1 int | L1 first yield before SSE |
| S6 | `t_server_done_ns` (L1 idx5) | server | 1 int | L1 done |
| S* | `t_first_token_sampled_ns` | server | full L2 only | L2 full trace (per-step) — **forbidden as default**; use L1 S4/S5 proxy unless L2 narrowly sampled |
| H | host sampler 0.3s/0.5s | psutil/NVML | interval | always |

**Chosen for gate**: **L0 OFF** (0 ints) for baseline stationarity. Then L1 minimal 6 ints (idx0-5) with gate `instrumentation effect ≤10% throughput delta` (LOCKED). Full L2 with per-token `thread.get_ident`+sorting is BANNED as default (observer effect in Stage3M). L1b (sampled 10-20%) and L1c (counters) are fallbacks if L1 shows gate breach.

Observer gate: `|thr_L1 - thr_OFF| / thr_OFF ≤10%` AND `|TTFT_server_OFF_proxy - TTFT_server_L1| ≤10%` on same single-V100 warm workload; otherwise L1 invalid.

## 3. L0 stationarity gate (locked, do not relax)

Per `LOCKED_STATIONARITY_PLAN` / Stage3E §7, philosophy central ≤5% tail ≤10%:

- **Central** (`median_ttft`, `median_lat`, `throughput`): `rel_range = (max-min)/median ≤5%` AND `CV = std/mean ≤5%` advisory AND `|rho|≤0.6` or `|slope|/median ≤2%/step` (drift). Also absolute floor 10 ms reported alongside %.
- **Tail** (`p95_ttft`, `p95_lat`): `rel_range ≤10%` (hard fail >15%), single outlier `|value-median|/median ≤20%`, same drift rule.
- **Instrumentation preservation**: L1 vs OFF throughput/TTFT delta ≤10% (locked killer).
- **Reps**: ≥5 per concurrency, balanced interleaved run order `c1,c4,c4,c1,c1,c4,c1,c4,c4,c1` seeded 6101-6105/6401-6405, fixed workload `synthetic 512/64`, `n=40` (2 warmup seq excluded +38 measured), pooled client, fixed `ThreadPoolExecutor(32)`, `torch 16/16`, `temperature 0.0`, `stream True`, `timeout 180`, uniform warmup `1× c=4 n=40 pooled OFF seed 8000` + per-run 2 excluded (not Protocol D).
- **Preservation**: `raw/` keeps `run_id_raw.json`, `run_id_requests.csv`, `run_id_system.csv`, plus `manifest.json` (hashes, command, config, PID/port, token stats `describe_workload` ~510.8±11.2), `trace_count`, `health_before/after`. Failures retained; silent deletion banned; exclusion must log reason.

**Why this gate**: puts noise of repeated measurements beside declared margin (5%/10%). When noise wider than margin, run has not tested idea → fix run, not smaller claim.

## 4. Q1/Q2 audit plan (client vs server)

- Collect both `TTFT_client` (harness) and `TTFT_server` (L1) per request in same run (joined by `X-Stage3-Run-Id`/`Request-Id`). Compute `Var(TTFT_client) = Var(TTFT_server)+Var(T_emit)+2Cov` and `CV_server` vs `CV_client` per concurrency. Prediction: `CV_server << CV_client` if emission/network dominates (H1/H5); else `CV_server ≈ CV_client` if runtime (H3/H4) or WDDM (H2) dominates.
- F-test `Var ratio = Var_client/Var_server`; significant if >2 with p<0.05 (permutation). Report alongside CV.

## 5. Q3 decomposition (phase slicing)

After L0 PASS on chosen env, slice `TTFT_server` using L1 6 ints:

- `executor_wait = S3 - S2` (queue + thread scheduling)
- `prepare_to_forward = S4 - S3` (includes tokenization/truncation before model)
- `model_to_yield = S5 - S4` (forward + sampling + decode of first token + SSE prep)
- `total_server = S5 - S1`
- `handler_to_submit = S2 - S1` (ingress + body parse)

Correlate each slice variance with `TTFT_server` (R²). Dominant slice = highest R² and largest `Var(slice)/Var(total)` share.

## 6. Controls (minimal causal)

Single-variable perturbations, L0 OFF unless slicing:

- client `pooled vs per_request`, streaming vs non-stream, `use_cache True vs False`, `GC enabled vs gc.disable()`, `torch threads 16 vs 8`, `max_workers 32 vs 16`, `prefix_reuse 0 vs 1`, `input 512 vs 128`. Each requires 5 reps ×2 conds, interleaved Latin square, same gate. Not run until L0 baseline exists.

## 7. Analysis path (evaluator)

- Process raw → `processed/<run>_processed.json` via `harness.compute_processed` (stats: median/p95 throughput, latency, TTFT, TPOT, system CPU/RSS/threads/GPU). Gate script computes rel_range, CV, rho, slope, pairwise outlier. Host sampler confirms no thermal/threads spike (threads 67-68 stable, RSS ±6 MB, GPU 43C, CPU 10-18%).
- Comparison across envs is stationarity-only; absolute performance (rps) not ranked (different GPU/hardware).

## 8. Reuse & provenance

Reuse `workload/generator.py` (5ef1dcbf575c), harness, `hf_server.py` (39cc7cd90bac), host logging, minimal tracing interface; do NOT rewrite system. Every run logs command, SHA-256 per file, PID/port, workload token stats, hardware, run order.

## 9. Hard bounds

- Do NOT recover Full L2 as default.
- Do NOT relax 5%/10%/10% after seeing data; report as non-TTL if gate fails and propose next fix.
- Do NOT compare 4060 vs V100 absolute ms as speed claim.
- Do NOT auto-unfreeze A_01; even STATE A requires external review.
