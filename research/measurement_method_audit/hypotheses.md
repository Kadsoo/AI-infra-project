# Hypotheses — First-Token Variance Localization (competing, evidence-gated)

> Each hypothesis tagged [H*] with falsifier. Status = OPEN / REFUTED / SUPPORTED (needs replication). No pre-selection.

## H1 — Client/HTTP artifact (pooled vs per-request, streaming/SSE)
- **Claim**: TTFT jitter dominated by `httpx` client queuing / HTTP chunked encoding / SSE buffering.
- **Falsifier**: pooled client already reduces overhead 6 ms median p95 23 ms but c4 TTFT jitter 77 ms >> 6 ms; throughput stable; minimal_trace `L1_FIRST_CONTENT_YIELD - L1_HANDLER_ENTER` on server should have CV << client CV if client-driven.
- **Evidence so far**: discovery showed pooled client helps but does NOT eliminate instability (migration_summary §4). **NOT refuted**, but reduced to minority share. Needs server-side TTFT sidecar (Q2).
- **Test**: compare `TTFT_client = client_first_content - client_send` vs `TTFT_server = t_first_content_yield - t_handler_enter` per request; `F-test Var(client)/Var(server) > 2` supports H1. Instrumentation: L1 minimal (6 ints) with observer gate <10% throughput delta.

## H2 — WDDM / GPU scheduling jitter (platform-specific)
- **Claim**: Windows WDDM batches commands → additional service-time jitter J ~ Uniform[0, D_batch] inflating CV at all c, independent of prefill queue.
- **Falsifier**: same harness on Native Linux single-V100 shows CV ≤5% (pass) while Windows fails 87% at same workload/hash. WSL2-kernel fallback FAIL (87%) weakens but does not kill H2 because kernel≠userland+driver; Native Linux still needed.
- **Prediction**: CV(c) = sqrt(Var(J)+Var(queue(c))+Var(compute(c))) / E[TTFT(c)] → Windows offset at c=1 tail already 19% vs Linux ≈0.
- **Test**: paired c∈{1,4} 5 reps on single V100 Native Linux vs W with identical commit/workload/c. Mark UNKNOWN until probe. Source-balance caveat: GPU 4060 vs V100 32G difference => compare *stationarity*, not absolute ms.

## H3 — Runtime dispatch / asyncio scheduling (server ingress → executor)
- **Claim**: `asyncio.to_thread` with fixed executor `max_workers=32` + `torch 16/16` queues first-token prepare/forward; `executor_wait = t_prepare_worker_start - t_prepare_submit` variable.
- **Evidence refs**: `hf_server.py: _stream_real` per-token `to_thread`, `FIXED_EXECUTOR`; 3M-D GC+sleep improved 67%→20% but not to 5%; simple warmup worsened to 87% (migration_summary §4) suggesting runtime state, not just warmup.
- **Falsifier**: `Var(executor_wait) << Var(TTFT_server)` across 5 reps → H3 not dominant.
- **Test**: L1 decomposition `TTFT_server = executor_wait + (first_forward_begin - executor_start) + model_to_token`; correlate `executor_wait` with `TTFT_server` (R²). L0 OFF baseline needed before L1 to avoid conflation.

## H4 — Prefill / forward execution (model compute) + Python GC / allocator
- **Claim**: `past_key_values` growth per step + Python GC of per-request dicts + `past_key_values` allocation jitter drives heavy-tailed first-token. Prior Stage 3M-D saw RSS 190→60 MB sawtooth and GC helped 3.3x (67%→20%). Stage 3E RSS stable (±6 MB) but still FAIL → allocator sawtooth not sole driver this run.
- **Falsifier**: `use_cache=False` or isolated prefill-only executor should cut CV if true; otherwise H4 refuted.
- **Test**: control `use_cache=True vs False`, GC enabled vs disabled, `torch.set_num_threads` 16 vs 8; measure server-side `model_to_token = first_content_yield - first_forward_begin`.

## H5 — Token serialization / streaming emission
- **Claim**: `StreamingResponse` + `sse-starlette` `async yield` + WS -> client line framing adds variance after token ready, conflated into client TTFT but not server `t_first_token_sampled`.
- **Evidence**: `hf_server.py` marks `t_first_token_sampled` (inside worker) vs `t_first_content_yield` (before SSE `yield`) — gap is serialization. Prior [HYPOTHESIS] only.
- **Falsifier**: `TTFT_server (handler→first_token_sampled)` stable while `handler→first_content_yield` unstable → emission contributes.
- **Test**: same as H1 server-side split; requires L1 timestamps 3 and 4.

## H6 — Workload/arrival & batching confound (measurement-specific)
- **Claim**: closed `Semaphore(concurrency)` + 512/64 synthetic + 2 warmup excluded leaves queueing mis-modeled; tail 19% at c=1 suggests even low conc unstable without queue.
- **Evidence**: c1 median 3.28% PASS but p95 19.78% FAIL → tail instability even without contention; throughput p95_lat PASS → not pure queue.
- **Test**: vary `arrival_distribution` closed vs Poisson, `prefix_reuse` 0 vs 1, input length 512 vs 128; Latin-square interleaving already used for order (c1,c4,c4,c1...), rho 0.43 → order not driver.

## H7 — General runtime/framework (reproduces on both platforms)
- **Claim**: variance is framework-specific (`asyncio.to_thread` + HF `use_cache` per-token loop) and reproduces on any OS/GPU, not just WDDM.
- **Falsifier**: needs V100 Native Linux L0 FAIL to support; WSL2 FAIL 87% is partial support but Native still blocked → remains OPEN.
- **Implication**: would merit new RQ (framework pathology) not A_01 knee; do NOT automatically open new RQ per instructions.

## Priority order for next probes (information gain)
1. **Server-side TTFT stability (Q2)** via L1 6-int on Windows with observer gate — cheapest, distinguishes H1/H5 from H2/H3/H4.
2. **Single-V100 L0 OFF** (c=1,4 ×5) — distinguishes H2 (platform) from H5/H3/H4 (general).
3. **Phase decomposition** (executor_wait vs model_to_token) — localizes H3 vs H4 vs H5.
4. **Causal controls** (GC on/off, use_cache, pooled vs per-req, streaming vs non-stream) — confirms mechanism.

## Labels
All above are [HYPOTHESIS] until measured; code refs are [SOURCE-CODE FACT] (hf_server.py locks), prior gates are [EXPERIMENTAL OBSERVATION], frozen docs are [DOCUMENTED FACT]. No claim promoted without phase timing or A/B.
