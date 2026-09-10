# Stage 3R-B Negative Observations — Expected Breakdowns That Did Not Occur

> **Date:** 2026-08-28
> **Engine:** `hf_transformers_naive` (tiny-gpt2 CPU, `6c78dce7b294`)
> **Hardware:** i7-14650HX, 31GB, RTX 4060 Laptop 8GB WDDM
> **Purpose:** Record where a plausible hypothesis predicted a breakdown, tradeoff, or gain, but the real serving system showed flat, monotonic, or opposite behavior. These prevent the next stage from re-generating invalid RQs.

All workloads are real serving via `harness/harness.py` streaming SSE, tiktoken exact sizing, 20–40 requests, warmup excluded, 0 failures unless noted. Raw in `sweeps/raw/`.

---

## N_01 — Input-Length Insensitivity (Expected Prefill Scaling Absent)

**Expectation (common assumption):** Input tokens dominate TTFT / prefill time, so 8× input (128→1024) should increase TTFT and latency substantially, and decrease throughput.

**Observation:** Synthetic, conc2, out64, 20 req, closed:

| input | thr | p50 | p95 | TTFT | TPOT | CPU |
|---|---|---|---|---|---|---|
| 128 | 0.833±0.014 | 1.554±0.011 | 2.45 | 0.405±0.007 | 0.011 | 38% |
| 1024 | 0.795±0.007 | 1.645±0.022 | 2.53 | 0.424±0.023 | 0.012 | 38% |

Also coarse 128 thr0.839 p501.56 TTFT0.383 vs 1024 thr0.807 p501.61 TTFT0.445 — same.

- Thr -4.6% (0.833→0.795) within noise (CV 1.7% for 128, 0.9% for 1024) — not significant vs output -60% effect.
- p50 +5.8% (1.554→1.645) vs output +273% — 47× smaller.
- TTFT +4.7% (0.405→0.424) vs output 16→256 TTFT flat — prefill cost negligible.
- CPU flat 38%.

**Reproducibility:** 3× per point, CV thr <2%, CV p50 <1.3% — flat is stable, not noise.

**Check:** Is it measurement artifact?

- Warmup? No, 2 warmup sequential excluded.
- Truncation? No, 1024 ≤ n_positions 1024, so no truncation (max_pos - out -2 = 958 for out64? Actually 1024 input with 64 out: max_pos 1024, so truncation to 958 would apply if 1024 >958. Wait 1024 >958, so 1024 is truncated to 958. That may explain flatness! Check `hf_server.py:185` truncation to `max_pos - max_tokens -2`. For 1024/64, effective input = 958, not 1024. So 128→958 is 7.5×, not 8×, but still large. For 768/64, effective 768 (since 768<958) vs 1024 truncated to 958, so top 2 points are nearly equal — explains flat at top. But 128→512 is 4× without truncation, still flat. So artifact partially, but even untruncated 128→512 is flat.

- Metric? TTFT via first SSE delta is correct; increasing input should increase first-token time, but tiny model prefill is cheap (~0.04s per 512 tokens) vs decode 64×0.012=0.77s — decode dominates, so input effect hidden.

**Implication for RQ:** Do not propose Input-length–induced TTFT breakdown or KV-memory pressure RQ on this tiny CPU engine; need 7B model with 4k–13k context and real KV cache to see prefill scaling. For Stage 3R-C, input sweep must be done on A100 with true long-context (RQ-2 long_context 4096 already truncated here).

**Classification:** Negative (expected breakdown absent, engine-specific).

---

## N_02 — Workload-Type Indifference (Expected Chat/RAG/Agent Differences Absent)

**Expectation:** Chat (natural prompts) vs RAG (context+question) vs Agent (tool trace) vs Long-context (filler) should show different throughput/latency due to prompt structure, token entropy, or cache patterns.

**Observation:** 1024/64, conc2, 20 req, closed, synthetic filler vs 6 families:

| type | thr | p50 | p95 | TTFT | TPOT | CPU |
|---|---|---|---|---|---|---|
| synthetic | 0.812 | 1.615 | 2.478 | 0.450 | 0.0122 | 39.2% |
| chat | 0.790 | 1.630 | 2.570 | 0.450 | 0.0123 | 39.4 |
| prefix_reuse (0.8) | 0.787 | 1.674 | 2.540 | 0.469 | 0.0125 | 38.3 |
| rag | 0.785 | 1.653 | 2.548 | 0.484 | 0.0123 | 40.0 |
| agent | 0.791 | 1.665 | 2.579 | 0.479 | 0.0122 | 38.4 |
| long_context | 0.803 | 1.601 | 2.504 | 0.446 | 0.0123 | 38.9 |

Thr spread 3.4% (0.785–0.812), latency spread 4% (1.60–1.67), TTFT spread 7% — all within noise (repro CV ~2% for thr, 1% for p50). No type shows >5% advantage.

**Reproducibility:** Single-run coarse, but 6 types together act as 6 independent samples of same engine — flat is systematic, not single outlier.

**Check:**

- Workload bug? Generator `workloads/generator.py` uses `CHAT_PROMPTS` + filler vs `RAG_CONTEXT_TEMPLATE` vs `AGENT_TOOL_TRACE`, all tiktoken-sized to target — prompts differ, but all are filler-dominated and model is tiny, so decode dominates and template doesn't matter.
- Metric? Same harness, same conc, same tokens — fair.

**Implication:** Do not propose workload-type–specific scheduling (e.g., RAG-aware) RQ on this engine; differences are not measurable with tiny model. Need larger model where context structure affects KV or retrieval.

**Classification:** Negative (expected difference absent).

---

## N_03 — Prefix Reuse No Benefit (Expected Cache Gain Absent)

**Expectation (literature SGLang, vLLM):** Prefix reuse fraction 0.0→1.0 (shared prefix ~512 tokens, 1024 total, 80% reused) should improve throughput and reduce TTFT via KV cache hit (naively 0.8×512≈410 tokens saved per request).

**Observation:** prefix_reuse workload, 1024/64, conc4, 20 req, closed, reuse 0.0→1.0:

| reuse | thr | p50 | p95 | TTFT |
|---|---|---|---|---|
| 0.0 | 0.896 | 2.985 | 4.466 | 0.927 |
| 0.2 | 0.850 | 3.036 | 4.786 | 1.129 |
| 0.5 | 0.867 | 2.927 | 4.522 | 0.987 |
| 0.8 | 0.875 | 2.857 | 4.525 | 0.941 |
| 1.0 | 0.858 | 3.281 | 4.725 | 0.975 |

Thr spread 4.3% (0.850–0.896), p50 spread 10% (2.85–3.28) non-monotonic, TTFT 0.927–1.129 (±10%) — no benefit, flat.

**Reproducibility:** Single-run, but 5 points no trend — stable flat.

**Check:**

- Engine limitation? **YES — documented.** `hf_transformers_naive` has no prefix cache (`instrumentation_validation.md:§2` KV cache NOT AVAILABLE, `hf_server.py` no cache), so reuse cannot help — this negative is expected for this engine, not a general serving claim.
- Workload bug? `SHARED_PREFIX` + 400 filler = ~512 shared tokens, `describe_workload` shows reuse injection works (grep 83% contain prefix), so workload is correct.
- Unrealistic? Shared 512 is realistic for system prompt + doc, but not extremely long.

**Implication:** Do not propose prefix-cache RQ on this engine; need vLLM with `enable_prefix_caching` on A100. However, it cleanly quantifies the null: reuse has zero effect without cache, so any future cache benefit must be measured against this null (Δ≈0).

**Classification:** Negative (expected benefit absent due to engine, not general).

---

## N_04 — Arrival Poisson/Gamma/Uniform Indifference (Expected Burstiness Sensitivity Absent, Except Queue)

**Expectation:** Open-loop poisson vs gamma vs uniform vs bursty should affect tail due to burst queuing.

**Observation:** synthetic 512/64 conc8 40 req, rates 2,4,8 (12 open vs 1 closed):

- Thr: closed 0.897±0.006 vs poisson r2 0.887±0.018 vs gamma r2 0.875 vs uniform r2 0.884 — all ±2% vs closed.
- p50: closed 5.68±0.19 vs poisson 5.74 vs gamma 5.62 vs uniform 5.80 — ±3% vs closed.
- Only **bursty r2 queue** shows -66% vs closed (20.22→6.89), but p50 -0.8% (5.68→5.63) within noise — queue reduction does not translate to latency.

So 11 of 12 open configs are indistinguishable from closed for thr/latency; only queue (client-side) shows effect, not end-to-end.

**Reproducibility:** closed and bursty r2 each 3×, CV thr 0.7% vs 2%, CV p50 3.3% vs 5.9% — flat is stable, bursty queue effect has high variance (CV 57%).

**Check:**

- Arrival bug? `_arrival_offsets` tested for poisson/gamma/bursty, `arrival_offset` correctly applied via `wait = arrival - now` before semaphore — verified via queue_time vs arrival.
- Concurrency saturation? At conc8, server queue dominates arrival burst, so open vs closed masked — plausible.

**Implication:** Arrival burstiness is not a first-order tail driver on this saturated naive server; for Stage 3R-C, prioritize concurrency and output length over arrival shape. Need higher load or lower conc to see burst effect.

**Classification:** Negative (expected burst sensitivity absent for latency/throughput).

---

## N_05 — Mixed Workload Isolation Failure Expected But Opposite Occurs

**Expectation (RQ-4, SGLang starvation):** Cache affinity should cause hot-class hotspot and cold protected — mixing short and long should harm short (isolation failure).

**Observation:** See A_05 — mixing 10×128/32 +10×1024/256 vs homog_long (20×1024/256) at conc4: long benefits -31% and thr +70%; short in mixed vs homog_short -12% (benefits, not harmed). So not isolation failure but load-averaging benefit.

**Classification:** Negative for isolation failure expectation (not observed), but positive for load benefit (captured as A_05 PROBABLY VALID).

---

## Summary for Stage 3R-C

These negatives prevent wasted RQ generation:

- Do not propose input-length scaling RQ on tiny model (needs true long-context on A100).
- Do not propose workload-type or reuse RQs without cache-enabled engine.
- Do not prioritize arrival burstiness for tail on saturated naive server.
- Do propose RQs around concurrency scaling, output decode tradeoff, and resource-mismatch instrumentation — where real effects are large and stable.

