# LOCKED PLAN — RQ-1 (P/D vs Colocated Crossover)

> **Freeze date:** 2026-08-27. This plan is frozen after adversarial review (see `design_review.md`). Stage 3B may NOT modify these items because results look bad. If modification becomes scientifically necessary, create `LOCKED_PLAN_v2.md` with an explicit change rationale; never overwrite this file.

## 0. Freeze scope

The following are frozen and versioned here: Primary Hypothesis, Sub-Hypotheses, Main Metrics, Baseline, Success Criterion, Falsification Criterion, and the experiment kill-gate order (E1 → E2 → E3 → E4 → E5, from `experiment_plan.md`). Everything else in `hypothesis.md` / `experiment_plan.md` (workload parameters, instrumentation detail, confounder handling) is implementation-level: refinable in Stage 3B only with a written record, never silently.

## 1. Primary Hypothesis (frozen)

> Under cross-node placement at ≤25 Gbps interconnect with mixed prompt lengths (1k-class and 8k–13k-class) and bursty arrivals (inter-arrival-time coefficient of variation CV ≥ 2), the KV movement plus cache-affine queueing mechanism (transfer time + queue time on the TTFT critical path) becomes ≥ 30% of E2E TTFT at P95, causing the P/D configuration's P99 TTFT to exceed 1.5× the colocated chunked-prefill baseline at equal GPU allocation and equal SLO, while aggregate throughput at equal load stays within 5% of the baseline (reported guard, not a conjunct).

Revision note: the ≤5% P50-TTFT conjunct was removed pre-freeze (design_review R1) — it was arithmetically incompatible with the ≥30% fraction claim at 25 Gbps.

## 2. Sub-Hypotheses (frozen)

- **H1 (transfer materiality):** at ≥8k prompts, ≤25 Gbps cross-node, no NVLink co-location → P95 transfer fraction of E2E TTFT ≥ 30% (P50 ≥ 15%); NVLink keeps P95 ≤ 5%.
- **H2 (queue hotspot, tested at transfer ≈ 0 / NVLink):** CV 2–3 + cache-affine pairing → P99 queue fraction ≥ 20% and P99 TTFT ≥ 1.5× colocated; CV = 1 keeps P99 ≤ 1.1× colocated.
- **H3 (baseline overtake):** at joint condition (25 Gbps, CV ≥ 2, cache-affine), colocated ≥ 100% of P/D SLO goodput at equal GPU count, or P/D needs ≥ 1.25× GPUs for parity; benign cell (NVLink, CV = 1) reversed: P/D ≥ 1.3× colocated goodput.

## 3. Main Metrics (frozen)

TTFT P50/P95/P99; TPOT P50/P95/P99; transfer-time fraction of E2E TTFT (P50/P95, critical path); queue-time fraction of E2E TTFT (P95/P99); SLO goodput (attainment ≥ 95%); GPU allocation; cache hit rate (guard); aggregate throughput (guard).

## 4. Baseline (frozen)

Colocated chunked prefill (Sarathi-style, tau = 1024, FCFS + join-shortest-queue dispatch), same 2 GPUs, same trace (5,000 requests, fixed order/seed), same SLO, same mean rate. Fairness: only phase split and placement differ; hit-rate guard: if P/D hit rate exceeds colocated by > 10 points, enable the equivalent prefix cache in the colocated baseline and rerun.

## 5. Success Criterion (frozen)

| Hypothesis | Success = |
|---|---|
| H1 | P95 transfer fraction ≥ 30% at 25 Gbps (8k–13k) AND P95 fraction ≤ 5% at NVLink |
| H2 | at CV = 3 (NVLink): P99 queue fraction ≥ 20% AND P99 TTFT ≥ 1.5× colocated; at CV = 1: P99 ≤ 1.1×; load-balanced CV = 3: P99 ≤ 1.2× |
| H3 | at joint: colocated goodput ≥ 100% of P/D at equal GPUs, OR P/D needs ≥ 1.25× GPUs; benign: P/D ≥ 1.3× colocated goodput |

## 6. Falsification Criterion (frozen)

| Hypothesis | Falsified iff |
|---|---|
| H1 | measured P95 transfer fraction < 30% at 25 Gbps 8k–13k (corrected prefill calibration), OR 25 Gbps-vs-NVLink P95 TTFT gap < 25% of NVLink value (revised pre-freeze: old <10%/<10% lines were physically unreachable) |
| H2 | at CV = 3 (NVLink): P99 TTFT ≤ 1.1× colocated AND P99 queue fraction < 15%; OR load-balanced CV = 3 (same bandwidth) also ≥ 1.5× |
| H3 | at joint: P/D ≥ 100% of colocated goodput at equal GPUs with P99 TTFT ≤ 1.1× colocated |

Outcome bands (frozen): fraction ≥ 30% → H1 supported; < 30% → H1 falsified. "Neither" is not an allowed verdict at E4 for H1/H2. Both-fail at joint → double-allocation rerun (pre-registered first-class cell) before any conclusion.

## 7. Kill-gate order (frozen)

E1 (analytical screening, no GPU; can kill H1) → E2 (queue simulator, no GPU; can kill H2) → E3 (transfer microbenchmark; can kill H1 on hardware; doubles as E4 step-0) → E4 (vLLM serving, 3× A100 2+1; can kill H1/H2/H3) → E5 (GPU-allocation sweep; H3 allocation leg only if E4 supported H3). A kill at any gate stops the pipeline. SLO: TTFT P95 ≤ 4.0 s, TPOT P95 ≤ 0.15 s, attainment ≥ 95%.

## 8. Freeze rules

1. No threshold, baseline, metric, or kill condition in §1–§7 may change during Stage 3B based on experimental outcomes.
2. Any necessary change requires `LOCKED_PLAN_v2.md` with: what changed, why, what evidence motivated it, and whether it loosens or tightens the original claim.
3. Null results are valid outcomes and terminate the pipeline per the kill gates; they are recorded, not re-run into submission.
4. Claims are scoped as operating-boundary measurements (the matched-harness crossover), not mechanism discoveries (design_review Q8/Q9 accepted as-is).