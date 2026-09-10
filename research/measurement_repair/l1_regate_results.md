# L1 Overhead Re-Gate Results — Phase B

> **Date:** 2026-08-28  
> **Gate Status:** **SKIPPED — PHASE A NON-STATIONARY**  
> **Locked Criteria (not evaluated):** median ≤5%, individual ≤10%, knee multiplier ≤10%  
> **Reason:** Per LOCKED_STATIONARITY_PLAN.md §5.6 and §11, Phase B may only run if Phase A is STATIONARY — PASS (or PARTIALLY with p95_ttft declared usable). Phase A was **NON-STATIONARY** (see `stationarity_results.md`), therefore L1 vs L0 comparison would conflate instrumentation with environment drift and is prohibited.

---

## 1. Decision Gate

| Phase | Classification | Detail |
|---|---|---|
| Phase A (L0 OFF stationarity) | **NON-STATIONARY — FAIL** | c1 median_ttft 5.35% >5%, p95_ttft 10.28% >10%, thr 5.45% >5%; c4 median_ttft 67.7% >5%, p95_ttft 36.4% >15% single outlier 29.6% |
| Phase B (L1 overhead) | **SKIPPED** | Not executed; no matched OFF→L1 pairs collected in this stage. Prior real gate data (20/40 warm, measurement_repair/processed/overhead_gate.json) shows L1 median +8.2% and individual 19% outlier — but that data was collected before stabilization (per_request client, unbounded thread growth) and cannot be re-used as Phase B evidence under new stabilization. |

---

## 2. What Would Have Been Measured (Locked Design, Not Executed)

- **Workload:** synthetic 512/64, 40 req (2 warmup +38 measured), closed, stream True, pooled client, fixed ThreadPool 32, torch 16.
- **Pairs:** 3 matched pairs per concurrency (seeds 4101/4102/4103 for c1, 4401/4402/4403 for c4), interleaved OFF→L1 / L1→OFF balanced round-robin, 12 runs total (6 OFF +6 L1).
- **Metrics:** median_ttft, p95_ttft, median_latency, p95_latency, throughput, knee multiplier c4/c1 p95_ttft, absolute Δ ms and relative Δ %.
- **Thresholds:** median |Δ| ≤5%, individual |Δ| ≤10%, multiplier |rel_change| ≤10%, std <10%.

Design is preserved in `LOCKED_STATIONARITY_PLAN.md §6` and `run_stage3mb.py:regate_specs()` for a future re-run if environment ever stably passes.

---

## 3. Why Skipping Is Correct

Per §17 Environment vs Instrumentation: instrumentation effect can only be isolated if baseline L0→L0 variance is small. Phase A shows L0→L0 p95_ttft variance at c4 is **36.4% relative range, CV 13.2%**, which dwarfs expected L1 effect (mock L1 was 3.9% median). Any observed L1−OFF delta inside 5% would be indistinguishable from noise; a 8–15% delta could be either environment or tracing. Reporting an overhead gate under these conditions would be **misleading** and violates §10 gate integrity.

---

## 4. Prior L1 Reference (Not Phase B Evidence)

For context only (from `measurement_repair_summary.md §3`, pre-stabilization, per_request client):

- **Mock L1 (ideal host):** PASS — c1 p95 TTFT median +3.90% (≤5%), individual ≤5.22% (≤10%), multiplier -3.58% (≤10%).
- **Real L1 20-warm & 40-warm (per_request, unbounded threads, before pooled fix):** FAIL narrow — median +3.3–8.2% (>5%), individual 17–19% (>10%), multiplier -6.8% PASS. Absolute Δ 15 ms on 80 ms baseline. Classification **FAIL — ENVIRONMENT NON-STATIONARY**, not instrumentation alone.

After stabilization (pooled, fixed 32, torch 16), mock still expected PASS but real must be re-proven; the old 8.2% cannot be declared PASS without a new paired Phase B on a stationary host.

---

## 5. Artifacts

```
research/measurement_repair/raw/regate/        ← empty (no Phase B runs collected)
research/measurement_repair/processed/regate/  ← no overhead_gate.json produced this stage
```

Previous gate artifacts retained at `research/measurement_repair/raw/repair-real-gate-40-warm_*` and `processed/overhead_gate.json` (last FAIL) for audit, but not promoted to Phase B.

---

## 6. Next Step

Re-gate is **blocked** until L0 stationarity achieves STATIONARY — PASS. See `stage3mb_summary.md §7` for minimal next validation experiment (single-variable controls), not a blind retry of 12-run paired gate.

*— End regate (skipped) —*
