# L1 Overhead Re-Gate Results — Phase B (Independent Validation)

> **Date:** 2026-08-29T07:45:30Z  
> **Gate Status:** **SKIPPED — PHASE A NON-STATIONARY**  
> **Locked Criteria (not evaluated):** median ≤5%, individual ≤10%, knee multiplier ≤10%  
> **Reason:** Per LOCKED_INDEPENDENT_STATIONARITY_PLAN.md §5.7 and §6, Phase B may only run if Phase A is **STATIONARY — PASS**. Phase A was **NON-STATIONARY — FAIL** (c1 central 5.57% >5%, c4 median 20.3% >5% plus drift, c4 p95 29.9% >10%), therefore L1 vs L0 comparison would conflate instrumentation with environment non-stationarity and is prohibited. This matches the pre-registered non-invertible gate order.

---

## 1. Decision Gate

| Phase | Classification | Detail |
|---|---|---|
| Phase A (L0 OFF stationarity, Protocol D, new PID 34892) | **NON-STATIONARY — FAIL** | c1 median_ttft 5.57% >5% (CV 3.10% >3%); c4 median_ttft 20.3% >5% (CV 10.5% >3%, drift rho 0.62 significant), p95_ttft 29.9% >15% (CV 16.1% >7%, single outlier 21.3% >20%) |
| Phase B (L1 overhead, Protocol D) | **SKIPPED** | Not executed; no matched OFF→L1 pairs collected in this independent session. Protocol D's environment cannot yet be proven stationary, so overhead cannot be isolated. |

---

## 2. What Would Have Been Measured (Locked Design, Not Executed)

- **Workload:** synthetic 512/64, 40 req (2 warmup +38 measured), closed, stream True, pooled client, fixed ThreadPool 32, torch 16, Protocol D warmup+GC+sleep before each run, same PID 34892, port 8040.
- **Pairs:** 3 matched pairs per concurrency (seeds 5101/5102/5103 for c1, 5401/5402/5403 for c4), interleaved order balanced as per LOCKED plan §6.2 (12 runs total, 6 OFF +6 L1):
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
- **Metrics:** median_ttft, p95_ttft, median_latency, p95_latency, throughput_rps, knee multiplier `c4/c1 p95_ttft`, absolute Δ ms and relative Δ %.
- **Thresholds:** median |Δ| ≤5%, individual |Δ| ≤10%, multiplier |rel_change| ≤10%, std <10%.
- **Output:** `raw/l1_regate/` and `processed/l1_regate/overhead_gate.json` would have contained paired deltas.

Design is preserved in `LOCKED_INDEPENDENT_STATIONARITY_PLAN.md §6` and `run_stage3md.py:regate_specs_md()` for a future re-run **only if** Protocol D ever achieves STATIONARY.

---

## 3. Why Skipping Is Correct

Per §6.4 Environment vs Instrumentation separation: instrumentation effect can only be isolated if baseline **L0→L0 variance** is small. Phase A shows L0→L0 p95_ttft variance at c4 is **29.9% rel_range, CV 16.1%**, which dwarfs any plausible L1 effect (prior mock L1 was 3.9% median). Any observed L1−OFF delta inside 5% would be indistinguishable from noise; an 8–15% delta could be either environment or tracing. Reporting an overhead gate under these conditions would be **misleading** and violates pre-registered gate integrity.

Stage 3M-B had same issue (c4 TTFT 67% variance) and correctly skipped Phase B. This independent session replicates that pattern with Protocol D: TTFT still dominant variance source, even though throughput and latency are stable (CV 0.6–2%).

---

## 4. Prior L1 Reference (Not Phase B Evidence)

For context only (pre-stabilization and candidate session, not independent evidence):

- **Mock L1 (ideal host, prior):** PASS — c1 p95 TTFT median +3.9% (≤5%), individual ≤5.2% (≤10%), multiplier -3.58% (≤10%).
- **Candidate Protocol D at PID 56704 (Stage 3M-C):** Claimed c4 median TTFT CV 1.29%, p95 CV 1.55%, central ≤5% and tail ≤10% PASS, no drift. **Not reproduced** in new PID 34892 where CVs are 10.5% and 16.1%.
- No independent L1 data exists for new PID; previous real L1 data (40-warm, per_request or pooled) cannot be promoted to Phase B because host was different PID and not proven stationary.

---

## 5. Artifacts

```
research/measurement_repair/independent_validation/raw/l1_regate/        ← empty (no Phase B runs collected, per spec §20)
research/measurement_repair/independent_validation/processed/l1_regate/  ← no overhead_gate.json produced this phase
```

Previous gate artifacts retained at `research/measurement_repair/raw/regate/` (empty in 3M-B) and at `research/measurement_repair/processed/stationarity/` (old sessions) for audit, but not promoted.

---

## 6. Next Step

Phase B re-gate is **blocked** until L0 stationarity achieves **STATIONARY — PASS** under Protocol D (or a further modified protocol). See `stage3md_summary.md §7` for minimal next validation experiment.

*— End regate (skipped, pre-registered) —*
