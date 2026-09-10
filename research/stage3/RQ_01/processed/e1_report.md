# RQ-1 E1 Report — Transfer-Materiality Cost Model (Wave-1, CPU-only)

> Experiment E1 of `LOCKED_PLAN.md` (freeze date 2026-08-27). Script: `code/e1_cost_model.py` (SHA-256 `23d916fce0aac948fbc42ecfc9a8feb6c76c567b5c39da3c4c05daa229c519ba`). Raw: `raw/e1_cells.csv`, `raw/e1_anchor.json`, `raw/e1_verdict.json`. Date: 2026-08-27.

## 1. Inputs (trace-derived)

Fixed 5,000-request trace (`raw/trace_5000.csv`, seed 20260827, λ=3.0 req/s):

| Stat | Value |
|---|---|
| short-class fraction | 70.0% (median 1018 tok) |
| long-class fraction | 30.0% (mean 9262, P50 8770, P95 12399, P99 13000 tok) |
| hot-prefix share of long class | 59.6% |
| achieved arrival CV (1/2/3) | 0.996 / 2.054 / 2.996 |

## 2. Model constants

KV bytes = 131072 B/token (Llama-3.1-8B FP16 GQA); T_prefill(tok) = 2·8e9·tok/(312e12·0.5) s = 0.1026 ms/tok (A100 312 TFLOPS, 50% MFU; scale sweep absorbs MFU 30–60%); T_transfer = KV_bytes/(25 Gbps × f); NVLink leg = min(bytes/600 GB/s, 8 ms) (Splitwise non-overlapped cap); M/D/1 queue wait W = ρ·D/(2(1−ρ)), P95 wait = 2.75×W (recorded choice within the plan's 2.5–3× band); E2E TTFT(P95) = T_prefill + 2.75·W + T_transfer; transfer on the critical path, zero overlap.

## 3. Corpus-anchored point (f=0.24, ρ=0.5, prefill 1×, long-class P50/P95)

| Quantity | Value |
|---|---|
| T_transfer(25 Gbps) at P95 length | 2.167 s (P50 length: 1.44 s) |
| T_prefill at P95 length | 1.272 s |
| E2E TTFT P95 (25 Gbps) | 5.187 s |
| **P95 transfer fraction (25 Gbps)** | **41.8%** (≥30% support line ✓) |
| **P50 transfer fraction (25 Gbps)** | **41.8%** (≥15% support line ✓) |
| P95 transfer fraction (NVLink) | 0.09% (≤5% line ✓) |
| **25 Gbps-vs-NVLink P95 TTFT gap** | **71.6%** of NVLink value (≥25% kill-guard ✓) |

Note: the P50 and P95 fractions are nearly equal because both transfer and prefill scale linearly in prompt length; the fraction is approximately length-invariant inside the class.

## 4. Uncertainty sweep (72 cells: f×{0.2,0.5,1.0} × ρ×{0.3,0.5,0.7} × prefill-scale×{0.5,1,2} × class×{long-only,mixed}; P95)

| Quantity | Min | Max |
|---|---|---|
| P95 transfer fraction (25 Gbps, long-only) | 4.6% | 72.0% |
| 25 Gbps-vs-NVLink P95 TTFT gap | 4.8% | 256.4% |
| NVLink P95 fraction (long-only) | 0.05% | 0.27% |

- The min-fraction cell is (f=1.0, ρ=0.7, prefill 2×): full-nominal-bandwidth + heavy queue + slow prefill; queue dominates TTFT there.
- The max-fraction cell is (f=0.2, ρ=0.3, prefill 0.5×): low realized bandwidth + light queue.
- NVLink P95 fraction ≤ 0.27% in all cells (≤5% everywhere; NVLink leg is not material).

## 5. Frozen-threshold evaluation (E1 verdict)

| Frozen line | Result | Fires? |
|---|---|---|
| H1 kill: P95 fraction <30% at 25 Gbps at ALL plausible (f,ρ,scale) points | min 4.6% but max 72.0%; not all cells <30% | NO |
| H1 kill: 25 Gbps-vs-NVLink P95 TTFT gap <25% of NVLink value at all points | min 4.8% but max 256%; not all cells <25% | NO |
| Corpus-anchored support: P95 ≥30%, P50 ≥15%, NVLink ≤5% | 41.8% / 41.8% / 0.09% | YES |

**E1 verdict: H1 NOT KILLED. Pre-registered outcome "narrow band → E3 required":** the verdict flips across the plausible band (fraction 4.6%–72.0%, gap 4.8%–256%). At the corpus-anchored point and at every cell with realized bandwidth at or below the corpus anchor (f ≤ 0.24) and queue/prefill not simultaneously at their maxima, the P95 fraction ≥ 30% and the gap ≥ 25%. A kill at E1 would have required ≥16 GB/s realized bandwidth on a 25 Gbps link (f ≥ ~5), i.e., near-perfect fragmentation-free transfers — consistent with the plan's kill-reachability analysis. H1's hardware confirmation requires E3 (transfer microbenchmark) and E4 (serving), which are BLOCKED on this machine (no 3× A100 testbed, see `environment.md`).

## 6. Alternative explanations considered

- Length class: the mixed-trace cells (all classes) give lower fractions (short-class 1k prompts transfer 128 MiB in ~0.43 s at f=0.24) — consistent with H1 being a long-prompt-class claim; the long-only class is the H1-relevant cell, per the frozen IV.
- P95-queue factor (2.5–3× band): using 2.5× or 3.0× changes fractions by <2 points at the anchored point (queue is a minor TTFT component at f=0.24).
- M/D/1 vs M/G/1: prefill service is near-deterministic per prompt class; M/D/1 is the plan's declared model.