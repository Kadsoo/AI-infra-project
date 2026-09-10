# Stage 3R-B Anomaly Index — Empirical Observations

> **Date:** 2026-08-28
> **Engine:** `hf_transformers_naive` (tiny-gpt2, CPU, `6c78dce7b294`)
> **Hardware:** i7-14650HX, 31.78GB, RTX 4060 Laptop 8GB WDDM, driver 596.21
> **Status:** 5 candidate anomalies, 5 reviewed, 3 VALID, 1 PROBABLY VALID, 1 ARTIFACT for latency (queue part VALID)
> **Cross-system:** All are `tiny-cpu` single-system (vLLM/SGLang BLOCKED on Windows, see `environment.md:§4`); no ranking reversal across systems can be claimed.

| ID | Observation (facts only) | System | Effect (vs baseline) | Reproduced (N reps, CV) | Realism | Reviewer Status |
|---|---|---|---|---|---|---|
| A_01 | Concurrency 1→8 (synthetic 512/64, closed, 40 req) thr plateau + tail explosion: thr 0.673→0.873 (+30% then flat -1.4% 4→8), p50 0.619→5.817 ×9.4, p95 0.669→8.829 ×13.2, TTFT 0.06→2.84 ×47, TPOT ×12. At 1→64 thr +4% p50 ×42.5. Fine sweep 1→16 thr plateau at c≈3, p50 0.634→11.217 ×17.7. | hf_transformers_naive (tiny-cpu) | p50 9.4× (1→8), 42.5× (1→64), thr -1.4% 4→8 (scaling failure), tail 13–237× | 3× fresh per point, CV thr <2%, CV p50 <6% — **REPRODUCED**; also seen in baseline, sweepA, fineConc (4 datasets) | Realistic (conc 4–8, 512/64 typical, closed-loop standard) | **VALID** |
| A_02 | Output 16→256 (512 in, conc2, 20 req) thr -60.7% (1.071→0.421), p50 ×3.73 (1.049→3.910), p95 ×2.43, tok thr ×6.3 (17.1→107.7), TTFT flat 0.39→0.43, TPOT flat 0.011 | hf_transformers_naive (tiny-cpu) | thr -60%, latency +273%, token thr +530%, TTFT flat | 3× per point, CV thr 0.2–4.7%, CV p50 <1.1% — **REPRODUCED** | Realistic (output 32–256 typical chat) | **VALID** |
| A_03 | Resource mismatch: tail explodes 17–42× (conc 1→16/64) while CPU stays 30–46% (+44% relative vs 4250% tail), RAM +6% (16.6→17.6GB), GPU flat 0–2%. Same mismatch for output sweep CPU +42% vs thr -60% vs tok +530%. | hf_transformers_naive (tiny-cpu) | Tail 1670–4250% vs CPU -5% to +44% — 100× mismatch | Pattern across 30+ runs, 3 sessions — **REPRODUCED** (within-session CV <7% for CPU) | Realistic (metrics are standard) but engine-specific (CPU inference, WDDM) | **VALID** (as instrumentation gap) |
| A_04 | Arrival pattern (512/64, conc8, 40 req) closed vs bursty r2: coarse single hinted thr -13.6% p50 -31% queue -81%; 3× repro shows thr -1.1% (0.897→0.887, CV 2%), p50 -0.8% (5.68→5.63, CV 5.9%) within noise, queue -66% (20.22→6.89, CV 57%) — only queue effect remains, latency effect not reproduced. Other distributions (poisson/gamma/uniform) flat ±2% thr ±3% latency vs closed. | hf_transformers_naive (tiny-cpu) | Coarse: p50 -31% (not reproduced), repro: p50 -0.8% (n.s.), queue -66% (repro but high variance) | **NOT REPRODUCED for latency/throughput** (3×, CV > effect), PROBABLY VALID for queue | Realistic (poisson/bursty open-loop) | **ARTIFACT for latency claim** / **PROBABLY VALID for queue** |
| A_05 | Mixed workload (20 req, conc4/8): 10×128/32 +10×1024/256 interleaved vs homog_long (20×1024/256) thr +70% (0.485→0.826 c4, 0.483→0.822 c8), p50 -54% overall, long-class p50 -31% (6.546→4.512 c4) and -38% (12.815→7.915 c8); short-class in mixed vs homog_short -12% (c4) / -6% (c8) — short not harmed, long benefits. Vs homog_short, mixed thr -27% p50 +34% — between. | hf_transformers_naive (tiny-cpu) | Long thr +70%, latency -31–38%, short -6–12% (no harm) | 2 conc points same direction, single-run + 3× homog (CV <2%) — **PROBABLY VALID** (needs 3× per-class) | Realistic (short/long mix is RAG-like, 1:1 synthetic but plausible) | **PROBABLY VALID** |

**Legend:**
- **VALID:** stable, large effect >> noise, realistic, no measurement artifact found, raw traceable, reproduced with fresh process.
- **PROBABLY VALID:** directionally stable but needs 3× per-class/full gate or high variance (queue 57% CV) — treat as suggestive.
- **ARTIFACT:** initial effect disappears after 3× repro within noise — not a real anomaly.

**Cross-system column:** All `hf_transformers_naive (tiny-cpu)` — **System-Specific** (cannot claim shared vs system-specific without vLLM; vLLM BLOCKED). Must be re-anchored on Linux A100 with `engine: vllm` before claiming general serving bottleneck.

**For Stage 3R-C:** Only **VALID** (A_01, A_02, A_03) are strong enough for direct follow-up with instrumentation; **PROBABLY VALID** (A_05) is useful as negative for RQ-4 (shows isolation failure absent); **ARTIFACT** (A_04 latency) should not enter RQ generation.

Raw: `sweeps/raw/*_processed.json`, `sweeps/raw/*_requests.csv`, `sweeps/raw/*_system.csv`, configs in `sweeps/configs/`, server log `logs/server_out.log`, environment_hash `6c78dce7b294` in every run.

