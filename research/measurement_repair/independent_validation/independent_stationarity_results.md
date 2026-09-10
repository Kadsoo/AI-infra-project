# Independent Stationarity Results — Phase A (L0 OFF, Protocol D, New PID)

> **Date:** 2026-08-29T07:45:21.392539+00:00  
> **Classification:** NON-STATIONARY  
> **Session:** independent-3md-01  
> **Protocol D:** Frozen — 1×c4 n40 warmup + gc.collect + sleep2s  

## 1. Summary

**Overall:** NON-STATIONARY — FAIL  
**Failed checks:**
- c1 median_ttft central rel_range 5.6% >5%
- c4 median_ttft central rel_range 20.3% >5%
- c4 median_ttft chronological drift rho 0.62 slope 0.0030578326923076916 significant
- c4 p95_ttft tail rel_range 29.9% >15% (limit 10% strict, 15% fail)
- c4 p95_ttft single outlier 21.3% >20%

**Serving PID (new):** 34892 (distinct from Stage 3M-C 56704 and Stage 3M-B 19372)  
**Port:** 8040 (isolated)  

## 2. Per-Concurrency Metrics

### c=1 — 3 valid / 0 invalid (Protocol D, L0 OFF)

| Metric | Median | Mean | Min | Max | Range | Rel Range | CV | Max Pairwise Δ | Drift slope | Rho | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| median_ttft | 0.0486 | 0.0479 | 0.0462 | 0.0489 | 0.0027 | 5.57% | 3.10% | 4.94% | -0.000267 | -0.25 | FAIL |
| p95_ttft | 0.0641 | 0.0631 | 0.0604 | 0.0647 | 0.0043 | 6.71% | 3.69% | 5.76% | -0.000167 | -0.10 | PASS |
| median_lat | 0.2615 | 0.2614 | 0.2604 | 0.2624 | 0.0019 | 0.74% | 0.37% | 0.41% | -0.000316 | -0.45 | PASS |
| p95_lat | 0.2831 | 0.2811 | 0.2756 | 0.2847 | 0.0092 | 3.24% | 1.74% | 2.67% | -0.000268 | -0.08 | PASS |
| thr | 3.7518 | 3.7464 | 3.7210 | 3.7665 | 0.0455 | 1.21% | 0.62% | 0.82% | -0.004781 | -0.29 | PASS |
| tok_thr | 240.1121 | 239.7706 | 238.1428 | 241.0570 | 2.9142 | 1.21% | 0.62% | 0.82% | -0.306009 | -0.29 | PASS |

**Raw per-run values (Protocol D):**

| Order | Seed | Round | Warmup Seed | median_ttft | p95_ttft | median_lat | p95_lat | throughput | tok_thr | RSS pre (MB) | Threads | invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5101 | 1 | 8001 | 0.0489 | 0.0641 | 0.2624 | 0.2831 | 3.7518 | 240.1 | 495 | 185 | False |
| 4 | 5102 | 2 | 8004 | 0.0462 | 0.0604 | 0.2604 | 0.2756 | 3.7665 | 241.1 | 489 | 184 | False |
| 5 | 5103 | 3 | 8005 | 0.0486 | 0.0647 | 0.2615 | 0.2847 | 3.7210 | 238.1 | 496 | 184 | False |

### c=4 — 3 valid / 0 invalid (Protocol D, L0 OFF)

| Metric | Median | Mean | Min | Max | Range | Rel Range | CV | Max Pairwise Δ | Drift slope | Rho | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| median_ttft | 0.0615 | 0.0646 | 0.0600 | 0.0724 | 0.0125 | 20.29% | 10.52% | 17.73% | 0.003058 | 0.62 | FAIL |
| p95_ttft | 0.1307 | 0.1252 | 0.1029 | 0.1420 | 0.0391 | 29.88% | 16.07% | 21.29% | 0.004490 | 0.31 | FAIL |
| median_lat | 0.8894 | 0.8887 | 0.8780 | 0.8986 | 0.0206 | 2.31% | 1.16% | 1.28% | -0.000884 | -0.12 | PASS |
| p95_lat | 1.0024 | 1.0090 | 0.9929 | 1.0318 | 0.0388 | 3.87% | 2.01% | 2.93% | -0.000439 | -0.03 | PASS |
| thr | 4.3163 | 4.3371 | 4.3040 | 4.3911 | 0.0872 | 2.02% | 1.09% | 1.73% | -0.013444 | -0.40 | PASS |
| tok_thr | 276.2426 | 277.5761 | 275.4531 | 281.0326 | 5.5795 | 2.02% | 1.09% | 1.73% | -0.860431 | -0.40 | PASS |

**Raw per-run values (Protocol D):**

| Order | Seed | Round | Warmup Seed | median_ttft | p95_ttft | median_lat | p95_lat | throughput | tok_thr | RSS pre (MB) | Threads | invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 5401 | 1 | 8002 | 0.0615 | 0.1029 | 0.8986 | 0.9929 | 4.3911 | 281.0 | 674 | 185 | False |
| 3 | 5402 | 2 | 8003 | 0.0600 | 0.1420 | 0.8780 | 1.0318 | 4.3040 | 275.5 | 493 | 183 | False |
| 6 | 5403 | 3 | 8006 | 0.0724 | 0.1307 | 0.8894 | 1.0024 | 4.3163 | 276.2 | 679 | 183 | False |


## 3. Workload Token Comparability

- l0-independent-3md-01_01_c1_seed5101_off_o1: input {'mean': 511.075, 'min': 510, 'max': 512} / output None / seed 5101 / described prompt_hash ? / total_tokens measured ?
- l0-independent-3md-01_02_c4_seed5401_off_o2: input {'mean': 510.8, 'min': 510, 'max': 512} / output None / seed 5401 / described prompt_hash ? / total_tokens measured ?
- l0-independent-3md-01_03_c4_seed5402_off_o3: input {'mean': 511.075, 'min': 510, 'max': 512} / output None / seed 5402 / described prompt_hash ? / total_tokens measured ?
- l0-independent-3md-01_04_c1_seed5102_off_o4: input {'mean': 511.175, 'min': 510, 'max': 512} / output None / seed 5102 / described prompt_hash ? / total_tokens measured ?
- l0-independent-3md-01_05_c1_seed5103_off_o5: input {'mean': 511.2, 'min': 510, 'max': 512} / output None / seed 5103 / described prompt_hash ? / total_tokens measured ?
- l0-independent-3md-01_06_c4_seed5403_off_o6: input {'mean': 511.025, 'min': 510, 'max': 512} / output None / seed 5403 / described prompt_hash ? / total_tokens measured ?

## 4. Host State Validation (Protocol D Convergence)

| Run | Conc | Seed | Order | Threads before | Threads after | RSS before (MB) | RSS after (MB) | RSS after GC sleep (MB) | CPU % | GPU util % | invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| l0-independent-3md-0 | 1 | 5101 | 1 | 185 | 185 | 495 | 498 | 495 | 13.7 | 14.0 | False |
| l0-independent-3md-0 | 4 | 5401 | 2 | 185 | 183 | 674 | 673 | 674 | 19.5 | 14.0 | False |
| l0-independent-3md-0 | 4 | 5402 | 3 | 183 | 184 | 493 | 495 | 493 | 12.2 | 15.0 | False |
| l0-independent-3md-0 | 1 | 5102 | 4 | 184 | 184 | 489 | 683 | 489 | 8.0 | 14.0 | False |
| l0-independent-3md-0 | 1 | 5103 | 5 | 184 | 184 | 496 | 504 | 496 | 7.0 | 15.0 | False |
| l0-independent-3md-0 | 4 | 5403 | 6 | 183 | 183 | 679 | 673 | 679 | 17.8 | 13.0 | False |

**Thread convergence:** Check Δ threads ≤5 across formals; **RSS convergence:** Check pre-run RSS range/CV.

**Pre-run RSS stats:** median 495 MB, range 190 MB, rel_range 38.37%, CV 17.08%
**RSS vs p95_TTFT correlation (Pearson):** 0.49 (do NOT claim causality; report only stability)

## 5. Variance Decomposition

- c=1 p95_ttft by seed:  seed 5101 mean 0.0641 n=1; seed 5102 mean 0.0604 n=1; seed 5103 mean 0.0647 n=1;
  - order correlation rho=-0.10 for p95_ttft
- c=4 p95_ttft by seed:  seed 5401 mean 0.1029 n=1; seed 5402 mean 0.1420 n=1; seed 5403 mean 0.1307 n=1;
  - order correlation rho=0.31 for p95_ttft

## 6. Artifacts

```
research/measurement_repair/independent_validation/raw/l0/
research/measurement_repair/independent_validation/processed/l0/
```

*— End independent stationarity —*
