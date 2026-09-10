# Stationarity Results — Phase A (L0 OFF)

> **Date:** 2026-08-28T15:27:41.585097+00:00  
> **Classification:** NON-STATIONARY  

## 1. Summary

**Overall:** NON-STATIONARY — FAIL  
**Failed checks:**
- c1 median_ttft central rel_range 5.4% >5%
- c1 p95_ttft tail rel_range 10.3% >10% (partial)
- c1 median_lat central rel_range 7.4% >5%
- c1 thr central rel_range 5.4% >5%
- c1 tok_thr central rel_range 5.4% >5%
- c4 median_ttft central rel_range 67.7% >5%
- c4 p95_ttft tail rel_range 36.4% >15% (limit 10% strict, 15% fail)
- c4 p95_ttft single outlier 29.6% >20%

## 2. Per-Concurrency Metrics

### c=1 — 6 valid / 0 invalid

| Metric | Median | Mean | Min | Max | Range | Rel Range | CV | Drift slope | Rho | Gate |
|---|---|---|---|---|---|---|---|---|---|---|
| median_ttft | 0.0605 | 0.0610 | 0.0601 | 0.0633 | 0.0032 | 5.35% | 2.09% | -0.000263 | -0.62 | FAIL |
| p95_ttft | 0.0761 | 0.0749 | 0.0700 | 0.0778 | 0.0078 | 10.28% | 4.09% | -0.000258 | -0.25 | PARTIAL |
| median_lat | 0.4123 | 0.4200 | 0.4080 | 0.4385 | 0.0305 | 7.40% | 3.39% | -0.003300 | -0.70 | FAIL |
| p95_lat | 0.4391 | 0.4443 | 0.4356 | 0.4579 | 0.0223 | 5.08% | 2.27% | -0.002224 | -0.67 | PASS |
| thr | 2.3822 | 2.3469 | 2.2622 | 2.3920 | 0.1298 | 5.45% | 2.61% | 0.014489 | 0.71 | FAIL |
| tok_thr | 152.4618 | 150.2047 | 144.7828 | 153.0883 | 8.3054 | 5.45% | 2.61% | 0.927280 | 0.71 | FAIL |

**Raw per-run values:**

| Order | Seed | Round | median_ttft | p95_ttft | median_lat | p95_lat | throughput | invalid |
|---|---|---|---|---|---|---|---|---|
| 1 | 4101 | 1 | 0.0617 | 0.0760 | 0.4385 | 0.4563 | 2.2622 | False |
| 2 | 4102 | 1 | 0.0633 | 0.0778 | 0.4380 | 0.4579 | 2.2743 | False |
| 5 | 4102 | 2 | 0.0601 | 0.0700 | 0.4120 | 0.4376 | 2.3920 | False |
| 6 | 4101 | 2 | 0.0607 | 0.0762 | 0.4080 | 0.4356 | 2.3800 | False |
| 9 | 4101 | 3 | 0.0601 | 0.0769 | 0.4125 | 0.4403 | 2.3844 | False |
| 10 | 4102 | 3 | 0.0603 | 0.0722 | 0.4107 | 0.4378 | 2.3888 | False |

### c=4 — 6 valid / 0 invalid

| Metric | Median | Mean | Min | Max | Range | Rel Range | CV | Drift slope | Rho | Gate |
|---|---|---|---|---|---|---|---|---|---|---|
| median_ttft | 0.0934 | 0.0989 | 0.0765 | 0.1397 | 0.0632 | 67.71% | 22.81% | 0.001830 | 0.24 | FAIL |
| p95_ttft | 0.1664 | 0.1740 | 0.1551 | 0.2157 | 0.0606 | 36.42% | 13.19% | 0.000929 | 0.12 | FAIL |
| median_lat | 1.0586 | 1.0532 | 1.0254 | 1.0723 | 0.0468 | 4.42% | 1.85% | 0.002099 | 0.32 | PASS |
| p95_lat | 1.1389 | 1.1411 | 1.1165 | 1.1745 | 0.0580 | 5.09% | 1.67% | 0.000155 | 0.02 | PASS |
| thr | 3.6950 | 3.7022 | 3.6523 | 3.7610 | 0.1087 | 2.94% | 1.40% | -0.001752 | -0.10 | PASS |
| tok_thr | 236.4824 | 236.9430 | 233.7472 | 240.7025 | 6.9553 | 2.94% | 1.40% | -0.112113 | -0.10 | PASS |

**Raw per-run values:**

| Order | Seed | Round | median_ttft | p95_ttft | median_lat | p95_lat | throughput | invalid |
|---|---|---|---|---|---|---|---|---|
| 3 | 4401 | 1 | 0.1054 | 0.1828 | 1.0723 | 1.1449 | 3.6535 | False |
| 4 | 4402 | 1 | 0.0851 | 0.1606 | 1.0254 | 1.1165 | 3.7566 | False |
| 7 | 4402 | 2 | 0.0765 | 0.1722 | 1.0347 | 1.1745 | 3.7610 | False |
| 8 | 4401 | 2 | 0.0998 | 0.1574 | 1.0516 | 1.1397 | 3.6523 | False |
| 11 | 4401 | 3 | 0.1397 | 0.2157 | 1.0656 | 1.1331 | 3.7271 | False |
| 12 | 4402 | 3 | 0.0870 | 0.1551 | 1.0697 | 1.1382 | 3.6630 | False |


## 3. Variance Decomposition (Coarse)

Seed vs repetition vs order variances are estimated by comparing groups. Largest contributor is flagged.

- c=1 p95_ttft by seed:  seed 4101 mean 0.0764 n=3; seed 4102 mean 0.0734 n=3;
  - order correlation rho=-0.25 for p95_ttft
- c=4 p95_ttft by seed:  seed 4401 mean 0.1853 n=3; seed 4402 mean 0.1626 n=3;
  - order correlation rho=0.12 for p95_ttft

## 4. Environment vs Instrumentation Note

Phase A contains only L0→L0 variance, i.e., pure environment noise floor without tracing. This is the baseline against which any Phase B instrumentation effect must be judged.

## 5. Artifacts

```
research/measurement_repair/raw/stationarity/
research/measurement_repair/processed/stationarity/
```

*— End stationarity —*
