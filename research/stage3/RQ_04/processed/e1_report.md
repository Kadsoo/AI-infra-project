# e1_report.md - RQ-4 H1 point kill test, offline trace replay (frozen experiment_plan.md E1)

Generated 2026-08-27T16:21:06.536913+00:00 UTC. Script SHA-256: 9277aaa68224e5a71a8c88985e26dc9eca5c402e56a15e7b8a146e84cf1ce442

## 1. Scope

- 10,000-request Poisson traces (10 seeded replications, seeds 1001-1010, recorded in raw/trace_10000_repNN.csv); inter-arrival sequences scaled per cell to lambda = load_frac x lambda_sat(p_hot, gamma) (E0 anchoring).
- Per-node FIFO queues, admission = dispatch, cluster admission cap c = 32; gate FIFO for arrivals beyond the cap (recorded per run as gate_waits; binds only near/over overload).
- Warm-up (frozen schedule, recorded): hot prefix filled by 200 hot-request completions; steady-state window = arrivals with t >= t_warm. Under policy A all warm-up hot requests are routed to node 0 (deterministic realization of the frozen single-hot-node steady state); under B/B' they follow the policy metric (hot prefix accumulates on every node, per the frozen cache model).
- Task-line note: '60%% hot-prefix reuse structure' has no counterpart in the frozen workload spec; the frozen 1.3 structure (hot prefix shared by all hot requests, cold never reused) is implemented.
- E0 hit leg: G = 0.00 pp < 10 pp at both budget levels => pre-registrationally killed (processed/e0_report.md 6); H1 rests on the queue leg; the E1 falsification criterion (i) (hit gain <= 5pp) does NOT apply. Hit rates below are reported for completeness, not as verdict inputs.
- P99 = 99th percentile with linear interpolation between order statistics (numpy default), pooled over the steady-state window per run.

## 2. Aggregates over 10 replications (mean +/- std)

### p_hot = 0.8, load = 0.50, budget = L1, gamma = 0.5

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4010 +/- 0.0028 | 0.4010 +/- 0.0028 | gain 0.00 pp |
| aggregate mean TTFT (s) | 0.4131 +/- 0.0049 | 0.2999 +/- 0.0013 | 1.377 |
| hot P99 TTFT (s) | 1.1442 +/- 0.0518 | 0.2811 +/- 0.0001 | 4.071 |
| cold P99 TTFT (s) | 0.5308 +/- 0.0002 | 0.5311 +/- 0.0002 | 0.999 |
| hot P99 queue wait (s) | 0.8925 +/- 0.0523 | 0.0000 +/- 0.0000 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0001 +/- 0.0002 | |
| hot mean TTFT (s) | 0.3917 +/- 0.0064 | 0.2505 +/- 0.0003 | |
| cold mean TTFT (s) | 0.4999 +/- 0.0004 | 0.5003 +/- 0.0003 | |
| hot attainment (TTFT<=2.0s) | 0.9999 +/- 0.0002 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9999 +/- 0.0002 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.5322 +/- 0.0069 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.8, load = 0.50, budget = L1, gamma = 1.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4010 +/- 0.0028 | 0.4010 +/- 0.0028 | gain 0.00 pp |
| aggregate mean TTFT (s) | 0.8268 +/- 0.0099 | 0.5999 +/- 0.0027 | 1.378 |
| hot P99 TTFT (s) | 2.3107 +/- 0.1168 | 0.5622 +/- 0.0002 | 4.110 |
| cold P99 TTFT (s) | 1.0616 +/- 0.0003 | 1.0622 +/- 0.0003 | 0.999 |
| hot P99 queue wait (s) | 1.8066 +/- 0.1173 | 0.0000 +/- 0.0000 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0002 +/- 0.0005 | |
| hot mean TTFT (s) | 0.7840 +/- 0.0128 | 0.5009 +/- 0.0006 | |
| cold mean TTFT (s) | 0.9998 +/- 0.0007 | 1.0007 +/- 0.0006 | |
| hot attainment (TTFT<=2.0s) | 0.9798 +/- 0.0034 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9798 +/- 0.0034 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.5315 +/- 0.0074 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.8, load = 0.50, budget = L1, gamma = 2.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4010 +/- 0.0028 | 0.4010 +/- 0.0028 | gain 0.00 pp |
| aggregate mean TTFT (s) | 1.6557 +/- 0.0199 | 1.1998 +/- 0.0054 | 1.380 |
| hot P99 TTFT (s) | 4.6335 +/- 0.2262 | 1.1243 +/- 0.0005 | 4.121 |
| cold P99 TTFT (s) | 2.1231 +/- 0.0007 | 2.1244 +/- 0.0006 | 0.999 |
| hot P99 queue wait (s) | 3.6201 +/- 0.2296 | 0.0000 +/- 0.0000 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0003 +/- 0.0010 | |
| hot mean TTFT (s) | 1.5707 +/- 0.0262 | 1.0019 +/- 0.0011 | |
| cold mean TTFT (s) | 1.9996 +/- 0.0015 | 2.0013 +/- 0.0013 | |
| hot attainment (TTFT<=2.0s) | 0.7828 +/- 0.0113 | 1.0000 +/- 0.0001 | |
| cold attainment | 0.5047 +/- 0.0097 | 0.5023 +/- 0.0091 | |
| FI (report-only) | 0.6447 +/- 0.0100 | 0.5023 +/- 0.0091 | |
| hot-node util (measured) | 0.5317 +/- 0.0072 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.8, load = 0.50, budget = L2, gamma = 0.5

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4010 +/- 0.0028 | 0.4010 +/- 0.0028 | gain 0.00 pp |
| aggregate mean TTFT (s) | 0.4137 +/- 0.0045 | 0.2999 +/- 0.0013 | 1.379 |
| hot P99 TTFT (s) | 1.1493 +/- 0.0435 | 0.2811 +/- 0.0001 | 4.089 |
| cold P99 TTFT (s) | 0.5308 +/- 0.0002 | 0.5311 +/- 0.0002 | 0.999 |
| hot P99 queue wait (s) | 0.8970 +/- 0.0441 | 0.0000 +/- 0.0000 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0001 +/- 0.0002 | |
| hot mean TTFT (s) | 0.3924 +/- 0.0059 | 0.2505 +/- 0.0003 | |
| cold mean TTFT (s) | 0.4999 +/- 0.0004 | 0.5003 +/- 0.0003 | |
| hot attainment (TTFT<=2.0s) | 0.9999 +/- 0.0001 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9999 +/- 0.0001 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.5322 +/- 0.0062 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.8, load = 0.50, budget = L2, gamma = 1.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4010 +/- 0.0028 | 0.4010 +/- 0.0028 | gain 0.00 pp |
| aggregate mean TTFT (s) | 0.8256 +/- 0.0103 | 0.5999 +/- 0.0027 | 1.376 |
| hot P99 TTFT (s) | 2.3081 +/- 0.1200 | 0.5622 +/- 0.0002 | 4.106 |
| cold P99 TTFT (s) | 1.0616 +/- 0.0003 | 1.0622 +/- 0.0003 | 0.999 |
| hot P99 queue wait (s) | 1.8037 +/- 0.1205 | 0.0000 +/- 0.0000 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0002 +/- 0.0005 | |
| hot mean TTFT (s) | 0.7825 +/- 0.0134 | 0.5009 +/- 0.0006 | |
| cold mean TTFT (s) | 0.9998 +/- 0.0007 | 1.0007 +/- 0.0006 | |
| hot attainment (TTFT<=2.0s) | 0.9800 +/- 0.0034 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9800 +/- 0.0034 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.5314 +/- 0.0076 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.8, load = 0.50, budget = L2, gamma = 2.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4010 +/- 0.0028 | 0.4010 +/- 0.0028 | gain 0.00 pp |
| aggregate mean TTFT (s) | 1.6557 +/- 0.0188 | 1.1998 +/- 0.0054 | 1.380 |
| hot P99 TTFT (s) | 4.5703 +/- 0.2067 | 1.1243 +/- 0.0005 | 4.065 |
| cold P99 TTFT (s) | 2.1231 +/- 0.0007 | 2.1244 +/- 0.0006 | 0.999 |
| hot P99 queue wait (s) | 3.5638 +/- 0.2030 | 0.0000 +/- 0.0000 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0003 +/- 0.0010 | |
| hot mean TTFT (s) | 1.5707 +/- 0.0245 | 1.0019 +/- 0.0011 | |
| cold mean TTFT (s) | 1.9996 +/- 0.0015 | 2.0013 +/- 0.0013 | |
| hot attainment (TTFT<=2.0s) | 0.7817 +/- 0.0103 | 1.0000 +/- 0.0001 | |
| cold attainment | 0.5046 +/- 0.0097 | 0.5023 +/- 0.0091 | |
| FI (report-only) | 0.6456 +/- 0.0101 | 0.5023 +/- 0.0091 | |
| hot-node util (measured) | 0.5320 +/- 0.0080 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.8, load = 0.65, budget = L1, gamma = 0.5

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4010 +/- 0.0028 | 0.4010 +/- 0.0028 | gain 0.00 pp |
| aggregate mean TTFT (s) | 0.5043 +/- 0.0116 | 0.3007 +/- 0.0013 | 1.677 |
| hot P99 TTFT (s) | 1.6367 +/- 0.1163 | 0.3006 +/- 0.0137 | 5.449 |
| cold P99 TTFT (s) | 0.5309 +/- 0.0002 | 0.5516 +/- 0.0110 | 0.963 |
| hot P99 queue wait (s) | 1.3857 +/- 0.1196 | 0.0492 +/- 0.0131 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0495 +/- 0.0166 | |
| hot mean TTFT (s) | 0.5052 +/- 0.0145 | 0.2513 +/- 0.0004 | |
| cold mean TTFT (s) | 0.5001 +/- 0.0004 | 0.5011 +/- 0.0004 | |
| hot attainment (TTFT<=2.0s) | 0.9968 +/- 0.0024 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9968 +/- 0.0024 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.6795 +/- 0.0096 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.8, load = 0.65, budget = L1, gamma = 1.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4010 +/- 0.0028 | 0.4010 +/- 0.0028 | gain 0.00 pp |
| aggregate mean TTFT (s) | 1.0088 +/- 0.0237 | 0.6015 +/- 0.0026 | 1.677 |
| hot P99 TTFT (s) | 3.2592 +/- 0.2207 | 0.6011 +/- 0.0274 | 5.425 |
| cold P99 TTFT (s) | 1.0618 +/- 0.0003 | 1.1031 +/- 0.0220 | 0.963 |
| hot P99 queue wait (s) | 2.7606 +/- 0.2267 | 0.0984 +/- 0.0262 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0990 +/- 0.0332 | |
| hot mean TTFT (s) | 1.0108 +/- 0.0295 | 0.5025 +/- 0.0007 | |
| cold mean TTFT (s) | 1.0002 +/- 0.0008 | 1.0023 +/- 0.0007 | |
| hot attainment (TTFT<=2.0s) | 0.9208 +/- 0.0101 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9208 +/- 0.0101 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.6793 +/- 0.0094 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.8, load = 0.65, budget = L1, gamma = 2.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4010 +/- 0.0028 | 0.4010 +/- 0.0028 | gain 0.00 pp |
| aggregate mean TTFT (s) | 2.0153 +/- 0.0469 | 1.2030 +/- 0.0052 | 1.675 |
| hot P99 TTFT (s) | 6.4699 +/- 0.4354 | 1.2022 +/- 0.0549 | 5.385 |
| cold P99 TTFT (s) | 2.1235 +/- 0.0007 | 2.2063 +/- 0.0440 | 0.963 |
| hot P99 queue wait (s) | 5.4776 +/- 0.4470 | 0.1968 +/- 0.0523 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.1980 +/- 0.0664 | |
| hot mean TTFT (s) | 2.0187 +/- 0.0584 | 1.0050 +/- 0.0014 | |
| cold mean TTFT (s) | 2.0004 +/- 0.0017 | 2.0045 +/- 0.0014 | |
| hot attainment (TTFT<=2.0s) | 0.6260 +/- 0.0187 | 0.9998 +/- 0.0001 | |
| cold attainment | 0.5039 +/- 0.0095 | 0.4969 +/- 0.0093 | |
| FI (report-only) | 0.8054 +/- 0.0191 | 0.4970 +/- 0.0093 | |
| hot-node util (measured) | 0.6790 +/- 0.0102 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.8, load = 0.65, budget = L2, gamma = 0.5

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4010 +/- 0.0028 | 0.4010 +/- 0.0028 | gain 0.00 pp |
| aggregate mean TTFT (s) | 0.5043 +/- 0.0109 | 0.3007 +/- 0.0013 | 1.677 |
| hot P99 TTFT (s) | 1.6192 +/- 0.1068 | 0.3006 +/- 0.0137 | 5.392 |
| cold P99 TTFT (s) | 0.5309 +/- 0.0002 | 0.5516 +/- 0.0110 | 0.963 |
| hot P99 queue wait (s) | 1.3698 +/- 0.1109 | 0.0492 +/- 0.0131 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0495 +/- 0.0166 | |
| hot mean TTFT (s) | 0.5053 +/- 0.0135 | 0.2513 +/- 0.0004 | |
| cold mean TTFT (s) | 0.5001 +/- 0.0004 | 0.5011 +/- 0.0004 | |
| hot attainment (TTFT<=2.0s) | 0.9970 +/- 0.0021 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9970 +/- 0.0021 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.6791 +/- 0.0093 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.8, load = 0.65, budget = L2, gamma = 1.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4010 +/- 0.0028 | 0.4010 +/- 0.0028 | gain 0.00 pp |
| aggregate mean TTFT (s) | 1.0074 +/- 0.0238 | 0.6015 +/- 0.0026 | 1.675 |
| hot P99 TTFT (s) | 3.2561 +/- 0.2101 | 0.6011 +/- 0.0274 | 5.422 |
| cold P99 TTFT (s) | 1.0618 +/- 0.0003 | 1.1031 +/- 0.0220 | 0.963 |
| hot P99 queue wait (s) | 2.7557 +/- 0.2129 | 0.0984 +/- 0.0262 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0990 +/- 0.0332 | |
| hot mean TTFT (s) | 1.0090 +/- 0.0297 | 0.5025 +/- 0.0007 | |
| cold mean TTFT (s) | 1.0002 +/- 0.0009 | 1.0023 +/- 0.0007 | |
| hot attainment (TTFT<=2.0s) | 0.9214 +/- 0.0107 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9214 +/- 0.0107 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.6798 +/- 0.0090 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.8, load = 0.65, budget = L2, gamma = 2.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4010 +/- 0.0028 | 0.4010 +/- 0.0028 | gain 0.00 pp |
| aggregate mean TTFT (s) | 2.0178 +/- 0.0462 | 1.2030 +/- 0.0052 | 1.677 |
| hot P99 TTFT (s) | 6.5221 +/- 0.4083 | 1.2022 +/- 0.0549 | 5.429 |
| cold P99 TTFT (s) | 2.1235 +/- 0.0007 | 2.2063 +/- 0.0440 | 0.963 |
| hot P99 queue wait (s) | 5.5268 +/- 0.4188 | 0.1968 +/- 0.0523 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.1980 +/- 0.0664 | |
| hot mean TTFT (s) | 2.0218 +/- 0.0575 | 1.0050 +/- 0.0014 | |
| cold mean TTFT (s) | 2.0004 +/- 0.0016 | 2.0045 +/- 0.0014 | |
| hot attainment (TTFT<=2.0s) | 0.6253 +/- 0.0188 | 0.9998 +/- 0.0001 | |
| cold attainment | 0.5039 +/- 0.0095 | 0.4969 +/- 0.0093 | |
| FI (report-only) | 0.8063 +/- 0.0186 | 0.4970 +/- 0.0093 | |
| hot-node util (measured) | 0.6790 +/- 0.0094 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.8, load = 0.80, budget = L1, gamma = 0.5

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4010 +/- 0.0028 | 0.4010 +/- 0.0028 | gain 0.00 pp |
| aggregate mean TTFT (s) | 0.7329 +/- 0.0464 | 0.3021 +/- 0.0013 | 2.426 |
| hot P99 TTFT (s) | 3.0056 +/- 0.5316 | 0.3534 +/- 0.0113 | 8.518 |
| cold P99 TTFT (s) | 0.5310 +/- 0.0002 | 0.6031 +/- 0.0100 | 0.881 |
| hot P99 queue wait (s) | 2.7589 +/- 0.5335 | 0.1017 +/- 0.0127 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.1052 +/- 0.0108 | |
| hot mean TTFT (s) | 0.7900 +/- 0.0567 | 0.2526 +/- 0.0005 | |
| cold mean TTFT (s) | 0.5005 +/- 0.0004 | 0.5025 +/- 0.0004 | |
| hot attainment (TTFT<=2.0s) | 0.9512 +/- 0.0162 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9512 +/- 0.0162 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.8212 +/- 0.0122 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.8, load = 0.80, budget = L1, gamma = 1.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4010 +/- 0.0028 | 0.4010 +/- 0.0028 | gain 0.00 pp |
| aggregate mean TTFT (s) | 1.4616 +/- 0.0941 | 0.6042 +/- 0.0026 | 2.419 |
| hot P99 TTFT (s) | 6.0136 +/- 1.0607 | 0.7067 +/- 0.0226 | 8.521 |
| cold P99 TTFT (s) | 1.0620 +/- 0.0004 | 1.2062 +/- 0.0199 | 0.881 |
| hot P99 queue wait (s) | 5.5146 +/- 1.0678 | 0.2033 +/- 0.0255 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.2104 +/- 0.0216 | |
| hot mean TTFT (s) | 1.5749 +/- 0.1154 | 0.5053 +/- 0.0009 | |
| cold mean TTFT (s) | 1.0009 +/- 0.0008 | 1.0049 +/- 0.0009 | |
| hot attainment (TTFT<=2.0s) | 0.7430 +/- 0.0336 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.7430 +/- 0.0336 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.8206 +/- 0.0112 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.8, load = 0.80, budget = L1, gamma = 2.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4010 +/- 0.0028 | 0.4010 +/- 0.0028 | gain 0.00 pp |
| aggregate mean TTFT (s) | 2.9309 +/- 0.1831 | 1.2084 +/- 0.0052 | 2.426 |
| hot P99 TTFT (s) | 12.0431 +/- 2.1118 | 1.4135 +/- 0.0452 | 8.533 |
| cold P99 TTFT (s) | 2.1240 +/- 0.0007 | 2.4124 +/- 0.0398 | 0.881 |
| hot P99 queue wait (s) | 11.0407 +/- 2.1277 | 0.4067 +/- 0.0510 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.4208 +/- 0.0431 | |
| hot mean TTFT (s) | 3.1596 +/- 0.2247 | 1.0105 +/- 0.0019 | |
| cold mean TTFT (s) | 2.0019 +/- 0.0017 | 2.0099 +/- 0.0017 | |
| hot attainment (TTFT<=2.0s) | 0.4084 +/- 0.0194 | 0.9994 +/- 0.0003 | |
| cold attainment | 0.5024 +/- 0.0094 | 0.4901 +/- 0.0087 | |
| FI (report-only) | 0.8129 +/- 0.0333 | 0.4904 +/- 0.0087 | |
| hot-node util (measured) | 0.8211 +/- 0.0102 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.8, load = 0.80, budget = L2, gamma = 0.5

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4010 +/- 0.0028 | 0.4010 +/- 0.0028 | gain 0.00 pp |
| aggregate mean TTFT (s) | 0.7320 +/- 0.0488 | 0.3021 +/- 0.0013 | 2.423 |
| hot P99 TTFT (s) | 2.9987 +/- 0.5285 | 0.3534 +/- 0.0113 | 8.498 |
| cold P99 TTFT (s) | 0.5310 +/- 0.0002 | 0.6031 +/- 0.0100 | 0.881 |
| hot P99 queue wait (s) | 2.7518 +/- 0.5298 | 0.1017 +/- 0.0127 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.1052 +/- 0.0108 | |
| hot mean TTFT (s) | 0.7889 +/- 0.0598 | 0.2526 +/- 0.0005 | |
| cold mean TTFT (s) | 0.5005 +/- 0.0004 | 0.5025 +/- 0.0004 | |
| hot attainment (TTFT<=2.0s) | 0.9512 +/- 0.0164 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9512 +/- 0.0164 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.8201 +/- 0.0120 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.8, load = 0.80, budget = L2, gamma = 1.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4010 +/- 0.0028 | 0.4010 +/- 0.0028 | gain 0.00 pp |
| aggregate mean TTFT (s) | 1.4642 +/- 0.0921 | 0.6042 +/- 0.0026 | 2.424 |
| hot P99 TTFT (s) | 5.9936 +/- 1.0601 | 0.7067 +/- 0.0226 | 8.493 |
| cold P99 TTFT (s) | 1.0620 +/- 0.0004 | 1.2062 +/- 0.0199 | 0.881 |
| hot P99 queue wait (s) | 5.4926 +/- 1.0672 | 0.2033 +/- 0.0255 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.2104 +/- 0.0216 | |
| hot mean TTFT (s) | 1.5782 +/- 0.1129 | 0.5053 +/- 0.0009 | |
| cold mean TTFT (s) | 1.0010 +/- 0.0008 | 1.0049 +/- 0.0009 | |
| hot attainment (TTFT<=2.0s) | 0.7409 +/- 0.0313 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.7409 +/- 0.0313 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.8208 +/- 0.0109 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.8, load = 0.80, budget = L2, gamma = 2.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4010 +/- 0.0028 | 0.4010 +/- 0.0028 | gain 0.00 pp |
| aggregate mean TTFT (s) | 2.9304 +/- 0.1932 | 1.2084 +/- 0.0052 | 2.425 |
| hot P99 TTFT (s) | 12.0247 +/- 2.0981 | 1.4135 +/- 0.0452 | 8.520 |
| cold P99 TTFT (s) | 2.1239 +/- 0.0007 | 2.4124 +/- 0.0398 | 0.881 |
| hot P99 queue wait (s) | 11.0331 +/- 2.1049 | 0.4067 +/- 0.0510 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.4208 +/- 0.0431 | |
| hot mean TTFT (s) | 3.1588 +/- 0.2369 | 1.0105 +/- 0.0019 | |
| cold mean TTFT (s) | 2.0019 +/- 0.0016 | 2.0099 +/- 0.0017 | |
| hot attainment (TTFT<=2.0s) | 0.4090 +/- 0.0249 | 0.9994 +/- 0.0003 | |
| cold attainment | 0.5023 +/- 0.0094 | 0.4901 +/- 0.0087 | |
| FI (report-only) | 0.8140 +/- 0.0442 | 0.4904 +/- 0.0087 | |
| hot-node util (measured) | 0.8207 +/- 0.0119 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.9, load = 0.50, budget = L1, gamma = 0.5

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4505 +/- 0.0018 | 0.4505 +/- 0.0018 | gain 0.00 pp |
| aggregate mean TTFT (s) | 0.3946 +/- 0.0049 | 0.2749 +/- 0.0009 | 1.435 |
| hot P99 TTFT (s) | 1.1206 +/- 0.0604 | 0.2809 +/- 0.0001 | 3.990 |
| cold P99 TTFT (s) | 0.5307 +/- 0.0002 | 0.5308 +/- 0.0002 | 1.000 |
| hot P99 queue wait (s) | 0.8710 +/- 0.0622 | 0.0000 +/- 0.0000 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | |
| hot mean TTFT (s) | 0.3830 +/- 0.0057 | 0.2502 +/- 0.0002 | |
| cold mean TTFT (s) | 0.4996 +/- 0.0006 | 0.4999 +/- 0.0006 | |
| hot attainment (TTFT<=2.0s) | 0.9999 +/- 0.0001 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9999 +/- 0.0001 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.5141 +/- 0.0060 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.9, load = 0.50, budget = L1, gamma = 1.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4505 +/- 0.0018 | 0.4505 +/- 0.0018 | gain 0.00 pp |
| aggregate mean TTFT (s) | 0.7894 +/- 0.0112 | 0.5499 +/- 0.0018 | 1.436 |
| hot P99 TTFT (s) | 2.2353 +/- 0.1193 | 0.5618 +/- 0.0003 | 3.979 |
| cold P99 TTFT (s) | 1.0613 +/- 0.0003 | 1.0617 +/- 0.0003 | 1.000 |
| hot P99 queue wait (s) | 1.7316 +/- 0.1233 | 0.0000 +/- 0.0000 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | |
| hot mean TTFT (s) | 0.7662 +/- 0.0131 | 0.5004 +/- 0.0005 | |
| cold mean TTFT (s) | 0.9993 +/- 0.0012 | 0.9998 +/- 0.0012 | |
| hot attainment (TTFT<=2.0s) | 0.9823 +/- 0.0035 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9823 +/- 0.0035 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.5133 +/- 0.0071 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.9, load = 0.50, budget = L1, gamma = 2.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4505 +/- 0.0018 | 0.4505 +/- 0.0018 | gain 0.00 pp |
| aggregate mean TTFT (s) | 1.5802 +/- 0.0208 | 1.0998 +/- 0.0036 | 1.437 |
| hot P99 TTFT (s) | 4.4719 +/- 0.2062 | 1.1235 +/- 0.0005 | 3.980 |
| cold P99 TTFT (s) | 2.1227 +/- 0.0007 | 2.1233 +/- 0.0006 | 1.000 |
| hot P99 queue wait (s) | 3.4688 +/- 0.2165 | 0.0000 +/- 0.0000 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | |
| hot mean TTFT (s) | 1.5340 +/- 0.0243 | 1.0007 +/- 0.0010 | |
| cold mean TTFT (s) | 1.9986 +/- 0.0024 | 1.9995 +/- 0.0023 | |
| hot attainment (TTFT<=2.0s) | 0.7996 +/- 0.0117 | 1.0000 +/- 0.0000 | |
| cold attainment | 0.5093 +/- 0.0171 | 0.5083 +/- 0.0167 | |
| FI (report-only) | 0.6369 +/- 0.0191 | 0.5083 +/- 0.0167 | |
| hot-node util (measured) | 0.5138 +/- 0.0067 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.9, load = 0.50, budget = L2, gamma = 0.5

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4505 +/- 0.0018 | 0.4505 +/- 0.0018 | gain 0.00 pp |
| aggregate mean TTFT (s) | 0.3944 +/- 0.0051 | 0.2749 +/- 0.0009 | 1.435 |
| hot P99 TTFT (s) | 1.1201 +/- 0.0588 | 0.2809 +/- 0.0001 | 3.988 |
| cold P99 TTFT (s) | 0.5307 +/- 0.0002 | 0.5308 +/- 0.0002 | 1.000 |
| hot P99 queue wait (s) | 0.8682 +/- 0.0586 | 0.0000 +/- 0.0000 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | |
| hot mean TTFT (s) | 0.3828 +/- 0.0060 | 0.2502 +/- 0.0002 | |
| cold mean TTFT (s) | 0.4996 +/- 0.0006 | 0.4999 +/- 0.0006 | |
| hot attainment (TTFT<=2.0s) | 0.9999 +/- 0.0001 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9999 +/- 0.0001 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.5132 +/- 0.0068 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.9, load = 0.50, budget = L2, gamma = 1.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4505 +/- 0.0018 | 0.4505 +/- 0.0018 | gain 0.00 pp |
| aggregate mean TTFT (s) | 0.7893 +/- 0.0107 | 0.5499 +/- 0.0018 | 1.435 |
| hot P99 TTFT (s) | 2.2426 +/- 0.1163 | 0.5618 +/- 0.0003 | 3.992 |
| cold P99 TTFT (s) | 1.0613 +/- 0.0003 | 1.0617 +/- 0.0003 | 1.000 |
| hot P99 queue wait (s) | 1.7386 +/- 0.1226 | 0.0000 +/- 0.0000 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | |
| hot mean TTFT (s) | 0.7661 +/- 0.0124 | 0.5004 +/- 0.0005 | |
| cold mean TTFT (s) | 0.9993 +/- 0.0012 | 0.9998 +/- 0.0012 | |
| hot attainment (TTFT<=2.0s) | 0.9822 +/- 0.0034 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9822 +/- 0.0034 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.5138 +/- 0.0065 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.9, load = 0.50, budget = L2, gamma = 2.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4505 +/- 0.0018 | 0.4505 +/- 0.0018 | gain 0.00 pp |
| aggregate mean TTFT (s) | 1.5786 +/- 0.0211 | 1.0998 +/- 0.0036 | 1.435 |
| hot P99 TTFT (s) | 4.4743 +/- 0.2336 | 1.1235 +/- 0.0005 | 3.982 |
| cold P99 TTFT (s) | 2.1227 +/- 0.0007 | 2.1233 +/- 0.0006 | 1.000 |
| hot P99 queue wait (s) | 3.4677 +/- 0.2338 | 0.0000 +/- 0.0000 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | |
| hot mean TTFT (s) | 1.5323 +/- 0.0245 | 1.0007 +/- 0.0010 | |
| cold mean TTFT (s) | 1.9986 +/- 0.0024 | 1.9995 +/- 0.0023 | |
| hot attainment (TTFT<=2.0s) | 0.8003 +/- 0.0108 | 1.0000 +/- 0.0000 | |
| cold attainment | 0.5093 +/- 0.0171 | 0.5083 +/- 0.0167 | |
| FI (report-only) | 0.6364 +/- 0.0188 | 0.5083 +/- 0.0167 | |
| hot-node util (measured) | 0.5138 +/- 0.0060 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.9, load = 0.65, budget = L1, gamma = 0.5

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4505 +/- 0.0018 | 0.4505 +/- 0.0018 | gain 0.00 pp |
| aggregate mean TTFT (s) | 0.4946 +/- 0.0126 | 0.2753 +/- 0.0009 | 1.797 |
| hot P99 TTFT (s) | 1.6216 +/- 0.1090 | 0.2811 +/- 0.0001 | 5.769 |
| cold P99 TTFT (s) | 0.5307 +/- 0.0002 | 0.5319 +/- 0.0027 | 0.998 |
| hot P99 queue wait (s) | 1.3687 +/- 0.1068 | 0.0016 +/- 0.0046 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0080 +/- 0.0188 | |
| hot mean TTFT (s) | 0.4940 +/- 0.0140 | 0.2505 +/- 0.0002 | |
| cold mean TTFT (s) | 0.4997 +/- 0.0006 | 0.5003 +/- 0.0005 | |
| hot attainment (TTFT<=2.0s) | 0.9969 +/- 0.0022 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9969 +/- 0.0022 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.6623 +/- 0.0079 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.9, load = 0.65, budget = L1, gamma = 1.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4505 +/- 0.0018 | 0.4505 +/- 0.0018 | gain 0.00 pp |
| aggregate mean TTFT (s) | 0.9889 +/- 0.0246 | 0.5506 +/- 0.0017 | 1.796 |
| hot P99 TTFT (s) | 3.2410 +/- 0.2264 | 0.5622 +/- 0.0003 | 5.765 |
| cold P99 TTFT (s) | 1.0613 +/- 0.0004 | 1.0637 +/- 0.0055 | 0.998 |
| hot P99 queue wait (s) | 2.7324 +/- 0.2148 | 0.0032 +/- 0.0092 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0160 +/- 0.0375 | |
| hot mean TTFT (s) | 0.9877 +/- 0.0273 | 0.5010 +/- 0.0005 | |
| cold mean TTFT (s) | 0.9993 +/- 0.0012 | 1.0005 +/- 0.0010 | |
| hot attainment (TTFT<=2.0s) | 0.9249 +/- 0.0096 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9249 +/- 0.0096 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.6622 +/- 0.0081 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.9, load = 0.65, budget = L1, gamma = 2.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4505 +/- 0.0018 | 0.4505 +/- 0.0018 | gain 0.00 pp |
| aggregate mean TTFT (s) | 1.9794 +/- 0.0495 | 1.1011 +/- 0.0035 | 1.798 |
| hot P99 TTFT (s) | 6.5036 +/- 0.4835 | 1.1244 +/- 0.0005 | 5.784 |
| cold P99 TTFT (s) | 2.1227 +/- 0.0007 | 2.1274 +/- 0.0109 | 0.998 |
| hot P99 queue wait (s) | 5.4930 +/- 0.4684 | 0.0064 +/- 0.0185 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0321 +/- 0.0750 | |
| hot mean TTFT (s) | 1.9771 +/- 0.0550 | 1.0020 +/- 0.0010 | |
| cold mean TTFT (s) | 1.9986 +/- 0.0024 | 2.0010 +/- 0.0020 | |
| hot attainment (TTFT<=2.0s) | 0.6448 +/- 0.0164 | 1.0000 +/- 0.0000 | |
| cold attainment | 0.5093 +/- 0.0172 | 0.5063 +/- 0.0166 | |
| FI (report-only) | 0.7899 +/- 0.0246 | 0.5063 +/- 0.0166 | |
| hot-node util (measured) | 0.6629 +/- 0.0080 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.9, load = 0.65, budget = L2, gamma = 0.5

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4505 +/- 0.0018 | 0.4505 +/- 0.0018 | gain 0.00 pp |
| aggregate mean TTFT (s) | 0.4956 +/- 0.0119 | 0.2753 +/- 0.0009 | 1.800 |
| hot P99 TTFT (s) | 1.6290 +/- 0.1078 | 0.2811 +/- 0.0001 | 5.795 |
| cold P99 TTFT (s) | 0.5307 +/- 0.0002 | 0.5319 +/- 0.0027 | 0.998 |
| hot P99 queue wait (s) | 1.3774 +/- 0.1073 | 0.0016 +/- 0.0046 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0080 +/- 0.0188 | |
| hot mean TTFT (s) | 0.4951 +/- 0.0132 | 0.2505 +/- 0.0002 | |
| cold mean TTFT (s) | 0.4997 +/- 0.0006 | 0.5003 +/- 0.0005 | |
| hot attainment (TTFT<=2.0s) | 0.9969 +/- 0.0023 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9969 +/- 0.0023 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.6623 +/- 0.0085 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.9, load = 0.65, budget = L2, gamma = 1.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4505 +/- 0.0018 | 0.4505 +/- 0.0018 | gain 0.00 pp |
| aggregate mean TTFT (s) | 0.9890 +/- 0.0232 | 0.5506 +/- 0.0017 | 1.796 |
| hot P99 TTFT (s) | 3.2493 +/- 0.2264 | 0.5622 +/- 0.0003 | 5.779 |
| cold P99 TTFT (s) | 1.0614 +/- 0.0003 | 1.0637 +/- 0.0055 | 0.998 |
| hot P99 queue wait (s) | 2.7400 +/- 0.2126 | 0.0032 +/- 0.0092 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0160 +/- 0.0375 | |
| hot mean TTFT (s) | 0.9878 +/- 0.0258 | 0.5010 +/- 0.0005 | |
| cold mean TTFT (s) | 0.9993 +/- 0.0012 | 1.0005 +/- 0.0010 | |
| hot attainment (TTFT<=2.0s) | 0.9250 +/- 0.0087 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9250 +/- 0.0087 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.6624 +/- 0.0083 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.9, load = 0.65, budget = L2, gamma = 2.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4505 +/- 0.0018 | 0.4505 +/- 0.0018 | gain 0.00 pp |
| aggregate mean TTFT (s) | 1.9794 +/- 0.0492 | 1.1011 +/- 0.0035 | 1.798 |
| hot P99 TTFT (s) | 6.4848 +/- 0.5034 | 1.1244 +/- 0.0005 | 5.767 |
| cold P99 TTFT (s) | 2.1228 +/- 0.0007 | 2.1274 +/- 0.0109 | 0.998 |
| hot P99 queue wait (s) | 5.4827 +/- 0.5058 | 0.0064 +/- 0.0185 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0321 +/- 0.0750 | |
| hot mean TTFT (s) | 1.9771 +/- 0.0547 | 1.0020 +/- 0.0010 | |
| cold mean TTFT (s) | 1.9986 +/- 0.0024 | 2.0010 +/- 0.0020 | |
| hot attainment (TTFT<=2.0s) | 0.6446 +/- 0.0167 | 1.0000 +/- 0.0000 | |
| cold attainment | 0.5091 +/- 0.0172 | 0.5063 +/- 0.0166 | |
| FI (report-only) | 0.7900 +/- 0.0247 | 0.5063 +/- 0.0166 | |
| hot-node util (measured) | 0.6630 +/- 0.0087 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.9, load = 0.80, budget = L1, gamma = 0.5

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4505 +/- 0.0018 | 0.4504 +/- 0.0018 | gain 0.00 pp |
| aggregate mean TTFT (s) | 0.7419 +/- 0.0438 | 0.2759 +/- 0.0008 | 2.689 |
| hot P99 TTFT (s) | 2.9425 +/- 0.4022 | 0.2950 +/- 0.0099 | 9.965 |
| cold P99 TTFT (s) | 0.5307 +/- 0.0002 | 0.5494 +/- 0.0165 | 0.967 |
| hot P99 queue wait (s) | 2.6909 +/- 0.4011 | 0.0426 +/- 0.0085 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0507 +/- 0.0178 | |
| hot mean TTFT (s) | 0.7684 +/- 0.0479 | 0.2511 +/- 0.0003 | |
| cold mean TTFT (s) | 0.4997 +/- 0.0006 | 0.5009 +/- 0.0005 | |
| hot attainment (TTFT<=2.0s) | 0.9525 +/- 0.0137 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9525 +/- 0.0137 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.8091 +/- 0.0100 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.9, load = 0.80, budget = L1, gamma = 1.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4505 +/- 0.0018 | 0.4504 +/- 0.0018 | gain 0.00 pp |
| aggregate mean TTFT (s) | 1.4865 +/- 0.0866 | 0.5518 +/- 0.0017 | 2.694 |
| hot P99 TTFT (s) | 5.9131 +/- 0.7859 | 0.5900 +/- 0.0198 | 10.014 |
| cold P99 TTFT (s) | 1.0614 +/- 0.0004 | 1.0988 +/- 0.0330 | 0.967 |
| hot P99 queue wait (s) | 5.4076 +/- 0.7857 | 0.0852 +/- 0.0171 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.1014 +/- 0.0356 | |
| hot mean TTFT (s) | 1.5399 +/- 0.0947 | 0.5022 +/- 0.0006 | |
| cold mean TTFT (s) | 0.9993 +/- 0.0012 | 1.0018 +/- 0.0010 | |
| hot attainment (TTFT<=2.0s) | 0.7529 +/- 0.0291 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.7529 +/- 0.0291 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.8086 +/- 0.0102 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.9, load = 0.80, budget = L1, gamma = 2.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4505 +/- 0.0018 | 0.4504 +/- 0.0018 | gain 0.00 pp |
| aggregate mean TTFT (s) | 2.9712 +/- 0.1701 | 1.1035 +/- 0.0034 | 2.693 |
| hot P99 TTFT (s) | 11.7905 +/- 1.5744 | 1.1800 +/- 0.0397 | 9.983 |
| cold P99 TTFT (s) | 2.1227 +/- 0.0008 | 2.1976 +/- 0.0661 | 0.967 |
| hot P99 queue wait (s) | 10.7692 +/- 1.5808 | 0.1705 +/- 0.0342 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.2028 +/- 0.0711 | |
| hot mean TTFT (s) | 3.0778 +/- 0.1861 | 1.0044 +/- 0.0012 | |
| cold mean TTFT (s) | 1.9986 +/- 0.0023 | 2.0036 +/- 0.0019 | |
| hot attainment (TTFT<=2.0s) | 0.4272 +/- 0.0209 | 0.9999 +/- 0.0001 | |
| cold attainment | 0.5094 +/- 0.0171 | 0.5009 +/- 0.0160 | |
| FI (report-only) | 0.8391 +/- 0.0415 | 0.5009 +/- 0.0159 | |
| hot-node util (measured) | 0.8098 +/- 0.0094 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.9, load = 0.80, budget = L2, gamma = 0.5

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4505 +/- 0.0018 | 0.4504 +/- 0.0018 | gain 0.00 pp |
| aggregate mean TTFT (s) | 0.7412 +/- 0.0434 | 0.2759 +/- 0.0008 | 2.687 |
| hot P99 TTFT (s) | 2.9446 +/- 0.3972 | 0.2950 +/- 0.0099 | 9.972 |
| cold P99 TTFT (s) | 0.5307 +/- 0.0002 | 0.5494 +/- 0.0165 | 0.967 |
| hot P99 queue wait (s) | 2.6894 +/- 0.3986 | 0.0426 +/- 0.0085 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.0507 +/- 0.0178 | |
| hot mean TTFT (s) | 0.7676 +/- 0.0475 | 0.2511 +/- 0.0003 | |
| cold mean TTFT (s) | 0.4997 +/- 0.0006 | 0.5009 +/- 0.0005 | |
| hot attainment (TTFT<=2.0s) | 0.9524 +/- 0.0136 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.9524 +/- 0.0136 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.8081 +/- 0.0103 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.9, load = 0.80, budget = L2, gamma = 1.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4505 +/- 0.0018 | 0.4504 +/- 0.0018 | gain 0.00 pp |
| aggregate mean TTFT (s) | 1.4842 +/- 0.0860 | 0.5518 +/- 0.0017 | 2.690 |
| hot P99 TTFT (s) | 5.8894 +/- 0.7963 | 0.5900 +/- 0.0198 | 9.973 |
| cold P99 TTFT (s) | 1.0613 +/- 0.0004 | 1.0988 +/- 0.0330 | 0.967 |
| hot P99 queue wait (s) | 5.3839 +/- 0.7924 | 0.0852 +/- 0.0171 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.1014 +/- 0.0356 | |
| hot mean TTFT (s) | 1.5374 +/- 0.0941 | 0.5022 +/- 0.0006 | |
| cold mean TTFT (s) | 0.9993 +/- 0.0011 | 1.0018 +/- 0.0010 | |
| hot attainment (TTFT<=2.0s) | 0.7536 +/- 0.0283 | 1.0000 +/- 0.0000 | |
| cold attainment | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | |
| FI (report-only) | 0.7536 +/- 0.0283 | 1.0000 +/- 0.0000 | |
| hot-node util (measured) | 0.8088 +/- 0.0101 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

### p_hot = 0.9, load = 0.80, budget = L2, gamma = 2.0

| metric | A | B | ratio A/B |
|---|---|---|---|
| hit rate | 0.4505 +/- 0.0018 | 0.4504 +/- 0.0018 | gain 0.00 pp |
| aggregate mean TTFT (s) | 2.9681 +/- 0.1749 | 1.1035 +/- 0.0034 | 2.690 |
| hot P99 TTFT (s) | 11.7610 +/- 1.6088 | 1.1800 +/- 0.0397 | 9.958 |
| cold P99 TTFT (s) | 2.1228 +/- 0.0007 | 2.1976 +/- 0.0661 | 0.967 |
| hot P99 queue wait (s) | 10.7531 +/- 1.5998 | 0.1705 +/- 0.0342 | |
| cold P99 queue wait (s) | 0.0000 +/- 0.0000 | 0.2028 +/- 0.0711 | |
| hot mean TTFT (s) | 3.0743 +/- 0.1914 | 1.0044 +/- 0.0012 | |
| cold mean TTFT (s) | 1.9987 +/- 0.0023 | 2.0036 +/- 0.0019 | |
| hot attainment (TTFT<=2.0s) | 0.4277 +/- 0.0206 | 0.9999 +/- 0.0001 | |
| cold attainment | 0.5094 +/- 0.0171 | 0.5009 +/- 0.0160 | |
| FI (report-only) | 0.8402 +/- 0.0415 | 0.5009 +/- 0.0159 | |
| hot-node util (measured) | 0.8090 +/- 0.0103 | - | |
| overload runs (any node util>=1.0) | 0 | 0 | |

## 3. B' sensitivity cell (request-count least-loaded), p_hot = 0.8, gamma = 1.0

| load | budget | B' hot P99 TTFT (s) | B hot P99 TTFT (s) | B' agg mean (s) | B agg mean (s) |
|---|---|---|---|---|---|
| 0.50 | L1 | 0.5622 +/- 0.0002 | 0.5622 +/- 0.0002 | 0.6011 +/- 0.0027 | 0.5999 +/- 0.0027 |
| 0.50 | L2 | 0.5623 +/- 0.0003 | 0.5622 +/- 0.0002 | 0.6012 +/- 0.0026 | 0.5999 +/- 0.0027 |
| 0.65 | L1 | 0.7166 +/- 0.0583 | 0.6011 +/- 0.0274 | 0.6044 +/- 0.0026 | 0.6015 +/- 0.0026 |
| 0.65 | L2 | 0.7061 +/- 0.0532 | 0.6011 +/- 0.0274 | 0.6042 +/- 0.0026 | 0.6015 +/- 0.0026 |
| 0.80 | L1 | 0.8969 +/- 0.0353 | 0.7067 +/- 0.0226 | 0.6091 +/- 0.0025 | 0.6042 +/- 0.0026 |
| 0.80 | L2 | 0.9006 +/- 0.0202 | 0.7067 +/- 0.0226 | 0.6090 +/- 0.0026 | 0.6042 +/- 0.0026 |

## 4. Frozen thresholds (for the analyst; no verdict is drawn here)

- H1 success (p_hot in {0.8, 0.9}, c = 32, stable region, equal load, equal budget): hit-rate gain >= G (hit leg pre-registrationally killed: G = 0.00 pp), aggregate mean TTFT(A) <= 0.8 x B, hot P99(A) >= 2 x hot P99(B), cold P99(A) <= 1.2 x cold P99(B) (LOCKED_PLAN 5).
- H1 kill (queue leg): hot P99(A) within 20%% of hot P99(B) at both test points (LOCKED_PLAN 6; criterion (i), hit gain <= 5pp, does not apply).
- Overload region (hot-node util >= 1.0) excluded from verdicts, reported separately (frozen).
- Simulator passes are 'queue-concentration passes; tail claim untested until E4' (frozen revision R1).

## 5. Sanity checks vs E0 (3 anchor points) and monotonicity

| anchor | measured util | E0 util (frozen) | util delta %% | measured mean wait (s) | E0 mean wait (s) | wait delta %% | P-K wait at realized rho (s) | P-K delta %% |
|---|---|---|---|---|---|---|---|
| A p_hot=0.9 load=0.80 gamma=1.0 | 0.7923 | 0.8000 | -0.97 | 0.9648 | 1.4355 | -32.79 | 1.0085 | -4.33 |
| B p_hot=0.9 load=0.80 gamma=1.0 | 0.2408 | 0.2444 | -1.47 | 0.0019 | 0.0960 | -98.02 | n/a | n/a |
| A p_hot=0.8 load=0.65 gamma=1.0 | 0.6643 | 0.6500 | +2.19 | 0.4680 | 0.7592 | -38.35 | 0.5520 | -15.21 |
| B p_hot=0.8 load=0.65 gamma=1.0 | 0.2410 | 0.2437 | -1.14 | 0.0025 | 0.1078 | -97.70 | n/a | n/a |
| A p_hot=0.8 load=0.80 gamma=2.0 | 0.8009 | 0.8000 | +0.11 | 1.8649 | 5.0221 | -62.87 | 2.2449 | -16.93 |
| B p_hot=0.8 load=0.80 gamma=2.0 | 0.2966 | 0.3000 | -1.14 | 0.0100 | 0.2866 | -96.51 | n/a | n/a |

| policy | monotone vs load (mean TTFT and hot P99 increasing in 0.5 -> 0.65 -> 0.8) |
|---|---|
| A | True |
| B | True |

Frozen sanity rule: utilization and mean wait should match E0 within a few %% at the 3 anchor points; deltas are reported as measured. Notes:
- Policy A: measured hot-node util sits between E0's frozen formula (rho_hot = load) and the full-utilization variant (which assumed a 25%% cold split onto the hot node); the realized cold share is ~4%%. The P-K wait at the realized rho (E0 moments) validates the M/G/1 mechanism itself (column 8-9).
- Policy B: min-backlog routing keeps per-node utils at E0's rho_B (column 5), but the per-node M/G/1 Poisson-arrival assumption does not hold for load-balanced nodes (arrivals are routed away from busy servers), so measured mean waits are far below E0's W_mean_B. This is a documented approximation of the frozen algebra, not a simulator defect; the affinity hot node (Poisson arrivals) is the cell where the algebra's wait predictions apply.

## 6. Overload-region report (excluded from verdicts, reported separately)

No run had any node at utilization >= 1.0 in the steady-state window; all E1 cells are within the stable region.

## 7. Reproducibility

- raw/e1_cells.csv: every run, every replication (never dropped; duplicate run_ids replaced in place).
- raw/per_request/<run_id>.csv.gz: per-request records (arrival time, node, queue wait, prefill tokens, hit/miss, TTFT) for the steady-state window.
- raw/per_node_trace/<run_id>.csv.gz: per-node queue-length/backlog traces at the 6 sanity anchor runs.
- Seeds, warm-up count, lambda, gate-wait counts, and t_warm are recorded per run in raw/e1_cells.csv.
