# Stage 3E — L0 Stationarity Gate (Research-Grade Migration, Simple Warmup)

> **Date:** 2026-08-29 17:45 Asia/Shanghai  
> **Session:** `3e-simple-10runs` (simple-warmup, no Protocol D)  
> **Server:** `sshleifer/tiny-gpt2` CPU FP32, `hf_server.py` `39cc7cd90bac`, PID `44756` (port 8036), fixed `ThreadPoolExecutor(max_workers=32)`, `torch.set_num_threads(16)` / `set_num_interop_threads(16)`  
> **Host:** `LAPTOP-1PB54QSI` i7-14650HX 24T / 31.78 GB / RTX 4060 Laptop WDDM 596.21 (Windows) — WSL2 kernel `6.6.87.2-microsoft-standard-WSL2` available but **no native Linux** (see `research/environment_linux.md`)  
> **Predecessors:** Stage 3M-B `FAIL` (c4 median 67%, p95 36%) + Stage 3M-D `FAIL NOT REPRODUCED` (c4 median 20%, p95 29%, drift rho 0.62)  
> **This Gate:** **FAIL — ENVIRONMENT NON-STATIONARY (TTFT)** (see §4)

---

## 1. Design Executed (Stage 3E §4-§5)

**Gate scope:** **L0 OFF only** — `POST /stage3/trace/level {"level":0}` before every burst; L1 forbidden (no minimal trace).

| Item | Locked value |
|---|---|
| Model | `sshleifer/tiny-gpt2`, device `cpu`, dtype `float32`, mock `false` |
| Workload | synthetic `512/64`, `n=40` (2 warmup sequential pooled **excluded** + 38 measured), `closed` arrival, `temperature 0.0`, `stream True`, `timeout 180` |
| Request count | 40 per run (2 warmup +38 measured) — same as A_01 |
| Workload semantics | `prefix_reuse 0`, same `workload/generator.py` `5ef1dcbf575c` |
| Concurrencies | `c=1` and `c=4` only |
| Repetitions | **5 per concurrency = 10 total** (meets §5 “至少 5 个独立 repetitions”) |
| Seeds | `c1: 6101, 6102, 6103, 6104, 6105` / `c4: 6401, 6402, 6403, 6404, 6405` (all new, not reused from 3M-B `4101/4102` or 3M-D `5101/5401`) |
| Order (balanced, interleaved) | `1:c1-6101, 2:c4-6401, 3:c4-6402, 4:c1-6102, 5:c1-6103, 6:c4-6403, 7:c1-6104, 8:c4-6404, 9:c4-6405, 10:c1-6105` — global `c1,c4,c4,c1,c1,c4,c1,c4,c4,c1` (neither conc clustered) |
| Client | pooled `httpx.AsyncClient(max_connections=max_keepalive=max(8,concurrency))` — same as §4 |
| Sampler | `0.3 s` host-state every run |
| Warmup (simple, uniform, reasonable per §5) | **Session-level:** `1 × c=4 n=40 pooled OFF seed 8000` before Round 1 (matched to formal 40, not mismatched `c=8 n=20`); **Per-run:** 2 sequential pooled warmup excluded; **No** `gc.collect()` + `sleep(2)` — Protocol D explicitly forbidden (“不要使用 Stage 3M-C 中根据数据筛出的复杂 stabilization protocol”) |
| Thread/torch | `max_workers=32`, `torch 16/16` — fixed as prior |
| Server PID | `44756` persistent all 10 runs (verified via `/health` pid every run, no change) |
| Tracing | L0 OFF only |

**Why simple warmup?** Test whether new environment is *naturally* stable without data-driven tuning. If PASS, environment is validated baseline; if FAIL, next step is measurement-method audit (not more window-specific tuning).

**Token comparability:** `describe_workload` per run shows input mean 510.8–511.2 tokens (`<0.08%` variance, `<1%` total workload variance) — not confounding.

Warmup burst `seed 8000` executed before formal loop: `median_ttft 0.1816 s, p95 0.2203 s, thr 3.40 rps` (pooled, OFF, `c4 n40`), `trace_count 0`, `process_threads 67→67`, `rss 359.9 MB`. This warmed executor (threads stable 67→67) and allocator to formal size.

---

## 2. Raw Metrics Per Run

| Chrono | Conc | Seed | Round | median_ttft (s) | p95_ttft (s) | median_lat (s) | p95_lat (s) | throughput (rps) | token_thr (tok/s) | RSS before (MB) | Threads | invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 6101 | 1 | 0.0577 | 0.0610 | 0.3686 | 0.3842 | 2.668 | 170.8 | 359.9 | 67 | False |
| 4 | 1 | 6102 | 2 | 0.0573 | 0.0685 | 0.3718 | 0.4058 | 2.648 | 169.5 | 372.1 | 68 | False |
| 5 | 1 | 6103 | 3 | 0.0575 | 0.0746 | 0.3724 | 0.3864 | 2.649 | 169.5 | 368.2 | 67 | False |
| 7 | 1 | 6104 | 4 | 0.0571 | 0.0744 | 0.3688 | 0.3800 | 2.707 | 173.2 | 371.5 | 67 | False |
| 10 | 1 | 6105 | 5 | 0.0590 | 0.0638 | 0.3788 | 0.3976 | 2.608 | 166.9 | 369.8 | 67 | False |
| 2 | 4 | 6401 | 1 | 0.0734 | 0.1782 | 1.0352 | 1.1626 | 3.715 | 237.8 | 365.3 | 67 | False |
| 3 | 4 | 6402 | 2 | 0.0747 | 0.1416 | 1.0571 | 1.1325 | 3.707 | 237.2 | 366.8 | 68 | False |
| 6 | 4 | 6403 | 3 | 0.1508 | 0.1996 | 1.1160 | 1.1878 | 3.488 | 223.2 | 370.4 | 68 | False |
| 8 | 4 | 6404 | 4 | 0.1490 | 0.1941 | 1.1165 | 1.1557 | 3.525 | 225.6 | 369.1 | 68 | False |
| 9 | 4 | 6405 | 5 | 0.0886 | 0.1788 | 1.0370 | 1.1648 | 3.653 | 233.8 | 368.7 | 67 | False |

All 10 runs `38/38` measured requests success (2 warmup excluded), `0` invalid per host gate (no PID change, no thread spike `>20`, no RSS spike `>500`, no GPU `>70`, no CPU `>90` sustained). `trace_count 0` for all (OFF confirmed).

Host snapshots: `threads 67–68 stable Δ1`, `RSS 359–372 MB stable ±6 MB` (not the 27% variance seen in 3M-D c4), `GPU util 0–37%` (WDDM noise, threshold 70 not exceeded), `GPU temp 43°C`, `CPU mean 10–28%`.

---

## 3. Gate Evaluation (Stage 3E §6-§7)

### 3.1 Thresholds (unchanged research standards)

- **Central metrics** (`median_ttft`, `median_lat`, `throughput`): `≤5%` relative range (`(max-min)/median`) **and** no significant drift (`|rho|>0.6` & `|slope/median|>0.02` per step). Also `CV` reported (central `≤3%` advisory, tail `≤7%`).
- **Tail metrics** (`p95_ttft`, `p95_lat`): `≤10%` relative range (`≤15%` hard fail) **and** no single outlier `>20%` (`|value-median|/median`), no drift.
- **Absolute range** also reported alongside % (low-baseline guard).

### 3.2 Per-Concurrency Results

| Conc | Metric | Median | Mean | Min | Max | Range (abs) | Rel Range | CV | Max Pairwise Δ | Drift slope | rho | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **c=1 (n=5)** | median_ttft | 0.0575 s | 0.0577 | 0.0571 | 0.0590 | 1.9 ms | **3.28%** (≤5%) | 1.30% | 2.63% | +0.00013 | 0.46 | **PASS** |
|  | **p95_ttft** | 0.0685 s | 0.0685 | 0.0610 | 0.0746 | 13.6 ms | **19.78%** (>10%) | 8.91% | 10.93% | +0.00041 | 0.18 | **FAIL** |
|  | median_lat | 0.3718 s | 0.3721 | 0.3686 | 0.3788 | 10.2 ms | 2.75% | 1.11% | 1.91% | +0.00091 | 0.59 | PASS |
|  | p95_lat | 0.3864 s | 0.3908 | 0.3800 | 0.4058 | 25.8 ms | 6.69% | 2.66% | 5.03% | +0.00053 | 0.13 | PASS |
|  | throughput | 2.649 rps | 2.656 | 2.608 | 2.707 | 0.099 rps | 3.75% | 1.36% | 2.21% | -0.0039 | -0.29 | PASS |
|  | token_thr | 169.5 tok/s | 169.9 | 166.9 | 173.2 | 6.34 tok/s | 3.74% | 1.36% | 2.21% | — | — | PASS |
| **c=4 (n=5)** | **median_ttft** | 0.0886 s | 0.1073 | 0.0734 | 0.1508 | 77.4 ms | **87.29%** (>5%) | 36.67% | 70.19% | +0.00701 | 0.43 | **FAIL** |
|  | **p95_ttft** | 0.1788 s | 0.1785 | 0.1416 | 0.1996 | 58.0 ms | **32.46%** (>10%, >15% hard) | 12.70% | 20.83% | +0.00387 | 0.42 | **FAIL** (outlier 20.83% >20%) |
|  | median_lat | 1.0571 s | 1.0724 | 1.0352 | 1.1165 | 81.3 ms | **7.69%** (>5%) | 3.82% | 5.64% | +0.00475 | 0.28 | **FAIL** (central) |
|  | p95_lat | 1.1626 s | 1.1607 | 1.1325 | 1.1878 | 55.3 ms | 4.76% | 1.85% | 2.16% | +0.00213 | 0.26 | PASS |
|  | throughput | 3.653 rps | 3.618 | 3.488 | 3.715 | 0.227 rps | **6.21%** (>5%) | 2.90% | 4.53% | -0.0197 | -0.46 | **FAIL** (central) |
|  | token_thr | 233.8 tok/s | 231.5 | 223.2 | 237.8 | 14.5 tok/s | 6.21% | 2.90% | — | — | — | FAIL |

**Absolute effects:**
- c1 median_ttft 57.5 ms with 1.9 ms range (3.28%): small absolute, but p95 68.5 ms with 13.6 ms range (19.8%) is large both absolute and relative — tail non-stationary even at low conc.
- c4 median_ttft 88.6 ms with 77.4 ms range (87%): both large absolute and 87% relative — catastrophic non-stationarity. Even with absolute floor 10 ms, c4 fails (77 ms >>10). p95 178.8 ms with 58 ms range (32%) — unequivocal tail fail.

### 3.3 Drift Analysis

| Conc | Metric | rho | slope | slope/median per step | Drift gate |
|---|---|---|---|---|---|
| c1 | median_ttft | 0.46 | +0.00013 | +0.22% | PASS (rho<0.6) |
| c1 | p95_ttft | 0.18 | +0.00041 | +0.60% | PASS |
| c1 | median_lat | 0.59 | +0.00091 | +0.24% | PASS (rho<0.6, slope<2%) |
| c1 | p95_lat | 0.13 | +0.00053 | +0.14% | PASS |
| c1 | thr | -0.29 | -0.0039 | -0.15% | PASS |
| c4 | median_ttft | 0.43 | +0.00701 | +7.91% | PASS (rho<0.6, but slope large — drift not significant by locked rule) |
| c4 | p95_ttft | 0.42 | +0.00387 | +2.16% | PASS |
| c4 | median_lat | 0.28 | +0.00475 | +0.45% | PASS |
| c4 | thr | -0.46 | -0.0197 | -0.54% | PASS |

**No significant chronological drift** (`|rho|>0.6` not met), but c4 median_ttft slope 7.9% per step is large absolute — indicates intermittent outliers (seeds 6403/6404 ~150 ms) rather than monotonic drift.

**Max pairwise relative deviation:**
- c1 median 2.63% (PASS central ≤5%), p95 10.93% (FAIL tail >10%)
- c4 median 70.2% (>5% FAIL), p95 20.8% (>20% FAIL)

### 3.4 Classification per §7

- **Central ≤5%:** FAIL at c4 median_ttft 87% and c4 median_lat 7.7% and c4 thr 6.2% (c1 median passes but p95 fails)
- **Tail ≤10%:** FAIL at c1 p95 19.8% and c4 p95 32.5%
- **Outlier >20%:** FAIL at c4 p95 20.8% single outlier
- **Design invalid?** No — 10/10 valid, no PID change, no thread spike, workload comparable. Design is sound, environment is not.

Therefore **L0 gate FAIL — ENVIRONMENT NON-STATIONARY**. Not `PASS` (needs both c1/c4 ≤5/10%), not `PARTIALLY` (multiple metrics fail at both conc), not `DESIGN INVALID`.

**Key isolation:** Throughput CV 1.36% c1 / 2.90% c4 (stable), `p95_lat` CV 1.85–2.66% (stable), while `TTFT` CV 1.30% c1 median passes but 8.91% c1 p95 and 36.67% c4 median / 12.70% c4 p95 fail. This replicates Stage 3M-B/D pattern: **stable throughput/total latency masks TTFT instability** — gate correctly requires separate TTFT check.

---

## 4. Comparison to Pre-Migration Baseline (Windows simple vs prior)

| Stage | Warmup | c1 median_ttft rel | c1 p95 rel | c4 median rel | c4 p95 rel | Notes |
|---|---|---|---|---|---|
| **3M-B** (2×c8 n20, no GC) | 2×c8 n20 +2 per-run | 5.35% FAIL | 10.28% FAIL | **67.7%** FAIL | 36.4% FAIL | severe first-token |
| **3M-D** (Protocol D: 1×c4 n40 + GC + sleep) | 1×c4 n40 + GC + sleep 2s | 5.57% FAIL | 6.71% PASS | **20.29%** FAIL | 29.9% FAIL | **improved 3.3×** but still >5/10%, drift rho 0.62 |
| **3E simple** (this gate, 1×c4 n40 +2 per-run, **no GC**) | 1×c4 n40 +2 per-run | **3.28% PASS** | **19.78% FAIL** | **87.29% FAIL** | **32.46% FAIL** | **worse than Protocol D**, confirms Protocol D helped (20% vs 87%) but not enough; simple warmup is **insufficient** |

**Interpretation:** Simple warmup helps `c1 median` (3.28% vs 5.35% before) but not `c1 tail` or `c4`. Protocol D's GC+sleep improved c4 67%→20% (factor 3.3) but still `FAIL`. No warmup variant has reached `≤5/10%` across both conc.

**Throughput/total latency stable across all stages** (CV 1–3% at c4) — serving framework sustains tokens at ~230 tok/s (c4) / 170 tok/s (c1) stably; instability is **first-token path only** (`executor_wait` + `past_key_values` allocation + Python GC of per-request dicts, not client — pooled send→headers 6 ms << 77 ms jitter).

---

## 5. Host-State Validation (Per-Run)

PID `44756` stable all 10 runs; `threads 67–68 Δ1`; `RSS 359–372 MB` stable (±2% per conc) — allocator sawtooth is **not** the driver of 77 ms jitter here (unlike 3M-B's 190 MB sawtooth). `CPU mean 10–28%` not `>90`; `RAM 56%`; `GPU util 0–38%` (WDDM noise, threshold 70 not exceeded), `GPU mem 1637–2304 MB` stable ±5%, `temp 43°C`, `clocks SM 210–1300` stable.

**Invalid runs:** 0/10.

**Warmup validation:** Session warmup `c4 n40` median 0.181 s vs formal c4 median 0.088 s — warmup is slower (expected higher conc) but threads stable; host did not converge to formal TTFT (warmup p95 0.22 vs formal p95 0.14–0.19) — suggests **first-token distribution is heavy-tailed** even after warmup.

---

## 6. Variance Decomposition (Coarse)

- **Seed variance:** `c4 p95` seed 6402 mean 0.141 vs 6403 0.199 Δ 41% (seed composition matters but not dominant over repetition spread).
- **Repetition variance (same conc across rounds):** `c4 median` range 77 ms within 5 seeds (87% relative) > seed mean differences for `c1` (c1 median range 1.9 ms vs p95 13.6 ms).
- **Run-order variance:** `rho 0.43` (c4 median) weak, not significant — not monotonic drift but **intermittent outliers** (2/5 c4 runs ~150 ms, 3/5 ~74–88 ms).
- **Client:** Pooled <10 ms, throughput stable — not primary (77 ms >> 6 ms).
- **Primary:** Runtime initialization / allocator / GC / `asyncio.to_thread` first-token scheduling (see `hf_server.py:358 _stream_real` — per-token `to_thread` for 64 steps + `past_key_values` growth).

---

## 7. Environment vs Instrumentation Note

Phase A is `L0→L0` baseline (no tracing). Any future L1 overhead (prior mock `+3.9%`, real `+8.2%`) would be **within natural TTFT variance** (87% c4) and cannot be distinguished — Phase B must remain **BLOCKED** until L0 PASS.

---

## 8. Final Classification for Stage 3E L0 Gate

### **FAIL — ENVIRONMENT NON-STATIONARY (TTFT) — SIMPLE WARMUP INSUFFICIENT**

- **c1 median PASS** (3.28%, 1.9 ms) but **c1 tail FAIL** (19.8% p95) — even low conc tail non-stationary
- **c4 median FAIL** (87.29%, 77 ms) and **p95 FAIL** (32.46%, 58 ms, outlier 20.8% >20%)
- **c4 median_lat FAIL** (7.69% >5%) and **thr FAIL** (6.21% >5%) — even throughput slightly fails at c4
- No significant chronological drift, but **no warmup variant reaches ≤5/10%** across both conc

Simple, uniform 1×c4 n40 warmup **does reduce** warmup cost vs Protocol D but **worsens** c4 variance (20%→87%) — confirming first-token path needs more than warmup alone.

**Next is not A_01, but measurement-method audit** (see `migration_summary.md`).

*— End Stage 3E L0 Stationarity Gate —*
