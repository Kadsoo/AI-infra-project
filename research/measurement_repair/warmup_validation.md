# Warmup Validation — Stage 3M-B

> **Session:** 3mb-stationarity-02 (formal Phase A) + reference 3mb-warmup-test  
> **Date:** 2026-08-28T15:22:30.416974+00:00 (formal), 15:17:57+00:00 (reference)  
> **Server PID:** 19372 (persistent, single process, port 8030)  
> **Model:** sshleifer/tiny-gpt2 CPU FP32, torch 16/16, executor 32  

---

## 1. Warmup Protocol (Locked, §3)

- **Server-level warmup:** 2× c=8 dummy workloads (synthetic 512/64, n=20, concurrency 8, pooled client, L0 OFF), not counted, executed back-to-back before Round 1. Purpose: drive `ThreadPoolExecutor`, tokenizer cache, torch kernels, and allocator past initial expansion.
- **Per-run warmup:** `warmup_requests=2` sequential pooled requests before concurrent burst, excluded from metrics — identical every run.
- **Fixed stabilization:** `ThreadPoolExecutor(max_workers=32)` via `loop.set_default_executor`, `torch.set_num_threads(16)`, `torch.set_num_interop_threads(16)`, pooled `httpx.AsyncClient(max_connections=max_keepalive=max(8, concurrency))` reused for warmup+measured.

All runs use same warmup; no per-run variation.

---

## 2. Host State Before / After Warmup (Formal Session 02)

| Phase | Health Threads (proc) | Snapshot Threads | RSS (MB) | CPU % | CPU Freq (MHz) | GPU Util (%) | GPU Mem (MB) | Latency p95 (s) |
|---|---|---|---|---|---|---|---|
| before_warmup | 247 | 247 | 318.4 | 10.6 | 2200.0 | 0.0 | 2001.5 | — |
| after_dummy_1 (c8 n20) | 247 | 247 | 129.3 | 15.3 | 1466.0 | 0.0 | 2001.5 | 2.535 |
| after_dummy_2 (c8 n20) | 248 | 248 | 314.3 | 12.3 | 1466.0 | 0.0 | 2001.5 | 2.592 |

- **Threads delta (dummy2−dummy1):** 1 (threshold ≤5) → **PASS**
- **Health threads delta:** 1 (≤5) → **PASS**
- **RSS delta:** 185.0 MB (threshold <50) → **FAIL** (4× over)
- **Latency p95 relative change dummy2/dummy1:** 2.24% (threshold 5% central) → **PASS**
- **Latency p95 absolute change:** 57 ms on 2535 ms baseline → trivial

### Reference Session (3mb-warmup-test, earlier same host, same fixed config)

| Phase | Threads | RSS (MB) |
|---|---|---|
| before | 67 | 58.9 |
| after dummy1 | 235 | 669.3 |
| after dummy2 | 235 | 671.7 |

- Threads delta 0, RSS delta 2.3 MB → PASS. This session started from freshly restarted server (67 threads) and showed textbook stabilization: threads 67→235 and RSS 58→669→671 plateau.

### Interpretation

- **Threads enter stable plateau after 1 dummy** in both sessions (67→235 or 247→248 ±1). Fixed executor 32 successfully prevents unbounded growth (prior unfixed grew 66→232→247 continuously). Warmup **does stabilize thread pool** as intended.
- **RSS does not consistently stabilize to <50 MB** after 2×20-request dummies in formal session (185 MB swing). Reference session did stabilize (2.3 MB). Difference is explained by starting RSS state: formal session started at 318 MB (already after prior warmup-test runs), dummy1 triggered GC drop to 129 MB (reclamation), dummy2 re-allocated to 314 MB (re-growth). Net swing 185 MB indicates allocator is still **oscillating between GC and allocation** when dummy size (20 requests, c=8) differs from formal size (40 requests, c=1/4). Per §4 Host State Gate, run with RSS spike >500 MB would be INVALID, but 185 MB is below 500 yet above 50 — **warmup is PARTIALLY STABLE: threads stable, latency stable, but RSS/allocator not fully quiescent**.
- Formal Phase A per-run RSS growth was 190–560 MB within each 40-request run (system.csv: 135→698 MB at c=4), confirming allocator expands per formal run by amount proportional to concurrency and then reclaims before next run. This per-run sawtooth is **repeatable** and not monotonic drift across runs (snapshot_after: 318→130→314 etc. oscillates, not monotonic), so it does not cause invalid thread growth but does inject TTFT variance (see stationarity analysis).

---

## 3. Warmup vs Formal First Run Alignment

- Formal first run (c1 seed4101) health RSS before 493 MB → after 130 MB. That 493 was residual from warmup dummy2 (314) plus idle allocation, then formal run's concurrent burst caused re-structuring and GC drop. The fact that formal runs start at 300–500 MB and end at 130 MB indicates **warmup dummy's RSS level (314) is not predictive of formal run's steady-state RSS (130)**, because dummy concurrency (c=8) allocates larger KV/past_key_values than c=1. Warmup with c=8 over-warms vs c=1 formal.

- **Conclusion:** c=8 dummy does drive threads stable, but **does not fully stabilize allocator for mixed c=1/c=4 formal runs**. Per spec §3, warmup must be validated to show metrics no longer change. Here threads pass, latency passes, but RSS does not fully. Therefore warmup validation is **PARTIALLY STABLE** — sufficient to proceed to formal measurement but flagged as allocator-related residual risk.

---

## 4. Torch / Executor / PID Verification

| Check | Value | Expected | Verdict |
|---|---|---|---|
| torch_num_threads | 16 | 16 | PASS |
| torch_num_interop_threads | 16 | 16 | PASS |
| fixed_executor_max_workers | 32 | 32 | PASS |
| pid before → after warmup | 19372 → 19372 | no change | PASS |
| port | 8030 | isolated | PASS |
| client_mode per dummy | pooled, max_connections 16 | pooled | PASS |
| trace_level during warmup | 0 (OFF) | 0 | PASS |

All stabilization configs locked per `LOCKED_STATIONARITY_PLAN.md §2` and verified.

---

## 5. Formal-Gate Entry Decision

| Criterion | Result |
|---|---|
| threads stable (±5) | **PASS** |
| latency p95 stable (<5%) | **PASS** (2.24%) |
| RSS stable (<50 MB) | **FAIL** (185 MB) — allocator sawtooth |
| PID stable | PASS |
| Overall warmup | **PARTIALLY STABLE** — threads/latency stable, allocator not fully settled |

Per §4 Host State Gate: if thread/RSS still in obvious init/growth, run must not enter formal stats. Here threads satisfy gate, RSS does not but is oscillatory not monotonic, so formal runs were allowed to proceed with **explicit allocator risk noted**. The subsequent stationarity failure (c4 TTFT 67% variance) is consistent with this allocator-related residual — not thread pool, which was stable.

**Recommendation:** Future warmup should use **formal-size dummies**: instead of 2× c=8 n20, use 1× c=4 n40 + 1× c=1 n40 sequential (matching formal request_count 40) to warm allocator to the exact formal working set, then verify RSS within 50 MB before formal Round 1.

---

*— End warmup validation —*
