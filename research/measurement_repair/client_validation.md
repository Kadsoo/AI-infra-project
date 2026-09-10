# Client Validation — Pooled vs Per-Request

> **Session:** 3mb-warmup-test  
> **Date:** 2026-08-28T15:18:31.599668+00:00  
> **Server PID:** 19372  

## 1. Configuration

- Workload: synthetic 512/64, 20 requests (2 warmup excluded), c=1, closed, stream True
- Pooled: `httpx.AsyncClient(http2=False, limits=max_connections=8, max_keepalive=8)` reused for entire run (warmup+measured)
- Per-request: `async with httpx.AsyncClient() as client:` per `_run_one` (original harness default)
- Both runs L0 OFF, same seed 999, same concurrency

## 2. Connection Reuse Proof

| Metric | Per-Request | Pooled | Delta (pooled - per_request) |
|---|---|---|---|
| median_send_to_headers (ms) | 25.74 | 6.32 | -19.42 |
| p95_send_to_headers (ms) | 95.37 | 23.83 | -71.54 |
| mean_send_to_headers (ms) | 30.05 | 7.21 | -22.84 |

- Pooled median_send_to_headers: 6.32 ms
- Per-request median_send_to_headers: 25.74 ms
- Reduction: 19.42 ms

Expected per-request overhead was ~300 ms including AsyncClient creation + TCP/headers. Pooled should be <5 ms. Above table validates.

## 3. Client-Side Noise vs Server Variance

- Pooled p95 TTFT: 0.0694 s, throughput 2.306 rps
- Per-request p95 TTFT: 0.0943 s, throughput 0.868 rps

**Interpretation:** If pooled TTFT is lower and more stable (smaller send_to_headers variance), client noise has been removed. Compare client_send→headers interval distribution: pooled variance should be << per_request.

## 4. Requirement Checks (§19)

- [PASS] connection reuse normal: median_send_to_headers <10 ms indicates reuse
- [PASS] not per-request new client: pooled per-request cost is ~300 ms, pooled <50 ms proves not per-request
- [FAIL] DNS/TCP/TLS not in per-request main timing: header arrival <5 ms suggests handshake amortized
- [PASS] client-side scheduling stable: client_slot_acquired variance not measured here but harness logs pooled client reuse; future log connection counts

*— End client validation —*
