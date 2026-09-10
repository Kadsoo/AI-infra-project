# RQ-1 — future solution notes (exploratory only, NOT implemented)

> Stage 3B instruction §13: record only; no implementation. Bottleneck candidates: (a) KV transfer is 30–40%+ of P95 TTFT at 25 Gbps for 8k–13k prompts (E1 anchored point); (b) no queue hotspot from cache-affine pairing at NVLink (E2) — the transfer leg, not the queueing leg, is the P/D liability.

Very short candidate directions (in no priority order):

- Fragmentation-aware KV transfer (fewer NCCL calls; the corpus's FlowKV 23,469→1 call pattern; Beluga sglist) — raises realized bandwidth, directly attacking the E1 fraction.
- Transfer/prefill overlap (Splitwise-style serialization removal) — only after measuring the unhidden path (E4 condition F serialized cell).
- Nothing for the queueing leg: H2's hotspot did not reproduce; don't engineer for a mechanism that didn't appear.
- Measure first: E3 microbenchmark (per-call overhead, fragmented-vs-contiguous, realized bandwidth) and E4 serving before any optimization work.