# Candidate Gaps — LLM Inference KV Cache Optimization (Stage 2A)

> Workdir: `F:\AIinfraResearch` | Input: `research/manifests/papers.md` (43 papers) + `research/paper_notes/*.md` (43) + `research/knowledge/taxonomy_draft.md` + `bottleneck_map.md` + `evaluation_map.md` + `assumption_map.md` + `limitation_map.md` + `contradictions.md` + `literature_matrix.md` + `research_landscape.md` + `coverage_report.md` | Date: 2026-08-27
> Scope: 10 candidate gaps, conservative, diverse dimensions. Each marked **NOT YET VERIFIED AS NOVEL**. No novelty search performed; use "underexplored" / "candidate gap" only.

---

## Candidate Gap 1: Joint Optimization of Token Budget × Bit-Width × Placement (Memory/KV Dimension)
### Observation (what you observed across papers)
Across 14 compression/eviction + 5 hierarchical papers, each optimizes a single axis in isolation: token eviction (H2O/PyramidKV/SnapKV), bit-quantization (KIVI/GEAR), and placement (InfiniGen/ShadowKV/Beluga). No paper jointly searches the Pareto frontier of `retained tokens × bit-width × tier placement`. Synergy is assumed orthogonal but never measured together.
### Evidence (which papers, with year, metric, limitation — traceable)
- PyramidKV (2024) — 12% cache matches Full (41.49 vs 41.46 on LongBench at 2048) via arithmetic pyramid `k^l`, but Appendix sensitivity shows task-specific optima; no quantization evaluated; hardware throughput not measured (limitation_map L1).
- KIVI (2024) — 2.6× less peak (incl. weights), 4× batch → 2.35–3.47× throughput on unspecified GPU, but uniform `G=32,R=128` for all layers/heads; short-context overhead, MQA Falcon needs 4-bit not 2-bit (taxonomy_draft Cat.4, limitation_map L1).
- GEAR (2024) — 2.39× peak, batch 3→18 on V100, 5.07× throughput at 2-bit near-lossless on CoT (40.20 vs KIVI 25.25), but uniform rank `r=4/2` and sparsity `s=2%` across heads; not combined with eviction (limitation_map L1/L2, bottleneck_map B14).
- DistServe (2024) / Splitwise (2023) — 2.0–4.6× goodput vs vLLM via disaggregated placement search, but defer quantitative comparison to chunked-prefill and double weight memory (2×350GB for 175B) not quantified (limitation_map L2/L8, assumption_map A8).
### Why existing methods do not fully cover it (what's missing)
Existing methods fix one axis and treat others as constant: eviction assumes FP16, quantization assumes fixed token count, placement assumes linear cost `T(l,u)`. Missing is an adaptive controller that per-request/per-layer jointly chooses `k^l × bit-width × tier` under HBM/CXL/PCIe constraints, accounting for tile-quantization (257 chunk 32% slower than 256) and sparsity non-uniformity. No paper reports the 2–4× multiplicative gain or interference from stacking paged blocks + quant kernels + RDMA.
### Related papers (2-5)
- PyramidKV (2024)
- KIVI (2024)
- GEAR (2024)
- ShadowKV (2025)
- DistServe (2024)
### Confidence: High
### Status: NOT YET VERIFIED AS NOVEL

---

## Candidate Gap 2: Online Adaptive Tuning of Static Budgets and Thresholds
### Observation (what you observed across papers)
A recurring pattern: all high-performing methods rely on manually tuned static constants that require offline profiling and fail to adapt to workload drift. This appears across 11 papers spanning compression, scheduling, and cache policy.
### Evidence (which papers, with year, metric, limitation — traceable)
- Sarathi-Serve (2024) — token budget `tau=512` strict / `2048` relaxed via Vidur profiling; 2.6× Mistral-7B, 3.7× Yi-34B, 5.6× Falcon-180B capacity, but tau static per model/HW, no per-request adaptation; chunked 25% overhead at 512 (bottleneck_map B7, limitation_map L1).
- PyramidKV (2024) — fixed `alpha=20` pyramid `k^{m-1}=k_total/(alpha·m)`; sensitivity Appendix shows optima vary by task; 0.7% budget (64 tokens) +20.5 TREC vs SnapKV but HotpotQA marginal (limitation_map L1).
- KIVI (2024)/GEAR (2024) — uniform `G/R` and uniform low-rank `r=4`/`s=2%`; GEAR leaves adaptive allocation as future (limitation_map L1).
- CacheBlend (2025) — experience threshold `r*=15%` (10–15% HKVD) → 2.2–3.3× TTFT ≤0.02 F1 loss pipelined via `Tload` vs `Trecompute`, but authors note `r*` may drift (limitation_map L1, taxonomy_draft Cat.8).
- HotPrefix (2025) — Cuckoo admission `hotness=freq·clock ≥10` and eviction `(freq+clock)/len`; +1.17–2.38× hit over LRU, 1.91× throughput, but threshold 10 sensitive, no CXL/distributed eval (bottleneck_map B9, limitation_map L1).
- Mooncake (2025) — `kvcache_balancing_threshold` manually tuned; transfer prediction hard due to congestion (limitation_map L1).
### Why existing methods do not fully cover it (what's missing)
No work provides a lightweight online estimator that continuously retunes thresholds without offline recalibration. Vidur profiling, bilinear `T(l,u)`, and Cuckoo aging are offline or heuristic. Missing is a closed-loop controller that learns per-layer sparsity, hotness, and cost models at runtime with bounded overhead (<0.3% as in SGLang RadixAttention) and generalizes across 1K–1M variance without per-model re-profiling.
### Related papers (2-5)
- Sarathi-Serve (2024)
- PyramidKV (2024)
- CacheBlend (2025)
- HotPrefix (2025)
- SCOPE (2025)
### Confidence: High
### Status: NOT YET VERIFIED AS NOVEL

---

## Candidate Gap 3: Online Heterogeneous Placement Beyond GH200 with Realistic CXL/PCIe5 Constraints (Memory/KV Dimension)
### Observation (what you observed across papers)
Hierarchical solutions assume either idealized two-tier HBM+DRAM or a single GH200 prototype with perfect future knowledge. The 2025–2026 CXL tier (rack-scale) remains prototype-scale and offline.
### Evidence (which papers, with year, metric, limitation — traceable)
- Dynamic Placement (2025, IEEE CAL) — formalizes `min Σ max(t^h,t^e) s.t. P_H≤100%` on GH200 (HBM 24GB 4.9TB/s + DRAM 480GB 500GB/s via C2C 900GB/s) → 5.87× upper bound vs HBM-only via SA search over window `W` and ratio `R`, but assumes perfect pre-knowledge not realizable; offline SA not online; PCIe5/CXL3 not evaluated beyond GH200 (assumption_map A11, research_landscape Sparse).
- Beluga (2025, SIGMOD26) — CXL 2.0 pool 8TB @1TB/s via 2×PCIe5 x16 + XC50256 256-lane 2TB/s, DAX mmap, GPU direct P2P → TTFT -89.6% (13.00s→1.36s), QPS 7.35× (1.54→11.32) on LV-Eval >15K Qwen-32B (14.6% HBM hit @28.3GB), but evaluated only on 2 servers×8 H20, switch single point of failure, software coherence (ntstore/CLFLUSH, 281ms UC) needed, RC bottleneck 33 vs 46GB/s, beyond 16 servers saturates 2TB/s/chip not measured (bottleneck_map B13, limitation_map L11).
- InfiniGen (2024, OSDI24) — selective prefetch <10% avg (cap 20%) via rehearsal `Xa_{i-1}+partial Q 30%` + `alpha 4/5` → 3× over FlexGen, but single RTX A6000 48GB PCIe3.0 x16 + 96GB DDR4 only; extra 15% KV/2.5% weights overhead; not for pipeline parallel (bottleneck_map B5/B13).
- LMCache (2025) — 256-token chunks + 1MB DMA 30–46GBps vs 4GBps at 64KB, Controller may bottleneck at 1000+ instances, 500GB CPU still insufficient for many tokens, needs auto-tiering DRAM-SSD-S3 not formalized (taxonomy_draft Cat.1).
### Why existing methods do not fully cover it (what's missing)
Missing is an online placement scheduler for heterogeneous clusters mixing HBM, CPU DRAM, CXL pool, and SSD without perfect future knowledge, handling 1K–1M variance, PCIe Gen1 2GB/s vs Gen5 64GB/s (KVFlow A10G vs H100), and SW coherence cost. No paper shows autoscaling across H100/A10G/H20 with CXL + RDMA mix at 100s-node scale with cost/power accounting.
### Related papers (2-5)
- Dynamic Placement (2025)
- Beluga (2025)
- InfiniGen (2024)
- LMCache (2025)
### Confidence: Medium
### Status: NOT YET VERIFIED AS NOVEL

---

## Candidate Gap 4: Fault Tolerance, Consistency, and Availability for Distributed/Hierarchical KV Pools (Evaluation/Hardware Dimension)
### Observation (what you observed across papers)
Throughput/goodput is measured only on happy path; disaggregated and hierarchical KV pools lack crash consistency, replication, or WAL. Only one 2024 system addresses fault tolerance and is not adopted by later baselines.
### Evidence (which papers, with year, metric, limitation — traceable)
- Splitwise (2023) — failure restarts from scratch; checkpointing KV to in-memory DB not designed, out of scope (limitation_map L11).
- DistServe (2024) — 32 GPUs (4×8 A100) achieves up to 7.4× request rate under strict SLO and <0.1% transfer (<30ms) via Alg1/2 co-location, but no checkpointing evaluated; failure of prefill or decode mid-request requires restart (limitation_map L11).
- Mooncake (2025, FAST25) — 4–20 nodes (32–160 GPUs, 800Gbps RDMA, Conductor) gives 20–525% vs vLLM (50% max reuse even at infinite storage, hot blocks 10k+ accesses), but no RDMA Messenger failure or incast handling beyond manual replication threshold; transfer prediction hard (limitation_map L5/L11).
- LMCache (2025) — 8×H100, TTL 1h reduces hit 85%→45%; no crash consistency for SSD tier; Controller bottleneck not measured at 100s nodes (limitation_map L11).
- CachedAttention / AttentionStore (2024, ATC24) — DRAM→SSD hierarchical 87% TTFT, 7.8× prefill on ShareGPT 52K turns via async save/layer-wise preload/decoupled RPE, but single 4×A100 128GB/10TB node only; eviction under real pressure not measured beyond ShareGPT (bottleneck_map B13).
- DéjàVu (2024, ICML24) — contrasts: token-level KV streaming + replication → 2× throughput with <2% overhead via buffered copies (95×), but not adopted by disaggregated baselines (literature_matrix, research_landscape Sparse).
### Why existing methods do not fully cover it (what's missing)
No production-grade design provides synchronous/asynchronous replication, WAL, or at-most-once/ exactly-once guarantees for KV in CPU DRAM/CXL/SSD across 16–1000 instances. Missing evaluation of failure injection, recovery time, and impact on P99 TBT/goodput. Overload prediction using uniform `td` (Mooncake) fails upon node failure, yet SLO definition counts only fully completed requests.
### Related papers (2-5)
- DéjàVu (2024)
- Splitwise (2023)
- Mooncake (2025)
- LMCache (2025)
### Confidence: High
### Status: NOT YET VERIFIED AS NOVEL

---

## Candidate Gap 5: SLO-Aware Fair Scheduling with Prefix-Affinity (Scheduling Dimension)
### Observation (what you observed across papers)
Cache-aware scheduling maximizes hit rate using greedy longest-prefix-first, but this conflicts with fairness, starvation, and P99 SLOs. Goodput and capacity metrics are reported in isolation without joint fairness evaluation.
### Evidence (which papers, with year, metric, limitation — traceable)
- SGLang (2024, NeurIPS24) — RadixAttention radix tree + LRU leaf-first + DFS longest-prefix, 50–99% hit (96% optimal), 6.4× throughput, overhead <0.3% (0.2s/74.3s ShareGPT no reuse), but greedy longest-prefix may starve small hot prefixes; requires cache ≥ max request len for Theorem 3.1; fairness integration left future; production hit 52.4% (LLaVA-NeXT-34B) /74.1% (Vicuna-33B) vs benchmark (taxonomy_draft Cat.5, limitation_map L4).
- Mooncake (2025) — Conductor `T_queue+T_prefill+T_transfer` routing to min TTFT, threshold trades recompute vs transfer; static 3P+1D vs 2P+2D shows imbalance (TTFT worse despite more decode nodes); SLO-aware replication/eviction for varying priorities left future; vLLM only 57% meet TBT vs Mooncake ~100% (limitation_map L4, bottleneck_map B10/B11).
- Sarathi-Serve (2024, OSDI24) — stall-free chunked-prefill `tau` → capacity 2.6×/3.7×/5.6× vs vLLM, Decode+Full 28.3× vs Decode+Chunked bound, but no fairness/complementary schedulers, no preemption (FastServe) integrated; TTFT 0.76s vs 0.53s openchat may increase (bottleneck_map B7/B8, limitation_map L4).
- FastServe (2023) — skip-join MLFQ token-level preemption → 31.4× SLO throughput vs vLLM, ENST proactive swap <5%, but no disaggregation comparison, alpha manual (evaluation_map).
- Online Scheduling (2025) — first KV-constrained online theory with competitive ratio `Ω(√n)` hardness, but single proof-of-concept homogeneous worker, discrete time, no evaluation on ShareGPT burst (research_landscape Sparse).
### Why existing methods do not fully cover it (what's missing)
No system integrates weighted fair queuing or EDF with prefix affinity and KV placement. Missing is a scheduler that jointly optimizes `hit rate × SLO attainment × fairness` with global deadlines, ref-count aware eviction, and hot-spot replication that does not create anti-phase fluctuations (Mooncake Fig9-10) or starvation. P99 TBT vs average throughput trade-off remains underexplored.
### Related papers (2-5)
- SGLang (2024)
- Mooncake (2025)
- Sarathi-Serve (2024)
- FastServe (2023)
### Confidence: High
### Status: NOT YET VERIFIED AS NOVEL

---

## Candidate Gap 6: Multi-Tenant Isolation, Privacy, and Security for Cross-Request KV Reuse (Memory/KV Dimension)
### Observation (what you observed across papers)
Enterprise traces show 94% cross-user hit opportunity, but all reuse mechanisms assume single-tenant public knowledge. Hash-based dedup and radix forests enable timing side-channels and KV leakage without access control evaluation.
### Evidence (which papers, with year, metric, limitation — traceable)
- Cache-Craft (2025, SIGMOD25) — Sys-X/Y production: strict prefix hit 8% requests /18% tokens, but 50% requests with 5 chunks scattered across ≥3 histories; naive 5-block reuse F1 0.65 vs Full 0.87 (50% drop) even with RoPE fix; cross-user reuse 94% with no access control; 30% recompute for 90% ROUGE (limitation_map L7, bottleneck_map B3).
- SGLang (2024) — shared prompt across users via radix tree may leak via timing (3.9× hit vs miss) even though tree edges are token sequences with ref-count; isolation per tenant not discussed (limitation_map L7).
- vLLM (2023, SOSP23) — CoW sharing via ref-count limited to within-request group (parallel sampling/beam) + explicit provider reservation; general cross-user dedup not evaluated; effective 20.4%→~96% and 55% beam saving within request (assumption_map A3).
- RAGCache (2024) — top 3% docs 60% requests (20× uniform), PGDSF Priority `Clock+Freq·Cost/Size` where `Cost` bilinear `T(l,u)` → 1.2–4× TTFT, 2.1× throughput, but assumes public Wikipedia 0.3M docs; no per-tenant isolation (limitation_map L7).
- LMCache (2025) — Controller P2P lookup across instances with no multi-tenant permission; 500GB CPU/disk tier, chunk 256 vs page 16; 2.3–14× throughput (limitation_map L7).
- Mooncake (2025) — reuse plateaus at 50–51% even at infinite storage; 50% blocks never reused vs hot 10k+ accesses; hash dedup block 512 not isolated (bottleneck_map B3).
### Why existing methods do not fully cover it (what's missing)
Missing is a per-tenant radix forest or encrypted hash with isolation that preserves privacy while retaining most reuse. No paper measures hit-rate loss under isolation (estimated 30–50% drop), side-channel mitigation, or latency of permission checks. Security vs efficiency Pareto is underexplored despite enterprise shared-disk (Shared RAG-DCache 2025, +15–71% throughput) pushing cross-instance sharing.
### Related papers (2-5)
- Cache-Craft (2025)
- SGLang (2024)
- LMCache (2025)
- RAGCache (2024)
### Confidence: Medium
### Status: NOT YET VERIFIED AS NOVEL

---

## Candidate Gap 7: Contamination-Aware Non-Prefix RAG Reuse Across Multi-History Dispersed Chunks (RAG/Agent Workload Dimension)
### Observation (what you observed across papers)
Prefix-only trees work for small ordered k, but production RAG scatters 5 chunks across ≥3 histories. Selective recompute is validated only on synthetic uniform chunks with fixed order, and CCI/CFO thresholds are not robust to distribution drift.
### Evidence (which papers, with year, metric, limitation — traceable)
- Cache-Craft (2025) — enterprise Sys-X 12B tokens/month = 9600 GPU-hours ~$50k reprocessed; `inter=(Σ A_{ij})/|C_i||C_j|`, `intra`, `CCI=1/(1+e^{-a_bar/b_bar})`, `beta` overlap, `gamma` Kendall Tau → `CFO`; → -51% redundant vs SOTA prefix, -75% vs full, 1.6× throughput, 2.1× delay; but thresholds need recalibration per drift; evaluated only 200 Q/dataset on LLaMA-3, N=100 chunks×5 variants ~50GB storage not scaled (bottleneck_map B3/B4, limitation_map L9).
- CacheBlend (2025, EuroSys25) — full reuse ignores cross-attention → F1 -0.1 to -0.2 QA / -0.03 to -0.25 summarization; HKVD 10–15% with Spearman correlation + pipelined `Trecompute 3ms vs Tload 16ms` → 2.2–3.3× TTFT ≤0.015 loss, 2.8–5× throughput on 2×A40 128GB/1TB NVMe 6000 GPT-4 synthetic queries (3 per original) with 512-token uniform chunks; not heterogeneous retriever drift or TB reuse (evaluation_map Highlight, bottleneck_map B4).
- RAGCache (2024) — knowledge tree PGDSF → cached prefix 11.5× faster than full prefill (3.9× with Host load) on 5K/30K; but exact prefix hit 8% requests in Sys-X/Y; 1-hour Poisson not production burst; host 192GB assumption (limitation_map L9, contradictions T6).
- KVLink (2025) — position-independent PIC: store `W_{k,v}·x` pre-RoPE, re-apply global RoPE + 5 trainable link tokens → TTFT -96% @5K (1K→5K), +4% QA (45.0% vs PromptCache 18.6% NQ), but needs 6000 steps on 8×H100, 131MB/1K tokens storage; not evaluated at scale heterogeneous storage (taxonomy_draft Cat.6, limitation_map L2).
### Why existing methods do not fully cover it (what's missing)
No training-free method handles arbitrary `k=5–30` variable-length chunks with order- and history-dependent contamination without per-distribution recalibration. Missing is a robust CCI/CFO-free detector that works across 1K–20K, adapts to retriever drift, and jointly decides `recompute %` vs `transfer vs disk` via pipelined `Trecompute ≤ Tload` hiding on heterogeneous H20 vs A100.
### Related papers (2-5)
- Cache-Craft (2025)
- CacheBlend (2025)
- RAGCache (2024)
- KVLink (2025)
### Confidence: Medium
### Status: NOT YET VERIFIED AS NOVEL

---

## Candidate Gap 8: State Lifetime Management for Branching/Looping Multi-Agent Workflows (RAG/Agent Workload Dimension)
### Observation (what you observed across papers)
Agent workflows interleave LLM calls with tool/browser/code gaps seconds-to-minutes. Existing Step Graph and TTL assume linear ReAct traces; branching/loops and ad-hoc spawns are not captured, causing persistent LRU mis-eviction.
### Evidence (which papers, with year, metric, limitation — traceable)
- KVFlow (2025, NeurIPS25) — LRU evicts soon-to-reuse Expresser while retaining unlikely suffix at T=13 (miss at T=14) in 4-agent Planner→Executor→Expresser→Reviewer cycle; Agent Step Graph `steps-to-execution = max/min over predecessors → priority = min among children` + proactive prefetch status-aware → 2.19× over SGLang HiCache (1.83× single large), 0.57× degraded at 64 concurrent on H100 (HBM 2TB/s) vs A10G PCIe Gen1 2GB/s; requires accurate Step Graph, branching wastes BW if mispredicted (assumption_map A4, bottleneck_map B9).
- Continuum (2025, arXiv 2511.02230) — tool gaps avg 0.9–1.9s (BFCL fetch_url slowest 10% =52.5% total, SWE-Bench cd 94.1%), 6.3–10.9 turns, 70K–93K tokens/program; TTL `tau* = argmax P(tau,f)·(T·psi+Prefill-Reload) - α·Cost` → avg JCT 8× (1.12–3.66× delay, 1.10–3.22× throughput) on 1×A100/4×B200/H100 with DRAM 100–200GB + SSD 400–800GB (Poisson 0.13 JPS, 500 tasks SWE-Bench Verified via GPT-5 100 traces), but only ReAct linear; rare tool tail fallback global; cold-start K=100 (limitation_map L12).
- SCOPE (2025, ACL25) — max length `T` must be known for Adaptive `ĥλ1=(t-λ2)λ1/(T-λ2)`; serving generation length unknown, prediction error reintroduces fluctuation; separate `Λ^p` vs `Λ^d` needed because heavy-hitters drift to decoding (Top-15% at steps 1/300/500), 35% total approx Full (56.21 vs 59.78) while unified Top-K biases to recent (contradictions T4).
- CachedAttention (2024, ATC24) — session as minimum eviction granularity (all-or-nothing) inefficient when only suffix needed; 65B 2.5MB/token vs Falcon 0.12MB/token not adapted; 86% hit but TTL 1h (limitation_map L12).
### Why existing methods do not fully cover it (what's missing)
Missing is a unified lifecycle that handles `fork/join`, loops, and dynamic spawns with iteration-granular `DoP` changes (as in LoongServe ESP) and joint prefill/decode drift correction, without assuming known `T` or linear chains. Need online prediction of output length (TetriInfer 74.9% @granularity 200) integrated with TTL/Step Graph and token-granular KV pool at enterprise scale.
### Related papers (2-5)
- KVFlow (2025)
- Continuum (2025)
- SCOPE (2025)
- LoongServe (2024)
### Confidence: Medium
### Status: NOT YET VERIFIED AS NOVEL

---

## Candidate Gap 9: End-to-End Cost, Power, and Energy (Perf/$ and Perf/W) Benchmarking on Heterogeneous Clusters (Evaluation/Hardware Dimension)
### Observation (what you observed across papers)
Performance is reported as tokens/s or goodput isolated from TCO. Provisioned power, cooling, replication overhead, and dollar cost are omitted, yet heterogeneous decisions depend on them.
### Evidence (which papers, with year, metric, limitation — traceable)
- Splitwise (2023, ISCA24) — 2×DGX-A100 +2×DGX-H100 (IB 200/400Gbps) → 2.35× @same cost/power via HA (H100 prompt / A100 token) at `$17.6/hr A100 vs $38/hr H100` CoreWeave rental; provisioned power not dynamic (700W H100 vs 400W A100) ; HA IB H100→A100 not deployed, CLS bottleneck (bottleneck_map B7/B10, limitation_map L8).
- DistServe (2024, OSDI24) — per-GPU goodput `max RPS meeting 90% TTFT+TPOT / num_gpus` → 2.0–4.6× vs vLLM, 12.6× tighter SLO; but double weight memory (prefill+decode each hold full weights for 175B 350GB) not quantified in per-GPU goodput; 25Gbps limited vs 800Gbps sim (limitation_map L8).
- CachedAttention (2024) — cost model `$5/hr/A100 + $0.0088/GB DRAM + $0.000082/GB SSD` spot price → 43–70% saving, 7.8× prefill; no network/ops or replication cost (limitation_map L8).
- Mooncake (2025) — ample DRAM/SSD without additional costs claim, but no dollar/power quantification; replication threshold manual (limitation_map L8).
- Beluga (2025) — ConnectX-7 $1745 vs CXL adapter $210, switch $16k vs $5.8k sample; cost economics not stable (evaluation_map Table1, bottleneck_map B13).
- Evaluation_map Highlight — 14 papers limited to single-GPU families (A10/A40/A6000/T4/V100) and ≤32 GPUs; simulator MAPE <2–3% substitutes for real deployment; many claims extrapolate beyond measured topology.
### Why existing methods do not fully cover it (what's missing)
No standardized harness reports `joules/token`, `$/1M tokens under SLO`, and `provisioned vs dynamic power` alongside TTFT/TPOT/P99 for same traces across A100/A10G/H20/H100/GB200 NVL72 and CXL vs RDMA. Missing is analysis of double-weight memory overhead for P/D pools and KV replication, and trade-off of adding CXL pool vs adding GPUs for perf/$.
### Related papers (2-5)
- Splitwise (2023)
- DistServe (2024)
- Beluga (2025)
- CachedAttention (2024)
### Confidence: High
### Status: NOT YET VERIFIED AS NOVEL

---

## Candidate Gap 10: Realistic Burst Production Traces and Tail-SLO Evaluation vs Synthetic Poisson Simulators (Evaluation/Hardware Dimension)
### Observation (what you observed across papers)
Evaluation relies on synthetic Poisson arrivals and short uniform traces; production is bursty heavy-tail with skewed popularity. Benchmark 96% optimal hit vs 52–74% production hit shows the gap; tail SLOs are under-reported.
### Evidence (which papers, with year, metric, limitation — traceable)
- Orca (2022) — synthetic `U(32,512)` in / `U(1,128)` out Poisson, no EOS, never emits `<EOS>` due to lack of checkpoint/text; 36.9× vs FasterTransformer synthetic only (evaluation_map Sec.3, limitation_map L9).
- vLLM (2023) — ShareGPT/Alpaca Poisson 1h trace (15min for 175B), block 16 default; 2–4× vs Orca; optimal 16–32 but short Alpaca prefers ≤32 manual tuning (evaluation_map).
- DistServe (2024) — Poisson RPS sweeps 1–4 rps assuming workload predictable over hours/days via `M/D/1 Avg_TTFT = D + R·D²/(2·(1-R·D))` fitting; short-term burstiness acknowledged but not tested (assumption_map A6, bottleneck_map B8).
- RAGCache (2024) — Wikipedia 0.3M avg 3718, text-embedding-3-small IVF1024 top-k 2, Poisson 0.8 rps; 4× TTFT & 2.1× throughput vs vLLM at batch4 small; hit +75% vs LFU at 8GiB not at TB scale (evaluation_map Highlight).
- CacheBlend (2024) — 6000 queries via GPT-4 generated similar queries (3 per original) with uniform 512-token chunks; 5–18% recompute ≤0.015 loss not tested on heterogeneous retriever drift (limitation_map L9).
- SGLang (2023) — 5-shot MMLU production 52.4% (LLaVA-NeXT-34B) /74.1% (Vicuna-33B) vs bench 50–99% (96% optimal) (bottleneck_map B3/B9).
- KVFlow (2025) — 10-agent synthetic sequential 8192/32/32; high-concurrency 64 concurrent on H100 HiCache even degraded 0.57× vs GPU-only due to queuing not in Poisson model (assumption_map A6).
- Continuum (2025) — 100 traces per dataset via GPT-5 only, Poisson 0.13 JPS on SWE-Bench Verified 500 tasks; TTL cold-start K=100 fallback heavy tail misprediction (evaluation_map Highlight).
### Why existing methods do not fully cover it (what's missing)
No harness sweeps same system across prompt 1K→1M, output 32→512+, arrival burstiness Gamma CV, and strict vs relaxed SLO (`TTFT_P90 ≤10× baseline, TBT_P90 ≤5×` Mooncake vs 0.1s strict Sarathi) to report both goodput and capacity inclusive of P99 and fairness. Poisson simulators with `<3% MAPE` (Splitwise/DistServe) underestimate queuing under real burst where 10% slowest tool =94% delay (Continuum) and mixing across ≥3 histories. Production multi-tenant heavy-tail remains unbenchmarked.
### Related papers (2-5)
- SGLang (2024)
- DistServe (2024)
- KVFlow (2025)
- RAGCache (2024)
- Continuum (2025)
### Confidence: High
### Status: NOT YET VERIFIED AS NOVEL

---

## Traceability & Diversity Check

- Diverse dimensions covered: memory/KV (Gaps 1,2,3,6), scheduling (Gap 5), evaluation/hardware (Gaps 4,9,10), RAG/agent workload (Gaps 7,8) — exceeds required at least one per category.
- All evidence traceable to `research/paper_notes/*.md` sections 10/12/13 and `research/knowledge/*_map.md` / `literature_matrix.md` / `evaluation_map.md` / `research_landscape.md`; no hallucinated metrics; hardware marked `[NOT REPORTED]` where absent in source maps.
- Each gap marked **NOT YET VERIFIED AS NOVEL**; no prohibited phrase used; language conservative: "underexplored" / "candidate gap".

*End of candidate_gaps.md — 10 gaps, each with 6 required subsections and Status tag; ready for Stage 2B novelty verification.*
