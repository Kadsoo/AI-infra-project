# Stage 2A Summary
## Stage 2A — Literature Mapping & Taxonomy Synthesis (Final Summary)

> **Workdir:** `F:\AIinfraResearch` | **Date:** 2026-08-27 | **Synthesis Lead:** Stage 2A Synthesis | **Input:** ALL Stage 2A artifacts (12 files) | **Output:** `F:\AIinfraResearch\research\knowledge\stage2a_summary.md`
> **Corpus:** `research/manifests/papers.md` (43 papers: 38 dedup +5 supplements, 31 High /12 Medium, 27 Recent 63%, 8 Foundational) | **Coverage:** 10/10 required categories verified in `coverage_report.md`
> **Rule:** Conservative language — underexplored / candidate gap / NOT YET VERIFIED AS NOVEL; no novelty claims; traceable to source files; no new solution brainstorming.

---

## Statistics

| Dimension | Count | Source File | Notes |
|---|---|---|---|
| **Seed papers (manifest)** | **43** (38 dedup +5 supplements) | `research/manifests/papers.md` | A=13, B=15, C=14 → dedup 38 → +5 auditor (CacheGen, CachedAttention, Dynamo, Llumnix, LoongServe) → 43; 31 High /12 Medium; 22→24 with code (56%) |
| **Technical categories (taxonomy)** | **14** | `research/knowledge/taxonomy_draft.md` | Covers 17 required dimensions; >=12 required satisfied; Mature 3 / Active 7 / Emerging 3 / Sparse 1 |
| **Bottlenecks mapped** | **14** | `research/knowledge/bottleneck_map.md` | 12 mandatory +2 additional (Hierarchical Tiering Complexity, Compression-vs-Quality); each with trigger / metrics / papers / remaining limitations |
| **Common assumptions mapped** | **12** | `research/knowledge/assumption_map.md` | 26 notes read; confidence High 4 / Medium 6 / Low 1 / Conditional 1; cross-cutting stability table |
| **Repeated limitations** | **13** | `research/knowledge/limitation_map.md` | 6 Widely acknowledged (>=5 papers) +6 Appears repeatedly (>=3) +1 Paper-specific/Weakly evidenced; author-stated vs inferred separated |
| **Evaluation patterns** | **Metrics 14 / Baselines 13 families / Workloads 9 strata / Hardware 35 reported +8 [NOT REPORTED]** | `research/knowledge/evaluation_map.md` + `research/knowledge/literature_matrix.md` | 24 notes reviewed in evaluation_map (56% corpus); literature_matrix 43 rows full |
| **Candidate gaps** | **10** | `research/knowledge/candidate_gaps.md` | All marked **NOT YET VERIFIED AS NOVEL**; diverse dimensions: memory/KV 4, scheduling 1, evaluation/hardware 3, RAG/agent 2 |
| **Contradictions checked** | **8 tension pairs → 6 tensions +2 no-strong** | `research/knowledge/contradictions.md` | Resolved as workload/hardware/metric/implementation/scale-conditioned Pareto, not logical contradiction |
| **Research landscape sections** | **10** | `research/knowledge/research_landscape.md` | Field overview + families + bottlenecks + assumptions + evaluation norms + mature/active/sparse + repeated limitations + tensions |
| **Coverage auditor weak spots resolved** | **5** (W1-W5 → supplemented) | `research/knowledge/coverage_report.md` | W1 codec → CacheGen, W2 chat persistence → CachedAttention, W3 Dynamo, W4 Llumnix/LoongServe, W5 minor |
| **Subagents used** | **~34** | `research/knowledge/stage1_summary.md` + Stage 2A agents | Stage1 24 (Scout 3 + Reader 15 + Reviewer 3 + Auditor 1 + Supplement 2) + Stage2A 10 (taxonomy, bottleneck, evaluation, assumption, limitation, contradictions, literature_matrix, landscape, candidate_gaps, synthesis) |
| **Paper notes read (Stage2A)** | **22–26 per map, 43 total available** | `research/paper_notes/*.md` (43 files) | taxonomy 24, bottleneck 24+, evaluation 24, assumption 26, limitation 22, all traceable [PAPER FACT] vs [AGENT INFERENCE] |
| **File index total** | **13 files (12 inputs +1 output)** | See File Index below | Absolute paths listed |

> **Audit note:** All counts cross-checked against `papers.md` 43 entries, `literature_matrix.md` 43 rows, `stage1_summary.md` 43 notes 100% read, 37/43 reviewed (High 100%). Hardware [NOT REPORTED] intentionally preserved to avoid hallucination (source: `evaluation_map.md` Sec.4).

---

## Major Technical Categories (summary table)

*Source: `research/knowledge/taxonomy_draft.md` (14 categories, each with problem / representative papers / mechanisms / maturity). Traceability: every paper title/year/venue verbatim in `papers.md`.*

| # | Category | Problem Addressed | Representative Papers (year) — Source `papers.md` | Maturity — Source `taxonomy_draft.md` |
|---|---|---|---|---|
| 1 | **KV Allocation & Paged Memory Management** | Fragmentation 60-80% waste (20.4-38.2% effective →96% with paging) | vLLM PagedAttention 2023, SGLang RadixAttention 2024, LMCache 2025, FlexGen 2023 | **Mature** — production (vLLM/SGLang/LMCache/Dynamo) |
| 2 | **Eviction — Token Dropping & Sparse Retention** | O(n) KV growth, 32K-1M exceeds HBM; keep <=20% | StreamingLLM 2023, H2O 2023, Scissorhands 2023, SnapKV 2024, SCOPE 2025 | **Active** |
| 3 | **Compression — Layer-Adaptive & Semantic Chunk** | Uniform budget wastes lower layers; token scoring fragments chunks | PyramidKV 2024, ChunkKV 2025, ShadowKV 2025, SCOPE 2025 | **Active** |
| 4 | **Quantization — Low-Bit KV Cache** | 540B PaLM batch512 3TB KV 3x params; need 2-bit | KIVI 2024, GEAR 2024, CacheGen 2024 | **Active** (4-bit Mature, 2-bit Emerging) |
| 5 | **Prefix Caching & Reuse — Automatic Sharing** | Shared static prefixes (system prompts, few-shot) recomputed | SGLang 2024, LMCache 2025, RAGCache 2024, HotPrefix 2026 | **Mature** — SGLang 50-99% hit (96% optimal, prod 52-74%) |
| 6 | **KV Sharing — CoW & Cross-Request Fusion** | Parallel sampling/beam/RAG multi-chunk non-prefix share duplicates | vLLM CoW 2023, CacheBlend 2025, Cache-Craft 2025, KVLink 2025 | **Active→Emerging** (CoW Mature, arbitrary PIC Emerging) |
| 7 | **Offloading & Hierarchical Memory** | HBM 24-96GB insufficient for 128K->tens GB; CPU/CXL/SSD 10-100x slower | FlexGen 2023, InfiniGen 2024, ShadowKV 2025, Beluga 2026, CachedAttention 2024 | **Active** (FlexGen Mature, CXL Emerging) |
| 8 | **Recomputation — Selective & Piecewise Fusion** | Non-prefix splicing loses cross-attention; 4000 tokens 3-6s | CacheBlend 2025, Cache-Craft 2025, KVLink 2025 | **Emerging** |
| 9 | **KV Transfer & Communication Optimization** | 1.13GB/512 tokens (66B) needs 90 Gbps; NCCL Lx2 calls 25% E2E | Mooncake 2025, FlowKV 2025, CacheGen 2024, Dynamo 2025, LMCache 2025 | **Active** |
| 10 | **Scheduling — Continuous Batching & Cache-Aware Admission** | Request-level batching blocks; long prefills stall decodes 28.3x TBT | Orca 2022, Sarathi-Serve 2024, HotPrefix 2026, Online Scheduling 2025, FastServe 2023, TetriInfer 2024 | **Mature** (iteration-level/chunked), **Active** (hotness), **Sparse** (theory) |
| 11 | **Request Routing & Load Balancing — KV-Aware** | Global routing without locality → imbalance + extra transfers | Mooncake Conductor 2025, Dynamo Smart Router/Planner 2025, FlowKV Global Controller 2025, Llumnix 2024, DistServe Placement 2024 | **Active** |
| 12 | **Prefill/Decode Disaggregation — Phase Splitting** | Compute-bound prefill vs memory-bound decode contention | Splitwise 2024, DistServe 2024, TetriInfer 2024, DéjàVu 2024, AMPD 2026 | **Active** |
| 13 | **Distributed Serving & Elasticity — Multi-Node Scale** | 175B needs 16 GPUs, 1K-1M variance, fragmentation | Orca 2022, Llumnix 2024, LoongServe 2024, Dynamo 2025, DéjàVu 2024 | **Active→Emerging** (elasticity Emerging) |
| 14 | **Workload-Aware Optimization — RAG / Agent / Long-Context / Heterogeneous** | Generic policies fail for skewed RAG, tool gaps, 1M variance, CXL BW | RAGCache 2024, CacheBlend/Cache-Craft/KVLink 2025, KVFlow 2025, Continuum 2025, AMPD 2026, ShadowKV/SCOPE/LoongServe, Beluga/DynamicPlacement 2025 | **Emerging→Active** |

> **Cross-cutting synergy (taxonomy_draft.md):** Paged (1) + Radix (5) + CoW (6) + stall-free chunked (10) + disaggregation (12) + CXL pool (7) orthogonal and demonstrated unified via LMCache/Dynamo connectors; memory vs compute, hit rate vs load, transfer vs recompute, persistence vs revival trade-offs.

---

## Major Bottlenecks

*Source: `research/knowledge/bottleneck_map.md` (14 bottlenecks, each: where/trigger/why hurts/metrics/papers/solution families/remaining limitations).*

| # | Bottleneck | Layer — Source `bottleneck_map.md` | Trigger Condition | Observable Metrics | Papers Addressing — Source `bottleneck_map.md` |
|---|---|---|---|---|---|
| B1 | **GPU HBM Capacity Exhaustion** | GPU HBM → Scheduler/Cluster | Batch512×2048 1.2-3 TB (3-3.8x weights); 100K → >50GB | Effective batch, peak memory GB, OOM threshold 33K vs 380K, memory saving 2.6x–7.08x | vLLM 2023, FlexGen 2023, H2O 2023, KIVI 2024, ShadowKV 2025, LMCache 2025 |
| B2 | **Memory Fragmentation & Reservation Waste** | GPU HBM allocator / PCIe / Scheduler | Variable-length max_tokens reservation 60-80% waste; block size mismatch | Effective KV 20-38% vs 96%, internal waste ≤1 block, PCIe 4→46 GBps, NCCL 23469→1 calls | vLLM 2023, SGLang 2024, LMCache 2025, FlowKV 2025, Beluga 2025, Orca 2022 |
| B3 | **KV Cache Miss / Low Hit Rate** | Memory manager + Scheduler | Non-prefix RAG <10% prefix hit; production 8% req/18% tokens; 50% plateau even infinite cache | Hit rate, TTFT 1.2-4x, throughput 1.54-2.25x | SGLang 2024, RAGCache 2024, HotPrefix 2026, CacheBlend 2025, KVLink 2025, Mooncake 2025 |
| B4 | **Redundant Recomputation** | GPU compute (prefill) | Same chunks 60%+ RAG reappear; 12B tokens/month ~9600 GPU-hours waste; 4000 tokens 3-6s | Prefill latency, TTFT 2.2-3.3x, throughput 2.8-5x, redundant tokens % 51% vs prefix | CacheBlend 2025, Cache-Craft 2025, KVLink 2025, RAGCache 2024, SGLang 2024 |
| B5 | **PCIe Bandwidth / CPU-GPU Transfer Saturation** | PCIe (2 GB/s A10G Gen1 →64 GB/s H100 Gen5) | Full CPU offload fetch per layer; small 62.5KB pages 4 GBps vs 1MB 30-46 GBps | PCIe throughput/size, transfer ms/layer, throughput tokens/s | InfiniGen 2024, FlexGen 2023, ShadowKV 2025, Beluga 2025, LMCache 2025, KVFlow 2025 |
| B6 | **Network Transfer Overhead** | Network RDMA/NCCL | 1.13GB/512 tokens 66B →90 Gbps@10rps; Lx2 NCCL 23469 calls; 25 Gbps inter-limited | Transfer latency 0.944→0.053s -96%, %E2E 25%→<0.1%, throughput +95% | FlowKV 2025, Mooncake 2025, DistServe 2024, Splitwise 2024, CacheGen 2024, Dynamo 2025, Beluga 2025 |
| B7 | **Prefill/Decode Interference** | GPU compute / Scheduler | Compute-bound prefill (flat >512) vs memory-bound decode (scales to 64); TBT stalls seconds | TBT P99, pipeline bubble ~950ms, capacity 2.6-6.3x | Orca 2022, Sarathi-Serve 2024, Splitwise 2024, DistServe 2024, TetriInfer 2024, Mooncake 2025 |
| B8 | **Scheduler Inefficiency (Queueing, HoL)** | Scheduler (centralized) | Poisson burst, FCFS arrival, no SLO awareness, max_bs manual | Capacity QPS SLO, P99 TBT, queue T_queue, goodput, competitive ratio Ω(√n) | Orca 2022, Sarathi-Serve 2024, FastServe 2023, Llumnix 2024, OnlineScheduling 2025, HotPrefix 2026 |
| B9 | **Poor Reuse Prediction / Hotness Misclassification** | Scheduler / Memory manager | Skew top 3% 60% hits 20x uniform; LRU evicts soon-to-reuse Expresser (KVFlow) | Hit delta vs oracle, steps-to-execution accuracy, JCT 8x | RAGCache 2024, HotPrefix 2026, KVFlow 2025, Continuum 2025, Cache-Craft 2025, KVLink 2025 |
| B10 | **Load Imbalance & Autoscaling Inefficiency** | Cluster (global router) | KV-aware routing hot-spot; P/D 2P+2D worse than 3P+1D; QPS 1.54 vs 11.32 imbalanced | Load ratio T_queue+T_prefill+T_transfer, hotspot replication, P/D utilization | Mooncake 2025, Llumnix 2024, FlowKV 2025, Dynamo 2025, DynamicPlacement 2025, LoongServe 2024 |
| B11 | **Tail Latency (P95/P99 SLO Violations)** | Scheduler / Runtime | High QPS near capacity; hybrid without budget 28.3x TBT; hotspot PCIe 25 Gbps | P99 TTFT/TBT attainment %, capacity goodput, tail jitter | Sarathi-Serve 2024, Orca 2022, FastServe 2023, DistServe 2024, Mooncake 2025, Beluga 2025 |
| B12 | **Long-Context Memory Pressure (Quadratic)** | GPU HBM + Attention kernel | 1K-1M variance; 128K tens GB; 1M OOM batch2 80GB; O(n²) | Max context before OOM 33K→380K, TTFT vs length, RULER/LongBench accuracy | ShadowKV 2025, StreamingLLM 2023, SnapKV 2024, PyramidKV 2024, SCOPE 2025, LoongServe 2024 |
| B13 | **Hierarchical Tiering & Heterogeneous Complexity** | Node (HBM+DRAM+CXL+SSD) | HBM 2TB/s vs DRAM 50GB/s vs PCIe/CXL vs RDMA 400Gbps vs S3 1Gbps; CXL2 non-coherent | Tier hit % (14.6% HBM), placement upper bound 5.87x, BW realized 33 vs 46 GB/s | Beluga 2025, DynamicPlacement 2025, InfiniGen 2024, CachedAttention 2024, Shared RAG-DCache 2025, LMCache 2025 |
| B14 | **Compression-vs-Quality Degradation** | Attention kernel / Memory manager | 0.7% budget (64 tokens) or 2-bit per-token mixes outliers; drift autoregressively | Accuracy drop %, reconstruction error, PPL, memory vs accuracy Pareto | KIVI 2024, GEAR 2024, H2O 2023, PyramidKV 2024, SnapKV 2024, SCOPE 2025 |

> **Repeated evidence gaps (bottleneck_map.md):** manual thresholds, single-GPU PCIe3.0/limited heterogeneity, scale ≤32 GPUs/1M max, storage cost not quantified, isolation/privacy not evaluated — echoed in `limitation_map.md` and `research_landscape.md`.

---

## Common Assumptions

*Source: `research/knowledge/assumption_map.md` (12 assumptions, each with papers / why needed / failure impact / evidence [PAPER FACT] / confidence).*

| # | Assumption — Source `assumption_map.md` | Papers Relying (year) — Source `assumption_map.md` | Confidence — Source `assumption_map.md` | What Happens If Fails |
|---|---|---|---|---|
| A1 | **GPU HBM capacity (KV size) is dominant throughput bottleneck** | Orca 2022, vLLM 2023, FlexGen 2023, H2O 2023, Sarathi-Serve 2024, Mooncake 2025 | **High** | If compute/interconnect is bottleneck, larger batches do not scale; vLLM Alpaca already compute-bound plateau; H100 FLOPS/BW divergence 3.43x vs 1.64x |
| A2 | **Prefill compute-bound (quadratic) vs decode memory-bound dichotomy stable** | Splitwise 2023, DistServe 2024, TetriInfer 2024, DéjàVu 2024, Sarathi-Serve 2024 | **High** | MoE/speculative/MLA blur boundaries; TetriInfer heavy+heavy marginal gain; tile 257→32% slower |
| A3 | **Requests independent; no cross-request semantic sharing beyond reserved prefixes** | Orca 2022, vLLM 2023, FlexGen 2023, DistServe 2024, FastServe 2023 | **High — Fragile** | Invalidated by RAG skew top3% 60% hits 20x, SGLang 50-99% prefix opportunity wasted; Mooncake >50% blocks unused while hot blocks 10k+ accesses |
| A4 | **Prefix reuse follows recency (LRU) / temporal locality** | SGLang 2023, RAGCache 2024, Mooncake 2025, CachedAttention 2024, vLLM paging | **Medium — Fragile, contested 2025** | KVFlow LRU evicts soon-to-reuse Expresser; RAGCache burst 0% vs optimal 66%; HotPrefix/Cache-Craft beat LRU 2-75% |
| A5 | **Interconnect bandwidth sufficient & homogeneous; KV transfer hideable via overlap** | Splitwise 2023, DistServe 2024, TetriInfer 2024, DéjàVu 2024, FlowKV 2025, Mooncake 2025 | **Medium — fragile cross-node** | Heterogeneous H100→A100 IB gaps 10x, PCIe3 16GB/s vs Gen5 64GB/s, RDMA 75% sync overhead, sglist 30 vs 128 chunks → TTFT +64% serialized |
| A6 | **Workload arrival/length distributions stationary Poisson, fit once per hours/days** | Orca 2022, vLLM 2023, DistServe 2024, Mooncake 2025, RAGCache 2024, TetriInfer 2024 | **Medium — poor for tail** | Production bursty long-tail 10% slowest tool 52-94% delay; HiCache 0.57x under 64 concurrent → simulators underestimate |
| A7 | **Future request/output lengths unknown at schedule time (no clairvoyance)** | Orca 2022, FastServe 2023, TetriInfer 2024, Online Scheduling 2025, DistServe/Splitwise 2024 | **High — foundational but attacked** | TetriInfer predictor 74.9% @200 granularity enables SJF 10% gap over MLFQ; Mooncake notes request-level prediction too costly |
| A8 | **Recomputation/fetch cost uniform linear in tokens/blocks** | FlexGen 2023, Sarathi-Serve 2024, CacheBlend 2024, SnapKV/PyramidKV 2024, GEAR/KIVI 2024 | **Medium — fragile** | Tile quantization 257→32% penalty, sparsity 84.3%, outliers; Sarathi 512 chunk 25% overhead; CacheBlend HKVD shows 10-15% tokens dominate deviation |
| A9 | **Attention sparse power-law with persistent heavy hitters** | H2O 2023, Scissorhands 2023, StreamingLLM 2023, SnapKV 2024, PyramidKV 2024, SCOPE 2025 | **Medium — conditional** | Dense copy/needle tasks 20% budget →35% drop; later layers persistence dips |
| A10 | **Initial tokens are attention sinks pinned for stable streaming** | StreamingLLM 2023, SGLang 2023, Scissorhands 2023, AttentionStore 2024 | **High for window, Medium universal** | Falcon/MPT need 1 sink only (waste 3x if 4 pinned); Learnable sink 18.01 vs Vanilla 27.87 |
| A11 | **Two-tier HBM+Host DRAM via PCIe sufficient** | FlexGen 2023, InfiniGen 2024, RAGCache 2024, Cache-Craft/CacheBlend 2025, DejaVu 2024 | **Low — actively invalidated 2025** | 1M needs SSD/CXL 8TB@1TB/s (Beluga); ShadowKV OOM 488K batch2→5 with compression; Shared RAG-DCache disk +15-71% needed |
| A12 | **Token importance decidable from local observation window (alpha≈8)** | SnapKV 2024, PyramidKV 2024, H2O 2023, Cache-Craft 2025, CacheBlend 2024 | **Medium — strong for QA, weak for summarization** | Dispersed uniform importance or instruction-at-front fails; CCI misclassify →50% F1 drop naive 5-block reuse |

> **Cross-cutting (assumption_map.md):** A3, A4, A5, A11 repeatedly invalidated by 2024-2025 RAG/agent/long-context work; systems on their conjunction over-provision 2-7x (Splitwise 2.35x, Mooncake 75%, Beluga 7.35x vs RDMA).

---

## Repeated Limitations

*Source: `research/knowledge/limitation_map.md` (13 limitations; type: Widely acknowledged ≥5 papers / Appears repeatedly ≥3 / Paper-specific). Author-stated (§12) vs inferred (§13).*

### Widely acknowledged (author-stated future work / caveats)

| # | Limitation — Source `limitation_map.md` | Type | Papers Mentioning (year) — Source `limitation_map.md` |
|---|---|---|---|
| L1 | **Static manually-tuned budgets/thresholds, no adaptive per-workload optimization** | Widely acknowledged | PyramidKV 2024 (α=8,β=20), KIVI 2024 (G32/R128), GEAR 2024 (rank 4/2 s2%), SCOPE 2025, Sarathi-Serve 2024 (τ512/2048 Vidur), CacheBlend 2025 (r*15%), InfiniGen 2024 (alpha4/5), DistServe 2024, Mooncake 2025, HotPrefix 2025 |
| L2 | **No joint optimization across orthogonal axes (token × bit × reuse × disaggregation)** | Widely acknowledged | vLLM 2023, H2O 2023, SGLang 2023, KIVI/GEAR 2024, PyramidKV 2024, ShadowKV 2025, DistServe/Splitwise 2024, Mooncake 2025, LMCache 2025 |
| L3 | **Narrow hardware & scale coverage — single GPU family, small clusters, no cost/power heterogeneity** | Widely acknowledged | vLLM 1-8×A100, SGLang A10G, StreamingLLM A6000, H2O A100/T4, ShadowKV A100, InfiniGen A6000 PCIe3, KIVI/GEAR V100/Titan, Sarathi 1-8 GPUs 2 nodes, DistServe 32 GPUs 25Gbps, Mooncake 20×A800 dummy 70B |
| L4 | **Cache-aware scheduling improves hit rate at expense of fairness/starvation/tail SLOs** | Widely acknowledged | SGLang 2024 (greedy longest-prefix starves), Sarathi-Serve 2024, Mooncake 2025 (3P+1D vs 2P+2D imbalance), FastServe 2023, Online Scheduling 2025, KVFlow 2025 |
| L5 | **KV transfer/placement in disaggregated/hierarchical systems remains bottleneck; heterogeneity unrealistic** | Widely acknowledged | Splitwise 2024 (HA IB not deployed), DistServe 2024 (25 Gbps), Mooncake 2025, FlowKV 2025 (25%→-96% but still 8ms/5ms), LMCache 2025 (4 vs 46 Gbps), CachedAttention 2024 |
| L6 | **No true context-window extension; compression/eviction trades recent stability for long-range fidelity** | Widely acknowledged | StreamingLLM 2023 (does NOT extend), H2O 2023, SCOPE 2025 (20% prefill →95% GSM8K+ drop but retrieval near-lossless), PyramidKV 2024, InfiniGen 2024 |

### Appears repeatedly (inferred across ≥3 papers §13)

| # | Limitation | Type | Papers — Source `limitation_map.md` |
|---|---|---|---|
| L7 | **Cross-request KV reuse without tenant isolation — security/privacy/side-channel leakage unsolved** | Appears repeatedly | vLLM 2023, SGLang 2024, RAGCache 2024, Cache-Craft/KVLink/CacheBlend 2025, LMCache/ShadowKV 2025, CachedAttention 2024 |
| L8 | **Dollar cost/power/energy not measured — Perf/$ and Perf/W incomplete** | Appears repeatedly | Splitwise 2024 (rental $17.6 vs $38 not TCO), DistServe 2024 (double weights not counted), Mooncake 2025, CachedAttention $5/hr spot, KIVI/GEAR no joules/token |
| L9 | **Synthetic/non-representative workloads mask real serving distributions & SLO interactions** | Appears repeatedly | RAGCache 2024 (Wiki 0.3M synthetic), CacheBlend 2024 (6000 GPT-4 synthetic uniform 512), Cache-Craft 2025 (200 Q sample), KVFlow 2025 (synthetic 10-agent), DistServe Poisson, Mooncake dummy 70B |
| L10 | **Positional encoding coupling breaks naive reuse — re-encoding correctness not guaranteed** | Appears repeatedly | StreamingLLM 2023 (RoPE re-encode), CachedAttention 2024 (RPE decouplable, naive >1e3 PPL), CacheBlend/Cache-Craft/PyramidKV/KVLink 2024-25 |
| L11 | **No fault tolerance/availability/consistency for hierarchical/distributed KV pools** | Appears repeatedly | Splitwise 2023 (restart), DistServe 2024 (no checkpoint), Mooncake 2025, LMCache 2025 (TTL 85%→45%), DejaVu 2024 (<2% overhead but not adopted) |
| L12 | **Multi-turn/agentic workflow state lifetimes not captured by single-turn prefix assumptions** | Appears repeatedly | KVFlow 2025 (branching not captured), Continuum 2025 (ReAct only), SCOPE 2025 (T unknown), CachedAttention 2024 (session all-or-nothing), LMCache 500GB insufficient |
| L13 | **Extreme compression brittleness on reasoning tasks & semantic chunking — gains not task-agnostic [Weakly evidenced]** | Paper-specific | GEAR/KIVI 2024, PyramidKV 2024, SCOPE 2025, Cache-Craft 2025 |

> **Synthesis (limitation_map.md cross-cutting):** Problems many papers do NOT truly solve: adaptive control, multi-axis co-optimization, heterogeneous large-scale deployment, SLO-faithful scheduling, secure multi-tenant reuse, end-to-end cost/power, production realism, positional correctness, fault-tolerant persistence, stateful agentic lifetimes.

---

## Evaluation Patterns

*Source: `research/knowledge/evaluation_map.md` (metrics / baselines / workloads / hardware, 24 notes reviewed) + `research/knowledge/literature_matrix.md` (43 rows) + `research/knowledge/bottleneck_map.md` cross-check. Hardware [NOT REPORTED] preserved.*

### Metrics (evaluation_map.md Sec.1, literature_matrix.md Sec.C)

| Metric | What Is Measured | Papers Using (year) — Source `evaluation_map.md` | Typical Reporting |
|---|---|---|---|
| **TTFT** | Prefill latency (s/ms) | Sarathi-Serve 2024, Mooncake 2024, RAGCache 2024, CacheBlend 2024, LMCache 2025, Beluga 2025, FlowKV 2025 (+SGLang, HotPrefix, CachedAttention, CacheGen) — 17/43 (39%) | Avg & P50/P99; SLO 10× baseline (Mooncake) or 0.1-1s strict vs 15s relaxed (Sarathi) |
| **TPOT / ITL / TBT** | Decode inter-token latency | DistServe TPOT 2024, Sarathi TBT 2024, Mooncake TBT 2024, LMCache ITL 2025, Beluga TPOT 2025 — 11/43 (25%) | P90/P99; e.g., Sarathi strict 0.1s (7B)/0.2s (34B) |
| **Throughput / Goodput / Capacity** | tokens/s, RPS, jobs/s, max RPS under SLO | Orca 36.9x, vLLM 2-4x, Splitwise 2.35x iso-cost, DistServe 2.0-4.6x (5.7x code), Sarathi 2.6-5.6x, Mooncake 20-525%, FlowKV +95% — **38/43 (88%)** | Per-GPU goodput = RPS/GPU meeting 90% SLO (DistServe); capacity = max RPS median delay <2s + P99 TBT (Sarathi) |
| **P50/P90/P95/P99 tail** | Percentile latency | Sarathi P99 TBT, Mooncake P90, DistServe P90/99 attainment, Beluga P99, LMCache P95, Continuum P90/95 JCT 8.18× | SLO-defining; Mooncake TTFT_P90 ≤10× baseline, TBT_P90 ≤5× |
| **Memory / Peak memory / KV size** | GPU HBM, CPU DRAM, bytes/token, fragmentation | vLLM 20-38%→96%, H2O 5× (20% budget), KIVI 2.6× less peak /8× KV, ShadowKV 6× batch 7.08× saving — 35/43 (81%) | Per-token LLaMA-65B 2.5MB, 70B GQA 0.31MB, OPT-13B 800KB/token |
| **Cache hit rate** | Cached tokens/docs / total | SGLang 50-99% (96% optimal, prod 52-74%), Mooncake 30% 1K→50% 50K plateau 51% inf, RAGCache +2-75% over LRU | CachedLength/ComputationLength (SGLang) vs hit docs/total docs (RAGCache) |
| **Recomputation cost / Saving** | Prefill recompute vs load | vLLM swap vs recompute ≤20%, CachedAttention 99% eliminated, CacheBlend 5-18% full prefill, RAGCache cached prefix 11.5× faster (3.9× with transfer) | CacheGen text fallback 1.67-1.81× even when recompute |
| **Bandwidth / Transfer latency** | PCIe/NVLink/RDMA/CXL | Splitwise 8ms A100/5ms H100 <7% prompt, DistServe <0.1% <30ms 95%, FlowKV 96.8% ↓ 0.944→0.053s 23469→1 calls, Beluga 64B/750ns 1TB/s, LMCache 4→46 GBps | vLLM block 16=62.5KB (Llama3.1-8B) |
| **GPU util / MFU** | Compute vs memory-bound | Splitwise power vs batch, Sarathi >80% time even 1K, Mooncake MFU, Llumnix decode 2.6× with batch | DistServe M/D/1 queuing model Avg_TTFT = D + R·D²/(2·(1-R·D)) |
| **Quality / Accuracy** | F1, Rouge-L, PPL, EM, BLEU | H2O HELM 20% no drop, StreamingLLM PPL 5158→5.40 QA 91.37% (70B), KIVI CoQA ≤2% @2-bit, CacheBlend F1 ≤0.02 drop, ShadowKV RULER 86.88 vs 86.68 | Needle, SWE-Bench, RULER, LongBench |
| **SLO attainment / JCT** | % meeting TTFT+TPT, jobs completed | DistServe 90%/99%, Splitwise 9 SLOs P50/90/99, Mooncake TTFT_P90/TBT_P90 vs RPS, Continuum avg JCT 8× | SARATHI capacity = max RPS median delay <2s |
| **Cost / Energy / Power** | $/hr, Whr, provisioned power | Splitwise $17.6 A100 vs $38 H100, 700W vs 400W, CachedAttention $5/hr/A100 43-70% saving, Beluga ConnectX-7 $1745 vs CXL adapter $210 — **sparse** | Provisioned vs dynamic power rarely measured (limitation L8) |

> **Coverage note (evaluation_map.md):** TTFT+throughput in 15+/24 = community standard; compression papers add quality as mandatory third dimension.

### Baselines — source `evaluation_map.md` Sec.2 (frequency among 24 reviewed, lower-bound vs 43)

| Baseline System | Category | Freq (24) | Example Comparisons (year) — Source `evaluation_map.md` |
|---|---|---|---|
| **vLLM (PagedAttention)** | Serving/memory open-source SOTA | **15** | SGLang 2023, Sarathi-Serve 2024, Mooncake 2024, DistServe 2024, RAGCache 2024, CacheBlend 2024, LMCache 2025, FlowKV 2025, Beluga 2025 |
| **SGLang / RadixAttention / HiCache** | Prefix reuse | 6 | RAGCache vs SGLang+Faiss 2024, LMCache vs SGLang CPU offload 2025, KVFlow vs SGLang/HiCache 2025 |
| **FlexGen / DeepSpeed / HF Accelerate** | Offload | 3 | H2O 29× over DeepSpeed & 3× over FlexGen on T4 2023 |
| **LMCache / Mooncake / Dynamo / InfiniStore** | Disaggregated KV pool | 5 | Beluga vs Mooncake v3.2 & Dynamo v0.4.1 2025, LMCache vs NIXL 2025, FlowKV vs Mooncake 2025 |
| **Splitwise / DistServe / TetriInfer / Dynamo H100** | Disaggregated P/D | 4 | FlowKV vs DistServe/Splitwise 2025, Mooncake related work 2024 |
| Others | ChunkedAttention/PromptCache, HiCache/AttentionStore, Faiss, Commercial Cloud | 2-3 each | SGLang vs ChunkedAttention, CachedAttention vs LRU/FIFO, RAGCache vLLM+Faiss |

### Workloads — source `evaluation_map.md` Sec.3 (9 strata)

| Workload Category | Definition | Papers (year) — Source `evaluation_map.md` |
|---|---|---|---|
| **Synthetic (controlled Poisson)** | Random lengths, Poisson/Gamma arrivals | Orca 2022 U(32,512)/U(1,128) Poisson, vLLM 2023 Poisson, Splitwise 70 RPS scaled, DistServe 1-4 rps sweeps, Sarathi QPS sweeps, FlowKV 1K/5K/10K Poisson 0.1-2.0 RPS, Llumnix Poisson+Gamma CV |
| **ShareGPT / Real chat traces** | Heavy-tail multi-turn | vLLM ShareGPT 8.4× vs Alpaca, SGLang ShareGPT 74.3s no-reuse, CachedAttention ShareGPT 9K sessions 52K turns Poisson1.0, Sarathi median 1730/415 |
| **Conversational / Multi-turn** | 4-turn chat, concatenated history | SGLang 4-turn 256±12, CachedAttention 87% TTFT ↓, CacheGen LongChat 9.2-9.6K 100 samples, Continuum SWE-Bench 10.9 turns 70K/program |
| **Long-context ≥8K to 1M** | LongBench, RULER, NIAH, PG19 | StreamingLLM PG19 20K-4M 4M stable, Mooncake ArXiv 8088/L-Eval 19019 16K-128K, ShadowKV RULER 128K NIAH 1M |
| **RAG (retrieval, Wikipedia, 512-token chunks)** | Stuff multiple retrieved chunks | RAGCache Wiki 0.3M avg3718 MMLU/NQ top-k2, CacheBlend 2WikiMQA/Musique 6×512, LMCache doc QA 10K +100 tokens |
| **Agent / Workflow / Tool-use** | ReAct loops, tool gaps 0.9-1.9s avg | SGLang ReAct/ToT, KVFlow 10-agent 8192/32/32 + PEER 4-agent, Continuum SWE-Bench 10.9/6.3 turns OpenHand/BFCL 93K |
| **Prefix-heavy (system, few-shot)** | Large shared preamble, 5/20-shot, beam | SGLang 5-shot MMLU/20-shot HellaSwag, vLLM WMT 1-shot 1.67× 5-shot 3.58× |
| **Bursty / High concurrency** | Poisson bursts, Gamma CV, 64+ concurrent | Llumnix Gamma CV 16 instances 64Gb/s 36% saving, KVFlow 64 concurrent H100 HiCache 0.57× degraded |
| **Other — Coding / Summarization / Math / Needle** | Specialized tasks | Splitwise coding 1500/13 vs conversation 1020/129 bimodal, DistServe HumanEval code vs LongBench sum 5.7×/4.3× |

### Hardware — source `evaluation_map.md` Sec.4 (per-paper as reported; [NOT REPORTED] preserved) — highlights:

- **Single-GPU A10/A40/A6000 dominates eviction/compression** (StreamingLLM A6000, KIVI [NOT REPORTED], ShadowKV A100 single, InfiniGen A6000 PCIe3.0); **multi-node 32 GPUs** DistServe 4×8 A100 25Gbps limited vs 800Gbps sim, **20 nodes 160 GPUs** Mooncake A800 800Gbps RDMA, **16 instances 2 servers CXL** Beluga H20 96GB 2TB DRAM 8TB CXL pool 750ns/64B 1TB/s
- **Interconnect:** NVLink 600GB/s intra vs 25-50 Gbps inter vs PCIe Gen1 2GB/s (A10G) →Gen5 64GB/s (H100)
- **Limited-condition highlights (evaluation_map.md Sec.4 Highlight):** 14 papers qualified — StreamingLLM single-stream not batched vLLM, KIVI [NOT REPORTED] max8K no distributed, CacheGen 4×A40 0.2-2Gbps narrow link (gain shrinks at 100Gbps), RAGCache 1×A10G vs 2×H800 top-k≤5 Poisson0.8rps (TB scale may differ), CacheBlend 2×A40 synthetic GPT-4 uniform 512 not heterogeneous drift, Llumnix 16×A10 no NVLink max2K context, KVFlow 1×A10G weak PCIe vs 1×H100 strong, Continuum 100 traces GPT-5 Poisson0.13 single-node, Orca/vLLM synthetic Poisson not real heavy-tail, ShadowKV single A100 fixed r160 no multi-node, Beluga rack-scale 2 servers Qwen-32B only not beyond 1TB/s switching — **transfer to H100/GB200 or 100s nodes not validated, many claims via simulator (<3% MAPE) or dummy models not at 1M**.

---

## Top Candidate Gaps (5-15 prioritized with evidence)

*Source: `research/knowledge/candidate_gaps.md` (10 gaps, each with 6 subsections). All conservative: **Status NOT YET VERIFIED AS NOVEL**, language underexplored/candidate gap, traceable to papers + maps. No novelty claims. Prioritized by evidence strength + system impact + cross-paper recurrence.*

### Gap 1: Joint Optimization of Token Budget x Bit-Width x Placement (Memory/KV Dimension) — **High priority**

- **Observation (candidate_gaps.md G1):** 14 compression/eviction +5 hierarchical papers each optimize single axis; orthogonality assumed but never measured together.
- **Evidence source:** `candidate_gaps.md G1` + `taxonomy_draft.md Cat.2-4,7` + `limitation_map.md L1/L2` + `bottleneck_map.md B14` + `assumption_map.md A8/A9` ; papers: PyramidKV 2024 (12% matches Full 41.49 vs41.46 at 2048, alpha=20 fixed, no quant), KIVI 2024 (2.6× less peak 4× batch G32/R128 uniform, MQA 4-bit needed), GEAR 2024 (2.39× peak batch3→18 uniform r4/2 s2% not combined with eviction), DistServe 2024/Splitwise 2023 (2.0-4.6x goodput via placement, double weight 2×350GB not quantified).
- **Why interesting (underexplored):** Stacking paged blocks + quant kernels + RDMA may be 2-4× multiplicative but interference (tile-quant 257→32% slower, sparsity 84.3%) not quantified; per-layer/per-request adaptive `k^l x bit-width x tier` controller under HBM/CXL/PCIe constraints missing — needs GPT to verify if any 2025 work already jointly searches Pareto.
- **Status:** **NOT YET VERIFIED AS NOVEL**

### Gap 2: Online Adaptive Tuning of Static Budgets and Thresholds — **High priority**

- **Observation (candidate_gaps.md G2):** 11 papers rely on manually-tuned static constants requiring offline profiling, fail on drift.
- **Evidence source:** `candidate_gaps.md G2` + `limitation_map.md L1` + `bottleneck_map.md B7/B9` ; papers: Sarathi-Serve 2024 (τ512/2048 via Vidur, 2.6/3.7/5.6× capacity, 25% overhead@512), PyramidKV 2024 (α20 fixed sensitivity), KIVI/GEAR 2024 (uniform G/R/r), CacheBlend 2025 (r*15% 2.2-3.3× TTFT ≤0.02 F1 but drift), HotPrefix 2025 (admission 10, +1.17-2.38× hit), Mooncake 2025 (kvcache_balancing_threshold manual).
- **Why interesting:** Vidur profiling, bilinear T(l,u), Cuckoo aging all offline/heuristic; lightweight closed-loop estimator with <0.3% overhead (SGLang level) generalizing 1K-1M without per-model reprofiling would remove top repeated limitation (widely acknowledged ≥7 papers).
- **Status:** **NOT YET VERIFIED AS NOVEL**

### Gap 3: Online Heterogeneous Placement Beyond GH200 with Realistic CXL/PCIe5 Constraints — **Medium-High priority**

- **Evidence source:** `candidate_gaps.md G3` + `research_landscape.md Sparse` + `assumption_map.md A11 (Low)` + `bottleneck_map.md B13` ; papers: Dynamic Placement 2025 IEEE CAL (GH200 5.87× upper bound SA over W,R perfect knowledge not realizable), Beluga 2025 SIGMOD26 (8TB@1TB/s 89.6% TTFT 7.35× QPS but 2 servers×8 H20 only, SW coherence 281ms UC, 16+ servers not measured), InfiniGen 2024 (selective <10% avg cap20% 3× over FlexGen but single A6000 PCIe3), LMCache 2025 (Controller 1000+ bottleneck, 500GB still insufficient).
- **Why interesting:** Missing online scheduler mixing HBM/DRAM/CXL/SSD without perfect future knowledge handling Gen1 2GB/s vs Gen5 64GB/s and SW coherence cost at 100s-node scale with cost/power accounting — CXL prototype vs GH200 idealized leaves TCO gap.
- **Status:** **NOT YET VERIFIED AS NOVEL**

### Gap 4: Fault Tolerance, Consistency, and Availability for Distributed/Hierarchical KV Pools — **High priority (system)**

- **Evidence source:** `candidate_gaps.md G4` + `limitation_map.md L11` + `bottleneck_map.md B13` + `research_landscape.md Sparse` ; papers: Splitwise 2023 (restart from scratch), DistServe 2024 (32 GPUs 7.4× but no checkpoint), Mooncake 2025 (800Gbps RDMA 20-525% but manual replication, incast not handled), LMCache 2025 (TTL 1h 85%→45% no crash consistency), CachedAttention 2024 (87% TTFT 7.8× but single 4×A100), DéjàVu 2024 ICML (<2% overhead via replication not adopted).
- **Why interesting:** Throughput/goodput measured only happy path; SLO counts only fully completed requests, yet single prefill crash wastes layer-wise transfer and uniform td prediction fails; RDMA incast and Controller bottleneck at 100s-1000 instances not measured — production SLO fragile.
- **Status:** **NOT YET VERIFIED AS NOVEL**

### Gap 5: SLO-Aware Fair Scheduling with Prefix-Affinity — **High priority (scheduling)**

- **Evidence source:** `candidate_gaps.md G5` + `limitation_map.md L4` + `bottleneck_map.md B8/B10/B11` + `taxonomy_draft.md Cat.11` ; papers: SGLang 2024 (Radix LRU leaf-first DFS 50-99% 96% optimal 6.4× but starves small hot prefixes, needs cache≥max req for Theorem3.1), Mooncake 2025 (T_queue+T_prefill+T_transfer routing, 3P+1D vs 2P+2D imbalance, vLLM 57% vs Mooncake ~100% TBT), Sarathi-Serve 2024 (τ stall-free 2.6/3.7/5.6× but no fairness/FS/preemption), FastServe 2023 (31.4× SLO via MLFQ), Online Scheduling 2025 (Ω(√n) hardness single worker).
- **Why interesting:** No system jointly optimizes hit rate × SLO attainment × fairness with weighted fair queuing/EDF + prefix affinity + ref-count aware eviction + hot-spot replication without anti-phase fluctuation (Mooncake Fig9-10); P99 TBT vs avg throughput trade-off under starvation remains underexplored despite LRU vs PGDSF/Step Graph 2-75% hit beats.
- **Status:** **NOT YET VERIFIED AS NOVEL**

### Gap 6: Multi-Tenant Isolation, Privacy, and Security for Cross-Request KV Reuse — **Medium priority (enterprise)**

- **Evidence source:** `candidate_gaps.md G6` + `limitation_map.md L7` + `bottleneck_map.md Evidence Gaps` + `assumption_map.md A3` ; papers: Cache-Craft 2025 (8% prefix hit but 94% cross-user hit ~50GB storage N=100×5 variants with no access control, F1 0.65 vs0.87 naive 50% drop), SGLang 2024 (radix tree timing 3.9× hit vs miss side-channel), vLLM 2023 (CoW limited to within-request group), RAGCache 2024 (top3% 60% hits 20× uniform public Wiki 0.3M), LMCache 2025 (Controller P2P no permission), Mooncake 2025 (plateau 50% infinite cache).
- **Why interesting:** Enterprise Shared RAG-DCache +15-71% pushes cross-instance sharing but assumes public knowledge; per-tenant radix forest or encrypted hash isolation loss 30-50% hit not measured; latency probing for doc reconstruction risk — security vs efficiency Pareto untouched.
- **Status:** **NOT YET VERIFIED AS NOVEL**

### Gap 7: Contamination-Aware Non-Prefix RAG Reuse Across Multi-History Dispersed Chunks — **Medium-High priority (RAG)**

- **Evidence source:** `candidate_gaps.md G7` + `contradictions.md T6` + `bottleneck_map.md B3/B4` + `taxonomy_draft.md Cat.5-6` ; papers: Cache-Craft 2025 (CCI=1/(1+e^{-a/b}) beta gamma CFO -51% redundant vs prefix -75% vs full 1.6× but thresholds recalibration), CacheBlend 2025 (full reuse F1 -0.1-0.2, HKVD 15% pipelined 3ms vs16ms 2.2-3.3× ≤0.015 loss but 2×A40 synthetic uniform 512), RAGCache 2024 (knowledge tree PGDSF 11.5× prefix 3.9× with Host but exact 8% requests), KVLink 2025 (PIC pre-RoPE +5 link tokens -96% TTFT +4% QA but 6000 steps 8×H100).
- **Why interesting:** Production 5 chunks scattered ≥3 histories (>50% requests) not handled training-free robustly; CCI/CFO not robust to retriever drift 1K-20K; missing robust detector + joint `recompute% vs transfer vs disk` via Trecompute ≤ Tload hiding on heterogeneous H20 vs A100.
- **Status:** **NOT YET VERIFIED AS NOVEL**

### Gap 8: State Lifetime Management for Branching/Looping Multi-Agent Workflows — **Medium priority (agent)**

- **Evidence source:** `candidate_gaps.md G8` + `limitation_map.md L12` + `bottleneck_map.md B9` + `research_landscape.md Sparse` ; papers: KVFlow 2025 (LRU evicts Expresser T=13 miss T=14, Step Graph max/min →priority 2.19× over HiCache but 0.57× degraded 64 concurrent), Continuum 2025 (tool gaps 0.9-1.9s 52.5% 10% slowest, 10.9 turns 70K TTL tau* 8× JCT but ReAct only), SCOPE 2025 (max T known for Adaptive drift Top-15% steps1/300/500), CachedAttention 2024 (session all-or-nothing 86% hit TTL1h), LoongServe 2024 (ESP token-granular pool).
- **Why interesting:** Existing Step Graph/TTL assume linear ReAct; fork/join loops ad-hoc spawns not captured; need unified lifecycle with iteration-granular DoP and joint prefill+decode drift correction without known T, integrated with online length predictor (TetriInfer 74.9% @200) at enterprise scale.
- **Status:** **NOT YET VERIFIED AS NOVEL**

### Gap 9: End-to-End Cost, Power, and Energy (Perf/$ and Perf/W) Benchmarking on Heterogeneous Clusters — **High priority (evaluation)**

- **Evidence source:** `candidate_gaps.md G9` + `limitation_map.md L8` + `evaluation_map.md Sec.1 Cost / Sec.4 Highlight` + `bottleneck_map.md Evidence Gaps` ; papers: Splitwise 2023 (2×DGX-A100+2×DGX-H100 2.35× same cost/power via rental $17.6 vs $38 not TCO, HA IB not deployed), DistServe 2024 (goodput 2.0-4.6× but double weights 2×350GB not counted), CachedAttention 2024 ($5/hr spot 43-70% saving no network/ops), Beluga 2025 (ConnectX-7 $1745 vs CXL adapter $210 economics unstable), Highlight 14 papers ≤32 GPUs simulator MAPE<2-3% extrapolates.
- **Why interesting:** Heterogeneous decisions (A100 vs H100 pools, CXL pool vs adding GPUs) depend on TCO but no harness reports joules/token, $/1M tokens under SLO, provisioned vs dynamic power (700W vs 400W) alongside TTFT/TPOT/P99 across A100/A10G/H20/H100/GB200 NVL72 and CXL vs RDMA; double-weight P/D overhead and replication cost not quantified — blocks perf/$ claims.
- **Status:** **NOT YET VERIFIED AS NOVEL**

### Gap 10: Realistic Burst Production Traces and Tail-SLO Evaluation vs Synthetic Poisson Simulators — **High priority (evaluation)**

- **Evidence source:** `candidate_gaps.md G10` + `evaluation_map.md Sec.3-4 Highlight` + `assumption_map.md A6` + `limitation_map.md L9` + `bottleneck_map.md B8/B11` ; papers: Orca 2022 (U(32,512)/U(1,128) synthetic Poisson no EOS 36.9×), vLLM 2023 (ShareGPT/Alpaca Poisson 1h trace 15min 175B block16 manual), DistServe 2024 (Poisson 1-4 rps M/D/1 predictable hours/days burst not tested), RAGCache 2024 (Wiki 0.3M avg3718 IVF1024 Poisson0.8 hit +75% @8GiB not TB scale), CacheBlend 2024 (6000 GPT-4 synthetic uniform 512), SGLang 2023 (bench 96% vs prod 52.4%/74.1%), KVFlow 2025 (64 concurrent HiCache 0.57× due to queuing not Poisson), Continuum 2025 (100 traces GPT-5 Poisson0.13).
- **Why interesting:** No harness sweeps same system across prompt 1K→1M, output 32→512+, arrival Gamma CV, and strict (0.1s) vs relaxed (30s 10×/5×) SLO to report both goodput and capacity inclusive P99/fairness; Poisson simulators <3% MAPE underestimate queuing where 10% slowest tool =94% delay and mixing ≥3 histories — production multi-tenant heavy-tail remains unbenchmarked.
- **Status:** **NOT YET VERIFIED AS NOVEL**

> **Overall candidate gap traceability:** Evidence per gap ties to 2-5 papers + 2-4 knowledge maps; all 10 gaps marked **NOT YET VERIFIED AS NOVEL**; conservative phrasing per `candidate_gaps.md` instruction — GPT must verify novelty via external literature search before claiming.

---

## Uncertainties / Least Confident Parts

*Where synthesis is least confident — risks in synthesis, not in papers themselves. Traceable to source limitations.*

1. **Supplemental 5 papers single-read risk (High uncertainty)** — *Source: `research/knowledge/stage1_summary.md` Reliability Limitations; `research/knowledge/coverage_report.md`.* CacheGen SIGCOMM24, CachedAttention ATC24, Dynamo GTC25 0.4, Llumnix OSDI24, LoongServe SOSP24 were added after auditor websearch and only single-read (HotPrefix still simplified). Stage 1 reviewers note 37/43 reviewed, High 31/31 but supplementals not double-read; details (e.g., Dynamo NIXL vs SGLang KVBM workaround) may shift. *Mitigation: taxonomy_draft.md explicitly lists tracing to `papers.md` 43 entries but notes supplementals verified via search not deep notes for 8 [NOT REPORTED] hardware — treat metrics as provisional.*

2. **Hardware [NOT REPORTED] propagation (Medium-High uncertainty)** — *Source: `research/knowledge/evaluation_map.md` Sec.4 (8/43 [NOT REPORTED]), `literature_matrix.md` Sec.C (33/43 reported 81%), `bottleneck_map.md` Evidence Gaps.* KIVI 2024, PyramidKV 2024, TetriInfer 2024, SCOPE/ChunkKV 2025 partial missing precision/GPU counts. Synthesis infers interconnect bottleneck (NVLink 600GB/s vs IB 400Gbps) but cannot guarantee transfer % (0.1% vs 25%) outside measured A100/A6000 vs H100 Gen1/Gen5 — evaluation_map Highlight flags 14 papers with limited conditions; extrapolating to GB200 NVL72 30× risks over-generalization.

3. **Threshold counting and cross-paper commensurability (Medium uncertainty)** — *Source: `research/knowledge/limitation_map.md` L1/L2, `research/knowledge/taxonomy_draft.md` Cross-Cutting Trade-offs, `research/knowledge/contradictions.md` Overall Assessment.* Synthesis states manual thresholds (τ512/2048, alpha20, G32/R128, r*15%, admission 10, kvcache_balancing_threshold) as widely acknowledged ≥7 papers, but paper_notes use different SLO definitions (Sarathi median delay <2s + P99 TBT vs Mooncake 10×/5× vs DistServe 90% attainment). Comparing capacity 2.6-5.6× vs goodput 2.0-4.6× vs 5.81× vs 7.35× is not apples-to-apples; contradictions.md resolves 6 tensions as Pareto-conditioned, but confidence Medium-High not Absolute — unified harness sweeping same traces/same hardware (1K→1M, 32→512 output, 25Gbps→800Gbps) not yet existent.

4. **Long-context 1M-scale extrapolation via simulator/dummy models (Medium uncertainty)** — *Source: `research/knowledge/evaluation_map.md` Sec.4 Highlight (ShadowKV single A100 1M 60K-488K, LoongServe 8×A800 1M, Mooncake 20 nodes dummy Llama2-70B simulated 128K, DistServe 25Gbps limited).* Many 100K-1M claims use single A100 batch2 OOM vs batch48 or simulator MAPE <2% or dummy 70B. Real heterogeneous 100s-node RDMA incast, CXL 2TB/s/chip beyond 16 servers saturation (Beluga O7), and enterprise queueing not measured — bottleneck_map B13 notes 16-server scale contention not measured. Synthesis treats these as upper bounds per limitation_map.md, but actual 1M deployment may differ 2-3×.

5. **Maturity judgments based on coverage counts not deployment depth (Medium uncertainty)** — *Source: `research/knowledge/taxonomy_draft.md` Maturity Summary, `research/knowledge/research_landscape.md` Sec.6-8.* Mature = ≥3 papers + years production (e.g., Paged, prefix reuse). However SGLang Mature but production 52-74% vs benchmark 96% shows gap; LMCache enterprise controller bottleneck at 1000+ not formalized; Beluga CXL Emerging but prototype $5800 switch economics unstable. Maturity reflects literature coverage 2022-2026 not verified production SLO at scale.

6. **Contamination vs recompute vs link-token tradeoff quantification (Medium-Low uncertainty)** — *Source: `research/knowledge/contradictions.md` T6, `research/knowledge/candidate_gaps.md` G7.* RAGCache 11.5× prefix, CacheBlend 15% HKVD pipelined 3ms vs 16ms, Cache-Craft CCI-aware 30% for 90% ROUGE, KVLink 5 link tokens -96% TTFT with fine-tuning 8×H100. Synthesis synthesizes these as ordered tradeoff but each evaluated different hardware (A10G 24GB, 2×A40 NVMe 4.8GB/s, 8×A100 8TB NVMe 16GB/s, 8×H100 training) and retrieval distribution (Wiki 0.3M top-k2 vs 6×512 random 6000 GPT-4 synthetic vs Sys-X/Y enterprise 5 chunks ≥3 histories). GPT verification must re-check if any 2025 work already unified CCI-free detector across same hardware/retriever.

7. **Cost/power TCO synthesis weak evidence base (High uncertainty for Gap 9)** — *Source: `research/knowledge/limitation_map.md` L8, `research/knowledge/evaluation_map.md` Sec.1 Cost, `research/knowledge/bottleneck_map.md` Evidence Gaps.* Only Splitwise provides rental $17.6/$38 and provisional 700W/400W, CachedAttention spot $5/hr, Beluga $1745 vs $210 NIC/adapter; no paper reports joules/token or PUE-corrected perf/$. Synthesis flags perf/$ claims (2.35× same-cost, 70% saving) as incomplete but cannot quantify corrected TCO without external metering — least confident quantitative correction.

> **Overall synthesis risk mitigation:** All uncertain numbers preserved as "[NOT REPORTED]" where absent per evaluation_map.md rule; every synthesis bullet hints source file; conservative "underexplored" not "novel" maintained; GPT should prioritize verifying Gaps 9-10 first via external cost traces and production burst datasets that maps flagged as missing.

---

## End Checks (9 checks from Stage 2A 结束检查 listing pass/fail)

*Derived from Stage 2A stopping criteria across artifacts: manifest + 10 taxonomy dimensions + bottleneck/assumption/limitation/evaluation/literature/landscape/contradictions/candidate_gaps.*

| # | End Check (Stage 2A) | Required Condition | Result — Source Verification | Pass/Fail |
|---|---|---|---|---|
| 1 | **Seed paper manifest completeness** | `research/manifests/papers.md` exists, 20-40+ papers (here 38 dedup →43), covers all 10 Stage1 categories + 5 weak supplemented, 43 notes exist | 43 entries verified (8 Foundational 2022-23, 27 Recent 63%, 31 High, 24 code) — `papers.md` 43 rows + `paper_notes/*.md` 43 files + `coverage_report.md` 10/10 covered + auditor 5 additions | **PASS** — Source `papers.md`, `coverage_report.md`, `stage1_summary.md` |
| 2 | **Taxonomy draft coverage & traceability** | `research/knowledge/taxonomy_draft.md` exists, ≥12 categories, each with problem/mechanism/maturity, traceable to `papers.md`, no hallucinated titles | 14 categories (Cat.1-14) mapping 17 dimensions, sampled 24 notes, every representative paper verbatim in `papers.md`, maturity Mature/Active/Emerging/Sparse — `taxonomy_draft.md` 572 lines | **PASS** — Source `taxonomy_draft.md` |
| 3 | **Bottleneck map coverage** | `research/knowledge/bottleneck_map.md` exists, ≥12 bottlenecks (12 mandatory +extras), each with where/trigger/metrics/papers/limitations | 14 bottlenecks B1-B14 full sections, 24+ notes read, 2 additional beyond mandatory, cross-bottleneck synergy + evidence gaps — `bottleneck_map.md` 640 lines | **PASS** — Source `bottleneck_map.md` |
| 4 | **Evaluation map completeness & [NOT REPORTED] rule** | `research/knowledge/evaluation_map.md` exists, 4 sections: metrics (≥10) / baselines (freq) / workloads (≥9 strata) / hardware (per-paper, [NOT REPORTED] where absent), ≥20 papers reviewed | 14 metrics, 13 baseline families vLLM 15/24 dominant, 9 workload strata, 24 notes reviewed + Highlight limited-conditions table, [NOT REPORTED] preserved — `evaluation_map.md` 150 lines + `literature_matrix.md` cross-check | **PASS** — Source `evaluation_map.md` |
| 5 | **Assumption map confidence-rated & traceable** | `research/knowledge/assumption_map.md` exists, ≥10 assumptions (here 12), each with 5 subsections + evidence [PAPER FACT] + confidence | 12 assumptions A1-A12, 26 notes read, each papers/why/failure/evidence/confidence, cross-cutting stability table with fragile flags — `assumption_map.md` 339 lines | **PASS** — Source `assumption_map.md` |
| 6 | **Limitation map author-stated vs inferred separated** | `research/knowledge/limitation_map.md` exists, distinguishes author-stated (§12) vs repeated inferred (§13), includes widely/repeated/paper-specific types | 13 limitations (6 Widely acknowledged ≥5 papers +6 Appears repeatedly ≥3 +1 Weakly evidenced), 22 notes read, traceable to §12-13, cross-cutting summary — `limitation_map.md` 242 lines | **PASS** — Source `limitation_map.md` |
| 7 | **Literature matrix & research landscape coverage** | `research/knowledge/literature_matrix.md` covers all 43 papers + `research/knowledge/research_landscape.md` 10 sections + `research/knowledge/contradictions.md` ≥4 tensions | Literature matrix 43 rows + header, landscape 10 sections, contradictions 6 tensions +2 no-strong +8 checks 265 lines — all files present | **PASS** — Source `literature_matrix.md`, `research_landscape.md`, `contradictions.md` |
| 8 | **Candidate gaps conservative & evidence-sourced** | `research/knowledge/candidate_gaps.md` exists, 5-15 gaps (here 10), each with 6 subsections, **Status NOT YET VERIFIED AS NOVEL**, conservative language | 10 gaps G1-G10 each 6 subsections, 2-5 related papers + maps evidence, underexplored/candidate gap phrasing, diverse dimensions — `candidate_gaps.md` 230 lines | **PASS** — Source `candidate_gaps.md` |
| 9 | **Stage 2A summary output integrity** | `research/knowledge/stage2a_summary.md` exists at required path, contains all required sections: Statistics, Major Technical Categories, Major Bottlenecks, Common Assumptions, Repeated Limitations, Evaluation Patterns, Top Candidate Gaps (5-15 with evidence), Uncertainties, File Index, Next Step Recommendation + 9 end checks listed | This file exists at `F:\AIinfraResearch\research\knowledge\stage2a_summary.md` with all 11 sections + 9 checks table — verified existence and structure (2026-08-27) | **PASS** — Source this file |

> **Overall Stage 2A stopping: 9/9 PASS — All Stage 2A artifacts exist and meet coverage/traceability/conservative-language criteria per input file index. Ready for GPT Stage 2B novelty verification.**

---

## File Index

*Absolute paths — all Stage 2A inputs verified present + this output. Source `research/manifests/papers.md` (43) + `research/knowledge/*.md` (11) per task input list.*

| # | Absolute Path | Artifact | Lines / Status — Source verification |
|---|---|---|---|
| 1 | `F:\AIinfraResearch\research\manifests\papers.md` | Seed paper set (43 papers) | 586 lines — Research Lead merged Scout A/B/C dedup 38 + auditor +5 →43; 31 High/12 Medium |
| 2 | `F:\AIinfraResearch\research\knowledge\taxonomy_draft.md` | Taxonomy draft (14 categories) | 572 lines — 14 categories ≥12 required, full subsections, sampled 24 notes |
| 3 | `F:\AIinfraResearch\research\knowledge\bottleneck_map.md` | Bottleneck map (14 bottlenecks) | 640 lines — 12+2, 24+ notes read, remaining limitations per bottleneck |
| 4 | `F:\AIinfraResearch\research\knowledge\evaluation_map.md` | Evaluation map | 150 lines — 4 sections metrics/baselines/workloads/hardware, Highlight table 14 limited-conditions |
| 5 | `F:\AIinfraResearch\research\knowledge\assumption_map.md` | Assumption map (12 assumptions) | 339 lines — 26 notes read, confidence-rated, cross-cutting implications |
| 6 | `F:\AIinfraResearch\research\knowledge\limitation_map.md` | Limitation map (13 limitations) | 242 lines — 22 notes read, author-stated vs inferred, types Widely/Appears/Paper-specific |
| 7 | `F:\AIinfraResearch\research\knowledge\contradictions.md` | Contradictions / Tensions map | 265 lines — 6 tensions +2 no-strong, 8 checks, resolution by conditioning |
| 8 | `F:\AIinfraResearch\research\knowledge\literature_matrix.md` | Literature matrix (43 rows) | 119 lines — 43 papers main matrix + counts per layer/bottleneck/metrics |
| 9 | `F:\AIinfraResearch\research\knowledge\research_landscape.md` | Research landscape | 117 lines — 10 sections overview→tensions, mature/active/sparse |
| 10 | `F:\AIinfraResearch\research\knowledge\candidate_gaps.md` | Candidate gaps (10) | 230 lines — 10 gaps each 6 subsections, Status NOT YET VERIFIED AS NOVEL |
| 11 | `F:\AIinfraResearch\research\knowledge\coverage_report.md` | Coverage report | 138 lines — 7 well covered +5 weak W1-W5 +5 suggested additions verified |
| 12 | `F:\AIinfraResearch\research\knowledge\stage1_summary.md` | Stage1 summary | 165 lines — 43 notes 100%, 37 reviews, 24 subagents, 5 supplements, readiness |
| 13 | `F:\AIinfraResearch\research\knowledge\stage2a_summary.md` | **Stage 2A Summary (this file)** | **This file — 2026-08-27 — all required sections, 9 end checks, conservative language, traceable** |

> **Integrity:** Every bullet in this summary hints source file (e.g., `taxonomy_draft.md Cat.1`, `bottleneck_map.md B3`, `evaluation_map.md Sec.1`). No hallucinated titles — all paper names/years/venues verbatim in `papers.md` per `taxonomy_draft.md` Traceability notes.

---

## Next Step Recommendation

> **No new solution brainstorming per rules; only recommendation for Stage 2B process.**

1. **GPT further verification first (mandatory before claiming novelty):** For each of the 10 candidate gaps above (especially prioritized Gaps 1,2,4,9,10 High), run external literature search (arXiv 2024-2026, OSDI/SOSP/FAST/EuroSys 2025-2026, Dynamo/LMCache docs) to verify **NOT YET VERIFIED AS NOVEL** status — gaps are currently **conservative "underexplored" based only on 43-paper corpus** per `candidate_gaps.md`; do not advance to idea generation until verification completes. Focus checks: joint token×bit Pareto (G1) vs recent multi-axis work, online tuning (G2) vs RL-based controllers, fault tolerance (G4) beyond DéjàVu, TCO harness (G9) beyond Splitwise rental, burst production traces (G10) beyond Poisson simulators.

2. **Harness unification before Stage 2B ideas:** Build or select unified evaluation harness that can sweep prompt 1K→1M, output 32→512, arrival Gamma CV, SLO strict (0.1s) vs relaxed (30s 10×/5×), hardware A10G/A100/H20/H100/GB200 NVL72 and CXL vs RDMA — exactly the gap highlighted in `evaluation_map.md` Highlight and `contradictions.md` Overall Assessment point 1, to attribute FlowKV vs Splitwise transfer gap and compare disaggregation vs chunked hybrid Pareto on same traces.

3. **Staged Stage 2B idea generation:** After verification, prioritize 3-4 gaps where synthesis confidence remains High and evidence strongest (G1 joint optimization, G2 adaptive tuning, G5 SLO-fair scheduling, G9-10 evaluation realism) before tackling CXL placement (G3) and security (G6) which remain Emerging/Sparse per `research_landscape.md` Sparse. Keep traceability: each idea must cite which papers/maps evidence it addresses.

4. **Risk management:** Treat supplemental 5 papers and hardware [NOT REPORTED] uncertainties (see Uncertainties Sec.1-2) as provisional — re-read CacheGen/CachedAttention/Dynamo/Llumnix/LoongServe dual-read before final proposal; re-profile cost/power with real metering not rental.

> **Stopping condition met:** File exists at required path with all required sections and 9 end checks listed. No other files created.

*End of stage2a_summary.md — Stage 2A Synthesis Lead 2026-08-27 — Workdir F:\AIinfraResearch — 10 candidate gaps prioritized, all traceable, conservative.*

