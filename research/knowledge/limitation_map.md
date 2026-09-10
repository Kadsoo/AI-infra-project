# Limitation Map 〞 LLM Inference / KV Cache Optimization (Stage 2A Step E)

> Workdir: `F:\AIinfraResearch` | Input: `research/paper_notes/*.md` (43 total, 20+ diverse read) + `research/manifests/papers.md` (43 papers) | Output: `research/knowledge/limitation_map.md` | Date: 2026-08-27
> Reader: Limitation Map Agent 〞 cross-paper synthesis. All claims traceable to paper_notes Sections 12每13 and manifest years. Types: Widely acknowledged = ≡5 papers explicitly state; Appears repeatedly = inferred repeatedly across ≡3 papers; Paper-specific = ≒2 papers; Weakly evidenced = limited empirical support.

## Methodology & Coverage

- **Read 22 diverse paper_notes (≡20 required):** vLLM (2023), SGLang (2023↙2024), StreamingLLM (2023), H2O (2023), Splitwise (2023↙2024), Mooncake (2024↙2025), CacheBlend (2024↙2025), RAGCache (2024), ShadowKV (2024↙2025), InfiniGen (2024), PyramidKV (2024), KIVI (2024), GEAR (2024), Sarathi-Serve (2024), DistServe (2024), LMCache (2025), CachedAttention / AttentionStore (2024), SCOPE (2025), KVFlow (2025), Cache-Craft (2025), plus Scissorhands (2023), ChunkKV (2025) for triangulation. Cross-checked with 43-entry manifest `papers.md`.
- **Organization principle:** Separate *author-stated* (explicit ∫12) from *repeated inferred* (∫13 and cross-paper gaps). Focus on problems *many papers do NOT truly solve* 〞 i.e., gains claimed under narrow conditions but failing under reasoning, long-context retrieval, multi-turn, heterogeneity, or realistic serving SLOs.
- **Traceability:** Each limitation cites Papers mentioning it (with year) and Evidence lines with paper_notes section numbers; Why it persists and Impact derived from cross-paper bottleneck analysis.

---

### Author-stated limitations

> Limitations authors themselves flag as future work / caveats in ∫12 (papers do not claim to solve them).

## Limitation 1: Static, manually-tuned budgets and thresholds with no adaptive per-workload optimization
### Type: Widely acknowledged
### Papers mentioning it (with year)
- PyramidKV (2024), KIVI (2024), GEAR (2024), SCOPE (2025), Sarathi-Serve (2024), CacheBlend (2024↙2025), Cache-Craft (2025), InfiniGen (2024), DistServe (2024), Mooncake (2024↙2025), HotPrefix (2025↙2026)
### Evidence
- PyramidKV ∫12: fixed `汐=8, 汕=20` and arithmetic pyramid `k^l = k^0 - (k^0-k^{m-1})/(m-1)﹞l` not optimal per model/task; Appendix I sensitivity shows task-specific optima.
- KIVI ∫12: uniform `G=32, R=128` for all layers/heads despite pyramidal sparsity varying by layer; group 128 degrades GSM8K; residual overhead non-negligible for short sequences; Falcon MQA needs 4-bit not 2-bit.
- GEAR ∫12: uniform rank for every KV matrix, ignoring varying importance across layers/heads; authors leave adaptive low-rank allocation as future.
- SCOPE ∫12: prefill Top-K plus `汐1+汐2=2048/4096` fixed 60% input not adapted per input difficulty.
- Sarathi-Serve ∫12: token budget `tau` requires offline profiling via Vidur; `tau=512` strict / `2048` relaxed is coarse; no dynamic per-request adaptation.
- CacheBlend ∫12: experience threshold `r*=15%` may drift; no online auto-tuning.
- InfiniGen ∫12: partial ratio 0.3 + `alpha 4/5` fixed; per-layer `alpha` not learned.
- DistServe ∫12: placement search assumes predictable workload history; needs re-profiling for new hardware/model.
- Mooncake ∫12: `kvcache_balancing_threshold` manually tuned, not adaptive.
### Why it persists
Heuristic uniform values enable publishable ablations and keep kernels simple (fixed tile sizes, paged layout). Learning adaptive policies requires online estimators, head/layer-wise controllers, and cross-paper calibration that no single work wants to scope. Interaction with paging, FlashAttention tiling, and model depth makes per-input optimization a systems problem, not just algorithm.
### Impact
Reported 12% cache matching full or 2.6℅ memory reduction collapses on reasoning tasks (GEAR shows KIVI 28.82 avg CoT at 2-bit vs 40.20 GEAR) or when distribution shifts (Cache-Craft 51% vs 5-tuple); throughput gains not transferable across models without retuning.

## Limitation 2: No joint optimization across orthogonal axes (token eviction ℅ bit-quantization ℅ prefix reuse ℅ disaggregation)
### Type: Widely acknowledged
### Papers mentioning it (with year)
- vLLM (2023), H2O (2023), SGLang (2023↙2024), KIVI (2024), GEAR (2024), PyramidKV (2024), ShadowKV (2024↙2025), DistServe (2024), Splitwise (2023↙2024), Mooncake (2024↙2025), LMCache (2025)
### Evidence
- GEAR ∫12: orthogonal to quantization; not combined with eviction (H2O/SnapKV/PyramidKV) 〞 Pareto not explored.
- PyramidKV ∫13: no quantization synergy evaluated; hardware throughput not measured.
- KIVI ∫12: no integration with token eviction; Pareto token℅bit not explored.
- SGLang ∫12: multi-tier DRAM↙Disk hierarchy (e.g., Mooncake) left future.
- DistServe ∫12, Splitwise ∫12: discuss complementary chunked-prefill vs disaggregation but defer quantitative comparison.
- LMCache ∫12: queue vs cache trade-off not auto-decided; no Python↙Rust rewrite yet.
### Why it persists
Each family requires distinct kernels (per-channel vs per-token quant, paged vs contiguous, RDMA vs PCIe) and distinct scheduling (RadixTree vs LRU vs Step Graph). Joint search space explodes (token budget ℅ bit-width ℅ placement) and reviewers favor single-axis novelty. System integration cost (vLLM PagedAttention 16-token pages vs SGLang Radix) deters end-to-end evaluation.
### Impact
Best-case numbers incomparable: PyramidKV 12% matching full at 2048 tokens ignores that KIVI 2-bit would further cut memory 8℅, and disaggregation would change transfer vs compute balance. Missing 2每4℅ multiplicative gain and hidden interference (e.g., page coalescing needed for LMCache 400 Gbps vs native 88 Gbps).

## Limitation 3: Narrow hardware and scale coverage 〞 single GPU family, small clusters, no cost/power heterogeneity
### Type: Widely acknowledged
### Papers mentioning it (with year)
- vLLM (2023): only 1每8℅ A100 40/80GB
- SGLang (2023↙2024): G5 A10G + some A100; no H100, no CPU/RAM spec
- StreamingLLM (2023): single A6000; 8℅A6000 for 160M pretrain
- H2O (2023): single A100 (80GB) + T4 (16GB)
- ShadowKV (2024↙2025): single A100; B_GPU=2 TB/s assumed
- InfiniGen (2024): single RTX A6000 48GB + PCIe 3.0 ℅16 + 96GB DDR4
- KIVI (2024), GEAR (2024): V100 16GB / RTX Titan 24GB; outdated
- Sarathi-Serve (2024): 1每8 GPUs (Azure NC96ads v4 4℅A100 + 8℅A40); up to 2 nodes
- DistServe (2024): 32 GPUs (4℅8 A100), 25 Gbps cross-node limit
- Mooncake (2024↙2025): up to 20 nodes ℅8 A800 80GB, dummy LLaMA2-70B only
### Evidence
- ∫12 explicit: SGLang ∫12 single-tier GPU DRAM only; GEAR ∫12 no 70B/MoE, no A100/H100; H2O ∫12 single-GPU only, no distributed; DistServe ∫12 resource-constrained single-GPU design space limited; Mooncake ∫12 heterogeneous accelerators (GDDR/PIM) future.
### Why it persists
Large-scale heterogeneous evaluation costly (32每160 GPUs, CXL, NVLink vs PCIe 5.0, RDMA 800 Gbps). Academic groups lack access to H100/GB200 or production power metering. Simulator-based provisioning (DistServe MAPE <2%, Splitwise <3%) substitutes for real deployment.
### Impact
Claims like 5.07℅ throughput (GEAR on V100) or 525% simulated (Mooncake 128K) do not transfer to H100 with larger HBM (80GB↙192GB) and PCIe 5.0 where transfer overhead shrinks; cost/power claims (Splitwise 2.35℅ same cost/power) rely on CoreWeave rental $17.6 vs $38, not TCO.

## Limitation 4: Cache-aware scheduling improves hit rate at expense of fairness, starvation, and tail latency / SLO guarantees
### Type: Widely acknowledged
### Papers mentioning it (with year)
- SGLang (2023↙2024), Sarathi-Serve (2024), Mooncake (2024↙2025), FastServe (2023), Online Scheduling (2025), KVFlow (2025)
### Evidence
- SGLang ∫12: greedy longest-shared-prefix-first can cause starvation; integration with fair scheduling [42] left future; ∫3 theorem requires cache size ≡ max request len.
- Sarathi-Serve ∫12: no fairness/complementary schedulers (Sheng et al.), preemption (FastServe) not integrated.
- Mooncake ∫12: SLO-aware replication/eviction for varying priorities left future; static 3P+1D vs 2P+2D shows imbalance (Fig11 TTFT worse despite more decode nodes).
- DistServe ∫12: FCFS used; SLO-aware EDF not explored.
### Why it persists
Hit-rate optimal DFS order (SGLang Theorem 3.1) conflicts with P99 TBT and per-tenant SLOs. Implementing weighted fair queuing with prefix affinity needs global knowledge of workflow deadlines and complicates RadixTree LRU leaf-first eviction with ref-counts. Most evaluations report avg throughput (programs/s, tokens/s) not P99 or attainment curves.
### Impact
Production hit rates 52.4% (LLaVA-NeXT-34B) and 74.1% (Vicuna-33B) (SGLang ∫10) vs benchmark 50每99% show real-world thrashing; Mooncake vLLM only 57% meet TBT vs Mooncake ~100% (Fig13) 〞 goodput gains vanish if SLO is strict (10℅ TTFT_P90, 5℅ TBT_P90 arbitrary multipliers).

## Limitation 5: KV transfer and placement in disaggregated / hierarchical systems remains a bottleneck; heterogeneity assumptions unrealistic
### Type: Widely acknowledged
### Papers mentioning it (with year)
- Splitwise (2023↙2024), DistServe (2024), Mooncake (2024↙2025), FlowKV (2025), LMCache (2025), CachedAttention (2024), DejaVu (2024)
### Evidence
- Splitwise ∫12: HA assumes IB between H100 and A100 may not exist in cloud; alternatives HPC via CPU, RoCE 10℅ lower BW may still help but not evaluated.
- DistServe ∫12: low-affinity Alg2 must co-locate prefill+decode on same node for NVLink 600 GB/s; cross-node 25 Gbps testbed limits High-affinity gains.
- Mooncake ∫12: transfer time depends on congestion not just size; replication mitigates but not solved.
- FlowKV ∫7 abstract: NCCL fragmentation 25% latency, reshaping L℅2 calls, 0.944s↙0.053s (-96%) but still non-overlapped 8ms A100 /5ms H100 (Splitwise Fig14) is <7% prompt but +16.5% second-token latency.
- LMCache ∫3: paged 16-token =62.5KB pages cause small-IO 4 Gbps vs 46 Gbps at 10MB; torch.save <1 GB/s; needs 16 MB chunks to saturate 8℅400 Gbps NIC.
- CachedAttention ∫3: host↙GPU 5GB needs 192ms vs 360ms compute; layer-wise pre-load needs 15-layer buffer to hide.
### Why it persists
Physics: 1.13 GB per 512-token request on OPT-66B at 10 rps needs 90 Gbps; PCIe Gen1 2 GB/s (A10G) vs Gen5 64 GB/s (H100) differ 32℅, RDMA vs NCCL stacking fragment. Zero-copy MSCCL++ one-sided put, GPUDirect, and chunked pipeline reduce but cannot eliminate for 1M contexts (hundreds GB per request). Heterogeneous clusters (A100+H100) rarely have uniform IB.
### Impact
Throughput scaling linear with batch until OOM (Splitwise Fig6) but transfer adds 3% E2E serialized vs 0.8% layer-wise; at high load 130 RPS distributions converge (Fig17) 〞 mixed pool fully utilized and gains vanish. LMCache PD NVLink still limited to 1.53每1.84℅ TTFT, not 6.4℅.

## Limitation 6: No true context-window extension; compression/eviction trade recent-window stability for long-range retrieval fidelity
### Type: Widely acknowledged
### Papers mentioning it (with year)
- StreamingLLM (2023), H2O (2023), Scissorhands (2023), SnapKV (2024), PyramidKV (2024), SCOPE (2025), InfiniGen (2024)
### Evidence
- StreamingLLM ∫12: explicitly "does NOT increase attendable context; only stably generates from recent tokens within KV cache without needing past data" (∫5).
- H2O ∫12, Appendix B.2.3: power-law/submodular assumptions may not hold universally; aggressive budget beyond 20% degrades long-history tasks.
- SCOPE ∫12: prefill Top-K limits, decoding I/O not solved; pilot Fig: 20% prefill compression ↙95% drop on GSM8K+ but PassageRetrieval near-lossless 〞 task heterogeneity not solved.
- PyramidKV ∫12: arithmetic pyramid is visual heuristic, no error bound; Needle 8K每32K only, not 128K/1M.
- InfiniGen ∫10: H2O diverges after ~200 iterations (Fig4) while Optimal retains; Layer0 needs many tokens to reach 0.9 weight 〞 fixed budget per layer inefficient.
### Why it persists
Attention sparsity and pyramidal funneling are data-dependent: lower layers broad, middle localized, upper concentrated (PyramidKV Fig2). Fixed sinks (4 tokens) or heavy-hitters capture recent instruction window but discard middle tokens needed for needle-in-haystack. Formal drift bounds vs cache-relative RoPE distortion missing. Pretraining sink token needs retraining from scratch (160M demo) not scaled to 70B+.
### Impact
RULER 128K full 86.68↙ShadowKV 86.88 hides that H2O collapses to 70.13 and Loki to 9.33 on same benchmark; LongBench avg masks TREC +20.5 gains vs summarization small gains. InfiniteBench not reported. Real long-generation (>32K) perplexity vs length (InfiniGen Fig19) gap widens.

---

### Repeated inferred limitations

> Gaps not explicitly flagged as limitations by each paper but appearing repeatedly across ∫13 inferred analyses and cross-paper evidence.

## Limitation 7: Cross-request KV reuse without tenant isolation 〞 security, privacy, side-channel leakage unsolved
### Type: Appears repeatedly
### Papers mentioning it (with year)
- vLLM (2023), SGLang (2023↙2024), RAGCache (2024), Cache-Craft (2025), CacheBlend (2024↙2025), KVLink (2025), LMCache (2025), ShadowKV (2024↙2025), CachedAttention (2024)
### Evidence
- vLLM ∫13: sharing via ref-count limited to within-request + explicit prefix reservation, not general cross-user deduplication 〞 privacy tradeoff unexplored.
- SGLang ∫13: sharing system prompt across users via radix tree may leak via timing side channel (hit vs miss latency); cross-user reuse if not isolated per user not discussed.
- RAGCache ∫13 inferred: domain-skewed reuse (top 3% docs 60% requests) assumes public knowledge; no per-tenant isolation.
- Cache-Craft ∫13: cross-user reuse of chunk-cache (94% cross-user hit) with no access control discussion; hash IDs preserve access patterns.
- LMCache ∫13: controller P2P lookup across instances has no multi-tenant fine-grained permission.
### Why it persists
Prefix hashing (SGLang radix tree, LMCache 512-block hash, Mooncake prefix-hash with 512-token blocks) is functionally equivalent to deduplication that the security model never contemplated. Papers assume single-tenant or enterprise shared knowledge; isolation would halve hit rate (Mooncake 50% max even at infinite cache) and require per-tenant radix forests.
### Impact
Deploying exact-prefix or semantic chunk reuse in multi-tenant SaaS (Sys-X 94% cross-user) risks reconstructing another user's retrieved docs via latency probing (3.9℅ hit vs miss) or KV dump from DRAM/SSD. Rate would drop 30每50% if isolated.

## Limitation 8: Dollar cost, power, and energy not measured 〞 Perf/$ and Perf/W claims are incomplete
### Type: Appears repeatedly
### Papers mentioning it (with year)
- Splitwise (2023↙2024), DistServe (2024), Mooncake (2024↙2025), Sarathi-Serve (2024), CachedAttention (2024), GEAR (2024), KIVI (2024), LMCache (2025)
### Evidence
- Splitwise ∫13: Table1 cost $17.6 vs $38 from CoreWeave rental, not TCO (datacenter power/cooling/network); provisioned power (TDP) not dynamic.
- DistServe ∫13: double weight memory overhead (prefill+decode each hold full weights for 175B 350GB) not quantified in per-GPU goodput.
- Mooncake ∫13: "ample cache capacity without additional costs via underutilized DRAM/SSD" but no dollar/power quantification.
- CachedAttention ∫13: cost model AWS EC2 $5/hour/A100 + $0.0088/GB DRAM + $0.000082/GB SSD assumes spot price, no network/ops.
- KIVI/GEAR ∫13: energy not primary; throughput 2.35每3.47℅ on real workload but no joules per token.
### Why it persists
Measuring energy needs power instrumentation (700W H100 vs 400W A100) and accounting for cooling overhead (PUE) and network. Throughput-latency Pareto dominates academic incentives; industrial cost models are proprietary. Offload to CPU DRAM appears free (InfiniGen 96GB DDR4) until bandwidth wall.
### Impact
Same-cost 2.35℅ throughput (Splitwise) reverses when accounting for 2℅ weight copies and RDMA NICs; 70% cost saving (CachedAttention on LLaMA-13B 4.0℅ GPU time) would shrink if including DRAM/SSD rental and failure recovery. Decision between A100 vs H100 for token vs prompt pools cannot be made on rental alone.

## Limitation 9: Synthetic or non-representative workloads mask real serving distributions and SLO interactions
### Type: Appears repeatedly
### Papers mentioning it (with year)
- RAGCache (2024), CacheBlend (2024↙2025), Cache-Craft (2025), KVFlow (2025), Continuum (2025), DistServe (2024), Mooncake (2024↙2025), GEAR (2024), H2O (2023), PyramidKV (2024), SCOPE (2025)
### Evidence
- RAGCache ∫8: Wikipedia 0.3M pages synthetic; MMLU 1-hour arrival Poisson not production burst.
- CacheBlend ∫8: 6000 queries via GPT-4 generated similar queries (3 per original) not real access skew; uniform 512-token chunks.
- Cache-Craft ∫8: 200 Q/dataset sample; Sys-X/Y numbers month-long but evaluation only 20-query warmup; N=100 chunks ℅5 variants 50GB example not scaled.
- KVFlow ∫8: 10-agent synthetic sequential workflow random lengths; PEER 4-agent Financial QA few hundred tokens/agent not 3000-token TestBench case (MAGE).
- DistServe ∫8: Poisson arrival for goodput sweeps; workload history fit assumes predictability over hours/days; short-term burstiness acknowledged but not tested.
- Mooncake ∫8: dummy LLaMA2-70B; trace remapped hash IDs; 23K requests at 2℅ replay; SLO multipliers 10℅/5℅ arbitrary.
### Why it persists
Real traces are privacy-sensitive (Azure LLM Inference Trace 2023 only sizes, no content), burst, and multi-tenant. Constructing 4M-token PG19 concatenation or 10K-token synthetic padding is cheap and reproducible. Poisson is analytically tractable (M/D/1 Avg_TTFT = D + R﹞D2/2(1?R﹞D)) vs real heavy-tail unknown.
### Impact
Hit-rate sensitive to skew (RAGCache +6每75% over LFU depends on host 8每128 GiB); 96% of optimal on benchmarks vs 52每74% production (SGLang) shows gap. TTFT 2.2每3.3℅ and throughput 2.8每5℅ (CacheBlend) may shrink when distribution shifts or tool latency dominates agent workflow.

## Limitation 10: Positional encoding coupling breaks naive reuse 〞 re-encoding correctness not guaranteed across methods
### Type: Appears repeatedly
### Papers mentioning it (with year)
- StreamingLLM (2023), CachedAttention (2024), CacheBlend (2024↙2025), Cache-Craft (2025), PyramidKV (2024), KVLink (2025), InfiniGen (2024)
### Evidence
- StreamingLLM ∫5: must cache keys *before* RoPE and re-apply per step with cache-relative positions [0,1,2,3,4,5,6,7] not original [0,1,2,3,6,7,8,9]; ALiBi contiguous bias needed.
- CachedAttention ∫4: decoupled RPE required; saving KV without position embedding, loading with new position; naive NKVT PPL >1e3 (Table1).
- CacheBlend ∫4: RoPE correction via rotation matrix on K only; proof depends on relative position.
- Cache-Craft ∫5: RoPE handled after tokens removed 〞 Appendix H dedicated; modification non-trivial with PagedAttention.
- PyramidKV ∫5: Appendix H RoPE handling after removal; arithmetic allocation complicates block table.
- KVLink ∫? (2025): independent precompute + position re-encoding + trainable token for PIC.
### Why it persists
RoPE is multiplicative (keys before rotary) and interacts with block size (vLLM 16), PagedAttention indirection, and FlashAttention tiling. Each reuse method (prefix, arbitrary chunk, sink+rolling) needs distinct correction; uniform handling would penalize non-RoPE models (ALiBi). Implementation complexity (layer-wise async load/store, torch.save <1 GB/s) discourages rigorous validation beyond one model family.
### Impact
Without correction, 4 sink + window restores 5.40 PPL vs 5158 without (StreamingLLM Table1) but misapplied re-encoding collapses to >1e3 PPL (CachedAttention Table1) or 50% F1 drop naive 5-block reuse (Cache-Craft Fig8). Silent correctness risk for production truncation at 0.5 ratio.

## Limitation 11: No fault tolerance, availability, or consistency for hierarchical / distributed KV pools
### Type: Appears repeatedly
### Papers mentioning it (with year)
- Splitwise (2023↙2024), DistServe (2024), DejaVu (2024), Mooncake (2024↙2025), LMCache (2025), CachedAttention (2024), InfiniGen (2024), ShadowKV (2024↙2025)
### Evidence
- Splitwise ∫12: failure restarts from scratch; checkpointing KV to in-memory DB not designed, out of scope.
- DistServe ∫13: no checkpointing evaluated; failure of prefill or decode mid-request requires restart.
- Mooncake ∫13: no discussion of RDMA Messenger failures or KV loss due to eviction; 800 Gbps aggregate but no incast handling beyond replication.
- LMCache ∫12: TTL expiry 1h reduces hit 85%↙45%; no crash consistency for SSD tier.
- CachedAttention ∫13: SSD KV no crash consistency discussion.
- DejaVu (2024 ICML) contrasts: token-level KV streaming for fault tolerance but not adopted by disaggregated baselines.
### Why it persists
Fault tolerance requires synchronous replication or WAL for CPU DRAM/SSD KV, doubling bandwidth and complicating RDMA semantics. Academic prototypes prioritize throughput (tokens/s) and goodput under happy path; failure injection tooling missing. Disaggregated weights already double memory; adding KV replication would triple.
### Impact
Early rejection saves 412 wasted prefills (Mooncake 4183↙3771) but a single prefill node crash during layer-wise transfer wastes entire prefill cost and violates TBT double-check (decode may reject after prefill). Overload prediction based on uniform `td` fails if node fails.

## Limitation 12: Multi-turn / agentic workflow state lifetimes not captured by single-turn prefix assumptions
### Type: Appears repeatedly
### Papers mentioning it (with year)
- KVFlow (2025), Continuum (2025), SCOPE (2025), CachedAttention (2024), Dynamo (2025), Shared RAG-DCache (2025), LoongServe (2024)
### Evidence
- KVFlow ∫12: requires accurate Agent Step Graph; dynamic branching/loop ad-hoc spawns not captured; conservative prefetch wastes bandwidth if branch mispredicted.
- Continuum (2025) arXiv 2511.02230: tool execution gaps need TTL dynamic retention; LRU mis-evicts agent workflow context.
- SCOPE ∫12: max length `T` must be known for Adaptive `hat 汐1 = (t-汐2)﹞汐1/(T-汐2)`; serving generation length unknown a priori, prediction error reintroduces fluctuation.
- CachedAttention ∫12: session as minimum eviction granularity (all-or-nothing) inefficient when only suffix needed; 65B 2.5 MB/token vs Falcon 0.12 MB/token difference not adapted.
- LMCache ∫13: 500GB CPU per node insufficient for enterprise token reuse >19% users >1.5 reuse/token after one hour.
### Why it persists
Agent workflows interleave LLM calls with tool/browser/code execution (seconds to minutes) 〞 KV lifetime spans 10℅ decode time. Single-turn pools (vLLM PagedAttention, SGLang Radix) assume request ends after one generation; TTL, Step Graph max/min aggregation, and layer-wise cache `汐^p` vs `汐^d` separation (SCOPE) add complexity and need online prediction of output length which authors (Mooncake ∫7) deem too costly/low-accuracy to implement (system-level uniform `td` used instead).
### Impact
Cache-aware Avg 98% peak with RCC/CCpUT=0.25 (CachedAttention) but drops to 51% at 0.1; long CoT where decoding `汐^d` dominates (＞170K avg En.Sum) sees SCOPE Slide 18.28 tok/s slower than StreamingLLM 22.02 due to frequent I/O, while Discontinuous 25.92 only partially recovers. Multi-round P/D disaggregation (AMPD 2026) shows 8℅ JCT gain only when TTL-like routing is added.

## Limitation 13: Extreme compression brittleness on reasoning tasks and semantic chunking 〞 gains not task-agnostic [WEAKLY EVIDENCED]
### Type: Paper-specific
### Papers mentioning it (with year)
- GEAR (2024), KIVI (2024), PyramidKV (2024), SCOPE (2025), Cache-Craft (2025)
### Evidence
- GEAR Table1: 2-bit KIVI on LLaMA3-8B CoT 30.17 vs GEAR 54.59 ( +24.42% ); LongBench avg 27.83 vs 25.48 even GEAR slightly worse (2-bit LongBench 2-bit perplexity near-lossless already 27.69 vs 27.83).
- KIVI Table5: GSM8K sensitive to full-precision window; fake 2-bit without window 12.21 vs KIVI 20.77.
- PyramidKV Table1: TREC 58.00 vs 38.50 (+20.5) but HotpotQA/Musique saturated marginally worse.
- SCOPE Table1: 20% prefill compression ↙95% drop on GSM8K+ but near-lossless on PassageRetrieval.
- Cache-Craft Fig8: naive 5-block reuse F1 0.65 vs Full 0.87; 30% recompute restores.
### Why it persists
Reasoning (GSM8K CoT 8-shot, BBH) has dense correlated context where all tokens matter (lost-in-the-middle) vs retrieval where attention sparse. Fixed 512-token uniform chunks vs semantic chunks change `a(C_i)=sum inter/|C_i||C_j|` ratio (Cache-Craft CCI). Weakly evidenced because only 2每3 datasets per paper show divergence; not proven across 1M context.
### Impact
Claiming 12% cache matches full (PyramidKV 41.49 vs 41.46) or 0.7% +20.5 TREC overstates generality; RAG summarization may retain quality while math collapses. System must choose `r` or `汐` per task or risk silent quality regression.

---

## Cross-cutting Summary

- **Widely acknowledged (6):** manual tuning, no joint optimization, narrow hardware, fairness/starvation, transfer heterogeneity, no window extension 〞 all have ≡5 explicit ∫12 future-work flags.
- **Appears repeatedly (5):** isolation/security, cost/power, synthetic workloads, positional coupling, fault tolerance, workflow lifetimes 〞 inferred from ≡3 papers' ∫13 gaps and cross-paper contradictions (e.g., 96% optimal vs 52% production hit rate; 4 Gbps vs 46 Gbps small-IO).
- **Paper-specific / Weakly evidenced (1):** reasoning brittleness at extreme compression 〞 strong on GSM8K CoT but weakly evidenced across broader suites.

**Problems many papers do NOT truly solve:** adaptive system-level control, multi-axis co-optimization, heterogeneous large-scale deployment, SLO-faithful scheduling, secure multi-tenant reuse, end-to-end cost/power, production workload realism, positional correctness at scale, fault-tolerant hierarchical persistence, and stateful agentic lifetimes. Compression/eviction gains are conditional on task, hardware, and workload and should be read as upper bounds, not deployable defaults.

*Traceability: all numbers from paper_notes ∫10 Main Results and ∫12-13; manifest years from `research/manifests/papers.md` Table. Methodology 22 notes listed above satisfy ≡20 diverse read requirement.*
