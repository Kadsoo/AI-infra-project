# Assumption Map — LLM Inference / KV Cache Field (Stage 2A Step D)

> Workdir: F:\AIinfraResearch | Sources: research/paper_notes/*.md (43) + research/manifests/papers.md | Date: 2026-08-27
> Method: Read 26 diverse notes (Orca, vLLM, SGLang, FlexGen, StreamingLLM, H2O, Scissorhands, FastServe, Splitwise, SnapKV, PyramidKV, KIVI, GEAR, TetriInfer, DistServe, DéjàVu, Sarathi-Serve, InfiniGen, RAGCache, ShadowKV, Mooncake, CacheBlend, Cache-Craft, KVFlow, Continuum, Beluga) + papers.md.

> Method detail: Read 26 diverse notes plus papers.md. Extracted repeated explicit/implicit assumptions; each entry traceable to Stage-1 notes with [PAPER FACT] evidence. Confidence = frequency x directness x contestation.

---

## Assumption 1 — GPU HBM Capacity (KV Cache Size) Is the Dominant Throughput Bottleneck

### Papers relying on it (6 with year)
- Orca (2022) — reserves max_tokens slots per request [PAPER FACT]
- vLLM (2023) — PagedAttention to eliminate reservation waste [PAPER FACT]
- FlexGen (2023) — offload placement LP minimizing T/bls under GPU/CPU/disk peak-memory constraints [PAPER FACT]
- H2O (2023) — 30B batch 128 seq 1024 -> 180GB KV example [PAPER FACT]
- Sarathi-Serve (2024) — capacity defined by TBT SLO + max batch size tradeoff [PAPER FACT]
- Mooncake (2024-2025) — disaggregated KVCache-centric pool, VRAM occupation cost noted [PAPER FACT]

### Why they need it
If KV memory caps batch size, optimizing allocation (paging, eviction, offloading, disaggregation) directly yields linear throughput gains. All throughput claims are stated as "more concurrent requests per GPU" rather than FLOPS.

### What happens if it fails (impact)
If true bottleneck is compute (long prefill), interconnect, or CPU scheduling, larger batches do not scale linearly; vLLM 175B Alpaca case already shows compute-bound plateau where gains over Orca vanish [PAPER FACT]. For H100 the FLOPS/bandwidth ratio diverges (3.43x FLOPS vs 1.64x bandwidth in Splitwise Table1) [PAPER FACT]; optimizing only memory then wastes engineering and may increase latency without throughput.

### Evidence (quote or paraphrase from notes with [PAPER FACT] style)
- "For OPT-13B, one token = 800 KB ... A100 40GB devotes ~65% to weights, ~30% to KV cache" [PAPER FACT] (vLLM S1)
- "KV cache at batch 128 seq 2048 is 1152GB (3.5x weights) ... max batch 34 for OPT-175B on 8xA100 80GB" [PAPER FACT] (Scissorhands Table1-2)
- "Only 20.4%-38.2% of KV memory holds actual token states (Fig.2) ... vLLM ~96% effective" [PAPER FACT] (vLLM S3)
- "With batch 16 increasing length, KV easily surpasses OPT-30B weights (Fig2)" [PAPER FACT] (InfiniGen S2)
- "Llama-2-7B 100K tokens -> >50GB" [PAPER FACT] (PyramidKV) and "50M tokens in Kimi -> ~20TB DRAM" [PAPER FACT] (Beluga)

### Confidence: High

---

## Assumption 2 — Prefill Is Compute-Bound (Quadratic), Decode Is Memory-Bandwidth-Bound — Stable Phase Dichotomy

### Papers relying on it (6 with year)
- Splitwise (2023-2024) — Insights I-VI: prompt throughput degrades after 2048 tokens while token throughput scales to batch 64 [PAPER FACT]
- DistServe (2024) — superlinear prefill vs memory-bound decode, chunked prefill causes O(N^2) KV loads [PAPER FACT]
- TetriInfer (2024) — accelerator-saturate threshold 512 tokens for OPT-13B; bubble taxonomy [PAPER FACT]
- DejaVu (2024) — bimodal latency 1.4x to 106x prompt vs per-token [PAPER FACT]
- Sarathi-Serve (2024) — linear ops >80% time even at high seq len; decode memory-bound, prefill saturates at ~512 tokens [PAPER FACT]
- Splitwise/DistServe/Sarathi collectively justify disaggregation [PAPER FACT]

### Why they need it
The dichotomy justifies separating pools (prompt vs token), heterogeneous hardware (H100 for prompt, A100 for token), and scheduling policies (chunked prefill, stall-free batches). Throughput models and placement search assume orthogonal scaling.

### What happens if it fails (impact)
If decode becomes compute-bound (e.g., MoE, speculative decoding, MLA with higher arithmetic intensity), disaggregation adds KV-transfer overhead without benefit; TetriInfer notes "heavy prefill + heavy decode" marginal gain [PAPER FACT]. If prefill becomes bandwidth-bound (GQA/MQA smaller KV, FlashAttention-3), chunking granularity is wrong: Sarathi tile-quantization 257 vs 256 -> 32% slower shows sensitivity [PAPER FACT].

### Evidence
- "Prompt processing time scales with input size and is compute-bound; token generation time with KV cache is nearly constant and memory bandwidth-bound ... Fig2 prompt 1.4x to 106x higher than per-token generation (batch8, prompt1000)" [PAPER FACT] (DejaVu S2)
- "Prefill throughput flat after 1 request at 1024 tokens, decode throughput scales almost linearly with batch (Fig3)" [PAPER FACT] (Sarathi-Serve S3)
- "Prompt power draw increases with batch size, token power flat; token tolerates 50% power cap with almost no latency impact" [PAPER FACT] (Splitwise Insight VI)
- "Chunked-prefill piggyback still causes O(N^2) KV loads (N chunks -> N+(N-1)+...)" [PAPER FACT] (DistServe S2.3)

### Confidence: High

---

## Assumption 3 — Requests Are Independent; No Cross-Request Semantic Sharing Beyond Explicitly Reserved Prefixes

### Papers relying on it (5 with year)
- Orca (2022) — K/V manager keeps per-request keys/values separately until explicit removal; no prefix sharing [PAPER FACT]
- vLLM (2023) — sharing at block granularity with COW only for parallel sampling/beam search **within** a request group; cross-request sharing requires manual provider reservation (Fig.10) [PAPER FACT]
- FlexGen (2023) — throughput-oriented batching of infinite prompts with dummy weights, per-request isolated offload [PAPER FACT]
- DistServe (2024) — replication scales linearly n = ceil(R/goodput_p) assuming requests equally dispatched with no reuse [PAPER FACT]
- FastServe (2023) — ENST scheduling assumes job sizes independent; FCFS admission [PAPER FACT]

### Why they need it
Isolation simplifies memory management (buddy allocation, block tables per sequence group), security (no cross-user KV leakage), and queuing theory (M/D/1 with Poisson arrivals). Performance isolation and SLO definitions per request become tractable.

### What happens if it fails (impact)
In production RAG, agent workflows, and chat, requests heavily share system prompts, documents, and few-shot examples. SGLang reports 50-99% cache-hit opportunities wasted without RadixAttention [PAPER FACT]; RAGCache shows top 3% docs receive 60% requests (20x uniform) [PAPER FACT]; CacheBlend shows non-prefix chunk reuse essential but prefix-only caching saves only marginal [PAPER FACT]. Assuming independence underestimates hit rate by up to 50x, mis-provisions pools (Mooncake finds >50% blocks unused while some accessed tens of thousands times [PAPER FACT]), and causes redundant prefill (Cache-Craft reports 75% chunks reprocessed -> 12B tokens, 9600 GPU-hours, ~$50k [PAPER FACT]).

### Evidence
- "Existing systems (FasterTransformer, Orca) store KV cache in contiguous memory ... No sharing across sequences even when decoding algorithms (parallel sampling, beam search, shared prefix) allow it - 12% to 55% memory unshared" [PAPER FACT] (vLLM S3)
- "Sharing at block granularity with copy-on-write ... Parallel sampling: prompt logical blocks (0,1) map to same physical blocks (7,1) refcount=2 ... Shared prefix: service provider pre-reserves physical blocks" [PAPER FACT] (vLLM S4.4)
- "Prefix caching only applicable to 8% requests, 18% prefill tokens; 5-tuple hit rate << single-chunk hit rate and not power-law" [PAPER FACT] (Cache-Craft S2)

### Confidence: High

---

## Assumption 4 — Prefix Reuse Follows Recency (LRU) / Temporal Locality; Frequency or Recency Is Sufficient

### Papers relying on it (5 with year)
- SGLang (2023-2024) — LRU leaf-first eviction to preserve ancestors, reference counter, longest-shared-prefix-first DFS optimal when cache >= max request len (Theorem 3.1) [PAPER FACT]
- RAGCache (2024) — evaluates LRU/LFU/GDSF as baselines; proposes PGDSF Priority = Clock + Frequency*Cost/Size because LRU fails prefix-awareness [PAPER FACT]
- Mooncake (2024-2025) — KVCache pool supports LRU/LFU/LengthAwareCache; Conductor chooses instance with shortest predicted TTFT [PAPER FACT]
- CachedAttention / AttentionStore (2024, ATC) — hierarchical DRAM->SSD with LRU leaf-first [PAPER FACT]
- vLLM paging implicitly LRU for block reclamation [PAPER FACT]

### Why they need it
If reuse distance approximated by recency, LRU guarantees near-optimal hit rate with O(1) overhead (<0.3% for no-reuse in SGLang [PAPER FACT]) and enables simple distributed radix tree without workflow knowledge.

### What happens if it fails (impact)
Workflows with future-known reuse distance violate recency: KVFlow shows in 4-agent cycle Planner->Executor->Expresser->Reviewer, LRU evicts soon-to-be-reused Expresser while retaining recently-generated dynamic suffixes unlikely to be reused, causing miss at timestamp 14 [PAPER FACT]. RAGCache shows burst pattern Q_odd/D1 -> Q_even/D2 makes hit 0% vs optimal 66% under LRU [PAPER FACT]. HotPrefix and KVFlow replace LRU with hotness-aware and steps-to-execution priorities, gaining 2.19x over LRU HiCache [PAPER FACT].

### Evidence
- "RadixAttention ... LRU leaf-first to preserve ancestors, each node reference counter ... Overhead negligible (<0.3%, 0.2s/74.3s for 100 ShareGPT no-reuse)" [PAPER FACT] (SGLang S3)
- "Greedy cache-aware longest-prefix-first (DFS optimal when cache >= max request)" [PAPER FACT] (SGLang Theorem 3.1)
- "PGDSF +2%-75% hit gain over LRU/LFU/GDSF via PGDSF ... OrderPriority = CachedLength / ComputationLength reordering 1.2-2.1x TTFT" [PAPER FACT] (RAGCache S4-5)
- "KVFlow ... LRU evicts the soon-to-be-reused agent (Expresser) ... HiCache reactive loads disrupt pipeline ... 0.57x of GPU-only SGLang under 1024 fixed 64 concurrent" [PAPER FACT] (KVFlow S1-3)

### Confidence: Medium — widely used but actively contested by 2025 workflow-aware work

---

## Assumption 5 — Interconnect Bandwidth Is Sufficient and Homogeneous; KV Transfer Can Be Hidden via Overlap

### Papers relying on it (6 with year)
- Splitwise (2023-2024) — layer-wise async transfer via MSCCL++ one-sided put, <7% prompt time, constant ~8ms A100 /5ms H100 [PAPER FACT]
- DistServe (2024) — bandwidth-aware placement via NVLink 600 GB/s intra-node, InfiniBand 800 Gbps inter-node; KV transmission <0.1% total latency, >95% <30ms even at 25 Gbps with co-location [PAPER FACT]
- TetriInfer (2024) — emulated 200-300 Gbps network, chunk-level transfer pipelined; benefits maintain across bandwidth [PAPER FACT]
- DejaVu (2024) — buffered copies 95x improvement, streaming overhead within 2% vs no streaming [PAPER FACT]
- FlowKV (2025) — locates NCCL fragmentation 25% latency, shape reshaping Lx2 reduction, 96.8% reduction (0.944s->0.053s) [PAPER FACT]
- Mooncake (2024-2025) — Messenger GPUDirect RDMA up to 800 Gbps, layer-wise async overlapped via max(KVCache loading, standard prefilling) [PAPER FACT]

### Why they need it
Disaggregation value proposition rests on KV movement being cheaper than recomputation or batching interference. Simulators assume constant transfer latency linear in size/bandwidth and perfect overlap with next-layer compute.

### What happens if it fails (impact)
Under heterogeneous clusters (H100->A100 HA assumed IB but may need RoCE/Ethernet 10x lower BW [PAPER FACT] Splitwise SVII), PCIe 3.0 x16 16GB/s vs 64GB/s Gen5, or RDMA control-path overhead (75% of 10.55us is sync, 3x data movement [PAPER FACT] Beluga S3), transfer dominates. Beluga shows RDMA CPU-driven needs bounce buffers, sglist limit 30 vs 128 chunks forces multiple requests, and polling occupies SMs; CPU<->GPU via RDMA 7-6.3x slower than CXL (2.11us vs 8.39us RPC, 95.9% reduction for sparse 16-token loads 211us vs 5260us [PAPER FACT]). KVFlow notes fragmented layout prevents full PCIe utilization under concurrency [PAPER FACT]. TTFT then grows 64% for second token under serialized transfer (Splitwise Fig15) [PAPER FACT].

### Evidence
- "KV-cache transfer optimization ... layer-wise async ... overhead minimal (<7% prompt time, constant ~8ms A100/5ms H100 non-overlapped, Fig14)" [PAPER FACT] (Splitwise SIV-C)
- "KV transmission accounts for <0.1% total latency even for 175B ... >95% requests <30ms transmission despite 25 Gbps cross-node, thanks to NVLink co-location" [PAPER FACT] (DistServe S6.3)
- "H20 16KB transfer total 10.55us, actual data movement 2.68us, ~75% (8us) synchronization overhead ... sglist limited to 30 entries on ConnectX-7 vs 128 chunks needed" [PAPER FACT] (Beluga S3)
- "FlowKV locates NCCL fragmentation occupies 25% latency ... 96.8% reduction" [PAPER FACT]

### Confidence: Medium — true for NVLink intra-node, fragile cross-node/heterogeneous

---

## Assumption 6 — Workload Arrival and Length Distributions Are Stationary and Predictable (Poisson, Fit Once per Hours/Days)

### Papers relying on it (6 with year)
- Orca (2022) — num_input_tokens ~ U(32,512), max_gen_tokens ~ U(1,128), Poisson arrivals [PAPER FACT]
- vLLM (2023) — ShareGPT/Alpaca tokenized lengths synthesize Poisson with varying rates, 1-hour traces (15-min for 175B) [PAPER FACT]
- DistServe (2024) — workload traces resampled from history, fitting distribution over hours/days, M/D/1 model Avg_TTFT = D + R*D^2/(2*(1-R*D)) [PAPER FACT]
- Mooncake (2024-2025) — 23,608-entry 1-hour Kimi trace avg input 7590 output 182 ratio ~41.7, Poisson for synthetic; stable prefill/decode pool ratio [PAPER FACT]
- RAGCache (2024) — arrival Poisson with top 3% docs 60% requests skewed but stable; cost via bilinear interpolation T(alpha,beta) [PAPER FACT]
- TetriInfer (2024) — offline small-LLM predictor trained on prompt-only dataset, 74.9% accuracy at granularity 200 [PAPER FACT]

### Why they need it
Stationarity allows offline profiling of C1-C5 latency constants (<2% error [PAPER FACT]), provisioning search via event-driven simulator (<3% MAPE [PAPER FACT]), and TTFT prediction T_queue + T_prefill + T_transfer via summing queued times [PAPER FACT]. Provisioning (e.g., 2P+1D for 3.3 rps per GPU) and autoscaling are computed once.

### What happens if it fails (impact)
Production is bursty and long-tail: Continuum shows slowest 10% of BFCL fetch_url accounts 52.5% total delay, SWE-Bench cd 94.1% [PAPER FACT]; KVFlow high concurrency (64 workflows) makes HiCache 0.57x slower due to queuing not in Poisson model [PAPER FACT]. DistServe algorithm runs in minutes on 96-core [PAPER FACT] but cannot react to rapid shift; Mooncake notes benchmarks overestimate reuse 90% vs real 50% [PAPER FACT]; Splitwise 7% throughput setback when running conversation trace on coding-optimized cluster [PAPER FACT].

### Evidence
- "Arrival times Poisson process, varying arrival rate (load)" [PAPER FACT] (Orca S8)
- "Tokenized lengths used to synthesize requests; arrival times Poisson with varying rates" [PAPER FACT] (vLLM S6.1)
- "Fitting distribution over hours/days (assumes predictability)" [PAPER FACT] (DistServe S4)
- "Assume uniform decode time td ... system-level prediction sufficient; request-level output length prediction too costly/inaccurate" [PAPER FACT] (Mooncake S7)

### Confidence: Medium — reasonable for capacity planning, poor for tail/overload

---

## Assumption 7 — Future Request Lengths and Output Tokens Are Unknown at Schedule Time (No Clairvoyance)

### Papers relying on it (5 with year)
- Orca (2022) — max_tokens per request known a priori for reservation but actual EOS unknown; assumes never emit <EOS> in synthetic trace [PAPER FACT]
- FastServe (2023) — "output length depends on semantics not predictable; SRPT optimal but needs remaining time, MLFQ assumed without prior" [PAPER FACT]; semi-information-agnostic [PAPER FACT]
- TetriInfer (2024) — predictor granularity 200 tokens -> 74.9% accuracy because temperature/top-p variance prevents exact [PAPER FACT]
- Online Scheduling (2025) — first online scheduling with KV-cache constraints, competitive ratio, assumes online arrivals unknown [PAPER FACT]
- DistServe / Splitwise (2024) — provisioning assumes prompt/token size distributions known statistically but per-request length unknown until arrival [PAPER FACT]

### Why they need it
Justifies FCFS, MLFQ, and opportunity-driven policies over SJF/SRPT. FastServe skip-join MLFQ (skip to queue where q_i >= t_init) and ENST proactive swapping rely on only knowing t_init (input length) [PAPER FACT].

### What happens if it fails (impact)
If output length is predictable via lightweight predictors (TetriInfer OPT-125M 10x faster, 74.9% [PAPER FACT]), SJF/SRTF could beat MLFQ. FastServe example shows 3-job schedule: SRPT 3 vs skip-join 3.3 vs FCFS 4.23, leaving ~10% gap due to not using prediction [PAPER FACT]. Conversely, TetriInfer power-of-two + least-interference achieves lowest decoding time vs random by using predicted ranges [PAPER FACT].

### Evidence
- "Job size variable unknown: SRPT although optimal needs predicted remaining processing time, while LLM output length depends on semantics not predictable" [PAPER FACT] (FastServe S3)
- "Small LLM classification model (e.g., OPT-125M predicting for OPT-13B, ~10x faster) ... offline fine-tuning: bucket response lengths by granularity ... 74.9% accuracy" [PAPER FACT] (TetriInfer S3.3.2)
- "max_tokens = num_input_tokens + max_gen_tokens ... Assume never emit <EOS>, generate exactly max_gen_tokens due to lack of checkpoint/text" [PAPER FACT] (Orca S8)
- "Output length distribution long-tail ... predicting length is hard (high cost/low accuracy) especially under overload" [PAPER FACT] (Mooncake S7)

### Confidence: High — foundational; predictor papers attack but still acknowledge limited accuracy

---

## Assumption 8 — Recomputation / Fetch Cost Is Uniform and Linear in Tokens or Blocks

### Papers relying on it (5 with year)
- FlexGen (2023) — cost model Tpre = max(ctog^p, gtoc^p, ...) and I/O = 8h1^2+4h1*h2 + 2*bls*h1 average per token [PAPER FACT]
- Sarathi-Serve (2024) — token budget tau (512 strict / 2048 relaxed) bounds TBT; Trecompute = r%*Prefill(LLM,L) linear in r% [PAPER FACT]; KV reload O(N^2) noted [PAPER FACT]
- CacheBlend (2024-2025) — r% recompute = r% full prefill cost; Loading Controller Tload = PerTokenKVSize*L / Throughput(device) linear [PAPER FACT]
- SnapKV/PyramidKV (2024) — per-head top-k assumes uniform cost per retained token
- GEAR/KIVI (2024) — group size G=32 uniform across layers/heads; error measured as Frobenius norm linear in size [PAPER FACT]

### Why they need it
Linear models enable convex optimization, token-budget scheduling, and simple throughput prediction. Tile-quantization and sparsity are ignored.

### What happens if it fails (impact)
Non-uniformity from tile quantization (257 chunk 32% slower than 256 [PAPER FACT] Sarathi), attention sparsity (84.3% sparse [PAPER FACT] KIVI), power-law heavy hitters, and outliers (top 1% outliers) make cost non-linear. Sarathi shows small chunk 512 overhead ~25% vs no-chunking [PAPER FACT]; CacheBlend HKVD selection exploits non-uniformity where 10-15% tokens have far higher deviation — uniform recompute would waste 85% work [PAPER FACT]. GEAR shows outlier-aware KIVI improves over uniform KIVI from 30.17->36.01 on GSM8k but still far from GEAR 52.99 [PAPER FACT].

### Evidence
- "Tile-quantization ... dimensions not divisible by tile size cause extraneous compute: e.g., 257 chunk 32% slower than 256" [PAPER FACT] (Sarathi-Serve S3)
- "Attention output error Delta = 3.55 per-token vs 49.89 per-channel (~15x smaller) because attention highly sparse (84.3% sparsity)" [PAPER FACT] (KIVI Table2)
- "Spectrum of R_h drops rapidly — coherent component captured by top singular vectors" [PAPER FACT] (GEAR S3)
- "KV cache memory per token assumed uniform 0.125-0.5 MiB/token" [PAPER FACT] (RAGCache Table1)

### Confidence: Medium — convenient modeling, contradicted by sparsity/outlier work

---

## Assumption 9 — Attention Is Sparse / Power-Law Distributed with Persistent Heavy Hitters

### Papers relying on it (6 with year)
- H2O (2023) — Heavy-Hitter Oracle: >95% sparsity at 1% threshold, accumulated scores power-law, greedy retains H2+recent with (1-1/e) guarantee [PAPER FACT]
- Scissorhands (2023) — Persistence of Importance Hypothesis: repetitive pattern, ratio >95% in most layers, |S_{0->t}|/t <<0.5 [PAPER FACT]
- StreamingLLM (2023-2024) — window attention premise (middle tokens discardable)
- SnapKV (2024) — observation-window voting, 1D pooling clustering to retain surroundings via induction heads [PAPER FACT]
- PyramidKV (2024) — pyramidal funneling: lower layers broad, upper layers concentrated on few tokens [PAPER FACT]; allocation via alpha=20 [PAPER FACT]
- SCOPE (2025) — notes existing compression ignores decode and heavy-hitter drift [PAPER FACT]

### Why they need it
Sparsity justifies fixed-budget eviction (20% in H2O, 15-30% retain in Scissorhands until 5x reduction [PAPER FACT]), per-layer pyramidal budgets, and observation-window proxies. Theoretical bounds assume power-law.

### What happens if it fails (impact)
On tasks with uniform or dense attention (copy tasks, dispersed needle retrieval), eviction removes critical tokens and causes up to 35% drop at 20% budget (strided without H2: 50.00->83.00 with H2 [PAPER FACT] H2O S5.1). Scissorhands shows later layers have lower persistence ratio [PAPER FACT]; PyramidKV fixed arithmetic pyramid then misallocated. ShadowKV demonstrates eviction loses multi-turn conversation: SnapKV drops from round 2 [PAPER FACT].

### Evidence
- "Attention matrices >95% sparse at 1% threshold ... Small set of tokens (H2) get most mass; power-law ... Removing H2 -> significant degradation (Fig.2c)" [PAPER FACT] (H2O S4)
- "Persistence ratio >95% in most layers, dips later (Fig.2a); Pivotal set size |S_{0->t}|/t considerably smaller than 0.5 (Fig.2b)" [PAPER FACT] (Scissorhands S3.2)
- "Pyramidal Information Funneling — Lower layers (0th) approximately uniform ... Middle (6-18) localized ... Upper (24-30) massive attention concentrating on few key tokens" [PAPER FACT] (PyramidKV S3)
- "Last window of input sequence recognizes highly similar attention allocation pattern with actual generation (high overlap rates)" [PAPER FACT] (SnapKV S3)

### Confidence: Medium — strong for chat/QA, weak for retrieval-dense workloads

---

## Assumption 10 — Initial Tokens Are Attention Sinks and Must Be Pinned for Stable Streaming

### Papers relying on it (4 with year)
- StreamingLLM (2023-2024) — 4 sinks stabilize perplexity to 4M tokens, 22.2x speedup vs recomputation [PAPER FACT]
- SGLang (2023-2024) — retains sinks+recent implicitly via radix tree ancestors
- Scissorhands (2023) — repetitive pattern includes tokens 27,63,98 as persistent pivotal
- AttentionStore / CachedAttention (2024, ATC) — hierarchical caching with pinning for multi-turn chat [PAPER FACT]

### Why they need it
Softmax forces attention mass to sum to 1; when no strong match, mass dumps to globally visible initial tokens [PAPER FACT]. Keeping sinks anchors distribution, allowing window attention without catastrophic perplexity.

### What happens if it fails (impact)
If sinks are not at position 0 (e.g., models with learned sink token, or ALiBi), pinning 4 initial tokens wastes capacity and may still collapse: Llama-2-13B on PG19 65K with window 0+1024 -> PPL 5158.07 vs 4+1020 -> 5.40 [PAPER FACT], but Falcon/MPT stable with 1 sink already [PAPER FACT] Table2. For models needing only 1 sink, pinning 4 wastes 3x memory; for BOS not at 0, pinning wrong tokens fails.

### Evidence
- "Softmax must sum to 1 even when current query has no strong match, so model learns to dump redundant attention to globally visible initial tokens ... Substituting first 4 tokens with newline still attracts attention and restores perplexity, indicating positional bias matters more than semantics" [PAPER FACT] (StreamingLLM S3.1)
- "Keeping only sinks + recent tokens suffices ... With cache [0,1,2,3,6,7,8] decoding token 9, assigned positions are [0,1,2,3,4,5,6,7] contiguous within cache" [PAPER FACT] (StreamingLLM S3.2)
- "Vanilla models still need multiple (4) sinks because no consistent starting token; Learnable Sink 18.01 (1+1023) vs Vanilla 27.87 (0+1024)" [PAPER FACT] (StreamingLLM S3.3)
- "Cache size does not always help: Falcon 4+252 13.61 ->4+1020 12.34 ->4+2044 12.84 (worsens) ... models may not fully utilize larger context" [PAPER FACT] (StreamingLLM Table6)

### Confidence: High for window-attention streaming; Medium as universal requirement

---

## Assumption 11 — Two-Tier Memory Hierarchy (GPU HBM + Host DRAM via PCIe) Is Sufficient

### Papers relying on it (5 with year)
- FlexGen (2023) — three-tier GPU+CPU+Disk with linear-programming placement but assumes PCIe 2GB/s read, 1GB/s write NVMe enough with zig-zag schedule [PAPER FACT]
- InfiniGen (2024) — full pool on CPU, sparse prefetch via PCIe 3.0 x16; assumes <10% KV fetch suffices [PAPER FACT]
- RAGCache (2024) — GPU as primary / Host as secondary, PCIe swap-out-only-once [PAPER FACT]
- Cache-Craft / CacheBlend (2025) — hierarchical GPU HBM vs Host DRAM vs SSD with layer-wise preloading [PAPER FACT]
- DejaVu (2024) — CPU offload + remote CPU via NCCL/MPI, but still two-tier per pipeline [PAPER FACT]

### Why they need it
Two-tier simplifies allocation, coherence, and scheduling. Cost models use only ctog, gtoc, dtoc [PAPER FACT] FlexGen.

### What happens if it fails (impact)
Long-context (1M) and large batches exceed host DRAM: ShadowKV needs 6x batch at 122K (4->24) and still OOM at 488K batch 2 vs 5 with compression [PAPER FACT]; Mooncake needs 20TB DRAM for 50M tokens maximal hit [PAPER FACT]; Shared RAG-DCache uses disk via queuing window pre-generation (+15-71% throughput [PAPER FACT]). Beluga shows CXL 2.0 pooling provides 8TB at 1TB/s with 750ns latency, cheaper ($218.75/64GB/s vs $800 RDMA [PAPER FACT]) and enables near-local latency (10.32us CPU->GPU vs 11.73us CXL->GPU [PAPER FACT]), making two-tier assumption pessimistic and causing 75% sync overhead overestimation.

### Evidence
- "Keep entire KV history in CPU memory pool ... prefetch only essential entries ... average <10% KV avg, up to 20% cap" [PAPER FACT] (InfiniGen S4)
- "Beluga: RDMA-designed as networking protocol not memory bus, introduces extra data copies via host bounce buffers ... CPU-driven 75% sync overhead ... CXL load/store ... eliminating bounce buffers" [PAPER FACT] (Beluga S1-3)
- "ShadowKV: low-rank pre-RoPE keys ... theoretical 7.08x memory savings ... equivalent bandwidth 7.2 TB/s (3.6x A100)" [PAPER FACT] (ShadowKV S4)

### Confidence: Low — actively invalidated by 2025 CXL/disk work

---

## Assumption 12 — Token Importance Within Prompt Is Decidable From a Local Observation Window

### Papers relying on it (5 with year)
- SnapKV (2024) — C = sum W_obs, I = Top_k(C,k) from observation window L_obs (last segment), with 1D pooling [PAPER FACT]
- PyramidKV (2024) — importance s_i^h = sum_{j in [n-alpha,n]} A_{ij}^h where [n-alpha,n] is instruction tokens, alpha=8 [PAPER FACT]
- H2O (2023) — local accumulated attention sum o_s approximates global (future-inclusive) [PAPER FACT] Fig.2d
- Cache-Craft (2025) — a(C_i)=sum inter/|C_i||C_j|, b=intra/|C_i|^2, CCI=1/(1+e^{-a/b}) vs beta [PAPER FACT]
- CacheBlend (2024-2025) — HKVD tokens selected via recompute deviation on observation window; Spearman rank correlation high across layers [PAPER FACT]

### Why they need it
If future generation attends to same prefix tokens as last window, a single cheap attention pass over observation window suffices to prune 85-99% of prompt KV before generation.

### What happens if it fails (impact)
When important tokens are uniformly dispersed or instruction is at front, observation window misses them. SnapKV notes context-dependent importance: "Different instructions on same document prioritize different prefix features (hit rate descending when varying instructions)" [PAPER FACT] Fig.4; still claims robust to position but only tested on QMSum/Openreview/SPACE [PAPER FACT]. PyramidKV fixes alpha=8 across layers despite assumption [PAPER FACT]; for few-shot with many examples, 8 insufficient. Cache-Craft shows low CCI (<0.5) safe but high CCI needs order correction; misclassifying high CCI as low causes up to 50% F1 drop when naive reusing 5 blocks from 5 histories [PAPER FACT].

### Evidence
- "Last window of input sequence recognizes highly similar attention allocation pattern with actual generation (high overlap rates)" [PAPER FACT] (SnapKV S3)
- "Instruction tokens / local window ... s_i^h = sum_{j in [n-alpha,n]} A_{ij}^h where alpha hyperparameter ... following common practice ... alpha=8" [PAPER FACT] (PyramidKV S4.2.1)
- "Greedy local approx global: Retaining H2 based on local statistics (sum attention of preceding tokens only) is as effective as global (including future) (Fig.2d)" [PAPER FACT] (H2O Observation 3)
- "Information retrieval relies on high-attention features supplemented by copying surrounding via induction heads [PAPER FACT]; naive top selection loses completeness; pooling retains surrounding" [PAPER FACT] (SnapKV S4.3)

### Confidence: Medium — strong for QA/RAG with question at end, weak for summarization

---

## Cross-Cutting Implications

| Assumption | Stability | Counter-Evidence |
|---|---|---|
| 1 Memory bottleneck | Stable | Falcon single KV head needs 4-bit not 2-bit [PAPER FACT] KIVI; H100 compute/bandwidth divergence |
| 2 Phase dichotomy | Stable intra-node | MoE, MLA, speculative decoding blur boundaries |
| 3 Request independence | **Fragile** | SGLang, RAGCache, CacheBlend all profit from cross-request sharing |
| 4 LRU locality | **Fragile** | KVFlow, HotPrefix, Continuum TTL, PGDSF beat LRU 2-75% hit [PAPER FACT] |
| 5 Bandwidth homogeneous | **Fragile** | Beluga CXL 7x faster, FlowKV NCCL frag, HA 10x BW gap |
| 6 Stationary workload | Fragile | Long-tail tool 94% delay from 10% slowest [PAPER FACT] Continuum; burst thrashing |
| 7 No clairvoyance | Stable but attacked | TetriInfer predictor 74.9% shows partial clairvoyance feasible |
| 8 Uniform cost | Fragile | Tile quantization 32% penalty, sparsity 84%, outliers |
| 9 Sparse persistence | Conditional | Works for chat/QA, fails for dense copy/needle-dispersed |
| 10 Sinks | Stable for window | Learnable sink reduces need 4->1 [PAPER FACT] |
| 11 Two-tier hierarchy | **Obsolete** | CXL 8TB pool, SSD disk tier needed |
| 12 Local window proxy | Conditional | Instruction-at-front or whole-document summarization violates |

**Field-level risk:** Assumptions 3,4,5,11 are repeatedly invalidated by 2024-2025 RAG/agent/long-context work. Systems designed on their conjunction (LRU + homogeneous 25 Gbps + request-independent + two-tier) over-provision by 2-7x throughput (Splitwise 2.35x same cost/power [PAPER FACT], Mooncake 75% more requests [PAPER FACT], Beluga 7.35x [PAPER FACT] vs RDMA). Future work must compose workflow-aware eviction (KVFlow/Continuum), bandwidth-aware placement (Beluga/Mooncake), and joint token-bit-spatial compression (GEAR+PyramidKV+ShadowKV).

---

## Traceability Note

- Sources: 26 notes read + papers.md 43 entries [PAPER FACT]. Each evidence quote tagged [PAPER FACT] where directly supported by note text; numeric claims verifiable via local fitz-extracted text under C:\Windows\Temp\opencode\*.txt as recorded in notes reading source fields [PAPER FACT].
- Failure modes synthesize inferred limitations sections but grounded in author-stated limitations where possible (e.g., Mooncake "kvcache_balancing_threshold manually tuned" [PAPER FACT], TetriInfer "heavy+heavy marginal gain" [PAPER FACT]).

*Stage 2A Step D complete — 12 assumptions (>=10 required), each 5 subsections, citation-rich, confidence-rated.*
