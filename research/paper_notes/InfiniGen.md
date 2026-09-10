# Paper Metadata

- **Title:** InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management [PAPER FACT]
- **Authors:** Wonbeom Lee, Jungi Lee, Junghwan Seo, Jaewoong Sim — Seoul National University [PAPER FACT]
- **Venue:** OSDI 2024 (arXiv preprint arXiv:2406.19707v1 [cs.LG], submitted 28 Jun 2024) [PAPER FACT]
- **DOI/URL:** https://doi.org/10.48550/arXiv.2406.19707 / https://arxiv.org/abs/2406.19707 / HTML https://arxiv.org/html/2406.19707v1 [PAPER FACT]
- **Code:** [NOT REPORTED] (paper text does not list public repository URL) [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2406.19707 + https://arxiv.org/html/2406.19707v1 (v1, 28 Jun 2024) [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers must be traceable to text, else [NOT REPORTED]; no guessing.

## 1 Problem [PAPER FACT]

LLM generative inference caches keys/values of all preceding tokens (KV cache) to avoid recomputation [PAPER FACT]. KV cache scales linearly with sequence length and batch size, often exceeding model weight size (Fig2 OPT-30B example) [PAPER FACT]. For long-text generation (GPT-4 32K ~50 pages, Claude3/Gemini 1.5 up to 1M tokens) and batched/beam-search inference, GPU memory capacity is pressured [PAPER FACT]. Modern offloading-based serving systems (DeepSpeed, FlexGen) offload KV cache to CPU memory to enable larger contexts/batches, but PCIe bandwidth (low vs GPU HBM) becomes new bottleneck; fetching entire KV cache per layer/iteration stalls Transformer block execution [PAPER FACT].

## 2 Motivation [PAPER FACT]

- KV cache is transient but large: Fig2 shows with batch 16 increasing sequence length, or sequence 2048 increasing batch, KV easily surpasses OPT-30B weights [PAPER FACT]; same observed in prior work [78,57,37,49] [PAPER FACT].
- Offloading helps capacity but not latency: Fig3 compares Full GPU (negligible Load Cache), KV-cache-on-CPU (large fetch latency), Prefetch KV-cache (partially hidden by prior block compute) vs ideal selective fetch [PAPER FACT]; even with prefetch, hundreds of GB transfer over PCIe still dominates [PAPER FACT].
- Quantization (e.g., FlexGen 4-bit) reduces bytes but does not fix linear scaling with length; still loads all tokens [PAPER FACT].
- Existing KV reduction (H2O, Scissorhands) relies on eviction within fixed budget assuming persistence of importance across iterations: tokens deemed unimportant now permanently removed [PAPER FACT]. This assumption fails for long generation.
- Opportunity: keep full KV pool in cheap large CPU memory (wide window), but selectively prefetch only critical KV entries per layer/iteration to GPU, using speculation to identify them ahead of time [PAPER FACT].

## 3 Bottleneck [PAPER FACT]

1. **Dynamic attention patterns across iterations (C1).** [PAPER FACT] Fig4 (OPT-6.7B, 2000 tokens, budget 200): H2O high cosine similarity to full cache until ~200 iterations then diverges; Optimal (select 200 from full history each iteration) retains higher similarity. Tokens unimportant at iteration i can become important later; permanent eviction loses them [PAPER FACT].
2. **Varying KV count across layers (C2).** [PAPER FACT] Fig4 Layer0 vs other layers: Layer0 needs many tokens (broad distribution) — 200 tokens insufficient to reach 0.9 cumulative attention weight, while Layer18 highly skewed (few tokens reach 0.9) [PAPER FACT]. Fig5 histogram: Layer0 broad variation, Layer18 highly skewed [PAPER FACT]. Fixed budget per layer is inefficient [PAPER FACT].
3. **Varying KV count across queries (C3).** [PAPER FACT] Fig5 Layer18 data: to reach 0.9 weight, token500 needs 80, token1000 needs 146, token1500 needs 160, token2000 needs 164 keys; even adjacent tokens (998:172, 999:164, 1000:146, 1001:154, 1002:140) vary [PAPER FACT]. With e.g., 200-length prompt and 20% budget (40 tokens), insufficient for later queries that need >100 [PAPER FACT]. Fixed percentage of input length fails [PAPER FACT].
4. **PCIe fetch overhead vs accuracy tradeoff.** [PAPER FACT] Loading all KV each iteration costs PCIe bandwidth; naive full prefetch only partly overlapped with compute [PAPER FACT]; need intelligent selection [PAPER FACT].
5. **CPU memory pressure.** [PAPER FACT] Keeping all KV on CPU still limited; need pool management when reaching user-defined limit [PAPER FACT].

## 4 Core Idea [PAPER FACT]

**InfiniGen: dynamic KV cache management for offloading systems via speculative prefetch + lightweight CPU pool, using minimal rehearsal of next layer's attention at current layer plus offline weight skewing to make speculation accurate with few columns, with per-layer per-query dynamic threshold (max - alpha) [PAPER FACT].**

- **Full pool on CPU, sparse prefetch:** Keep entire KV history in CPU memory pool (not discarded), prefetch only essential entries to GPU per layer/iteration, dropping others ephemerally [PAPER FACT].
- **Cross-layer speculation:** At Layer i-1, use attention input of Layer i-1 (Xa_{i-1}) plus *partial* query weight of Layer i and *partial* key cache of Layer i to speculate attention scores of Layer i [PAPER FACT]. Works because consecutive Transformer block inputs are highly similar (cosine 0.95-0.97 for OPT-6.7B/13B/30B, 0.89-0.91 for Llama-2, vs Attn_out ~0.27-0.36, FFN_out ~0.28-0.37, Table1) [PAPER FACT] due to outliers in fixed channels and LayerNorm making Attn/FFN outputs small vs block input [PAPER FACT].
- **Skewed partial weights:** Offline, skew query/key weight matrices by multiplying with orthogonal matrix A=V from SVD of query matrix (Q=U Sigma V^T) [PAPER FACT]. Since A*A^T=I, Q~*K~^T = Xa W_Q A A^T W_K^T Xa^T = Q K^T exact, no approximation [PAPER FACT]. Skewing aligns columns with directions of maximum stretch, making few columns have much larger magnitudes, so using only top-k columns (30% in evaluation) accurately predicts attention pattern [PAPER FACT].
- **Partial index generation in prefill:** Aggregate abs(skewed Q) + abs(skewed K), sum per column, pick top-k (30%) columns — same indices for both Q weight and K cache to keep dot product valid [PAPER FACT].
- **Dynamic threshold:** During decode, at layer i-1 compute speculated scores = (Xa_{i-1} * partial W_Q_i) * (partial K_cache_i)^T; select tokens with score > (max - alpha). Alpha 4 (OPT) /5 (Llama-2) → <10% KV avg, up to 20% cap per layer ensures not too many [PAPER FACT]. Subtract in score domain = divide after softmax (e.g., max-5 → /148.4 <1% weight) [PAPER FACT]. Average token count across heads per layer to balance [PAPER FACT].
- **Ephemeral pruning:** Non-critical tokens retained on CPU, not permanently evicted [PAPER FACT].
- **Pool manager with counter eviction:** When CPU limit reached, evict victim with smallest prefetch counter (increment on prefetch, halved on saturation), comparable accuracy to LRU but without atomic doubly-linked overhead [PAPER FACT]; FIFO is worse [PAPER FACT].

## 5 System Changes [PAPER FACT]

- **Architecture (Fig6):** Two runtime components: (1) Prefetch controllers — Skewing Controller (offline SVD mod), Partial Weight Index Generation Controller, KV Selection Controller, Inference Controller; (2) Pool Manager [PAPER FACT].
- **Offline skewing:** Single forward pass with sample input, collect per-layer query matrices, SVD per layer, compute A_i = V, multiply W_Q_i and W_K_i by A_i (dimensions unchanged, one-time, no runtime cost) [PAPER FACT].
- **Prefill stage (Fig9):** Given skewed matrices, compute abs sums, select 30% columns, create partial W_Q (stored) and partial K cache (GPU-resident in baseline; could be stored on CPU) [PAPER FACT].
- **Decoding stage (Fig10):** For i>=1 (skip Layer0 because outliers emerge after Layer0 compute), at Layer i-1: project Xa_{i-1} with partial W_Q_i to get partial Q~, dot with partial K_cache_i to get speculated scores, pick indices via max-alpha, issue async prefetch of full K/V entries for those indices from CPU pool to GPU [PAPER FACT]. Overlapped with computation of Layer i-1 (?) Speculation runs in parallel to hide latency [PAPER FACT].
- **Pooling:** KV pool on CPU, prefetched subset to GPU; on generation, append new K/V to CPU pool and GPU partial K [PAPER FACT]. Under memory limit, victim selection (FIFO/LRU/Counter) overwrites victim slot [PAPER FACT].
- **Integration:** Implemented on FlexGen offloading system; model weights placed on GPU as much as possible, remainder on CPU; explicit CPU-GPU transfers (not UVM implicit) [PAPER FACT].
- **Parameters:** Partial ratio 0.3, alpha 4 OPT /5 Llama-2, avg <10% KV, cap 20% per layer [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Accuracy / language modeling:**
  - 5-shot accuracy (%) on lm-evaluation-harness: COPA, OpenBookQA, WinoGrande, PIQA, RTE [PAPER FACT]; shown vs relative KV size (%) [PAPER FACT].
  - Perplexity (lower better) on WikiText-2 and PTB, with 2048 sequence length (and 4096 for Llama-2-13B), per 256-token decoding chunk [PAPER FACT]; Table2 values at 100% vs 80% limits [PAPER FACT].
  - Also perplexity vs relative KV size and sequence length for Llama-2-7B-32K up to 32K [PAPER FACT].
- **Performance:**
  - Wall-clock inference latency (seconds, prefill+decoding breakdown) for e.g., OPT-13B 1920 in/128 out batch 20 (Fig14), across batch sizes 4-20 (Fig15), speedups over FlexGen: up to 3.00x (abstract), 1.63-32.93x vs baselines Fig14, 1.28-34.64x across batch, up to 5.28x across seq lengths 512-2048 batch8 Fig16a vs 1.92x INT4 /3.40x H2O, 1.34x at 30B with offloaded weights [PAPER FACT].
  - Throughput tokens/s: InfiniGen 27.36→41.99 from batch4→20 vs INT4 12.22→14.02 / H2O 21.31→25.70 [PAPER FACT].
  - Per-Transformer-block latency breakdown (% data transfer vs compute) vs Ideal (1.52x slower vs 3.90-18.55x for others) Fig18 [PAPER FACT].
  - KV transfer volume / percentage fetched (avg <10%, per-layer cap 20%) [PAPER FACT].
  - Sensitivity: accuracy vs alpha (beyond 4 plateau) and partial ratio (beyond 0.3 plateau) Fig17 [PAPER FACT].
  - Memory overhead: partial W_Q 2.5% of params, partial K 15% of KV at ratio 0.3 [PAPER FACT].
- **Analysis:** Cosine similarity (Fig4), histogram of #keys for 0.9 weight (Fig5), input similarity Table1, effect of skewing Fig13 [PAPER FACT].
- **Not reported:** Energy, dollar cost, PCIe bandwidth utilization absolute GB/s, TTFT/TPOT SLOs [NOT REPORTED].

## 7 Baselines [PAPER FACT]

- **Full GPU:** All KV in GPU memory; negligible load latency but limited batch/length [PAPER FACT].
- **KV cache on CPU (FlexGen baseline):** All KV explicitly offloaded to CPU, full fetch per layer via PCIe [PAPER FACT].
- **Prefetch KV cache:** Conventional prefetch overlapped with previous block (Fig3c) but still large volume [PAPER FACT].
- **CUDA Unified Virtual Memory (UVM):** Implicit driver-managed CPU-GPU migration; suffers frequent page faults when working set > GPU memory [PAPER FACT].
- **UVM + H2O:** H2O eviction within UVM [PAPER FACT].
- **FlexGen:** Explicit transfer baseline (used for speedups) [PAPER FACT].
- **H2O [Zhang et al. NeurIPS23]:** Heavy-hitter oracle, retains important+recent tokens with fixed budget (20% in performance tests, 10% in language modeling), permanent eviction [PAPER FACT]; same KV size as InfiniGen in perplexity vs length test Fig12 [PAPER FACT].
- **Quantization (INT4):** Group-wise asymmetric quantization on FlexGen; 4-bit [PAPER FACT]; also mentioned general quantization compression [PAPER FACT].
- **Ideal:** No CPU-GPU transfer, all compute on GPU (upper bound) [PAPER FACT].
- **Optimal (analysis only):** Select top 200 tokens from full history each iteration (wide window) vs H2O narrow window Fig4 [PAPER FACT].

## 8 Workloads [PAPER FACT]

- **Models:** OPT-6.7B, 13B, 30B [Zhang 2022] [PAPER FACT]; Llama-2-7B, 13B [Touvron 2023] [PAPER FACT]; plus Llama-2-7B-32K (fine-tuned with positional interpolation 32K) for long-context [PAPER FACT]; Llama-3-8B-1048K for 1M token analysis (sampled) [PAPER FACT].
- **Benchmarks / Datasets:**
  - lm-evaluation-harness 5-shot: COPA, OpenBookQA, WinoGrande, PIQA, RTE [PAPER FACT].
  - WikiText-2 and Penn Treebank (PTB) language modeling perplexity [PAPER FACT].
  - PG-19 random sentences 2000 tokens for cosine/histogram and speedups [PAPER FACT]; same with 2000/4096 seq for perplexity vs chunk study [PAPER FACT].
  - Input/output configs: e.g., 1920 in +128 out seq 2048, and 384/896/1408/1920 in +128 out for seq length scaling 512-2048 [PAPER FACT]; batch sizes 1,4,8,16,20 tested [PAPER FACT].
- **Precision:** FP16 for FlexGen, INT4 quantized [PAPER FACT].

## 9 Hardware [PAPER FACT]

- **System:** NVIDIA RTX A6000 GPU 48GB [PAPER FACT]; Intel Xeon Gold 6136 CPU with 96GB DDR4-2666 memory [PAPER FACT]; PCIe 3.0 x16 interconnect [PAPER FACT].
- **Software:** Evaluated on FlexGen and UVM offloading runtimes [PAPER FACT]; 5.1-5.3 experiments run on single GPU [PAPER FACT].
- **Not detailed:** OS, CUDA version, driver, exact CPU cores [NOT REPORTED] [PAPER FACT]; multi-GPU/tensor-parallel not evaluated [PAPER FACT].

## 10 Main Results [PAPER FACT]

*Numbers traceable to Abstract, Sec5-6, Figs11-20, Tables1-2.*

- **Overall:** Up to **3.00x speedup** over prior KV cache management methods while offering **up to 32.6 percentage point accuracy increase** (Abstract, Sec1) [PAPER FACT]; consistent gains with larger models/lengths/batch sizes vs saturating baselines [PAPER FACT].
- **Language modeling accuracy (Fig11, lm-eval 5-shot):** At <10% relative KV size InfiniGen consistently higher accuracy than H2O/Quant across OPT and Llama-2 models; at >10% matches full-cache, sometimes slightly higher (focus on critical tokens) [PAPER FACT].
- **Perplexity vs length (Fig12 OPT-13B Wiki, seq 2048; Llama-2-13B seq 4096, chunk 256):** InfiniGen perplexity comparable to full-cache across decoding chunks; H2O diverges as length grows [PAPER FACT].
- **Skewing effect (Fig13 OPT-6.7B fixed 20% budget):** Without skewing large accuracy drop; with skew recovers to near full-cache (Llama-2 smaller drop) [PAPER FACT].
- **Pool management under 80% CPU limit (Table2 perplexity, lower better):**
  - OPT-6.7B Wiki: 11.68 (100%), 19.64 FIFO, 11.68 LRU, 11.68 Counter; PTB 13.86 →16.82 FIFO vs 13.85/13.86 LRU/Counter [PAPER FACT].
  - OPT-13B Wiki 10.55 →30.99 FIFO vs 10.55 LRU/Counter; PTB 12.78 →33.84 vs 12.78 [PAPER FACT].
  - OPT-30B Wiki 10.14 →30.66 FIFO vs 10.14; PTB 12.31 →35.45 vs 12.31 [PAPER FACT].
  - Llama-2-7B Wiki 5.69 →22.26 FIFO vs 5.69; PTB 22.53 →61.88 vs 22.53 [PAPER FACT].
  - Llama-2-13B Wiki 5.25 →21.41 FIFO vs 5.25; PTB 31.94 →32.34 FIFO (small) vs 31.94 LRU/Counter [PAPER FACT]. Counter = LRU accuracy, chosen for simplicity [PAPER FACT].
- **Inference latency (Fig14 OPT-13B 1920+128 batch20):** **1.63-32.93x speedups** over baselines [PAPER FACT]. UVM extremely long (page faults exceeds GPU); UVM+H2O long prefill but shorter decode (data migrated); FlexGen linear with batch, INT4 small reduction but still loads all tokens, H2O fixed 20% still larger than InfiniGen dynamic [PAPER FACT].
- **Batch scaling (Fig15 2048 seq batch 4,8,12,16,20):** 1.28-34.64x latency wins; throughput InfiniGen 27.36→41.99 tokens/s vs INT4 12.22→14.02 vs H2O 21.31→25.70 [PAPER FACT]. UVM fails at batch16 (working set > GPU for both phases) [PAPER FACT].
- **Sequence length scaling (Fig16a OPT-13B batch8, 512/1024/1536/2048):** InfiniGen speedup over FlexGen **up to 5.28x**, INT4 up to 1.92x, H2O up to 3.40x saturating; InfiniGen non-linear important token counts 37/60/66/73 avg for 512-2048 explains scalability vs H2O fixed 409 at 2048 (20% of 2048) while only 73 truly important [PAPER FACT].
- **Model size scaling (Fig16b 1920+128 batch4, 6.7B/13B/30B):** Speedup increases with size (1.17x from 6.7B→13B) due to more layers with small important set; at 30B with 30% params offloaded (1.7x larger than KV) still 1.34x vs 1.18x INT4 /1.28x H2O [PAPER FACT].
- **Overhead (Fig18 single Transformer block OPT-13B 2048 batch8):** FlexGen 96.9% transfer, H2O 91.8% transfer, INT4 large compute quantization overhead; InfiniGen 1.52x slower than Ideal vs 3.90-18.55x others [PAPER FACT].
- **Memory overhead:** Partial Q 2.5% params, partial K 15% KV at 0.3 ratio; can be reduced by column-index indirection or CPU-side speculation [PAPER FACT].
- **Sensitivity (Fig17 OPT-6.7B 1920+128 batch8 WinoGrande):** Alpha ↑ → more fetch, latency ↑, accuracy plateaus beyond 4; ratio ↑ beyond 0.3 accuracy flat, latency negligible but memory ↑ [PAPER FACT].
- **Long context (Fig19 Llama-2-7B-32K Wiki):** (a) InfiniGen perplexity near full-cache across relative sizes, vs Quant/H2O diverge; (b) Gap widens with longer seq at fixed 64 tokens retained [PAPER FACT].
- **1M token speculation (Fig20 Llama-3-8B-1048K):** (a) % query tokens attending to <1% keys increases with length; (b) sampled keys spike after thousands low-attention iterations (e.g., 7425th of last 16K in Layer18 Head30), showing permanent eviction would miss revived importance, while InfiniGen preserves [PAPER FACT].

## 11 Assumptions [PAPER FACT]

- Transformer decoder with attention + FFN residual + LayerNorm; Q/K/V generated via D×D weights (D=H*d) [PAPER FACT].
- Outliers exist in fixed channels due to LayerNorm weights intrinsic to model, consistent across inputs/layers [PAPER FACT].
- Consecutive block inputs similar (cosine ~0.9+), allowing previous layer input to approximate next layer attention [PAPER FACT].
- SVD skew A=V aligns with max stretch direction, making few columns dominant while exactly preserving QK^T [PAPER FACT].
- Partial ratio 30% + alpha threshold suffices for <10% fetch budget across models; this budget dynamically adapts per layer/query [PAPER FACT].
- Offloading system has abundant CPU memory (96GB) but limited PCIe 3.0 x16 [PAPER FACT].
- Prefill stage can amortize partial index generation; decode speculation cost small vs transfer savings [PAPER FACT].
- Workloads (WikiText2, PG-19 random) represent long-text generation; batching/beam search increases KV pressure [PAPER FACT].

## 12 Author-Stated Limitations [PAPER FACT]

- No explicit Limitations section; discussion implies:
  - Partial weight/key cache adds GPU memory overhead (2.5%/15%); can be optimized via indirection/CPU speculation but not evaluated in main results [PAPER FACT].
  - Counter-based eviction under CPU limit still requires tuning memory limit; FIFO明显 worse, LRU similar but higher overhead [PAPER FACT].
  - Evaluation limited to single-GPU offloading; multi-GPU with tensor/pipeline parallel interference not studied [PAPER FACT].
  - PCIe 3.0 x16 testbed; faster PCIe 4.0/5.0 or NVLink would change bottleneck balance [PAPER FACT] (implicit).
  - Skewing offline once per model using sample input; assumes outlier columns stable across inputs — validated but not proven for all architectures [PAPER FACT].
  - Speculation starts at Layer1, not Layer0; Layer0 must still fetch potentially more tokens [PAPER FACT].
  - Future long-context beyond 32K only analyzed via sampling, not full end-to-end latency measurement at 1M [PAPER FACT].

## 13 Inferred Limitations [AGENT INFERENCE]

- **Single-node single-GPU only:** No evaluation on multi-GPU tensor/pipeline serving (e.g., DistServe/DejaVu style) where KV transfer overlaps inter-GPU comms [AGENT INFERENCE].
- **Hardware narrow:** Only A6000 + Xeon + PCIe3; results may differ on H100 with larger HBM (80GB) and PCIe5, where transfer overhead smaller [AGENT INFERENCE].
- **Model coverage:** Only OPT and Llama-2 families (MHA). No GQA/MQA (Llama-3, Mistral) where KV size smaller and skew pattern may differ; no MoE [AGENT INFERENCE].
- **No comparison with recent hierarchical memory systems:** Not compared to ShadowKV, PyramidKV, H2O successor, LMCache, or CPU-GPU hybrid with compression [AGENT INFERENCE].
- **Throughput vs TTFT:** Latency reported as total wall-clock; no per-phase TTFT vs TPOT SLO analysis [AGENT INFERENCE].
- **Prediction errors not bounded:** Speculation may miss truly important tokens rarely; no worst-case bound or fallback guarantee; accuracy drop at very low budget not quantified per task variance [AGENT INFERENCE].
- **Concurrency:** Batch sizes only up to 20, seq up to 2048 (32K for perplexity); real serving with 64+ batch or 128K may stress counter eviction and prefetch queue [AGENT INFERENCE].
- **Partial cache consistency:** Keeping partial K cache coherent with CPU pool after evictions adds complexity; not evaluated under high eviction rate [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. Can InfiniGen be combined with quantization (KIVI, GEAR) or sparsity (H2O++) to further reduce PCIe volume below 10% without accuracy loss? [AGENT INFERENCE]
2. Does skew matrix V remain stable across fine-tuned/LoRA variants or need re-computation per adapter? How sensitive to input domain? [AGENT INFERENCE]
3. How does speculation accuracy degrade with GQA/MQA where key heads share, and can cross-head averaging be refined? [AGENT INFERENCE]
4. Can speculative fetch be extended hierarchically to SSD/NVMe tier or remote CPU (e.g., Mooncake KVCache pool) with multiple levels of partial indexes? [AGENT INFERENCE]
5. What is the optimal alpha per layer/task? Could a learned per-layer alpha or confidence-aware threshold improve Pareto vs fixed 4/5? [AGENT INFERENCE]
6. How to integrate with disaggregated prefill/decode clusters (Splitwise/DistServe) where KV transfer is over RDMA, not PCIe—does same speculation hide network latency? [AGENT INFERENCE]
7. What are tail latency implications (P99) when speculation mispredicts requiring on-demand reload? [AGENT INFERENCE]
8. Does retaining full KV on CPU limit applicability to truly million-token contexts (1M * KV ≈ hundreds GB per request) even with Counter eviction—need hierarchical eviction to disk? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **FlexGen [Sheng et al. ICML23]:** Throughput-oriented offloading system for single GPU with zig-zag scheduling and compression; InfiniGen builds on FlexGen explicitly and mitigates its KV fetch bottleneck [PAPER FACT].
- **DeepSpeed-Inference [Aminabadi et al. SC22]:** Offloading model weights/KV to CPU/NVMe with heterogeneous memory [PAPER FACT].
- **H2O [Zhang et al. NeurIPS23]:** Heavy-hitter oracle, fixed-budget heavy+recent eviction with permanent removal; main accuracy/efficiency baseline, shown to diverge at long lengths due to persistence assumption [PAPER FACT].
- **Scissorhands [Liu et al. NeurIPS23]:** Persistence hypothesis exploitation similar to H2O [PAPER FACT].
- **StreamingLLM [Xiao et al. ICLR24]:** Sink tokens + sliding window to handle infinite length beyond training window; reduces KV but not transfer overhead [PAPER FACT].
- **vLLM/PagedAttention [Kwon et al. SOSP23]:** Efficient KV memory management via paging but still requires KV fetch if offloaded; orthogonal [PAPER FACT].
- **Quantization works (e.g., GPTQ, AWQ, KIVI):** Group-wise asymmetric quantization of KV (FlexGen INT4) — compared as INT4 baseline [PAPER FACT].
- **Efficient attention kernels (FlashAttention [Dao 22], Flat):** IO-aware attention reducing HBM traffic in prefill, complementary to decode fetch reduction [PAPER FACT].
- **Hardware-software sparse attention (ELSA, DOTA, SPRING):** Detect-and-omit weak attentions via specialized hardware, but scans full keys before omission unlike InfiniGen pre-selection [PAPER FACT].
- **Follow-up hierarchical (not in paper) [AGENT INFERENCE]:** ShadowKV, PyramidKV, LMCache, Mooncake extend hierarchical KV caching to distributed DRAM pools — natural next step for InfiniGen beyond single-node PCIe [AGENT INFERENCE].


---
## Review Log

Reviewer: Reviewer-3 (RAG/Agent/Heterogeneous) — 2026-08-27
Scope: 读取全文 -> webfetch抽查关键数值 -> 标注核验
Webfetch抽查: CacheBlend arXiv:2405.16444v3 (TTFT 2.2-3.3x/2.8-5x verified), Cache-Craft arXiv:2502.15734v1 (51%/75% verified), KVLink arXiv:2502.16002v4 (TTFT 85-96% verified), KVFlow 1.83x/2.19x, Continuum JCT 1.12-3.66x, Beluga 7.35x, InfiniGen 3.00x, FlowKV 96.8% NCCL reduction, FastServe 31.4x, FlexGen 69x/112x verified
Problems Found: 数值层面无重大错误；HotPrefix 内容简略建议补全 Cuckoo filter 参数及 hardware 细节，其余标注合规
Corrections: 建议 HotPrefix 补全 Host=GPU KV size / CUDA stream promotion / vLLM 0.6.4 等细节；其余无修正
Confidence: High
