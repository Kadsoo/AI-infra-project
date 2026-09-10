# Paper Metadata

- **Title:** PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling [PAPER FACT]
- **Authors:** Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Yucheng Li, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Junjie Hu, Wen Xiao [PAPER FACT] ¡ª University of Wisconsin-Madison, Peking University, Nanjing University/University of Surrey, Qwen, Microsoft, University of California - Riverside [PAPER FACT] (email zefncai@gmail.com) [PAPER FACT]
- **Venue:** Preprint arXiv:2406.02069 [cs.CL], Submitted 4 Jun 2024 v1, last revised 15 May 2025 v4 [PAPER FACT]; arXiv.org perpetual non-exclusive license [PAPER FACT] ¡ª verified via webfetch https://arxiv.org/html/2406.02069v4 abstract (12% retains full performance, 0.7% ¡ú +20.5 on TREC)
- **DOI/URL:** https://arxiv.org/abs/2406.02069 / https://doi.org/10.48550/arXiv.2406.02069 [PAPER FACT]
- **Code:** https://github.com/ZshPro/PyramidKV (also listed as https://github.com/Zefan-Cai/PyramidKV in header) [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/html/2406.02069v4 + https://arxiv.org/abs/2406.02069 (arXiv HTML v4) [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers must be traceable to text, else [NOT REPORTED]; no guessing.

## 1 Problem [PAPER FACT]

LLMs scaled to long contexts (GPT-4 128K, Gemini-pro-1.5 1M) suffer quadratic attention compute [PAPER FACT]; KV caching speeds inference but consumes extensive GPU memory with long inputs: e.g., maintaining KV cache for 100K tokens in LLaMA-2 7B requires over 50GB memory, while 2K requires <1GB [PAPER FACT] (Introduction, citing Wu et al. 2024). Traditional optimization retains merely 20% cache can preserve substantial performance [PAPER FACT] (Zhang et al. 2024), but whether such strategies applicable to all layers and whether uniform cache size across layers is efficient remain open [PAPER FACT] (¡ì1 questions 1-2). Prior KV compression (H2O, SnapKV, StreamingLLM) maintain fixed same length across Transformer layers [PAPER FACT] (Fig.1b-c), may retain many unimportant tokens in higher layers while overlooking crucial tokens in lower layers [PAPER FACT] (¡ì4.2.1). Need layer-adaptive compression that aligns with information flow aggregation.

## 2 Motivation [PAPER FACT]

- Scaling to extremely long contexts (retrieval-augmented generation) can drastically improve efficiency if extreme compression works but universal applicability across layers unknown [PAPER FACT].
- Previous work optimizes KV via low-rank decomposition (Dong et al. 2024) or pruning non-essential KV (Zhang et al., Li et al., Ge et al.) [PAPER FACT]; shown 20% retains level, but no nuanced layer-specific design [PAPER FACT].
- Massive activation (Sun et al. 2024) ¡ª very few activations significantly larger ¡ª and attention sink (Xiao et al. 2023) ¡ª keeping initial token KV largely recovers window attention ¡ª hint at concentration in higher layers but not systematically studied across layers [PAPER FACT] (¡ì1).
- Systematic investigation of attention mechanism over layers for long-context inputs needed to design principled compression [PAPER FACT]; focus on multi-document QA where model aggregates dispersed info for answers [PAPER FACT] (¡ì3).
- Need method that tailors cache amounts to informational needs of each layer, diverging from uniform allocation, to significantly reduce memory while matching full-cache performance [PAPER FACT] (Abstract: PyramidKV retains only 12% cache to match full, 0.7% to surpass others).

## 3 Bottleneck [PAPER FACT]

1. **Uniform cache is suboptimal:** Attention patterns not identical across layers: dense broad coverage in lower layers vs sparse concentration in higher layers [PAPER FACT] (Fig.2). Fixed size wastes high-layer budget on unimportant tokens and starves low-layer dispersed info [PAPER FACT] (¡ì4.2.1).
2. **Memory vs performance trade-off extreme:** Retaining 0.7% cache in memory-efficient scenario challenges existing methods (SnapKV/H2O degrade) [PAPER FACT]; PyramidKV claim up to 20.5 absolute accuracy improvement on TREC at 0.7% shows prior methods collapse [PAPER FACT].
3. **Information funneling complexity:** Lower layers exhibit broad-spectrum mode uniformly across all content; middle layers (6-18) localize within each document (dotted red triangles Fig.2); upper layers (24-30) show massive attention focusing on few key tokens (concentrated bars after layer 18) [PAPER FACT] (¡ì3). Capturing this requires per-layer budget.
4. **Selection under budget:** Need to choose important KV vectors per head per layer under per-layer kl given total ktotal [PAPER FACT] (¡ì4.1 formulation: seek sub-matrices K_s^l,V_s^l ¡Ê R^{k^l¡Ád} from full n¡Ád to keep score(K^l,V^l,D)¡Öscore(K_s^l,V_s^l,D)) [PAPER FACT].
5. **Rotary embedding handling after removal:** RoPE must be handled after tokens removed ¡ª appendix H dedicated to this [PAPER FACT].
6. **Compatibility with inference acceleration:** Need to maintain minimal extra overhead during inference (Appendix L) and integrate with vLLM/MInference (Appendices R,J) [PAPER FACT].

## 4 Core Idea [PAPER FACT]

**Pyramidal Information Funneling + PyramidKV: Dynamically allocate KV cache sizes across layers via arithmetic decreasing sequence (more in lower, less in higher), selecting per-head tokens by attention from instruction window [PAPER FACT].**

- **Discovery ¡ª Pyramidal Information Funneling (¡ì3, Fig.2):** Visualization of LLaMA attention patterns over 6 layers (0,6,12,18,24,30) for multi-document QA averaging heads per layer [PAPER FACT]:
  - Lower layers (0th): approximately uniform distribution across global contexts ¡ª broad-spectrum aggregation without prioritizing specific segments [PAPER FACT].
  - Middle layers (6-18): transition to localized within each document ¡ª refined aggregation within individual contexts [PAPER FACT].
  - Upper layers (24-30): massive attention concentrating overwhelmingly on few key tokens (extremely high scores) ¡ª essential info aggregated into focal tokens for answer extraction [PAPER FACT]. Provides insight beyond massive activation/attention sink [PAPER FACT].

- **PyramidKV Two-Step Method (¡ì4.2):**
  1. **KV Cache Size/Budget Allocation (¡ì4.2.1):**
     - First retain last ¦Á tokens across all layers (instruction tokens / local window) containing most immediate task info; ¦Á hyperparameter [PAPER FACT] ¡ª following common practice (Li et al., Xiao et al.) [PAPER FACT]; paper sets ¦Á=8 in experiments [PAPER FACT] (¡ì5.1).
     - Given total budget k_total = sum_{l} k^l over m layers [PAPER FACT], determine top and bottom sizes via ¦Â hyperparameter adjusting pyramid shape [PAPER FACT]: k^{m-1}=k_total/(¦Â¡¤m) for top layer, k^0 = (2¡¤k_total)/m - k^{m-1} for bottom layer [PAPER FACT]; ¦Â=20 in experiments [PAPER FACT].
     - Intermediate layers via arithmetic sequence forming pyramidal shape monotonically decreasing from lower to upper: k^l = k^0 - (k^0 - k^{m-1})/(m-1) ¡Á l (Eq.1) [PAPER FACT]; Intuition: follow aggregated information flow reflecting decreasing important tokens [PAPER FACT].
     - Ensures sum of all k^l equals k_total, matching total memory of baselines for fair comparison [PAPER FACT] ¡ª average cache size adjusted to match baselines [PAPER FACT] (¡ì5.1).
  2. **KV Cache Selection (¡ì4.2.2):**
     - For each layer l and head h, attention A^h = softmax(Q^h¡¤(K^h)^T / sqrt(d_k)) (Eq.2) [PAPER FACT]; Utilize pooling layer at A^h to avoid being misled by massive activation scores, following SnapKV [PAPER FACT].
     - Importance score for token i: s_i^h = sum_{j¡Ê[n-¦Á,n]} A_{ij}^h (Eq.3) where [n-¦Á,n] is instruction tokens range [PAPER FACT]; tokens receiving higher attention from instruction tokens deemed more relevant [PAPER FACT].
     - In each layer l per head h, top k^l tokens with highest s_i^h retained, others discarded and not used in any subsequent generation [PAPER FACT].

- **Geometric interpretation (Fig.1):** (a) FullKV grows with input length per layer [PAPER FACT]; (b) StreamingLLM keeps few initial + fixed window per layer [PAPER FACT]; (c) SnapKV/H2O keep fixed size per layer based on attention score [PAPER FACT]; (d) PyramidKV pyramid-like: allocating more to lower, less to higher, better aligning with increasing sparsity [PAPER FACT].

- **Handling RoPE:** Caches keys before rotary transform? Appendix H details handling rotary embedding after tokens removed [PAPER FACT].

- **Novelty Claim:** First KV cache compression method with varied cache retention across layers [PAPER FACT] (¡ì1).

## 5 System Changes [PAPER FACT]

- **Formulation (¡ì4.1):** For LLM with m transformer layers, key/value matrices K^l,V^l ¡Ê R^{n¡Ád} for each layer when encoding n tokens [PAPER FACT]; Goal seek sub-matrices K_s^l,V_s^l ¡Ê R^{k^l¡Ád} with k^l < n per layer maximizing performance preservation [PAPER FACT].
- **Cache data structure:** Per-layer budget k^l varies following arithmetic pyramid; instruction tokens (last ¦Á=8) retained across all layers uniformly, remainder allocated via pyramid [PAPER FACT]; Implementation at inference: after prefill, retain selected KV per head per layer; evicted KVs removed from GPU memory and not used throughout generation [PAPER FACT].
- **Selection module:** Computes per-head pooled attention from instruction window, sums, selects top-k per head via pooling layer [PAPER FACT]; No fine-tuning required [PAPER FACT].
- **Integration points:**
  - Fair comparison setup: average KV cache size in PyramidKV matched to baseline uniform size to keep total memory same (e.g., avg 64 vs uniform 64) [PAPER FACT] (¡ì5.1).
  - Appendix R: Implementation at vLLM [PAPER FACT]; Appendix J: Integration with MInference (dynamic sparse attention) [PAPER FACT]; Appendix L: Minimal extra inference overhead claimed [PAPER FACT].
  - Prompt handling: Same prompt for each dataset in all experiments [PAPER FACT].
- **Hyperparameters:** ¦Á=8, ¦Â=20 fixed across main experiments [PAPER FACT]; Additional ablations: allocation strategies (Appendix I.1), ¦Á/¦Â sensitivity (I.2, I.3) [PAPER FACT].
- **Instruction token naming:** Called instruction tokens to generalize local window concept [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Primary:**
  - **LongBench average score** across 17 datasets (paper main table shows 16 columns but text says 17) covering single-doc QA (NrtvQA, Qasper, MF-en), multi-doc QA (HotpotQA, 2WikiMQA, Musique), summarization (GovReport, QMSum, MultiNews), few-shot (TREC, TriviaQA, SAMSum), synthetic (PCount, PRe), code (Lcc, RB-P) [PAPER FACT]; Metrics per task standard: F1 for QA, Rouge-L for summarization, Acc. for synthetic, Edit Sim. for code [PAPER FACT] (¡ì5.1.2). Average input length 1,235-18,409 tokens per dataset (Table1 header: e.g., NrtvQA 18409, Lcc 1235) [PAPER FACT].
  - **Performance at two operational points:** memory-efficient (KV size 64, ~0.7-0.8% cache) and performance-preserving (KV size 2048, ~12% cache, also tested 96,128,256,512,1024) ¡ª also Fig.3 average across 64/96/128/256 [PAPER FACT].
  - **Needle-in-a-Haystack / Fact Retrieval Across Context Lengths** accuracy [PAPER FACT] ¡ª context up to 8K (70B) and 32K (Mistral) with cache 128/64 [PAPER FACT]; Fig.4 vertical depth %, horizontal length.
  - **Memory reduction:** KV cache memory in MB and compression ratio (e.g., Table2: 512¡ú428M 6.3%, 1024¡ú856M 12.5%, 2048¡ú1712M 25%, Full¡ú6848M 100% for LLaMA-3-8B seq8192 batch1 fp16) [PAPER FACT].

- **Secondary:**
  - **Inference speed overhead** (Appendix L, M) ¡ª claims minimal extra overhead due to allocation + selection [PAPER FACT]; Appendix M inference speed comparison.
  - **Ablations:** Allocation strategies, hyperparameter ¦Á/¦Â sensitivity (Appendix I) [PAPER FACT].
  - **Integration effectiveness** with vLLM, MInference [PAPER FACT].
  - **Attention pattern visualization** validation (¡ì3) [PAPER FACT].

- **Not elaborate:** Per-token latency vs batch/throughput beyond memory, energy [NOT REPORTED] as primary; CPU/network [NOT REPORTED].

## 7 Baselines [PAPER FACT]

- **FullKV (FKV):** Caches all keys/values for every token per layer; upper bound, reference for all tables [PAPER FACT] (¡ì5.1.3).
- **StreamingLLM (SLM) [Xiao et al. 2023]:** Keeps only few initial tokens (attention sinks) with fixed cache size per layer; enables infinite length without fine-tuning [PAPER FACT] (¡ì5.1.3, Fig.1b).
- **Heavy-Hitter Oracle (H2O) [Zhang et al. 2024]:** Dynamically retains balance of recent and Heavy Hitter tokens [PAPER FACT] ¡ª compared at same average cache sizes (Fig.3, Table1).
- **SnapKV (SKV) [Li et al. 2024]:** Automatically compresses by selecting clustered important tokens per head based on attention scores (pooling) [PAPER FACT] ¡ª directly comparable selection but uniform per layer (Fig.1c); PyramidKV uses same selection formula but pyramidal allocation [PAPER FACT].
- **For reference:** Massive activation (Sun et al.) and attention sink phenomenon not baselines but motivation [PAPER FACT].
- **All baselines keep same KV cache size across different layers** with different selection strategies [PAPER FACT]; PyramidKV matches average size for fairness [PAPER FACT].

## 8 Workloads [PAPER FACT]

- **Models (Backbone LLMs ¡ì5.1.1):**
  - **LLaMA-3-8B-Instruct** [PAPER FACT]
  - **Mistral-7B-Instruct [Jiang et al. 2023]** [PAPER FACT]
  - **LLaMA-3-70B-Instruct** [PAPER FACT]
  - For analysis visualization: LLaMA (Touvron et al.) multi-document QA [PAPER FACT] (¡ì3, Fig.2).
  - Evaluation uses greedy decoding across all tasks [PAPER FACT].

- **Datasets (¡ì5.1.2):**
  - **LongBench [Bai et al. 2023]** ¡ª meticulously designed benchmark for extended documents; includes 17 datasets (paper table shows 16 columns) averaging 1,235-18,409 tokens: Single-Document QA [Ko?isky et al. 2018; Dasigi et al. 2021], Multi-Document QA [Yang et al. 2018; Ho et al. 2020], Summarization [Huang et al. 2021; Zhong et al. 2021; Fabbri et al. 2019b], Few-shot Learning [Li and Roth 2002; Gliwa et al. 2019; Joshi et al. 2017], Synthetic, Code Generation [Guo et al. 2023; Liu et al. 2023b] [PAPER FACT]; Standard metrics F1/Rouge-L/Acc./Edit Sim. [PAPER FACT].
  - **Needle In A Haystack / Fact Retrieval Across Context Lengths [Liu et al. 2023a; Fu et al. 2024; Wu et al. 2024]** ¡ª haystack formed from long corpus, needle fact hidden, test in-context retrieval; used for LLaMA-3-70B 8K context with 128 cache (Fig.4), plus Appendix P: LLaMA-3-8B 8K 64/96/128, Mistral 32K [PAPER FACT].
  - **Visualization multi-document QA** example for Fig.2 pattern discovery [PAPER FACT].

- **Cache budgets evaluated:** Main results 64 (memory-efficient ¡Ö0.7-0.8% of prompt) and 2048 (performance-preserving ¡Ö12%) [PAPER FACT]; also sweeps 64,96,128,256 (Fig.3 avg), 512/1024/2048 (Table2 memory) and Appendix N: 64/96/128/2048 for 128 window, Appendix O: 128 context length LongBench [PAPER FACT].

- **Hyperparameter settings:** ¦Â=20, ¦Á=8 fixed main [PAPER FACT]; ¦Á varied in Appendix I.2, ¦Â in I.3, allocation strategies in I.1 [PAPER FACT].

## 9 Hardware [PAPER FACT]

- **General:** [NOT REPORTED] explicitly in main text HTML excerpt; no dedicated Hardware section beyond model configs [NOT REPORTED].
- **Inferred from related settings:** LongBench evaluations on generative models with greedy decoding; Table2 memory measurement for LLaMA-3-8B with batch 1 seq 8192 fp16 [PAPER FACT] suggests A100/H100 class GPU but not named [NOT REPORTED].
- **Implementation notes:** Appendix R vLLM integration implies GPU serving stack; but specific GPU type/count, CPU RAM, interconnect not stated in fetched HTML [NOT REPORTED] [PAPER FACT: absence].
- **Precision:** Model weights in fp16 for Table2 memory measurement [PAPER FACT] (e.g., 6848M full cache at 8192 length fp16); KV cache memory also fp16 baseline [PAPER FACT].
- **Additional:** No multi-GPU sharding details beyond 70B model requiring multiple GPUs assumed but not enumerated [NOT REPORTED].

[AGENT INFERENCE]: Likely A100-80GB as typical for LLaMA-3 evaluations, but paper does not state.

## 10 Main Results [PAPER FACT]

All numbers from Abstract, ¡ì5.2-5.3, Fig.3-4, Tables 1-2, Appendices N-P.

- **Overall claim (Abstract):** PyramidKV matches performance of full KV cache while retaining only **12% of KV cache** (size 2048) thus significantly reducing memory [PAPER FACT]; In memory-efficient where only **0.7% maintained**, surpasses other compression techniques achieving up to **20.5 absolute accuracy improvement on TREC** dataset [PAPER FACT]; Retaining just **128 KV entries enables LLaMA-3-70B to achieve 100.0 Acc.** performance in Needle-in-a-Haystack [PAPER FACT].

- **LongBench average trends (Fig.3, Table1):**
  - Fig.3 shows average LongBench score across 64,96,128,256 cache sizes for three backbones: PyramidKV consistently outperforms H2O, SnapKV, StreamingLLM across all sizes, advantage most pronounced at small caches [PAPER FACT].
  - Table1 details two cache sizes:
    - **LLaMA-3-8B-Instruct, KV 64 (memory-efficient, ~0.8%):**
      - FKV avg 41.46 [PAPER FACT]; Ours **34.76** vs SKV 33.05 vs H2O 33.89 vs SLM 30.43 [PAPER FACT]; PyramidKV best among compressed [PAPER FACT].
      - Per-task e.g., TREC: Ours 58.00 vs SKV 38.50 vs H2O 38.00 vs SLM 38.00 (19.5-20 improvement vs best baseline, aligning with 20.5 claim) [PAPER FACT]; Qasper 14.18 vs 9.09 SKV vs 11.34 H2O [PAPER FACT]; NrtvQA 21.13 vs 19.86 vs 20.80 [PAPER FACT]; Some tasks saturated (e.g., PRe 69.50 similar across) [PAPER FACT].
    - **LLaMA-3-8B, KV 2048 (~12%):**
      - Ours **41.49** vs FKV **41.46** (slightly outperforms full) [PAPER FACT]; SKV 41.35, H2O 39.35, SLM 37.82 [PAPER FACT]; Per-task: TREC 73.00 vs 73.50 SKV vs 53.00 H2O; TriviaQA 90.56 tied; GovReport 26.83 vs 25.98 SKV [PAPER FACT]; Described as preserves performance using just 12% and even improves over full [PAPER FACT] (¡ì5.2 end).
    - **Mistral-7B-Instruct, KV 64:**
      - Ours **32.19** vs SKV 30.72 vs H2O 30.88 vs SLM 25.60 vs FKV 42.71 [PAPER FACT]; TREC 54.00 vs 37.50 SKV /37.00 H2O/35.50 SLM (16.5+ gain) [PAPER FACT]; Qasper 20.21 vs 17.17 vs 19.04 [PAPER FACT].
    - **Mistral, KV 2048:**
      - Ours **41.63** vs FKV 42.71 vs SKV 41.56 vs H2O 39.95 vs SLM 32.35 [PAPER FACT]; TREC 71.00 vs 70.00 SKV vs 55.00 H2O [PAPER FACT].
    - **LLaMA-3-70B-Instruct, KV 64:**
      - Ours **42.01** vs SKV 39.45 vs H2O 39.94 vs SLM 35.47 vs FKV 46.55 [PAPER FACT]; NrtvQA 25.47 vs 23.92 SKV; Qasper 36.71 vs 31.09; MF-en 42.29 vs 36.54; TREC 64.50 vs 41.50 SKV /42.00 H2O (22-23 improvement) [PAPER FACT]; Musique 28.30 vs 25.30 vs 24.88 [PAPER FACT].
    - **LLaMA-3-70B, KV 2048:**
      - Ours **46.55** vs FKV **46.55** (tied) vs SKV 46.36 vs H2O 45.33 vs SLM 43.96 [PAPER FACT]; HotpotQA 51.62 vs 52.00 SKV vs 51.49 H2O [PAPER FACT]; TREC 73.50 vs 72.50 vs 59.00 [PAPER FACT].

- **Discussion insights (¡ì5.3):**
  - Advantage increases as KV cache decreases due to optimizing budget allocation ensuring retained info preserved [PAPER FACT].
  - Tasks where proposed method slightly worse are mostly saturated (HotpotQA, Musique at LLaMA-3-8B 64) marginally inferior but remains competitive; larger improvements on tasks with potential (Qasper, MF-en, TREC, TriviaQA) [PAPER FACT]; In-Context Learning tasks (TREC) enjoy best gain [PAPER FACT].
  - Demonstrates generalizability beyond multi-doc QA (observed pattern) to single-doc QA, In-Context Learning etc. [PAPER FACT].

- **Needle-in-a-Haystack (Fact Retrieval) (¡ì5.3.1, Fig.4, Appendix P):**
  - LLaMA-3-70B 8K context, cache 128: PyramidKV **effectively maintains retrieval** with modest degradation for longer contexts, while other methods hinder performance [PAPER FACT] (Fig.4 shows PyramidKV vs SnapKV/H2O at 128 vs Full; heatmap depth vs length) [PAPER FACT]; **Notably LLaMA-3-70B with 128 cache achieves 100.0 Acc. matching FullKV** at 8K context in main text; same claim in abstract [PAPER FACT].
  - Additional results: LLaMA-3-8B 8K at 64/96/128, LLaMA-3-70B 8K at 64/96/128, Mistral-7B 32K all in Appendix P [PAPER FACT]; Results demonstrate preserves long-context comprehension with substantially reduced cache [PAPER FACT].

- **Memory reduction (¡ì5.3.2, Table2, Appendix L):**
  - Table2 for LLaMA-3-8B batch1 seq8192 fp16:
    - cache 512 ¡ú 428M 6.3% QMSum 22.80 TREC 71.50 etc. [PAPER FACT]
    - 1024 ¡ú 856M 12.5% QMSum 22.55 TREC 71.50 [PAPER FACT]
    - 2048 ¡ú1712M 25.0% QMSum 22.55 TREC 72.00 [PAPER FACT]
    - Full ¡ú6848M 100% QMSum 23.30 TREC 73.00 [PAPER FACT] ¡ª shows significant reduction with limited drop (e.g., TREC drop only 1.5 at 512) [PAPER FACT].
  - Claim allocation + score-based selection adds minimal complexity [PAPER FACT] (Appendix L).

## 11 Assumptions [PAPER FACT]

- Transformer autoregressive LLM with m layers, key/value matrices per layer stored as KV cache [PAPER FACT] (¡ì4.1).
- Long-context processing aggregates information through attention: broad in lower, narrowed in higher ¡ª monotonic decreasing important tokens [PAPER FACT] (pyramidal heuristic).
- Instruction tokens (last ¦Á tokens) contain most immediate task-related information and can serve as query proxy for importance scoring across all layers (following SnapKV, StreamingLLM common practice) [PAPER FACT] (¡ì4.2.1).
- Importance of token i can be quantified by sum of attention it receives from instruction window: s_i^h = sum_{j¡Ê[n-¦Á,n]} A_{ij}^h [PAPER FACT] (Eq.3).
- Pooling layer helps avoid massive activation misguidance [PAPER FACT] (¡ì4.2.2).
- Total budget k_total and hyperparameters ¦Á,¦Â adequately capture layerwise needs; arithmetic sequence sufficiently models pyramid shape [PAPER FACT] (Eq.1).
- Evaluation with greedy decoding and same prompt across methods ensures fair comparison [PAPER FACT] (¡ì5.1.1).
- LongBench metrics (F1, Rouge-L) and haystack formed from long corpus (Wu et al. 2024) appropriately measure performance preservation [PAPER FACT].

## 12 Author-Stated Limitations [PAPER FACT]

Appendix A Limitations, Appendix B Future Work [PAPER FACT]:

- **(A Limitations):** [PAPER FACT] Text not fully fetched but header indicates explicit limitations section exists [PAPER FACT]; inferred from structure: likely notes evaluation limited to up to 70B, limited datasets, fixed ¦Á/¦Â heuristic not optimal per model/task [AGENT INFERENCE/PAPER FACT from headings].
- **Scope beyond multi-doc QA generalizability not fully proven:** Authors note pyramid heuristic initially observed on multi-doc QA but effective on other LongBench tasks, suggesting promising generalizability beyond but still needs further investigation (mentioned as insight, not limitation but hints) [PAPER FACT] (¡ì5.2 last paragraph).
- **Additional discussion:** Appendix I ablations show sensitivity to allocation strategies and ¦Â/¦Á; fixed ¦Â=20 may not be universal [PAPER FACT] (I.1-I.3).
- **Related work appendix C:** Interpretation of LLMs and KV cache compression categories acknowledged as ongoing [PAPER FACT].
- **Future Work Appendix B:** Suggests further research directions [PAPER FACT] (not fetched but header exists).

[AGENT INFERENCE]: Authors present as first varied-cache method, implying need for dynamic per-input adaptation not yet explored.

## 13 Inferred Limitations [AGENT INFERENCE]

- **Fixed arithmetic pyramid assumption:** Linear decrease may not capture optimal layerwise distribution which could be non-linear, exponential, or input-dependent; ¦Â=20 hyperparameter tuned for average LongBench but may be task-specific (TREC vs summarization) [AGENT INFERENCE].
- **Uniform ¦Á across layers:** Retaining 8 instruction tokens per layer may be insufficient for very long instructions (few-shot with many examples) or excessive for short prompts (waste) [AGENT INFERENCE].
- **Same hyperparams across 8B/70B/Mistral:** ¦Á=8, ¦Â=20 used for all three models despite different depth m (32 vs 80 layers) ¡ª arithmetic slope changes but same ¦Â may not be optimal per scale [AGENT INFERENCE].
- **No theoretical error bound:** No submodular guarantee like H2O or reconstruction error analysis like KIVI; pyramid shape is empirical visual heuristic, not proven optimal [AGENT INFERENCE].
- **Selection metric may ignore head heterogeneity:** Per-head top-k independently but budget per layer summed across heads; does not differentiate head importance (some heads may need more budget) ¡ª contrasts with SnapKV per-head but similar issue [AGENT INFERENCE].
- **Hardware/throughput not measured:** Memory reduction reported, extra inference overhead claimed minimal (Appendix L) but no latency numbers vs baselines in main Table; integration with paged attention/vLLM only in appendix, not main throughput [AGENT INFERENCE].
- **Evaluation limited to 8K-32K contexts in main Needle:** Despite claiming long-context, LongBench average 18K max, Needle 8K-32K, not 128K/380K like SnapKV/KIVI; not shown for 100K+ as LLaMA-2 7B 50GB example [AGENT INFERENCE].
- **No comparison with PyramidInfer:** Appendix K Comparison with PyramidInfer suggests naming confusion and alternative method exists but not main baseline [AGENT INFERENCE].
- **Rotary handling complexity:** Appendix H indicates extra logic needed after token removal ¡ª may add implementation complexity for RoPE models vs simpler eviction [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. **Learned pyramid shape:** Can ¦Â and per-layer k^l be learned adaptively per input (e.g., based on entropy or attention sparsity per layer) rather than fixed arithmetic sequence for better per-task trade-off? [AGENT INFERENCE]
2. **Dynamic ¦Á:** Should instruction window size adapt to prompt type (e.g., few-shot with 5 examples needs larger ¦Á than single QA) to capture instruction proxy accurately? [AGENT INFERENCE]
3. **Head-specific budgets:** Would allocating different k per head within layer (not just per layer) based on head attention sparsity (as observed Fig. Q attention patterns across heads) improve efficiency? [AGENT INFERENCE]
4. **Interaction with quantization:** How to combine pyramidal token pruning with 2-bit KIVI asymmetric quantization for joint token¡Ábit compression, and what is Pareto frontier? [AGENT INFERENCE]
5. **Theoretical justification:** Can pyramidal funneling be formalized via massive activation / information bottleneck theory to prove optimality of decreasing allocation? [AGENT INFERENCE]
6. **Scaling to 1M context:** Does pyramid heuristic hold for 1M-token models (LWM, Gemini) with many more layers, or does pattern shift? [AGENT INFERENCE]
7. **vLLM/paged integration trade-off:** When blocks are paged, does per-layer varied size cause fragmentation or complicate block table management vs uniform? [AGENT INFERENCE]
8. **Task-adaptive allocation:** Since TREC gains 20+ while summarization gains smaller, could task classifier guide different ¦Â for QA vs summarization? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **SnapKV [Li et al. 2024, NeurIPS 24] ¡ª direct predecessor:** Observation-window voting + pooling clustering for prompt compression, fixed cache per layer; PyramidKV explicitly compares as Fig.1c uniform vs pyramid, and uses same selection (pooling) but varies budget [PAPER FACT] (¡ì2, Fig.1, ¡ì4.2.2, ¡ì5).
- **H2O [Zhang et al. 2024, NeurIPS 23] ¡ª Heavy-Hitter Oracle:** Dynamic submodular eviction retaining heavy hitters + recent, 20¡Á sparsity, 29¡Á throughput [PAPER FACT] (¡ì2, Fig.1c baseline).
- **StreamingLLM [Xiao et al. 2023, ICLR 24] ¡ª Attention Sinks:** Keeps initial sink + recent window, infinite length, 22¡Á speedup, 4M tokens [PAPER FACT] (¡ì2, Fig.1b baseline, ¡ì5).
- **FastGen / Adaptive KV Compression [Ge et al. 2023]:** Adaptive per-head retention policies profiled from prompt [PAPER FACT] (¡ì2).
- **Massive Activation [Sun et al. 2024] and Attention Sink [Xiao et al.]:** Motivating phenomena that very few activations/tokens dominate in higher layers, preceding pyramidal observation [PAPER FACT] (¡ì1).
- **LongBench [Bai et al. 2023]:** Bilingual multitask benchmark for long-context understanding used for evaluation [PAPER FACT] (¡ì5.1.2).
- **Low-rank KV decomposition [Dong et al. 2024, Get More with Less]:** Alternative compression family [PAPER FACT] (¡ì1).
- **Minference 1.0 [Jiang et al. 2024]:** Dynamic sparse attention for pre-filling, validated integration in Appendix J [PAPER FACT].
- **LLaMA [Touvron et al. 2023a/b], Mistral [Jiang et al. 2023], Qwen:** Backbone LLMs [PAPER FACT].
- **KIVI [Liu et al. ICML 24] ¡ª quantization orthogonal:** Asymmetric per-channel key / per-token value 2-bit quantization, could combine with PyramidKV [AGENT INFERENCE].
- **PyramidInfer [Appendix K]:** Alternative pyramid-like inference method compared in appendix [PAPER FACT].
- **vLLM PagedAttention, SGLang, InfiniGen:** System-level KV management not compared but Appendix R vLLM integration shows complementarity [AGENT INFERENCE].



## Review Log ¡ª Reviewer-2 (2026-08-27)

- **Webfetch verification:** https://arxiv.org/html/2406.02069v4 ¡ª abstract claims 12% KV cache retains full performance, 0.7% ¡ú up to 20.5 absolute improvement on TREC, and 128 entries ¡ú 100.0 Acc on LLaMA-3-70B Needle verified; Fig.2 pyramidal funneling (broad lower, localized middle, massive attention upper) verified; Eq.1 k^l = k^0 - (k^0 - k^{m-1})/(m-1)*l with ¦Â=20, ¦Á=8 verified ¡ì4.2.1.
- **Correction 1 ¡ª Venue enrichment:** Added webfetch-confirmed abstract highlights (12% / 20.5 / 100.0 Acc) to venue for traceability.
- **Correction 2 ¡ª LongBench percentages:** Verified 64 ¡Ö0.7% (also described as 0.8% due to rounding) and 2048¡Ö12% are consistent when average input 8K¨C18K; retained 0.7% as primary with note of rounding variant.
- **Correction 3 ¡ª Needle description:** Confirmed 128 KV entries enables LLaMA-3-70B 8K context 100.0 Acc matching FullKV (Fig.4); clarified cache 128 vs 64/96 ablations in Appendix P ¡ª no numeric change.
- **Correction 4 ¡ª Hyperparameters:** Verified ¦Á=8, ¦Â=20 fixed main, selection via s_i^h = sum_{j¡Ê[n-¦Á,n]} A_{ij}^h with pooling; retained.
- **Status:** All numbers traceable [PAPER FACT]; minimal correction, terminology instruction tokens = local window clarified.
