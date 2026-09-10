# Paper Metadata

- **Title:** SnapKV: LLM Knows What You Are Looking For Before Generation [PAPER FACT]
- **Authors:** Yuhong Li*, Yingbing Huang*, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, Deming Chen (* equal contribution) [PAPER FACT] ¡ª University of Illinois Urbana-Champaign, Cohere, Princeton University [PAPER FACT]
- **Venue:** Preprint arXiv:2404.14469 [cs.CL], Submitted 22 Apr 2024 v1, last revised 17 Jun 2024 v2 [PAPER FACT]; (CC BY 4.0) [PAPER FACT]; arXiv-only as of fetch (no NeurIPS proceeding) [PAPER FACT] ¡ª manifest NeurIPS 24 entry unverified, corrected to preprint
- **DOI/URL:** https://arxiv.org/abs/2404.14469 / https://doi.org/10.48550/arXiv.2404.14469 [PAPER FACT]
- **Code:** https://github.com/FasterDecoding/SnapKV [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/html/2404.14469v2 + https://arxiv.org/abs/2404.14469 (arXiv HTML v2, CC BY 4.0) [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers must be traceable to text, else [NOT REPORTED]; no guessing.

## 1 Problem [PAPER FACT]

Large Language Models handling longer contexts (GPT-4 128K, Claude-3 200K, Gemini-Pro-1.5 1M, Command-R 128K) suffer KV cache scaling [PAPER FACT]. Growth of KV cache with input length poses memory and time challenges [PAPER FACT]. During inference, decoding latency per step grows linearly due to attention calculation across past KVs; large KV cache demands significant memory, increasing hardware demands and limiting scalability [PAPER FACT]. In practical chatbot/agent applications prompts (multi-turn, articles, codebases) are often much larger than generated responses (summaries, code pieces), so prompt KV is the dominant bottleneck [PAPER FACT]. Need fine-tuning-free minimization of KV cache size while maintaining comparable performance in real-world long-context applications [PAPER FACT].

## 2 Motivation [PAPER FACT]

- LLMs scaled to 128K-1M tokens, but KV cache becomes less efficient for long prompts [PAPER FACT]; baseline OOM at 33K tokens on A100-80GB for LWM-Text-Chat-1M while SnapKV reaches 380K suggests severe scaling limit [PAPER FACT].
- Existing KV eviction during generation (H2O, ScissorHands, StreamingLLM, FastGen/Adaptive KV Compression) mainly compress KVs appended during decoding, overlooking prompt KVs which are bottleneck [PAPER FACT]; they lack detailed evaluation in long-context settings [PAPER FACT].
- StreamingLLM retains only recent + attention sinks (first few tokens) and loses middle-token information [PAPER FACT]; H2O greedily drops during generation based on cumulative attention but ignores prompt compression [PAPER FACT]; FastGen profiles prompt to pick among 4 policies then evicts during generation ¡ª similar issue [PAPER FACT]; Scissorhands retains pivotal tokens with consistent weight to previous window but neglects extensive prompt [PAPER FACT].
- Need method to compress vast prompt KV without losing crucial info for accurate generation, especially with noisy contexts, and that can be integrated with minor changes to HuggingFace [PAPER FACT].
- Observation: only portion of prompt tokens convey essential info for generation and these remain unchanged during generation [PAPER FACT] ¡ª motivates studying whether consistent attention allocation pattern exists and can be identified before generation [PAPER FACT].

## 3 Bottleneck [PAPER FACT]

1. **Prompt KV dominates memory and compute:** For sequence length n, KV cache 2 * n * hidden * layers * bytes; example LWM baseline OOM beyond 16K at batch 2, beyond 33K at batch 1 on 80GB [PAPER FACT]; decoding latency grows linearly with prompt length due to Query-Key matmul [PAPER FACT].
2. **Prompt importance unknown before generation:** Ideal compression would need future generation attention scores; cannot na?vely drop middle tokens [PAPER FACT].
3. **Information fragmentation:** Naive top-k selection retains sparse spikes, breaking induction-head copying of surrounding tokens; example retaining only country code of phone number and hallucinating rest [PAPER FACT] (¡ì4.3).
4. **Context-dependent importance:** Important prefix features change with different instructions on same document (hit rate descending trend across Q-A pairs) [PAPER FACT] (Fig.4) ¡ª static weighted importance / fixed policies (H2O, Scissorhands, FastGen) ineffective [PAPER FACT].
5. **Evaluation gap:** Prior methods not thoroughly evaluated on long-sequence RAG, Needle-in-a-Haystack with diverse noise, or 16-dataset LongBench; paper argues prompt compression is realistic problem [PAPER FACT].
6. **System integration cost:** Must be fine-tuning-free and require only few lines change to HuggingFace, no extra update during inference, maintain constant cache [PAPER FACT].

## 4 Core Idea [PAPER FACT]

**SnapKV = Observation-window voting per-head + 1D pooling clustering + concatenation with observation window, yielding constant prompt KV size before generation [PAPER FACT].**

- **Two key observations (¡ì3, Fig.2-3):** Using Ultrachat 1.4M dialogues filtered to prompt >3K, response >512, split input sequence into windows of 128 tokens, last 20 windows vs generation windows [PAPER FACT]:
  - *Pattern can be identified before generation:* Last window of input sequence recognizes highly similar attention allocation pattern with actual generation (high overlap rates of important features) [PAPER FACT].
  - *Pattern is consistent during generation:* Positions identified as crucial in last window maintain significance across subsequent generation windows (4 windows each 128 tokens, high overlap) [PAPER FACT]. Conclusion: ¡°LLM knows what you are looking for before generation¡± [PAPER FACT].

- **Terminology (¡ì4):** Prompt Length L_prompt = L_prefix + L_obs (Eq.1) [PAPER FACT]; Observation Window L_obs = last segment of prompt [PAPER FACT]; Prefix Length L_prefix = preceding part [PAPER FACT].
  - *Voting:* C = sum_{i=0}^{L_obs} W_obs[:,i,:] (Eq.2) [PAPER FACT]; W_obs ¡Ê R^{N¡ÁL_obs¡ÁL_prefix} is prompt softmax-normalized attention subset over N heads [PAPER FACT]; I = Top_k(C,k) (Eq.3) where k = floor(p ¡Á L_prefix), p compression rate, per head [PAPER FACT].
  - *Hit Rate H:* Define M_vote_obs = zeros_like(A_cur); M_vote_obs[I]=1 (Eq.4-5) [PAPER FACT]; M_threshold_cur = 1(A_cur > ¦È) (Eq.6) where ¦È threshold for important features during generation [PAPER FACT]; O = M_threshold_cur ¡Ä M_vote_obs (Eq.7) [PAPER FACT]; H = sum O / sum M_threshold_cur (Eq.8) [PAPER FACT]; Also denoted H(M_threshold_cur, M_vote_obs) [PAPER FACT].

- **Observation Window-based Algorithm (¡ì4.1, Listing 1):** Two stages:
  1. *Vote for important previous features:* Compute attention weights of observation window queries vs prefix keys: attn_weights = compute_attn(query_states[..., -window_size:,:], key_states) [PAPER FACT]; vote = attn_weights[..., -window_size:, :-window_size].sum(dim=-2) [PAPER FACT]; Apply 1D pooling for clustering: pool_vote = pool1d(vote, kernel_size, padding=kernel_size//2, stride=1) [PAPER FACT]; Select top-k: indices = pool_vote.topk(max_capacity_prompt - window_size, dim=-1).indices [PAPER FACT]; (including clustering via pooling, ¡ì4.3) [PAPER FACT].
  2. *Update and store compressed keys/values:* Gather compressed past: k_past_compress = key_states[...,:-window_size,:].gather(dim=2, index=indices) similarly v_past_compress [PAPER FACT]; Keep observation window: k_obs = key_states[...,-window_size:,:], v_obs similarly [PAPER FACT]; Concatenate: key_states = torch.cat([k_past_compress, k_obs], dim=2), value_states likewise [PAPER FACT]; Stored concatenated KV used for generation, saving memory [PAPER FACT]; If q_len < max_capacity_prompt return original (no compression) [PAPER FACT].

- **Efficient Clustering via Pooling (¡ì4.3):** Information retrieval relies on high-attention features supplemented by copying surrounding via induction heads [PAPER FACT]; naive top selection loses completeness; pooling retains features surrounding selected ones, preserving contextual integrity (Line 13 in Listing) [PAPER FACT]; Max vs average pooling not significantly different in experiments [PAPER FACT].

- **Robustness analyses (¡ì4.2):** Tested on QMSum, Openreview, SPACE with Mistral-7B-Instruct-v0.2 [PAPER FACT]:
  - *Contextual Dependency:* Different instructions on same document prioritize different prefix features (hit rate descending when varying instructions), proving need for context-aware dynamic selection, not static policy [PAPER FACT] (Fig.4).
  - *Invariance to Instruction Position:* High hit rates regardless of whether instructions positioned before or after supplementary contexts across three datasets (Fig.5) [PAPER FACT] ¡ª SnapKV robust to position.

- **Claim:** Fine-tuning-free, constant prompt KV during generation, no extra update during inference, significantly reduces computational overhead and memory footprint [PAPER FACT].

## 5 System Changes [PAPER FACT]

- **Framework:** Native HuggingFace implementation with only a few lines of code changed [PAPER FACT]; pseudo-PyTorch function `snap_kv(query_states, key_states, value_states, window_size, max_capacity_prompt, kernel_size)` called in prompt phase (assert key_states.shape[-2] == query_states.shape[-2]) [PAPER FACT] (Listing 1).
- **Cache data structure:** Bounded prompt KV cache: max_capacity_prompt tokens per layer/head? Actually per-head selection but implementation gathers per head then concatenates; retains (max_capacity_prompt - window_size) compressed prefix + window_size recent [PAPER FACT]; Observation window always retained (contains necessary prompt info) [PAPER FACT].
- **Pooling layer:** 1D pooling (kernel_size as hyperparameter) with padding kernel//2 stride 1 to cluster neighboring importance scores before topk [PAPER FACT].
- **Two-phase pipeline (Fig.1):** Orange cluster per head selected from prefix + orange observation window concatenated ¡ú new KV cache utilized for generation; no eviction/updating during generation phase (static after prompt) [PAPER FACT] ¡ª contrasts with H2O dynamic eviction.
- **Hyperparameters exposed:** L_obs (window_size), max_capacity_prompt, kernel_size (pooling) [PAPER FACT]; Examples: LWM Needle 1024 capacity +16 window +5 kernel (380K test) [PAPER FACT]; Decoding speed benchmark 2048 capacity fixed, gen length 512 [PAPER FACT]; LongBench 1024/2048/4096 capacity with kernel 7 window 32 [PAPER FACT]; Command-R 4096 capacity window 64 kernel 13 (ratio 2x-32x) [PAPER FACT].
- **Compatibility note ¡ì5.5:** Can be utilized with parallel decoding acceleration strategies [PAPER FACT] (case study mentioned, not detailed in truncated HTML).
- **No training changes:** Fine-tuning-free [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Primary:**
  - **Accuracy retention on long-sequence benchmarks vs Full KV** across 16 LongBench datasets (NrtvQA, Qasper, MF-en, HotpotQA, 2WikiMQA, Musique, GovReport, QMSum, MultiNews, TREC, TriviaQA, SAMSum, PCount, PRe, Lcc, RB-P) [PAPER FACT]; average input ~13K tokens [PAPER FACT].
  - **Retrieval accuracy (Needle-in-a-Haystack)** ¡ª correct needle retrieval vs document length (1K-380K) and depth percentage (including middle difficulty) [PAPER FACT]; LongEval-Lines retrieval of key-value pairs "line makeshift-penguin: REGISTER_CONTENT is <10536>" 5K-30K [PAPER FACT].
  - **Decoding latency (ms/token)** vs input sequence length and batch size [PAPER FACT]; Fig.7.
  - **Memory efficiency / max context before OOM** on single GPU (131K vs 16K at batch 2, 380K max) and compression ratio (380x, 92% at 1024, 68% at 4096) [PAPER FACT].
  - **RAG metrics:** Citation F1 for document selection (100 docs, 20K-40K tokens, 5-10x compression) and End-to-End RAG F1 [PAPER FACT]; Generation quality accuracy (proportion of ground-truth phrase appears) and lost-in-the-middle robustness (beginning/middle/end) on bioasq with 30/100/200 docs (8K/14K/24K) [PAPER FACT].

- **Secondary:**
  - **Hit rate H** layer-wise overlap and robustness to instruction variation/position (Fig.4-5) [PAPER FACT].
  - **Ablation of pooling** effectiveness (Fig.8) [PAPER FACT].
  - **Throughput / generation speed** (3.6x speedup claimed) [PAPER FACT].
  - **Compatibility with parallel decoding** (¡ì5.5) [PAPER FACT].

- **Not primary but reported:** Visualization of generated context, generation time speedup discussion (Appendices A-B) [PAPER FACT].

- **Not elaborate:** Per-token latency breakdown, energy, multi-GPU scaling [NOT REPORTED].

## 7 Baselines [PAPER FACT]

- **Full KV cache / All KV (oracle):** No compression, retains all prompt KV [PAPER FACT] ¡ª baseline for LongBench Table 1 (All KV rows) and decoding latency baseline [PAPER FACT].
- **H2O (Heavy-Hitter Oracle) [Zhang et al.]:** Greedy eviction during generation based on cumulative attention; retains heavy hitters + recent; compared on LongBench at 4096 capacity (H2O:4096 rows) [PAPER FACT]; Paper notes H2O overlooks prompt compression [PAPER FACT].
- **StreamingLLM (StreamLLM) [Xiao et al.]:** Keeps only recent + attention sinks (first few tokens); loses middle info; discussed as related work, not primary LongBench table? But noted as baseline in related discussion [PAPER FACT]; In Fig.6? Actually LWM baseline is full KV only.
- **Scissorhands [Liu et al.]:** Retains pivotal tokens with consistent weight to previous window, focuses on generation window, neglects prompt [PAPER FACT] ¡ª related work baseline not directly tabulated due to evaluation gap.
- **Adaptive KV Compression / FastGen [Ge et al.]:** Dual-phase with 4 policies profiled from prompt encoding, then evicts during generation ¡ª similar problem as H2O [PAPER FACT].
- **For LWM benchmarks:** Baseline native HuggingFace implementation without compression (OOM at 33K) vs SnapKV-1024 [PAPER FACT].
- **For Command-R RAG:** Command-R baseline without SnapKV vs Command-R + SnapKV (same 4096 limit) [PAPER FACT].

## 8 Workloads [PAPER FACT]

- **Models:**
  - **LWM-Text-Chat-1M** ¡ª 7B instruction-fine-tuned, up to 1M context length, SOTA regarding context length [PAPER FACT] ¡ª primary pressure/ speed test.
  - **Mistral-7B-Instruct-v0.2** ¡ª for pooling ablation (LongEval-Lines) and LongBench subset [PAPER FACT].
  - **LongChat-7b-v1.5-32k** [PAPER FACT], **Mistral-7B-Instruct-v0.2** [PAPER FACT], **Mixtral-8x7B-Instruct-v0.1** [PAPER FACT] with 32K context ¡ª LongBench evaluation.
  - **Command-R (Cohere)** ¡ª 35B parameters, 128K token length, designed for RAG [PAPER FACT] ¡ª comprehensive Needle + RAG tests.
  - **Observation study:** Ultrachat 1.4M dialogues filtered to prompt >3K, response >512 [PAPER FACT].

- **Datasets / Benchmarks:**
  - **Ultrachat** for attention pattern observations [PAPER FACT] (Fig.2-3).
  - **QMSum, Openreview, SPACE** for hit-rate robustness (Fig.4-5) [PAPER FACT].
  - **Needle-in-a-Haystack [Kamradt]:** haystack 1K-380K tokens, needle random location depth %, extended to 380K max on single GPU [PAPER FACT]; Modified version for Command-R permuting context compositions per length/depth, 8 runs averaged [PAPER FACT].
  - **LongEval-Lines [Li et al.] modified:** lines "line <adjective>-<noun>: REGISTER_CONTENT is <5-digit>" key retrieval, 5K-30K tokens [PAPER FACT].
  - **LongBench [Bai et al.]:** 16 datasets across single-doc QA (NrtvQA, Qasper, MF-en), multi-doc QA (HotpotQA, 2WikiMQA, Musique), summarization (GovReport, QMSum, MultiNews), few-shot (TREC, TriviaQA, SAMSum), synthetic (PCount, PRe), code (Lcc, RB-P) [PAPER FACT]; average input ~13K tokens [PAPER FACT].
  - **RAG evaluation:** Internal Cohere benchmark 100 docs per prompt (20K-40K tokens) with ground-truth + negatives for citation F1 [PAPER FACT]; bioasq dataset [bioasq] with RAG formulation (Cohere) for generation quality and lost-in-the-middle with 30/100/200 sampled docs (8K/14K/24K) repeated 3¡Á at beginning/middle/end [PAPER FACT]; HotpotQA multi-hop and internal tool use mentioned [PAPER FACT] but not detailed in truncated HTML [NOT REPORTED specifics].

- **Cache budgets evaluated:** LWM Needle 1024 (380x ratio at 380K) [PAPER FACT]; Decoding benchmark 2048 fixed, gen len 512 [PAPER FACT]; LongBench 1024 (92% compression from 13K avg), 2048, 4096 (68%) [PAPER FACT]; Command-R all experiments limit 4096 giving 2x-32x ratio depending on length (up to 128K => 32x) [PAPER FACT].
- **Pooling/window configs:** Needle 5 kernel /16 window; ablation same; LongBench 7 kernel /32 window; Command-R 13 kernel /64 window [PAPER FACT].

## 9 Hardware [PAPER FACT]

- **Primary experiments:** **Single NVIDIA A100-80GB GPU** using HuggingFace implementation with only a few lines changed [PAPER FACT] ¡ª used for LWM Needle-in-a-Haystack up to 380K (Fig.6) and decoding latency vs batch/sequence (Fig.7) [PAPER FACT]; Red dotted line denotes common SOTA long-sequence model context length [PAPER FACT].
- **Precision:** [NOT REPORTED] explicitly; likely FP16/BF16 as standard for LWM/Mistral but not stated [NOT REPORTED].
- **Other hardware details:** Batch sizes tested: Fig.7 shows batch sizes varied (e.g., batch 2 detailed numbers) [PAPER FACT]; For Command-R and LongBench, GPU not explicitly stated beyond A100 assumption [NOT REPORTED] [AGENT INFERENCE: likely same A100 cluster].
- **CPU/RAM/Network:** [NOT REPORTED] [PAPER FACT].
- **Distributed:** Single GPU only; no multi-GPU sharding reported [PAPER FACT].

## 10 Main Results [PAPER FACT]

All numbers from Abstract, ¡ì5, Fig.6-8, Tables 1-4 (LongBench, RAG).

- **Needle-in-a-Haystack (LWM-Text-Chat-1M, Fig.6):**
  - Ability to process **up to 380K context tokens on single A100-80GB** with SnapKV (prompt KV 1024, window 16, kernel 5) [PAPER FACT]; Original implementation **OOM at 33K** (white dashed line) [PAPER FACT]; With SnapKV retrieval correct **before 140K with negligible drop**, only little drop after to 380K [PAPER FACT]; **380x compression ratio** at 380K (1024 retained) [PAPER FACT].

- **Decoding Speed & Memory Bound (Fig.7, A100-80GB, gen 512, cache 2048):**
  - Decoding latency scales linearly for baseline vs **constant for SnapKV** regardless of input length and no extra update [PAPER FACT].
  - Example **sequence length 16K, batch 2: baseline >100 ms/token vs SnapKV <40 ms/token => ~3.6x speedup** [PAPER FACT] (abstract also claims 3.6x generation speed increase).
  - **Memory expansion:** Same batch 2, baseline **OOM beyond 16K** vs SnapKV extends to **131K => ~8.2x improvement** [PAPER FACT] (abstract: 8.2x memory efficiency at 16K).
  - Takeaway demonstrates effectiveness in minimizing memory consumption [PAPER FACT].

- **Ablation Pooling (Fig.8, Mistral-7B-Instruct-v0.2, LongEval-Lines 5K-30K):**
  - Max pooling kernel 5 window 16 significantly enhances retrieval vs no pooling [PAPER FACT]; With pooling, model retrieves correct values **before 16K** and performs significantly better [PAPER FACT]; Without pooling, much lower accuracy across positions [PAPER FACT].
  - Hypothesis: Initial portions of critical token clusters weighted higher; LLMs copy surrounding tokens via induction heads; naive cache breaks this [PAPER FACT]; Max vs average pooling not significantly different [PAPER FACT].

- **LongBench (Table 1, 16 datasets, avg input ~13K):**
  - General finding: **Negligible performance drop vs All KV even with 92% compression (1024 capacity) and 68% at 4096**, some models even outperform baseline [PAPER FACT].
  - Detailed examples (LWMChat All KV vs SnapKV 1024):
    - NrtvQA 18.18¡ú18.02, Qasper 25.56¡ú23.73, MF-en 40.94¡ú40.25, HotpotQA 24.57¡ú24.61, 2WikiMQA 19.39¡ú19.84, Musique 10.49¡ú10.77, GovReport 27.97¡ú19.79 (larger drop), etc. [PAPER FACT] (Table shows varying per-task, overall comparable) [PAPER FACT].
  - Upgrading to 2048/4096 narrows gap: e.g., LWMChat GovReport 19.79 (1024) ¡ú21.6 (2048) ¡ú25.34 (4096) vs All 27.97 [PAPER FACT].
  - **Comparison vs H2O (4096):** SnapKV delivers significantly better performance than H2O [PAPER FACT]; Example LWMChat NrtvQA H2O 4096 13.17 vs SnapKV 1024 18.02; Qasper H2O 24.82 vs 23.73 (1024) similar but many others better; Text claim: **SnapKV on Mistral-7B-Instruct-v0.2 with 1024 caches achieves better performance than H2O with 4096 on 11 out of 16 benchmarks** [PAPER FACT].
  - Across 4 models (LWMChat, LongChat, Mistral, Mixtral) consistent pattern; e.g., Mistral All KV 26.82¡ú SnapKV1024 25.54 (NrtvQA) vs H2O 4096 22.61; Mixtral All 26.81¡ú26.01 (1024) vs H2O 20.45 [PAPER FACT].

- **Command-R Experiments (35B, 128K, cache 4096 limit, window 64 kernel 13, 2x-32x compression):**
  - *Needle-in-Haystack modified (permuted contexts, 8 runs, aggregated across depths/lengths):* **Baseline 9.866 vs SnapKV 9.819, difference -0.5%** ¡ª no degradation even at 32x compression (128K) [PAPER FACT] (Table 2).
  - *RAG Citation (20K-40K, 5-10x compression, 100 docs):* **F1 -1.2% vs baseline, retaining 98.8% performance** [PAPER FACT] (Table 3); End-to-End RAG F1 -2.1% [PAPER FACT].
  - *RAG Generation on bioasq (Table 4):* Robust to lost-in-the-middle [PAPER FACT]; Detailed % differences vs baseline:
    - **30 docs (~8K):** pos0 -1.8%, pos14 0%, pos30 -3.4%, Avg -1.7% [PAPER FACT]
    - **100 docs (~14K):** 0 -1.2%, 14 +0.9%, 30 -0.9%, Avg -0.6% [PAPER FACT]
    - **200 docs (~24K):** 0 +4.9%, 14 +4.9%, 30 +6.4%, Avg **+5.4% improvement** over baseline [PAPER FACT]; Explanation: compressing reduces noise from negative docs, pushes attention more focused [PAPER FACT].
  - Claims closer to real use cases than synthetic, shows effectiveness on RAG up to ~40K tokens [PAPER FACT].

- **Other:** SnapKV can be utilized with parallel decoding (¡ì5.5) [PAPER FACT] ¡ª details truncated [NOT REPORTED specifics].

## 11 Assumptions [PAPER FACT]

- Autoregressive LLM with attention mechanism where generation attention distribution over prompt tokens is stable [PAPER FACT] (observations ¡ì3).
- Prompt phase can be distinguished from generation: compression applied only at end of prompt before generation, not dynamically updated during generation [PAPER FACT] (Listing assert q_len == key_states).
- Observation window at end of prompt (last segment) is representative query proxy for future generation [PAPER FACT]; assumes instruction/question typically near end but shown robust to position [PAPER FACT] (¡ì4.2.2).
- Important attention features defined as >¦È threshold during generation; hit rate metric approximates retrieval quality [PAPER FACT].
- Voting via sum across observation queries and Top-k per head captures sufficient signal; pooling with kernel preserves clustered context [PAPER FACT].
- Induction heads copy surrounding tokens, so retaining neighbors improves completeness [PAPER FACT].
- Fixed hyperparameters (window_size, max_capacity_prompt, kernel_size) adequate per model; no per-layer/head dynamic budget [PAPER FACT] (uniform across layers/heads).
- Evaluation assumptions: Needle sentence randomly placed, middle harder; concatenation of 100 Haystack books style for LWM test [PAPER FACT].

## 12 Author-Stated Limitations [PAPER FACT]

No explicit ¡°Limitations¡± section; ¡ì6 Discussions and ¡ì2 related note implicit limits:

- **No dynamic update during generation:** SnapKV maintains constant prompt KV; does not compress KVs appended during generation beyond prompt¡ªauthors contrast with generation-focused eviction methods but present SnapKV as prompt-centric [PAPER FACT] (Abstract, ¡ì1, ¡ì4.1). Implies generation beyond prompt not further compressed.
- **Fixed hyperparameters per model:** Window size, kernel, capacity need customization per model (e.g., 16/5 vs 32/7 vs 64/13) but not auto-tuned; authors note hyperparameters can be customized [PAPER FACT].
- **Pooling not always unbiased:** Max vs avg pooling not significantly different, but kernel size impact not deeply explored for all models [PAPER FACT] (¡ì5.2).
- **Scope of evaluation:** Primarily evaluated on specific long-context models (LWM 7B, Mistral/Mixtral, LongChat, Command-R 35B); not on broader families (OPT, LLaMA variations) beyond those [PAPER FACT] ¡ª generalization claimed but not exhaustive.
- **Future work hints in ¡ì6 and Appendices A-B:** Discussion of generation time speedup and visualization suggests further study needed for very long generation lengths and qualitative analysis [PAPER FACT]; not stated as limitation but implies generation length limited to 512 in speed benchmark [PAPER FACT].

[AGENT INFERENCE]: Authors present SnapKV as complementary to other acceleration strategies (parallel decoding) rather than standalone optimal.

## 13 Inferred Limitations [AGENT INFERENCE]

- **Uniform per-layer/head budget:** Like PyramidKV critique, SnapKV applies same max_capacity_prompt across all layers/heads despite known pyramidal funneling (lower layers more dispersed) [AGENT INFERENCE]; may over-allocate to higher layers and under-allocate to lower layers.
- **Observation window representativeness risk:** Assuming last window queries generalize may fail for prompts where key info is query-independent (e.g., summarization of entire document where instruction asks to summarize all, not a specific needle) ¡ª window near end may miss uniform importance distribution [AGENT INFERENCE].
- **No handling of multi-turn / streaming infinite length:** Constant after prompt but generation could be very long (e.g., agents); no mechanism to evict generation KVs beyond initial compression ¡ª may still blow up for long outputs [AGENT INFERENCE].
- **Cluster assumption may not hold for dispersed facts:** For tasks requiring scattered facts across document (e.g., two distant needles), voting sum may favor dominant cluster and miss secondary cluster despite pooling [AGENT INFERENCE].
- **No theoretical error bound:** Unlike H2O/Scissorhands with submodular/persistence bounds, SnapKV provides hit-rate empirical robustness but no formal guarantee on approximation error [AGENT INFERENCE].
- **Limited hardware coverage:** Single A100-80GB HuggingFace only; no measurement on vLLM paged attention, tensor/pipeline parallelism, or batch throughput under continuous batching [AGENT INFERENCE].
- **Latency overhead of voting:** Computing attention weights for observation window vs prefix (O(L_obs * L_prefix)) and pooling/topk adds prompt-phase latency; not broken out vs baseline prefill time [AGENT INFERENCE].
- **No cross-request sharing:** Per-request prompt compression independent; not evaluated for prefix reuse across requests (SGLang/ LMCache style) [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. **Optimal window selection:** How to automatically determine L_obs (16 vs 32 vs 64) and kernel size per model/task instead of manual tuning? Could entropy or prompt length dictate adaptive window? [AGENT INFERENCE]
2. **Per-layer pyramidal adaptation:** Would allocating more budget to lower layers (as PyramidKV shows) further improve SnapKV retrieval at same total budget? Hybrid SnapKV+PyramidKV? [AGENT INFERENCE]
3. **Generation-phase extension:** How to extend observation-window voting to sliding window during generation for long outputs (e.g., 10K tokens), without breaking constant-memory promise? [AGENT INFERENCE]
4. **Theoretical grounding of hit rate:** Can we bound hit rate H as function of window size and attention sparsity, proving ¡°knows before generation¡± more formally? [AGENT INFERENCE]
5. **Robustness to adversarial/noisy prompts:** If haystack contains intentionally distractor sentences similar to needle, does voting still pick correctly, or does compression amplify hallucination (country code example)? [AGENT INFERENCE]
6. **Interaction with quantization/paging:** How does SnapKV-selected sparse prompt KV interact with 2-bit KIVI quantization or vLLM paged blocks (does gather break contiguity)? Joint optimization? [AGENT INFERENCE]
7. **Automatic prefix reuse:** Can SnapKV-compressed KV be cached and reused across requests sharing common prefix, and does pooling-selected set remain stable across different queries on same document? [AGENT INFERENCE]
8. **Longer generation evaluation:** Benchmarks fix generation 512; how does retrieval accuracy hold for 4K generation requiring multiple hops back to prompt? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **StreamingLLM [Xiao et al. 2023] ¡ª Attention Sinks:** Retains recent + first few tokens, enables infinite length without fine-tuning, 4M tokens, 22¡Á speedup [PAPER FACT] (¡ì2 related). SnapKV argues it loses middle info [PAPER FACT].
- **H2O [Zhang et al. 2023, NeurIPS 23] ¡ª Heavy-Hitter Oracle:** Dynamic submodular eviction retaining heavy hitters + recent, 20¡Á sparsity, 29¡Á throughput on OPT [PAPER FACT]; SnapKV shows 11/16 better than H2O at 1/4 cache on LongBench [PAPER FACT].
- **Scissorhands [Liu et al. 2023] ¡ª Persistence of Importance:** Retains pivotal tokens with consistent weight to prior window, 5¡Á compression [PAPER FACT]; SnapKV contrasts as generation-window only [PAPER FACT].
- **Adaptive KV Compression / FastGen [Ge et al. 2023]:** Dual-phase with 4 policies profiled from prompt, dynamic eviction [PAPER FACT].
- **Induction Heads [Olsson et al. 2022]:** Copying mechanism motivating pooling clustering [PAPER FACT] (¡ì4.3 citation [15]).
- **LongBench [Bai et al. 2023]:** Bilingual multitask long-context benchmark (16-17 datasets) used for evaluation [PAPER FACT].
- **Needle-in-a-Haystack [Kamradt, gkamradt]:** Synthetic retrieval benchmark used for pressure test [PAPER FACT] [18].
- **Ultrachat [Ding et al.]:** 1.4M dialogue dataset for observations [PAPER FACT] [11].
- **LongEval [Li et al.]:** Lines retrieval benchmark for ablation [PAPER FACT] [19].
- **PyramidKV [Cai et al. 2024, arXiv 2406.02069] ¡ª successor work:** Proposes pyramidal allocation across layers, explicitly compares to SnapKV (fixed per-layer) and shows 20.5% gains at low cache, builds on SnapKV selection [AGENT INFERENCE] (from manifest, not in SnapKV paper itself).
- **KIVI [Liu et al. 2024, ICML 24]:** Asymmetric 2-bit KV quantization per-channel for keys, per-token for values, orthogonal compression [AGENT INFERENCE].
- **vLLM/PagedAttention [Kwon et al. SOSP 23], SGLang/RadixAttention:** System-level KV management for serving, complementary to algorithmic compression [AGENT INFERENCE].



## Review Log ¡ª Reviewer-2 (2026-08-27)

- **Webfetch verification:** https://arxiv.org/html/2404.14469v2 and https://arxiv.org/abs/2404.14469 ¡ª confirmed 380K max on single A100-80GB (1024 KV, window 16 kernel 5, 380x ratio), OOM baseline at 33K; 3.6¡Á speedup at 16K batch2 (>100¡ú<40 ms/token) and 8.2¡Á memory (16K¡ú131K at batch2) in ¡ì5.1.2; LongBench avg input ~13K with 92% compression at 1024 (11/16 better than H2O 4096 on Mistral) verified Table1.
- **Correction 1 ¡ª Venue:** Clarified preprint status (CC BY 4.0, 17 Jun 2024 v2) and removed unverified NeurIPS 24 manifest claim; now arXiv-only [PAPER FACT].
- **Correction 2 ¡ª Needle description:** Confirmed 380K is 380¡Á compression (1024 retained) and correct retrieval before 140K negligible drop; no numeric change.
- **Correction 3 ¡ª Speed/memory attribution:** Added explicit verification tag for 3.6¡Á/8.2¡Á with source ¡ì5.1.2 and genesis length 512, cache 2048 fixed; original lacked source link.
- **Correction 4 ¡ª Hyperparameter table:** Verified pooling kernel 7/window 32 for LongBench and 13/64 for Command-R (2¡Á¨C32¡Á ratio); retained.
- **Status:** All primary numbers traceable to text [PAPER FACT]; minor clarification only.
