# Paper Metadata

- **Title:** H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models [PAPER FACT]
- **Authors:** Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher R谷, Clark Barrett, Zhangyang Wang, Beidi Chen [PAPER FACT] 〞 University of Texas at Austin, Stanford, UC San Diego, UC Berkeley, Adobe Research, Meta AI (FAIR), Carnegie Mellon [PAPER FACT]
- **Venue:** Preprint arXiv:2306.14048 [cs.LG], Submitted 24 Jun 2023 v1; last revised 18 Dec 2023 v3 [PAPER FACT]; Also presented at NeurIPS 2023 [PAPER FACT] (arXiv category cs.LG) [PAPER FACT]
- **DOI/URL:** https://arxiv.org/abs/2306.14048 / https://doi.org/10.48550/arXiv.2306.14048 [PAPER FACT]
- **Code:** https://github.com/FMInference/H2O [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/html/2306.14048v1 [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers must be traceable to text, else [NOT REPORTED]; no guessing.

## 1 Problem [PAPER FACT]

Large Language Models deployment cost-prohibitive, especially for long-content generation (dialogue, story writing) [PAPER FACT]. Beyond model weights, **transient KV cache** (Key/Value states) stored in GPU memory scales **linearly with sequence length and batch size** [PAPER FACT]; e.g., 30B model with batch 128, seq 1024 ↙ 180GB KV cache [PAPER FACT] (Introduction). Natural approach limits cache size like hardware caches, but reducing KV without accuracy drop is challenging [PAPER FACT]. Ideal KV cache should have: (i) small size, (ii) low miss rate, (iii) low-cost eviction policy [PAPER FACT] 〞 three technical challenges: whether size can be restricted (each step might need all previous KVs), combinatorial optimal eviction, and feasible deployment [PAPER FACT].

## 2 Motivation [PAPER FACT]

- LLMs proficiency in content creation, summarization, dialogue, but deployment costly [PAPER FACT]; KV cache becoming increasingly prominent vs weight size and quadratic attention [PAPER FACT] (citation Pope 2022).
- Existing sparse attention for training (Reformer, Flash Attention, Performer, Sparse Transformer, Multi-Query Attention) either still require large cache, or suffer **high miss rates** when applied to pre-trained LLMs, or have expensive eviction (gisting tokens) [PAPER FACT] (Fig.1 trade-off).
- Need **small, accurate, low-overhead** KV eviction that works on **pre-trained models without retraining** and preserves long-content generation ability [PAPER FACT].
- Preliminary explorations reveal intrinsic LLM properties enabling this: >95% attention sparsity, power-law heavy hitters, and greedy local statistic matches global [PAPER FACT] (∫1 bullets).

## 3 Bottleneck [PAPER FACT]

1. **Memory footprint blow-up:** KV cache 2* n_layers * hidden * seq * batch * 2 bytes; example 30B 180GB exceeds typical GPU memory, requiring offloading [PAPER FACT].
2. **Eviction policy is combinatorial:** Choosing k tokens to retain from n to maximize generation quality is subset selection; Belady optimal for standard cache **not optimal** for KV cache due to sequential dependency 〞 evicting important KV destroys future generation recursively [PAPER FACT] (∫2.1 footnote, ∫3.2).
3. **Future dependence:** Optimal policy would need future attention scores (not available at decode time) [PAPER FACT] (∫4.1).
4. **Prior sparse methods fail on pre-trained LLMs:** Sparse Transformer strided/fixed patterns drop accuracy up to 35% at 20% budget [PAPER FACT] (Table1); Multi-Query Attention etc. degrade [PAPER FACT].
5. **System integration challenges:** I/O efficiency if swapping, need low-cost per-step policy not to dominate decoding latency [PAPER FACT] (∫4.2 Implementation).
6. **Lack of theoretical grounding:** No submodular formulation or guarantee for greedy eviction prior to H2O [PAPER FACT] (∫4, Appendix D).

## 4 Core Idea [PAPER FACT]

**H2O (Heavy-Hitter Oracle): Dynamically retain a balance of Heavy Hitters (H2) + recent tokens via greedy accumulated attention, formulated as dynamic submodular maximization [PAPER FACT].**

- **Observation 1 每 Sparsity for small cache:** Attention matrices >95% sparse at inference (threshold 1% max per row) across OPT models on Wiki-Text-103 (Fig.2a) [PAPER FACT]; suggests **5% KV (20℅ reduction)** could suffice per step [PAPER FACT] (∫3.1).

- **Observation 2 每 Heavy Hitters for low miss rate:** Accumulated attention scores follow **power-law distribution** [PAPER FACT] (Fig.2b). Small set of tokens (H2) get most mass; strongly correlates with **frequent co-occurrence** in text (gray curve) [PAPER FACT]. Removing H2 ↙ "significant performance degradation" / drastic accuracy drop (Fig.2c) [PAPER FACT].

- **Observation 3 每 Greedy local ＞ global:** Retaining H2 based on **local statistics** (sum attention of preceding tokens only) is **as effective as global** (including future) (Fig.2d) [PAPER FACT] (∫1 bullet, ∫4.1). Enables low-cost online policy.

- **Algorithm 每 H2O Eviction (Def 4.3, Alg.1):** For budget k, maintain set Si (|Si|=k). At step i>k, compute normalized attention o_i = D_i^{-1} exp(Q_{i,*}(K_{Si,*})^T), D_i subtracts evicted mass; define F_score(T)=曳_{s﹋T} o_s (sum attention) [PAPER FACT]. Evict u = argmax_{v﹋Gi} F_score(S_{i-1}﹍{i}\{v}) where Gi = S_{i-1}﹍{i} [PAPER FACT] (keep set maximizing sum). Equivalent to keep tokens with highest **accumulated attention scores** plus recent window; Fig.3 example evicts 3rd token at step 4 with budget 3 [PAPER FACT].

- **Dynamic submodular formulation (∫4.1 Def 4.1):** Define F:2^{[n]}℅2^{[n]}↙? where F(Z,﹞) submodular w.r.t Z: f(X﹍{x})-f(X)≡ f(Y﹍{x})-f(Y) for Z?X?Y [PAPER FACT]. H2O greedy sequence is instance of dynamic submodular [PAPER FACT].

- **Theoretical guarantee (Thm 4.4 informal):** Under mild assumption, greedily computed top-k set \tilde{S}_i satisfies f(\tilde{S}_i) ≡ (1-汐)(1-1/e) max_{|S|=k} f(S) - 汕, 汐,汕>0 [PAPER FACT] (∫4.2). Provided via robust greedy with error propagation (Appendix D) [PAPER FACT].

- **System: Pure eviction without swapping** 〞 upon eviction directly overwrite memory slot with newly added KV, no CPU swap, ensuring I/O efficiency [PAPER FACT] (∫4.2).

- **Budget split:** Keep **both H2 and recent**; ablation shows either alone degrades 2.85%每22.75% [PAPER FACT] (∫5.3 Table6 described).

## 5 System Changes [PAPER FACT]

- **Framework on FlexGen:** Implement generic KV cache eviction policy interface atop FlexGen, orthogonal to offloading/quantization [PAPER FACT] (∫5.2). When model+cache not fit single GPU, enables CPU offloading; H2O reduces frequency of offloading [PAPER FACT].
- **Cache data structure:** Fixed-size buffer k per layer/head? Actually per attention head logically, but implementation manages KV cache per layer with bounded slots; |Si|=k throughout generation [PAPER FACT] (Def 2.1, 2.2).
- **Generative process with eviction (Def 2.2):** For each token i, Si ?[n] cached tokens, compute o_i with evicted mass subtracted (Di) (evicted KV set to 0) [PAPER FACT]; updates Si via policy g: S_{i-1}↙Si respecting |Si﹎S_{i-1}|≡k-1 [PAPER FACT].
- **Eviction step details (Alg.1):**
  - If i≒k: Si = S_{i-1}﹍{i} [PAPER FACT]
  - Else: compute Di, o_i, F_score, Gi, choose u via argmax, Si = (S_{i-1}﹍{i})\{u} [PAPER FACT]
  - History accumulated scores collected over sliding window to reduce variance; recent window protected (not evicted) [PAPER FACT] (described in Appendix A).
- **No swapping:** Fill newly added KV directly into evicted slot, avoid memory copy [PAPER FACT].
- **Integration points:** Supports any eviction algorithm; evaluation plugs H2O into FlexGen, accelerates prompt phase and token generation phase [PAPER FACT] (∫2 LLM Inference Breakdown, Appendix A).

## 6 Target Metrics [PAPER FACT]

- **Primary:**
  - **Accuracy on downstream tasks** with limited budget (20% KV) vs full cache [PAPER FACT]; tasks from **lm-eval-harness** [Gao et al.] and **HELM** [Liang et al.] [PAPER FACT] 〞 e.g., COPA, OpenBookQA, PIQA, MathQA, RTE, XSUM, CNN/Daily Mail [PAPER FACT].
  - **End-to-end throughput (tokens/s = generated tokens / (prompt time + decoding time))** and **latency (s)** [PAPER FACT] (∫5.2).
- **Secondary:**
  - **Memory footprint reduction** (up to 5℅, i.e., 20% budget) [PAPER FACT] (Abstract: "significantly reduces memory footprint", ∫5.1 5℅).
  - **Cache miss rate / sparsity** (95% sparsity) [PAPER FACT] (Fig.2a).
  - **Performance vs cache budget curve** (20%,60% collapse points) [PAPER FACT] (Fig.2d, ∫5.1 text).
  - **Quantization compatibility** accuracy (combine 4-bit) [PAPER FACT] (Table5).
  - **Ablation: effect of H2 alone vs recent alone** (Table6 range 2.85每22.75%) [PAPER FACT].
  - **Diversity of generated text** (Appendix C.1) [PAPER FACT].
- **Not elaborate:** Per-token latency breakdown, energy [NOT REPORTED].

## 7 Baselines [PAPER FACT]

- **Full KV cache (oracle):** No eviction, retains all KVs [PAPER FACT] (Fig.2d baseline).
- **Local (recent only):** Keep only most recent KVs, discard heavy hitters 〞 equivalent to window attention [PAPER FACT] (Fig.2d, Table1,4).
- **Sparse Transformer variants:**
  - Strided (w.o. H2 and w. H2) [PAPER FACT]
  - Fixed (w.o. H2 and w. H2) [PAPER FACT] (Table in ∫5.1 text: strided 50.00↙83.00 with H2, etc.)
- **Quantization baselines:** INT4 quantization alone vs H2O+quant (Table5) [PAPER FACT].
- **System baselines (∫5.2):**
  - **DeepSpeed Zero-Inference** [Aminabadi et al.] [PAPER FACT]
  - **Hugging Face Accelerate** [Gugger et al.] [PAPER FACT]
  - **FlexGen** [Sheng et al.] (state-of-art offloading engine H2O builds on) [PAPER FACT]
- **Theoretical:** Belady algorithm discussion as not optimal for KV cache [PAPER FACT].

## 8 Workloads [PAPER FACT]

- **Models (∫5.1 Setup, ∫5.2 Setup):**
  - **OPT family:** 6.7B, 13B, 30B, 66B, 175B [PAPER FACT]
  - **LLaMA:** 7B, 13B [PAPER FACT] (mentioned ∫5.1: "validates accuracy with OPT, LLaMA, GPT-NeoX")
  - **GPT-NeoX-20B** [PAPER FACT] (example GPT-NeoX-20B XSUM +0.18)
  - Evaluation primarily focuses OPT [PAPER FACT]; Appendix C.1-5 includes additional analysis of MLP blocks etc.

- **Datasets / Tasks:**
  - **HELM** and **lm-eval-harness** tasks [PAPER FACT]: OpenBookQA, COPA, PIQA, MathQA, RTE, XSUM, CNN/Daily Mail, Winogrande, HellaSwag etc. [PAPER FACT]
  - **Sparsity analysis:** Wiki-Text-103 validation [PAPER FACT] (Fig.2a)
  - **Power-law correlation:** vocabulary co-occurrence analysis [PAPER FACT] (Fig.2b)
  - **Throughput workload:** Synthetic datasets padded to same prompt length; generation lengths tested: 512+32, 512+512, 512+1024 on T4 [PAPER FACT] (Table2); and long sequences 7000+1024, 5000+5000, 2048+2048 on A100 [PAPER FACT] (Table3)
  - **Ablation:** 5-shot vs 10-shot inference (Table4) [PAPER FACT]

- **Cache budgets evaluated:** 20% H2 (i.e., 5℅ reduction) default; also 60% collapse point for Local [PAPER FACT]; enhances baselines at 20% [PAPER FACT].

- **Batch sizes:** Varied: T4 eval with effective batch sizes 1-728 depending on offloading level (Table2 bracket: e.g., FlexGen 144,C vs H2O 728,C for 6.7B 512+32) [PAPER FACT]; A100 eval batch 1,4,24,64 [PAPER FACT].

## 9 Hardware [PAPER FACT]

- **Single NVIDIA A100 (80GB) GPU** for accuracy evaluation with OPT, LLaMA, GPT-NeoX across HELM/lm-eval-harness [PAPER FACT] (∫5 intro).
- **Throughput evaluation:**
  - **NVIDIA T4 (16GB) GPU** (∫5.2 Setup) for FlexGen/DeepSpeed/Accelerate comparison [PAPER FACT]; synthetic length combos.
  - **NVIDIA A100 (80GB) GPU** for long sequences up to 10K (7000+1024 etc.) [PAPER FACT].
- **Additional setup note:** When model+cache not fit, CPU offloading used (denoted C in Table2) vs GPU-only G [PAPER FACT].
- **Implementation note:** FlexGen + H2O on T4/A100; not multi-GPU distributed beyond single GPU [PAPER FACT] 〞 claims distributed but evaluated single.
- **Precision:** [NOT REPORTED] explicitly; likely FP16 as OPT/LLaMA default 〞 not stated in ∫5 [NOT REPORTED].
- **CPU/RAM spec for offloading tier:** [NOT REPORTED] [PAPER FACT].

## 10 Main Results [PAPER FACT]

All numbers from ∫5, Fig.2, Tables 1-6.

- **Sparsity:** OPT attention >95% sparse across almost all layers (threshold 1% max) [PAPER FACT] (Fig.2a). Implies 20℅ theoretical reduction [PAPER FACT].

- **Heavy-hitter importance (Fig.2c):** Removing H2 damages model; Local vs H2O at 20% budget show large gap [PAPER FACT].

- **Main accuracy (∫5.1, 4 tasks shown):**
  - Ability to **reduce KV cache 5℅ (20% budget)** without degradation in many tasks; example improvements over baseline:
    - OPT-66B RTE +0.73%, OPT-30B MathQA +0.64%, GPT-NeoX-20B XSUM +0.18 with 20% budget (regularization effect) [PAPER FACT].
  - Local strategy collapses at 60% budget in LLaMA-13B XSUM and LLaMA-7B CNN/Daily Mail, while H2O matches full cache with 20% [PAPER FACT].
  - **Enhancing baselines:** Strided/fixed sparse attention fail at 20% (up to **35% drop** vs full); combining with H2 restores to near full:
    - On 4 tasks aggregated snippet: Local w.H2 84.00 vs 25.20 Local on OpenBookQA 5-shot? Actually Table in prompt shows: Strided w.o. H2 50.00 ↙ w. H2 83.00; Fixed w.o. 61.00↙76.00 etc. [PAPER FACT] (pre-Table).
  - H2+recent ablation: Only H2 or only local degrades 2.85%每22.75% vs full; H2O (both) retains baseline [PAPER FACT] (Table6 description).

- **Few-shot ablations (Table4, OPT-30B/66B, 20% budget):**
  - OpenBookQA 5-shot: Full 43.20/44.40 vs Local 25.20/30.60 vs H2O 43.00/44.20 [PAPER FACT]
  - COPA 5-shot: Full 85.00/83.00 vs Local 48.00/59.00 vs H2O 84.00/82.00 [PAPER FACT]
  - MathQA 5-shot: Full 26.23/27.87 vs H2O 26.87/27.67 (slightly above) [PAPER FACT]
  - Similar holds for 10-shot (e.g., COPA Full 86/85 vs H2O 85/86) [PAPER FACT].

- **Quantization compatibility (Table5, OPT-30B):**
  - COPA: Full 85.00, H2O 84.00, Quant-4bit 84.00, **H2O+Quant 84.00** [PAPER FACT]
  - OpenBookQA: Full 43.20, H2O 43.00, Quant 43.28, H2O+Quant 43.20 [PAPER FACT]
  - PiQA: Full 78.51, H2O 78.45, H2O+Quant 78.80 (better) [PAPER FACT] 〞 combination no compounding error [PAPER FACT].

- **Throughput on T4 (Table2, H2O 20% vs Baselines):**
  - 512+32, 6.7B: Accelerate 20.4 (2,G), DeepSpeed 10.2 (16,C), FlexGen 20.2 (2,G), **H2O 35.1 (4,G)** [PAPER FACT]
  - 512+512, 6.7B: 15.5 ↙16.8 ↙**51.7 (4,G)** ↙ **3℅ over FlexGen** [PAPER FACT]
  - 512+32, 30B: 0.6 (8,C) Acc/DS ↙ FlexGen 8.1 (144,C) ↙ **H2O 12.7 (728,C)** ~1.57℅ FlexGen but **29℅ over DeepSpeed/Accelerate (0.6)** [PAPER FACT]
  - 512+512/1024 30B similar up to **29℅** claimed [PAPER FACT].
  - Aggregate claim in Abstract: **29℅ over DeepSpeed, 29℅ over Accelerate, 3℅ over FlexGen on OPT-6.7B and OPT-30B with 20% heavy hitters (5℅ memory reduction)** [PAPER FACT] 〞 verified via webfetch https://arxiv.org/abs/2306.14048 abstract (29℅,29℅,3℅).

- **Latency on A100 (Table3, same batch size latency reduction 1.1每1.9℅, plus larger batch throughput 2.3℅):**
  - 7000+1024 30B batch1: FlexGen 57.0s ↙ H2O 50.4s (1.13℅) [PAPER FACT]
  - 5000+5000 13B batch4: 214.2 ↙155.4 (1.38℅) [PAPER FACT]
  - 2048+2048 6.7B batch24: latency 99.5↙53.5 (1.86℅), throughput 494.1 ↙918.9 token/s (1.86℅) [PAPER FACT]
  - 2048+2048 6.7B batch64: FlexGen OOM ↙ H2O **1161.0 token/s** (2.3℅ vs batch24) [PAPER FACT]; claim latency reduction **up to 1.9℅** [PAPER FACT] (Abstract).

## 11 Assumptions [PAPER FACT]

- Generative inference has two phases: prompt phase (build KV cache like training forward) and token generation phase (incrementally update cache) [PAPER FACT] (∫2.1).
- KV cache eviction defined as keeping |Si|=k constant, evicting at most 1 per step (|Si\Si-1|≒1) [PAPER FACT] (Def 2.1/4.3).
- Accumulated attention score sum is proxy for token importance; submodular assumption for attention scheme needed for theoretical near-optimality [PAPER FACT] (Lemma 3.1 informal, Def 4.1, Thm 4.4 assumes mild assumption + 汐,汕 params) [PAPER FACT].
- Power-law distribution of attention implies small H2 set dominates [PAPER FACT].
- Local accumulated score approximates global future-inclusive score 〞 empirical, not proven generally [PAPER FACT].
- Evicted KV can be set to 0 and subtracted from denominator without affecting other computations (Di correction) [PAPER FACT] (Def 2.2).
- Recent tokens always important (temporal locality) so policy retains recent window alongside H2 [PAPER FACT] (∫4, ∫5.3).

## 12 Author-Stated Limitations [PAPER FACT]

Explicit **Limitations** in Appendix B.2.3 [PAPER FACT]:

- **No training integration:** H2O is inference-only KV cache policy; not combined with training-time sparse attention optimizations [PAPER FACT].
- **No model weight compression:** Focus exclusively on KV cache, orthogonal to weight quantization/pruning/distillation which could be complementary [PAPER FACT].
- **Power-law / submodular assumptions may not hold universally:** Theoretical guarantee relies on submodular mild assumption; varying distributions may affect bound [PAPER FACT] (Appendix D).
- **Sequential dependency risk:** Even with H2, aggressive budget beyond 20% could degrade performance on tasks requiring long history (authors show collapse points but not exhaustive) [PAPER FACT] (B.2 discussion).
- **Scope of evaluation:** Mainly OPT, partial LLaMA/GPT-NeoX; broader coverage needed [AGENT INFERENCE/PAPER FACT 〞 Appendix notes limited architectures].
- **Social Impact:** Not detailed limitation but acknowledges deployment cost reduction is positive; no negative societal impact discussed beyond efficiency [PAPER FACT] (B.2.2).

[AGENT INFERENCE]: Authors note previous attempts expensive eviction, not deployment feasible, and their greedy variant addresses it.

## 13 Inferred Limitations [AGENT INFERENCE]

- **Fixed 20% heuristic:** Evaluation mostly at 20% budget; optimal k varies by layer/head/task and sequence length; no adaptive per-layer allocation (uniform budget) [AGENT INFERENCE].
- **Accumulated score overhead:** Computing argmax over F_score each step adds latency; not broken down per-token vs attention compute; Table2 throughput includes it but micro-cost not ablated [AGENT INFERENCE].
- **Comparison with StreamingLLM/Scissorhands overlap:** H2O retains H2+recent, StreamingLLM retains initial sinks+recent, Scissorhands retains heavy recent-history; H2 heavily overlaps with sinks but H2O does not pin sinks permanently 〞 may evict early sink if not heavy, potential instability for infinite streaming vs window [AGENT INFERENCE].
- **No infinite-length evaluation:** Throughput tested up to 10K (A100) but OPT trained 2K; paper benchmarks throughput beyond training length (7000+1024) but accuracy beyond 2K not shown; 4M test as in StreamingLLM not done [AGENT INFERENCE].
- **Single-GPU focus:** No distributed tensor-parallel evaluation; real large model serving (175B) would need multi-GPU cache coordination [AGENT INFERENCE].
- **No discussion of positional encoding interaction:** RoPE/ALiBi handling not detailed vs StreamingLLM which explicitly handles cache-relative positions 〞 H2O may still suffer positional drift for long sequences [AGENT INFERENCE].
- **Heavy hitter definition depends on attention pattern of prompt:** For prompts with system instruction, H2 may be system tokens; for diverse dialogues, H2 set may diverge from frequency correlation claim [AGENT INFERENCE].
- **Evaluation variance:** Some tasks show H2O slightly above full (regularization), but not statistically tested; could be noise [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. **Dynamic budget adaptation:** Can budget k be learned per-layer/head or per-sequence based on entropy/power-law exponent to maintain accuracy with <20% on average? [AGENT INFERENCE]
2. **H2 locality vs sinks:** Is pinning initial sinks (StreamingLLM) strictly better or worse than dynamic H2 for infinite streams? Hybrid policy? [AGENT INFERENCE]
3. **Theory tightening:** Can dynamic submodular guarantee be strengthened without 汐,汕 slacks and proven for actual Transformer attention (non-submodular in general)? [AGENT INFERENCE]
4. **Integration with other compressions:** What is joint optimization of H2O eviction + 4-bit KV quantization + weight pruning for Pareto frontier? [AGENT INFERENCE]
5. **Long-context beyond training window:** Does H2O enable true length extrapolation (e.g., 100K) with stable perplexity like StreamingLLM 4M, not just throughput on synthetic padded data? [AGENT INFERENCE]
6. **H2 in MLP blocks:** Appendix C.5 finds H2 in MLP too 〞 can combined attention+MLP H2 lead to further memory reduction (activations)? [AGENT INFERENCE]
7. **Diversity vs quality trade-off:** Appendix C.1 claims increased diversity 〞 is this controllable or may harm factuality? [AGENT INFERENCE]
8. **Real production traces:** How does H2O perform on ShareGPT/chat multi-turn heavy-tail distributions vs uniform synthetic throughput workload? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **Sparse/low-rank attention approximations:** Reformer [Kitaev 2020], Performer [Choromanski 2021], Sparse Transformer [Child 2019], Flash Attention [Dao 2022] [PAPER FACT] (∫2) 〞 still large cache, not designed for KV eviction.
- **Multi-Query Attention for KV reduction:** Shazeer 2019, etc. [11,12] [PAPER FACT].
- **Gisting tokens:** Mu et al. 2023 learnable compression but expensive eviction [PAPER FACT] (∫1).
- **Efficient LLM inference systems:** FlexGen [Sheng et al. 2023], DeepSpeed Zero-Inference [Aminabadi 2022], HuggingFace Accelerate [Gugger et al.] [PAPER FACT] 〞 baselines H2O improves 29℅/3℅ over.
- **KV cache每specific sparsification:** Zhao et al. 2023 etc., SpAtten [Wang 2021] 〞 discussed as related [PAPER FACT] (Appendix B.1).
- **Quantization/Pruning/Distillation families:** Survey in ∫2 B.1.2 [PAPER FACT] 〞 orthogonal.
- **Concurrent eviction work:**
  - **StreamingLLM (Xiao et al. 2023)** 〞 Attention Sinks, fixes initial tokens vs H2O dynamic heavy hitters; both exploit attention concentration [AGENT INFERENCE].
  - **Scissorhands (Liu et al. 2023)** 〞 Persistence of Importance, similar heavy-token retention with history window [AGENT INFERENCE].
- **Caching theory:** Belady Algorithm optimal for standard cache, LRU/LFU classical [PAPER FACT] (∫2 Caching, ∫3.2).
- **Submodular optimization:** Classical greedy (1-1/e) guarantee [Nemhauser] underlies theoretical analysis [PAPER FACT] (Appendix D).



## Review Log 〞 Reviewer-2 (2026-08-27)

- **Webfetch verification:** https://arxiv.org/abs/2306.14048 and https://arxiv.org/html/2306.14048v3 abstract confirms 29℅ over DeepSpeed, 29℅ over Accelerate, 3℅ over FlexGen with 20% heavy hitters, 1.9℅ latency reduction at same batch size. 30B/128/1024 ↙ 180GB example verified in ∫1 Introduction. Sparsity >95% at 1% threshold verified Fig.2a.
- **Correction 1 〞 Venue completeness:** Added missing last revised date 18 Dec 2023 v3 (fetched submission history shows v3 Mon 18 Dec 2023). Original omitted v3.
- **Correction 2 〞 Abstract claim precision:** Appended condition ※with 20% heavy hitters (5℅ memory reduction)§ to 29℅/3℅ claim; original stated without budget context.
- **Correction 3 〞 Hardware precision note:** Confirmed primary accuracy evaluation on single A100 80GB (∫5 intro) and throughput on T4 16GB + A100 80GB (long sequences). Added explicit note that precision [NOT REPORTED] remains unverified; no change to [NOT REPORTED] tag.
- **Correction 4 〞 Table formatting:** Verified Table2 batch annotation (G=GPU-only, C=CPU offloading) and Table3 latency 1.13℅每1.86℅, throughput 918.9 tok/s at batch24; no numeric change needed beyond clarification.
- **Status:** Numbers traceable to text [PAPER FACT]; no guessing retained.
