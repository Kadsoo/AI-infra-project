# Paper Metadata

- **Title:** GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM [PAPER FACT]
- **Authors:** Hao Kang*, Qingru Zhang* (equal), Souvik Kundu, Geonhwa Jeong, Zaoxing Liu, Tushar Krishna, Tuo Zhao (correspondence hkang342@gatech.edu etc.) [PAPER FACT] ¡ª Georgia Tech, Intel, Univ. of Maryland [PAPER FACT]
- **Venue:** Preprint arXiv:2403.05527 [cs.LG,cs.AI,cs.CL], Submitted 8 Mar 2024 v1, rev 30 Sep 2024 v4 [PAPER FACT]; non-exclusive license [PAPER FACT] ¡ª verified via webfetch https://arxiv.org/html/2403.05527v4 (near-lossless 2-bit, up to 24.42% over SOTA, 2.39¡Á memory, 2.1¡Á~5.07¡Á throughput)
- **DOI/URL:** https://arxiv.org/abs/2403.05527 / https://doi.org/10.48550/arXiv.2403.05527 / HTML https://arxiv.org/html/2403.05527v4 [PAPER FACT]
- **Code:** https://github.com/HaoKang-Timmy/GEAR [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2403.05527 + https://arxiv.org/html/2403.05527v4 [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers traceable else [NOT REPORTED].

## 1 Problem [PAPER FACT]
Efficient serving of LLMs for generative inference relies on KV caching to avoid recomputation, storing Key/Value tensors and reusing them [PAPER FACT] (¡ì1,¡ì2). However KV memory grows rapidly with model size and sequence length, making inference memory-bound and limiting throughput [PAPER FACT] (Abstract,¡ì1). Example 30B LLM, input 1024, batch 128 ¡ú KV cache up to 180 GB [PAPER FACT] (¡ì1 citing Zhang 2023b). Offloading to CPU/NVMe (DeepSpeed Inference Aminabadi 2022, FlexGen Sheng 2023) still incurs PCIe bottleneck [PAPER FACT] (¡ì1).

## 2 Motivation [PAPER FACT]
- Token-dropping (H2O Zhang 2023b, Scissorhands Liu 2023, Ge 2023) and quantization (FlexGen Sheng 2023 per-token group-wise, KIVI Liu 2024, KVQuant Hooper 2024 per-channel K / per-token V) are two families [PAPER FACT] (¡ì1).
- Existing methods near-lossless on simple NLU but deteriorate on **complex generative tasks requiring longer responses/reasoning** (math Cobbe 2021, CoT Wei 2023, BBH Suzgun 2022) at high ratio (4-bit/2-bit or dropping >50% tokens) [PAPER FACT] (¡ì1 footnote: compression ratio = FP16 size / compressed size).
- Root cause: non-trivial approximation error compounded autoregressively each step, deviating logits (Fig.1a error LLaMA3-8B GSM8k-CoT 2-bit, Fig.1b logit diff, Fig.1c accuracy vs error) [PAPER FACT] (¡ì1).
- Simple tasks tolerate large error (few tokens, few important tokens); CoT prompts dense correlated info, small error magnified [PAPER FACT] (¡ì1).
- Goal: efficient error-reduction framework augmenting any quantization to achieve near-lossless at high ratio, orthogonal plug-and-play [PAPER FACT] (Abstract,¡ì3).

## 3 Bottleneck [PAPER FACT]
1. Quantization error at ultra-low precision (2-bit) with coarse grouping large [PAPER FACT] (¡ì2,¡ì3).
2. Outlier entries with very large magnitudes widen ¦¤=(max-min)/(2^b-1), inflating error for majority [PAPER FACT] (¡ì3 citing Kim 2023, Xiao 2023).
3. Coherent vs incoherent residual components: R = X-(D^ + S) has shared structure vs sparse outliers; single technique insufficient (Fig.2a error per method still high) [PAPER FACT].
4. Spectrum of R_h drops rapidly ¡ú coherent component captured by top singular vectors shared among tokens (Fig.2b) [PAPER FACT].
5. Iterative alternating optimization (LoftQ Li 2023) minimizing Eq.3 accurate but latency unacceptable [PAPER FACT] (¡ì3).
6. Fine-grained grouping (KIVI g=64) ¡ú many FP16 scales/zero-points + residual buffer multiple of group size (128) ¡ú overhead; coarse grouping reduces overhead but increases error [PAPER FACT] (¡ì2).
7. Streaming nature: new tokens generated autoregressively, need buffered compression [PAPER FACT] (¡ì3 Streaming Buffer).

## 4 Core Idea [PAPER FACT]
**GEAR = Quantized backbone D? + Low-rank L + Sparse S to minimize ||X - D? - L - S||_F, decoupling coherent vs incoherent error, plus streaming buffer. Lite GEAR-L uses only L [PAPER FACT].**
- Objective min_{D?,L,S} ||X - D? - L - S||_F for X¡Ê{K_t,V_t}¡ÊR^{n¡Ád} [PAPER FACT] (Eq.3).
- (i) D?: Apply existing quantization to majority (~98%) entries of similar magnitude to ultra-low precision [PAPER FACT] (Abstract). Choose efficient **KCVT backbone**: per-channel Key (group size n per channel) and per-token Value (group size d per token) ¡ª coarse per-vector, no fine-grained grouping, greatly reducing scale/zero overhead vs KIVI [PAPER FACT] (¡ì2,¡ì4). Alternative **KIVI backbone** (g=64 per-channel K/per-token V fine-grained, residual nb=64) used for 2-bit where KCVT degenerates [PAPER FACT] (¡ì4).
- (ii) S = Filter_s(X): Per-vector outlier filtering ¡ª extract both top s/2% max and bottom s/2% min per channel (if X=K) or per token (if X=V), store FP16 sparse, else 0 (Eq.4) [PAPER FACT] (Eq.4). Then D? = Quant_b^{(scheme)}(X - S) (Eq.5) [PAPER FACT] (Eq.5). Fix **s=2%** (1% top+1% bottom per vector) [PAPER FACT] (¡ì4). Represented via two index vectors + value vector (3¡Á per non-zero) [PAPER FACT] discussion.
- (iii) L: Head-wise low-rank of residual R = X-(D?+S)¡ÊR^{n¡Ád}. Reshape to H heads: R_h¡ÊR^{n¡Ád_H}, d_H=d/H, exploiting diverse per-head channel ranges (Tenney 2019, Voita 2019) [PAPER FACT] (¡ì3). R_h ¡Ö ¦² ¦Ò_i u_i m_i^T; L_h = A_h B_h^T = SVDSolver_r(R_h), A_h¡ÊR^{n¡Ár}, B_h¡ÊR^{d_H¡Ár}, r<<n,d_H (e.g. n=1024 d_H=128 ¡ú r=4 sufficient) [PAPER FACT] (Eq.6). Solver: power iteration (Vogels 2019) with QR final iter (Appendix 8 Alg.2) [PAPER FACT] (¡ì3, App.8). L=Concat(L_h), batch-wise & head-wise [PAPER FACT].
- Synergy: D? entry-wise, L vector-wise coherent, S sparse incoherent; fully exploit potentials [PAPER FACT] (Abstract,¡ì3).
- Streaming Buffer: Store new KV vectors FP16 in small buffer B size **n_b=20**; when full every n_b steps compress buffered tokens; low-rank only on buffered tokens with ultra-low **r=2** during decoding [PAPER FACT] (¡ì1,¡ì3). KIVI buffer must be multiple of group size; KCVT arbitrary small 20 improves speed [PAPER FACT].
- Kernel optimization: Fuse dequant with matmul via CUDA; streaming for K/V both; low-rank forward as down-projection q_h^T B_h then up (q_h^T B_h) A_h^T reducing complexity [PAPER FACT] (¡ì4).
- Plug-and-play: Orthogonal to any quantization, Fig.2c shows improvement regardless of backbone [PAPER FACT].

## 5 System Changes [PAPER FACT]
- MHA: L layers, H heads, X¡ÊR^{n¡Ád}, MHA=Concat(H^{(i)})W_o, H^{(i)}=Softmax(Q^{(i)}K^{(i)T}/¡Ìd_H)V^{(i)}, Q^{(i)}=X W_{q_i} etc., d_H=d/H [PAPER FACT] (Eq.1 ¡ì2).
- Prefill & decoding: Prefill X_0¡ÊR^{n¡Ád}¡úK_0,V_0¡ÊR^{n¡Ád} cached; each step t compute q_t,k_t,v_t¡ÊR^d, append K_t=K_{t-1}¡¬k_t, V_t=V_{t-1}¡¬v_t, attention q_t with K_t,V_t [PAPER FACT] (¡ì2).
- Group-wise quantization: per-token with g, ¦¤_i=(max-min)/(2^b-1), Quant=ceil((X_{G_i}-min)/¦¤_i) [PAPER FACT] (Eq.2). Per-channel variant similar grouping along token dim [PAPER FACT].
- GEAR Algorithm (Appendix 7 Alg.1): For each X¡Ê{K0,V0} prefill: S=Filter_s, D?=Quant, R=X-D?-S, for each head h L_h=SVDSolver_{r_p}(R_h) (r_p=4), L=Concat, replace X with D?+L+S. Decoding: for t=1..n_g if t mod n_b==0 then for X¡Ê{K_B,V_B} compute S,D?,L_h with r_g=2 and replace, append K_t=K_{t-n_b}¡¬K_B; else push k_t,v_t to buffers [PAPER FACT] (App.7).
- SVDSolver Power Iteration (App.8 Alg.2): random init A¡ÊR^{n¡Ár},B¡ÊR^{d¡Ár}; for l<L: if l==L-1 QR(B), A=X B, if l==L-1 QR(A), B=X^T A, l++ [PAPER FACT].
- Stack: PyTorch [Paszke 2019] + HuggingFace Transformers [Wolf 2019]; keep other tensors FP16 [PAPER FACT] (¡ì4). Fused CUDA kernel for dequant+matmul tiling [PAPER FACT].
- Hyperparameters fixed: s=2%, r=4 prefill, r=2 decode buffer, n_b=20; 4-bit uses KCVT, 2-bit uses KIVI (g=64 nb=64) [PAPER FACT] (¡ì4). Reason KCVT effective 4-bit, degenerates 2-bit requiring fine-grained [PAPER FACT].
- Weight compression: To maximize batch, compress weights to 8-bit bitsandbytes for efficiency tests [PAPER FACT] (¡ì4.2).

## 6 Target Metrics [PAPER FACT]
- **Primary:**
  - Accuracy on generative reasoning (CoT): **GSM8k [Cobbe 2021], AQuA [Ling 2017], BBH [Suzgun 2022]** 8-shot CoT (Fu 2023) 256 gen; prefill avg 900/1304/1021 [PAPER FACT] (¡ì4.1, App.10). Also **GSM8k 5-shot** (non-CoT 672 prefill 96 gen) and **LongBench [Bai 2023]** 21 tasks avg (F1, Rouge-L, Acc, Edit Sim, avg prefill 3642) + subsets QMSum/SAMSum/GovReport [PAPER FACT] (App.10 Table5).
  - KV size (% remaining vs FP16) averaged per dataset/model [PAPER FACT] (Tables1-2).
  - Inference efficiency: peak memory (GB), wall-clock time (s), throughput (tokens/s) vs batch size; max batch; max sequence length [PAPER FACT] (Fig.3, Table6).
- **Secondary:**
  - Approximation error ||X-compressed||_F and logit diff (Fig.1) [PAPER FACT].
  - Ablation: vary s and r (Fig.4a), token recovery p% (Fig.4b), compression ratio sweep (Fig.4c), outlier-aware vs GEAR (Table8) [PAPER FACT].
  - Time breakdown quant vs low-rank vs sparsity vs other % (Fig.3a) [PAPER FACT].
  - KV component memory distribution: quantized ints + SZ FP16 + buffer + sparsity + low-rank (Fig.6) [PAPER FACT] (App.11.3).
  - Token dropping H2O at 50% (Table10 App.14) [PAPER FACT].
- Not elaborate: perplexity WikiText2/C4 excluded due calibration need [PAPER FACT]; Energy/performance/Watt [NOT REPORTED].

## 7 Baselines [PAPER FACT]
- **FP16 16-bit full KV** baseline 100% size [PAPER FACT] (Table1).
- **Per-token group-wise quantization (FlexGen Sheng 2023)** fine-grained g=64, 4/2-bit [PAPER FACT] (¡ì4).
- **KIVI [Liu 2024]** SoTA 2-bit ¡ª per-channel Key / per-token Value fine-grained, residual n_b; tested g=64 nb=64 (also g=32 nb=128 Table2) [PAPER FACT].
- **KCVT quantization** variant of KIVI **without fine-grained** (coarse per-vector, K group n, V group d), lower overhead [PAPER FACT] (¡ì2,Baselines) ¡ª used as 4-bit backbone.
- **H2O [Zhang 2023b]** token dropping via accumulated attention, 50% KV (Table10) [PAPER FACT] (¡ì4,App14).
- **Outlier-aware KIVI s=2% (KIVI)** ablation to isolate low-rank vs sparse [PAPER FACT] (App.12 Table8).
- **GEAR-L r=4 (KCVT/KIVI)** lite only low-rank, no sparse [PAPER FACT].
- **GEAR s=2% r=4 (KCVT or KIVI g=64)** full [PAPER FACT].
- KVQuant [Hooper 2024] non-uniform calibration discussion ¡ª excluded as not plug-and-play [PAPER FACT] (¡ì4 note).

## 8 Workloads [PAPER FACT]
- **Models [PAPER FACT] (¡ì4,¡ì4.2):** LLaMA3-8B [Meta 2024], LLaMA2-7B/13B [Touvron 2023b], Mistral-7B [Jiang 2023]; efficiency uses LLaMA2-7B; error analysis uses LLaMA2-7B first-layer Value cache random GSM8k example [PAPER FACT] (Fig.2 description).
- **Datasets [PAPER FACT] (App.10 Table3-5):**
  - GSM8k 8-shot CoT: 1319 examples, prefill 900, gen 256 [PAPER FACT]
  - AQuA 8-shot CoT: 254 examples, prefill 1304, gen 196 [PAPER FACT]
  - BBH 3-shot CoT: 6511 examples (23 subsets 6.5k), prefill 1021, gen 196 [PAPER FACT]
  - GSM8k 5-shot standard: 1319 examples, prefill 672, gen 96 [PAPER FACT]
  - LongBench: 4750 examples agg, avg prefill 3642, gen 256, 21 tasks including NarrativeQA 18409 F1, Qasper 3619 F1, MultiFieldQA-en 4559 F1, HotpotQA 9151 F1, 2Wiki 4887 F1, MuSiQue 11214 F1, DuReader 15768 Rouge-L, GovReport 8734 Rouge-L, QMSum 10614 Rouge-L, MultiNews 2113 Rouge-L, VCSUM 15380 Rouge-L, TREC 5177 Acc, TriviaQA 8209 F1, SAMSum 6258 Rouge-L, LSHT 22337 Acc, PassageCount 11141 EM, Retrieval-en 9289 EM, Retrieval-zh 6745 EM, LCC 1235 Edit Sim, RepoBench-P 4206 Edit Sim ¡ª en/zh noted, 150-500 each [PAPER FACT] (Table5)
  - LongBench subsets separately QMSum/SAMSum/GovReport [PAPER FACT] (Table2)
- Compression budgets 4-bit and 2-bit evaluated; KV size % remaining: Per-token 4-bit 34.2%, KCVT 27.1%, KIVI 34.2%, GEAR-L 29.0% (KCVT), GEAR 31.0% (KCVT); 2-bit Per-token 21.7%, KIVI 21.7%, GEAR-L 23.6%, GEAR 27.6% [PAPER FACT] (Table1, Table9); GSM8k5-shot/LongBench: KIVI 38.2% (4-bit) vs GEAR-L 30.4% etc. [PAPER FACT] (Table2)
- Prompts: CoT prompts Fu 2023 containing 8-shot multi-step reasoning (Fig.7 colored red question, green common, blue answer) [PAPER FACT] (App.15).

## 9 Hardware [PAPER FACT]
- Primary: Single **NVIDIA V100 16GB** ¡ª input 1000, gen 500, LLaMA2-7B, weights 8-bit bitsandbytes to maximize batch, increase until OOM, report peak/throughput between FP16 and 2-bit KIVI/GEAR/GEAR-L; batch 1,2,3 (FP16 max3) vs 1,4,8,12,16,18 (quantized max18) [PAPER FACT] (¡ì4.2, Table6). Peak measured at batch18 on larger GPU to accommodate FP16 [PAPER FACT] (App.11.1).
- Secondary: Single **NVIDIA RTX Titan 24GB** LLaMA2-7B comparing GEAR-L Prefill / GEAR-L / GEAR (Fig.5) [PAPER FACT] (App.11.2).
- Stack: PyTorch + HuggingFace + CUDA fused kernels + FlashAttention for longer seq max length [PAPER FACT] (¡ì4.2, App.11.4).
- Precision: other tensors FP16; KV 4/2-bit; weights 8-bit only for efficiency max-batch [PAPER FACT] (¡ì4).
- CPU/RAM for offloading [NOT REPORTED] [PAPER FACT].
- Detailed numbers: FP16 batch3 120s 11.44GB 12.5 tok/s; GEAR-2bit batch18 163s 14.63GB 55.21 tok/s (max) vs KIVI 56.6 at 18; GEAR-L 63.38 at 18 etc. [PAPER FACT] (Table6 App.11.1). Titan shows 2.10¡Á throughput over FP16 [PAPER FACT] (App.11.2).

## 10 Main Results [PAPER FACT]
All numbers from ¡ì4, Tables1-2, Figures3-4, Appendices 11-13.
- **CoT hard (Table1 8-shot CoT 256 gen, Avg over 9 model¡Ádataset combos):**
  - Overall All Ave.: FP16 40.52 ¡ú Per-token g64 4-bit 31.94, KCVT 34.92, KIVI 35.05, **GEAR-L(KCVT) 40.02**, **GEAR(KCVT) 40.80** (4-bit near-lossless +0.28 over FP16); With **2-bit** Per-token 7.67 (collapse N.A. for LLaMA2-13B), KIVI 25.25, **GEAR-L(KIVI) 38.34**, **GEAR(KIVI g64) 40.20** (¡ÖFP16, +14.95 over best baseline KIVI 25.25) [PAPER FACT].
  - Per-model 2-bit vs FP16: LLaMA3-8B FP16 54.21/38.19/53.66 (GSM8k/AQuA/BBH) vs **GEAR 54.59/38.19/50.30** (avg 47.69 vs 48.69 FP16) vs **GEAR-L 52.62/38.19/51.44**, KIVI 30.17/25.36/30.92 (avg 28.82) [PAPER FACT]; LLaMA2-13B FP16 30.34/21.65/40.79 vs GEAR 30.27/23.62/39.67 vs KIVI 16.60/17.72/29.43 [PAPER FACT]; Mistral-7B FP16 42.84/35.04/47.92 vs GEAR 43.14/33.96/48.03 vs KIVI 23.35/22.44/31.28 [PAPER FACT].
  - 4-bit: LLaMA3-8B FP16 40.52 vs GEAR 40.80 best, GEAR-L 40.02 vs Per-token 31.94 / KCVT 34.92 / KIVI 35.05 [PAPER FACT].
  - Claimed improvement **up to 24.42% over SOTA baselines at 2-bit** (abstract) [PAPER FACT]; body says 14.95% avg over best baseline across models/datasets at 2-bit [PAPER FACT] (¡ì1,¡ì4.1).
- **Easy tasks without CoT (Table2):**
  - GSM8k 5-shot: LLaMA2-7B FP16 13.50 vs KIVI g64 13.41 vs GEAR-L 12.51 vs GEAR 13.19 (KIVI already near-lossless); LLaMA3-8B FP16 49.89 vs Per-token 45.64 vs KCVT 43.14 vs KIVI 48.37 vs **GEAR 49.43** vs GEAR-L 47.23 [PAPER FACT]; 2-bit: Per-token 0.08/0.83 collapse, KIVI g32 nb128 12.74/42.54, GEAR-L 12.63/47.01, **GEAR 13.04/49.96** (+7.42 over KIVI on 8B 42.54) [PAPER FACT].
  - LongBench w. LLaMA2-7B 21 tasks avg: FP16 26.82 (100%) vs Per-token 27.31 (31.6%) KCVT 26.06 (25.7%) KIVI 27.58 (31.6%) GEAR-L 27.65 (27.3%) **GEAR 27.80** (29.3%) at 4-bit; At **2-bit** Per-token 27.69 (17.5%) KIVI (g32) 27.83 (19.7%) GEAR-L 27.90 (19.1%) GEAR 25.48 (23.1%) ¡ª 2-bit per-token already 27.69 near-lossless, GEAR on par but per-task GEAR better on QMSum 20.59 vs 20.50, SAMSum 43.22 vs 42.71, GovReport 27.73 vs 25.72 [PAPER FACT].
  - Conclusion token-quant already near-lossless on easy tasks (27.7), GEAR still improves 49.96 vs 42.54 on hard 5-shot [PAPER FACT].
- **Inference efficiency (Fig.3, Table6, ¡ì4.2):**
  - Peak memory reduction **up to 2.39¡Á** vs FP16 at batch18 [PAPER FACT] (Abstract 2.39¡Á, ¡ì1).
  - Batch size max from **3 (FP16) ¡ú 18 (GEAR/KIVI)** on V100 16GB (6¡Á) [PAPER FACT] (Fig.3b).
  - Throughput **2.10¡Á ~5.07¡Á** vs FP16 (abstract 2.1¡«5.07¡Á, ¡ì4.2 Fig.3c up to 5.07¡Á) [PAPER FACT]; Detailed FP16 batch3 12.5 tok/s vs GEAR-L batch18 63.38 (5.07¡Á), GEAR 55.21 (4.42¡Á), KIVI 56.6 (4.53¡Á) [PAPER FACT] (Table6). RTX Titan 24GB shows **2.10¡Á** over FP16 [PAPER FACT] (App.11.2).
  - Max sequence length (App.11.4 Table7): Batch1 FP16 max 5319 vs **GEAR(KIVI) 7291** (+2k ~37% longer) [PAPER FACT].
  - Time breakdown (Fig.3a): low-rank+sparse negligible; primary model forward; quant fused [PAPER FACT].
  - KV component distribution (Fig.6): KCVT small buffer overhead due large group n/d; KIVI larger residual+SZ due g=64 [PAPER FACT] (App.11.3).
- **Ablation (Fig.4, Table8, ¡ì4.3):**
  - Sparsity s and rank r (Fig.4a LLaMA3-8B GSM8k-CoT 2-bit): s=2% and r=4 adequate near-lossless; further increase not significant but adds overhead; discarding low-rank significantly degenerates, discarding sparse hurts less (grouping partially remedies) [PAPER FACT].
  - Outlier-aware vs GEAR (Table8): 2-bit Outlier-A s2%(KIVI) improves over KIVI (e.g. LLaMA3-8B GSM8k 30.17¡ú36.01, AQuA 25.36¡ú36.22, BBH 30.92¡ú36.59) but far from GEAR (52.99-54.59) [PAPER FACT] (App.12 Table8).
  - Apply error reduction to p% tokens (Fig.4b): decreasing p% recent prefill tokens degrades GEAR-L [PAPER FACT].
  - Different compression ratios (Fig.4c): GEAR/GEAR-L consistently outperform across remaining sizes [PAPER FACT].
  - Token dropping comparison (Table10): LLaMA2-7B GSM8k CoT FP16 16.33 vs H2O 50% KV 6.82 vs **GEAR(KCVT) 32.4% KV 16.14** near-lossless [PAPER FACT] ¡ª dropping cannot achieve high ratio on reasoning.

## 11 Assumptions [PAPER FACT]
- KV cache streaming sequential; optimization-based GPTQ unsuitable due overhead [PAPER FACT] (¡ì3).
- Round-to-nearest uniform group-wise quantization with ¦¤_i and zero-point min sufficient; tuning-free no calibration [PAPER FACT] (¡ì2 Eq.2).
- Key cache channel-wise outlier pattern persists fixed channels ¡ª per-channel Key isolation beneficial [PAPER FACT] (¡ì2).
- Top/bottom s/2% per-vector outlier extraction captures incoherent error; remaining entries similar magnitudes quantizable ultra-low precision [PAPER FACT].
- Residual R after D?+S low-rank coherent shared among tokens, rapidly decaying spectrum; rank 4 (prefill) and 2 (decode buffer) sufficient independent of n [PAPER FACT] (Fig.2b,¡ì3).
- Heads encode diverse info, head-wise decomposition necessary (Tenney, Voita, Zhang) [PAPER FACT].
- Small streaming buffer n_b=20 sufficient to amortize latency with negligible memory [PAPER FACT].
- Exact key/value passed to next layers during prefill while only quantized retained valid [PAPER FACT] (Fig.3).
- Power iteration Solver few iterations approximates top-r SVD accurate enough [PAPER FACT] (App.8).
- FP16 other tensors isolated KV effect; 8-bit weights for max-batch does not confound relative comparison [PAPER FACT].

## 12 Author-Stated Limitations [PAPER FACT]
No explicit Limitations but ¡ì6 + ¡ì6.1 Limitations and Broader Societal Impact states:
- **Uniform rank allocation:** identical rank for every Key/Value matrix, ignoring varying importance across layers/heads [PAPER FACT] (¡ì6.1). Empirically adaptive budget across matrices could further improve [PAPER FACT].
- Future work: leave adaptive low-rank allocation as future exploration [PAPER FACT].
- Broader impact positive: reducing inference energy and carbon footprint, democratizing LLM serving; improving efficiency across hardware; increasing user responsibility [PAPER FACT] (¡ì6.1).
- Scope: focus on complex generative reasoning; simple tasks already handled by baselines but GEAR still applicable [PAPER FACT] (¡ì4.1).

## 13 Inferred Limitations [AGENT INFERENCE]
- Sparse CSR overhead at s=2% (2%¡Á3¡Ö6% extra) + index compute may dominate large batches 128, n=10k and fragments memory [AGENT INFERENCE].
- Power iteration per prefill + per buffer adds latency tail; worst-case long prompt 8k with many heads/layers could be tens of ms, not broken out vs FlashAttention tiling [AGENT INFERENCE].
- Uniform s=2% and r=4 across layers/heads; early sparse vs late dense differences not exploited; adaptive likely better as authors admit [AGENT INFERENCE].
- No integration with token eviction (H2O/SnapKV/PyramidKV) ¡ª joint token¡Ábit Pareto not explored [AGENT INFERENCE].
- Limited model scale: tested only up to 13B (LLaMA2-13B) and 8B; no 70B+ or MoE where KV dominates more and rank requirement may change [AGENT INFERENCE].
- No extreme long context (32k-128k): LongBench avg 3642, max length test only 7k vs 5k; not show 100k retrieval as KIVI NIAH [AGENT INFERENCE].
- Hardware narrow: only V100 16GB (outdated) and RTX Titan 24GB via custom CUDA; no A100/H100/TPU or multi-GPU tensor-parallel paged KV (vLLM) evaluation [AGENT INFERENCE].
- Comparison with calibration-based KVQuant excluded intentionally (Hessian calibration) but that method shows near-lossless 3-bit with 1% outliers; GEAR may underperform vs calibration at 2-bit perplexity [AGENT INFERENCE].
- Decoding buffer recompression every 20 steps causes jitter; per-20 outlier sorting O(n log n) + SVD may cause latency variance for streaming SLAs [AGENT INFERENCE].
- Sorting per channel/token for outlier extraction overhead not broken out separate from sparsity time [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]
1. Adaptive rank & sparsity per-layer/head/group based on error spectrum or attention entropy? [AGENT INFERENCE]
2. Joint compression token¡Ábit¡Ácross-layer sharing (MiniCache) Pareto optimal? [AGENT INFERENCE]
3. Extreme 1-bit ternary (1.58-bit) with larger rank still near-lossless? [AGENT INFERENCE]
4. Theoretical bound on autoregressive generation deviation as function of ||R-L|| plus quantization step and steps compounding? [AGENT INFERENCE]
5. Distributed serving: partitioning L and S across tensor/pipeline parallel workers and integrating with paged attention (vLLM) without fragmenting? [AGENT INFERENCE]
6. Kernel fusion with FlashAttention-3: fused dequant+low-rank+sparse matmul tiling within Flash kernel to avoid separate HBM passes? [AGENT INFERENCE]
7. Dynamic buffer sizing adapting to generation length (larger for long story, smaller for interactive chat)? [AGENT INFERENCE]
8. Cross-task generalization: s=2% r=4 tuned on GSM8k-CoT transfer to code/summarization where Fig.6 suggests different distribution? Table2 LongBench 2-bit GEAR slightly worse (25.48 vs 27.83) ¡ª why? [AGENT INFERENCE]
9. Compatibility with GQA/MQA: Mistral/LLaMA3 use GQA tested but Falcon MQA not tested where KV smaller; would s/r need scaling? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]
- Weight quantization: GPTQ [Frantar 2023], SqueezeLLM Dense-and-Sparse [Kim 2023] ¡ª dense-quant + sparse outliers inspiration [PAPER FACT] (¡ì3,¡ì5). SmoothQuant [Xiao 2023] (8-bit activation/KV) and Atom [Zhao 2023] (4-bit KV channel-wise) limited to 8/4-bit simple tasks [PAPER FACT] (¡ì5,¡ì9).
- KV-specific quantization: **FlexGen [Sheng 2023]** per-token group-wise 4-bit baseline [PAPER FACT]; **KIVI [Liu 2024]** per-channel K/per-token V 2-bit SoTA + KCVT variant [PAPER FACT]; **KVQuant [Hooper 2024]** non-uniform calibration 3-bit excluded [PAPER FACT] (¡ì4 note).
- Token dropping/eviction: **H2O [Zhang 2023b]** heavy-hitter via accumulated attention [PAPER FACT] (Table10); **Scissorhands [Liu 2023], FastGen [Ge 2023], SparQ [Ribar 2023]** ¡ª alternative line but sensitive on reasoning and FlashAttention-incompatible [PAPER FACT] (¡ì5,¡ì9).
- Efficient inference systems: **DeepSpeed Inference [Aminabadi 2022], FlexGen** offloading; **FlashAttention [Dao 2022]** tiling [PAPER FACT] (¡ì1,¡ì4.2,¡ì5).
- LLM families: LLaMA [Touvron 2023a], LLaMA2 [Touvron 2023b], Mistral 7B [Jiang 2023], LLaMA3 [Meta 2024], PaLM [Chowdhery 2022], OPT [Zhang 2022], GPT-4 [OpenAI 2023] [PAPER FACT].
- Reasoning & CoT: Chain-of-thought [Wei 2023], Hub [Fu 2023], GSM8k [Cobbe 2021], AQuA [Ling 2017], BBH [Suzgun 2022] [PAPER FACT].
- LongBench [Bai 2023] bilingual multitask long-context benchmark [PAPER FACT].
- LoftQ [Li 2023] iterative alternating optimization for Eq.3 but rejected for latency [PAPER FACT] (¡ì3).
- PowerSGD [Vogels 2019] low-rank gradient compression ¡ª power iteration solver [PAPER FACT] (App.8).
- Chunk-based eviction distinct from quantization: ChunkKV [Liu 2025, arXiv:2502.00299] semantic-preserving chunk eviction vs GEAR quantization+low-rank+sparse [AGENT INFERENCE].
- Complementary: vLLM PagedAttention [Kwon 2023] and StreamingLLM could theoretically combine [AGENT INFERENCE].



## Review Log ¡ª Reviewer-2 (2026-08-27)

- **Webfetch verification:** https://arxiv.org/html/2403.05527v4 ¡ª abstract near-lossless at 2-bit, up to 24.42% over SOTA, 2.39¡Á peak memory, 2.1¡Á~5.07¡Á throughput verified; ¡ì4.1 Table1 40.52 FP16 ¡ú GEAR 40.80 (4-bit) and 40.20 (2-bit) vs KIVI 25.25 (+14.95) verified; ¡ì4.2 V100 16GB max batch 3¡ú18 (6¡Á) and Table6 FP16 batch3 12.5 tok/s vs GEAR-L 63.38 (5.07¡Á) verified; error reduction via D?+L+S with s=2% r=4/2 nb=20 verified ¡ì3.
- **Correction 1 ¡ª Venue enrichment:** Added abstract key numbers to venue for traceability.
- **Correction 2 ¡ª Backbone clarification:** Confirmed 4-bit uses KCVT (per-vector coarse) and 2-bit uses KIVI g=64 nb=64; original already correct, added explicit backbone tag verification.
- **Correction 3 ¡ª Throughput precision:** Verified V100 16GB vs RTX Titan 24GB distinction (V100 5.07¡Á max, Titan 2.10¡Á); retained.
- **Correction 4 ¡ª KV size %:** Verified KV sizes 34.2% (Per-token 4-bit), 27.1% KCVT, 31.0% GEAR etc. in Table1; Table2 easy tasks 4-bit/2-bit sizes ¡ª retained.
- **Status:** Numbers traceable [PAPER FACT]; minor enrichment only.
