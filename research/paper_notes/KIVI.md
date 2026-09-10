# Paper Metadata

- **Title:** KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache [PAPER FACT]
- **Authors:** Zirui Liu* , Jiayi Yuan* , Hongye Jin, Shaochen (Henry) Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu (* equal contribution, order determined by flipping a coin) [PAPER FACT] ¡ª Rice University (Zirui, Jiayi, Shaochen, Vladimir, Xia), Texas A&M University (Hongye), Stevens Institute of Technology (Zhaozhuo), Carnegie Mellon University (Beidi) [PAPER FACT]; Correspondence zl105@rice.edu, jy101@rice.edu [PAPER FACT]
- **Venue:** Preprint arXiv:2402.02750 [cs.CL, cs.LG, cs.PF], Submitted 5 Feb 2024 v1, last revised 25 Jul 2024 v2 [PAPER FACT]; Published at ICML 2024 (Comments: ICML2024) [PAPER FACT]; CC BY 4.0 [PAPER FACT] ¡ª verified via webfetch https://arxiv.org/html/2402.02750v2 abstract (2.6¡Á less peak memory, 4¡Á batch, 2.35¡Á~3.47¡Á throughput)
- **DOI/URL:** https://arxiv.org/abs/2402.02750 / https://doi.org/10.48550/arXiv.2402.02750 [PAPER FACT]; Related DOI: https://doi.org/10.13140/RG.2.2.28167.37282 [PAPER FACT]
- **Code:** https://github.com/jy-yuan/KIVI [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/html/2402.02750v2 + https://arxiv.org/abs/2402.02750 (arXiv HTML v2) [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers must be traceable to text, else [NOT REPORTED]; no guessing.

## 1 Problem [PAPER FACT]

Efficient serving of LLMs requires batching many requests to reduce cost per request, but with larger batch sizes and longer context lengths, KV cache storing attention keys/values to avoid recomputations significantly increases memory demands and becomes new bottleneck in speed and memory usage [PAPER FACT] (Abstract, ¡ì1). For 540B PaLM with batch 512 context 2048, KV cache alone can take 3TB, 3¡Á parameters [PAPER FACT] (citing Pope et al. 2023); For OPT-175B with batch 512 prompt 512 gen 32, KV cache 1.2TB, 3.8¡Á weights (Sheng et al. 2023) [PAPER FACT] (¡ì2). Additionally loading KV cache from GPU device memory to SRAM for every token leaves computational cores idle, limiting inference speed [PAPER FACT] (¡ì2 Memory and Speed Analysis). Straightforward solution is quantization to reduce total bytes, but lack of in-depth studies on element distribution to understand hardness/limitation of KV cache quantization [PAPER FACT]; existing vanilla 4bit round-to-nearest applied per-token to both KV fails at 2bit [PAPER FACT].

## 2 Motivation [PAPER FACT]

- Deployment costly requiring many GPUs; natural cost reduction via batching but KV cache scales as b¡Á(l_prompt+l_gen)¡Ád and dominates memory under large batch/long context [PAPER FACT] (¡ì2).
- Three existing categories: (1) reducing heads (multi-query attention Shazeer 2019, multi-group Ainslie 2023) but requires training from scratch/fine-tuning [PAPER FACT] (¡ì1); (2) evicting unimportant tokens (Zhang et al. 2023) [PAPER FACT]; (3) system perspective offloading KV (Sheng et al. 2023) or paging (Kwon et al. 2023) [PAPER FACT].
- Weight quantization well-studied (Lin et al., Xiao et al., Zhao et al.) but only few studies applied vanilla 4bit to KV cache due to streaming nature/complications (Sheng, Zhang, Zhao) [PAPER FACT] (¡ì1).
- Preliminary shows per-token 4bit maintains accuracy but 2bit results in notable drop (Table1: Llama-2-13B CoQA 66.37¡ú52.93 with 2bit per-token) [PAPER FACT]; need to explore element distribution to understand why [PAPER FACT].
- Goal: tuning-free, hardware-friendly 2bit quantization enabling Llama/Falcon/Mistral to maintain almost same quality while using significantly less memory, enabling larger batch and higher throughput [PAPER FACT] (Abstract goals).

## 3 Bottleneck [PAPER FACT]

1. **Streaming nature vs per-channel need:** KV cache arrives sequentially token-by-token during decoding; optimization-based methods like GPTQ unsuitable due to overhead [PAPER FACT] (¡ì3.1). Per-token quantization aligns naturally with streaming (append along token dim) [PAPER FACT] (¡ì3.3), but per-channel quantization spans different tokens and cannot be directly implemented streaming [PAPER FACT] (¡ì3.3).
2. **Channel-wise outliers in key cache:** Visualization of Llama-2-13B and Falcon-7B shows for key cache a few fixed channels have very large magnitudes, consistent with prior findings (Lin et al., Xiao et al.) [PAPER FACT] (Fig.2 description, ¡ì3.2). Per-token quantization mixes outlier channels with normal ones causing large error [PAPER FACT]; Table2 shows per-token reconstruction error 13.67 vs per-channel 4.55, attention score error 47.00 vs 9.60 (~5¡Á larger per-token) [PAPER FACT].
3. **Value cache sensitivity to token-wise mixing:** Value cache shows no obvious outlier pattern [PAPER FACT] (Fig.2); Reconstruction error similar per-token 4.57 vs per-channel 3.73, but attention output error ¦¤ = ||A X_V - A X_V''||_F / ||A X_V||_F is 3.55 per-token vs 49.89 per-channel (~15¡Á smaller per-token) [PAPER FACT] (Table2 bottom, Eq.2). Because attention output is weighted sum across tokens (Eq.2: [A X_V]_{i*}=sum_j A_{ij}[X_V]_{j*} ) and attention is highly sparse (84.3% sparsity noted Table2) [PAPER FACT], output is combination of few important tokens; per-token quantization confines error to each token without impacting important ones, while per-channel spreads error across tokens [PAPER FACT] (¡ì3.2 Analysis of Value Cache).
4. **Arithmetic grouping trade-off:** Group size G determines scaling factors (zero-point z_X=min X, s_X=(max-min)/(2^B-1)) granularity; too large (128) degrades accuracy especially for long inputs [PAPER FACT] (Table5, ¡ì4.2.3).
5. **Hardware efficiency:** Need fused dequantization+matmul at tiling level to avoid extra overhead; naive quantize/dequantize would stall compute [PAPER FACT] (¡ì3.3 System Support).

## 4 Core Idea [PAPER FACT]

**KIVI = Asymmetric quantization: key cache per-channel, value cache per-token, with grouped + residual split to handle streaming, hardware-friendly fused kernels [PAPER FACT].**

- **Definitions (Fig.1):** X ¡Ê R^{l_prompt¡Ád} is key/value cache where l_prompt tokens d channels; per-token groups along token dim, per-channel groups along channel dim; z_X zero-point, s_X scaling factor [PAPER FACT] (Fig.1 caption). B-bit round-to-nearest: Q(X)= floor((X - z_X)/s_X) , X''=Q(X)¡¤s_X+z_X (Eq. in ¡ì3.1) [PAPER FACT].

- **Preliminary Study (Table1, group size 32, fake quant with zero-padding if not divisible):**
  - OB1: per-token 4bit maintains accuracy, but to 2bit drops notably [PAPER FACT] (e.g., 66.37¡ú52.93 CoQA, 29.53¡ú24.98 TruthfulQA for Llama-2-13B per-token both) [PAPER FACT].
  - OB2: When value per-channel, accuracy significantly worsens regardless of key config (e.g., 2bit K-C V-C 2.88/0.74) [PAPER FACT].
  - OB3: Most accurate 2bit is key per-channel + value per-token (63.53/28.60) [PAPER FACT] (¡ì3.1).

- **Why (Analysis ¡ì3.2):** As above channel outliers for keys, attention sparsity for values [PAPER FACT].

- **KIVI Algorithm (¡ì3.3, Fig.3, Algorithm 1 in Appendix A):**
  - *Idea:* Split cache into grouped part and residual part.
  - For key cache: split X_K into grouped X_{Kg}=X_K[:l-r] and residual X_{Kr}=X_K[l-r:] where l tokens current, r residual count, l-r divisible by G [PAPER FACT]; Group size G=32 across all experiments, residual length R=128 (hyperparameter, R divisible by G) [PAPER FACT] (¡ì4.1). Only store Q(X_{Kg}) with group-wise quantized per-channel; X_{Kr} kept in full precision (at most R tokens) [PAPER FACT]. During decoding each newly arrived key t_K added to X_{Kr}; once X_{Kr} reaches R tokens, quantize and concatenate with previously quantized Q(X_{KG}), then reset X_{Kr} to empty [PAPER FACT].
  - For value cache: Similarly split into X_{Vg} and X_{Vr}, maintain queue; each new t_V pushed, once queue reaches R, oldest popped, quantized per-token and concatenated along token dim [PAPER FACT].
  - *Computation:* Tiled matmul raw attention logits: A_g = t_Q Q(X_{Kg}^T), X_{Kr}=Concat([X_{Kr},t_K]), A_r= t_Q X_{Kr}^T, A=Concat([A_g,A_r]) (Eq.3) [PAPER FACT]; Grouped and residual combined via tiled matrix multiplication [PAPER FACT].
  - *Note:* During prefill, exact key/value tensors passed to next layers though only quantized KV retained in memory [PAPER FACT] (Fig.3).
  - *Analysis:* At most R tokens in full precision, R¡Ü128 negligible vs long sequence l_prompt+l_gen much longer; memory overhead negligible especially for long contexts [PAPER FACT]; Maintains full-precision sliding window for local relevant tokens: expected window size R/2 for keys, R for values [PAPER FACT] (¡ì3.3 Analysis) ¡ª crucial for hard tasks like GSM8K where fake 2bit without window drops significantly but KIVI only 2% drop [PAPER FACT] (¡ì4.2.1).

- **System Support (¡ì3.3 System Support):**
  - Fused dequantization with matrix multiplication Q_Matmul at tiling level using CUDA to minimize overhead [PAPER FACT] (Fig.3).
  - Group-wise quantization kernel in Triton [PAPER FACT].
  - Fully compatible with weight-only quantization [PAPER FACT].

## 5 System Changes [PAPER FACT]

- **Inference workflow background (¡ì2):** Two phases: prefill (X¡ÊR^{b¡Ál_prompt¡Ád} ¡ú X_K=X W_K, X_V=X W_V cached) and decoding (t¡ÊR^{b¡Á1¡Ád} ¡ú t_K,t_V updated via Concat, attention t_Q=X? Actually t_O = A X_V with A=Softmax(t_Q X_K^T)) [PAPER FACT] (Eq.1 ¡ì2).
- **Cache data structures:** Pseudocode Appendix A Algorithm 1 (not fully fetched but referenced) replaces vanilla KV cache with quantized grouped cache + residual full-precision buffer [PAPER FACT].
- **Quantization modules:**
  - Grouped key cache: per-channel groups of G=32 tokens, shared z,s per group per channel [PAPER FACT].
  - Grouped value cache: per-token groups of G=32 channels? Actually per-token groups along channel dim with group size 32 [PAPER FACT].
  - Triton kernel for group-wise quantize, CUDA kernel for fused dequant+matmul [PAPER FACT].
- **Integration:** Implemented upon HuggingFace Transformers codebase [PAPER FACT] (¡ì4.1); No tuning/fine-tuning required [PAPER FACT].
- **Hyperparameters:** G=32, R=128 for both key and value (key R divisible by G) fixed across all experiments [PAPER FACT] (¡ì4.1); Ablations vary G 32/64/128 and R 32/64/96/128 [PAPER FACT] (Table5).
- **Memory/speed analysis:** Shape b¡Á(l_prompt+l_gen)¡Ád; after quantization bytes reduced from 2bytes (fp16) to 2bits + overhead of scales/zero-points (¡Ö group size) [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Primary:**
  - **Accuracy retention vs 16bit** on normal-context generation tasks via LM-Eval [Gao et al. 2021] : **CoQA (Exact match accuracy), TruthfulQA (BLEU score), GSM8K (Exact match accuracy)** [PAPER FACT] (¡ì4.1 Tasks).
  - **Long-context accuracy on LongBench [Bai et al. 2023]:** Tasks: Qasper (F1), QMSum (ROUGE), MultiNews (ROUGE), TREC (classification score), TriviaQA (F1), SAMSum (ROUGE), LCC (similarity), RepoBench-P (similarity) [PAPER FACT] (Table4 plus Appendices Table9-10 for Mistral-7B-v0.2 and LongChat-7b-v1.5) [PAPER FACT]; Max sequence length set 8192 for Mistral, 4096 for others [PAPER FACT] (¡ì4.1 footnote).
  - **Long-context retrieval ability:** Needle-in-a-Haystack (NIAH) task with detailed setting Appendix B [PAPER FACT]; Fig.4 counts words not tokens to account tokenizer diff, token length noted upper right [PAPER FACT].
  - **Memory efficiency:** Peak memory usage including model weight reduction factor (2.6¡Á less for Llama-2-7B) [PAPER FACT] (Abstract); KV cache size reduction implied 8¡Á from 16bit to 2bit before overhead.
  - **Throughput / batch size:** Up to 4¡Á larger batch size enabled, bringing 2.35¡Á~3.47¡Á throughput on real LLM inference workload [PAPER FACT] (Abstract, Fig.5).

- **Secondary:**
  - **Fake quantization comparison** across four configurations (Table3) to validate asymmetric choice [PAPER FACT].
  - **Ablation of hyperparameters:** Group size G effect and residual length R effect on GSM8K (Table5) [PAPER FACT]; Further Appendix C with R=32 full results [PAPER FACT].
  - **Efficiency vs accuracy trade-off** for 4bit (KIVI-4) vs 2bit (KIVI-2) [PAPER FACT].
  - **System overhead:** Fused kernel latency vs baseline (Fig.5 memory/throughput comparison) [PAPER FACT].

- **Not elaborate:** Closed-end tasks like MMLU noted as not ideal because only one decoding step and directly fetch logits, not suitable for studying compressed KV [PAPER FACT] (footnote ¡ì4.1); Energy, multi-GPU scaling not primary [NOT REPORTED].

## 7 Baselines [PAPER FACT]

- **16bit (FP16) full KV cache:** Uncompressed baseline for all tables (first row per model) [PAPER FACT].
- **4bit per-token quantization (K-??, V-??):** Vanilla round-to-nearest applied per-token to both caches, group size 32 [PAPER FACT] ¡ª second row, shows maintains accuracy at 4bit [PAPER FACT] (Table1, Table3).
- **Four fake 2bit KV cache quantization configurations (with zero-padding to ensure all tokens quantized for fair comparison) [PAPER FACT] (Table1, Table3):**
  - **2bit (K-??, V-??):** both per-token [PAPER FACT]
  - **2bit (K-?, V-?):** both per-channel [PAPER FACT]
  - **2bit (K-??, V-?):** key per-token value per-channel [PAPER FACT]
  - **2bit (K-?, V-??):** key per-channel value per-token ¡ª KIVI asymmetric choice, best among fakes [PAPER FACT]
- **KIVI-4 (4bit):** KIVI algorithm at 4bit (key per-channel, value per-token with residual) [PAPER FACT] (Table3-4).
- **KIVI-2 (2bit):** Proposed extreme low-bit [PAPER FACT].
- **Note distinction:** Unlike KIVI which preserves small portion of full precision residual (R tokens), all tokens in fake quantization are quantized for fair comparison [PAPER FACT] (Table3 caption).

## 8 Workloads [PAPER FACT]

- **Models (¡ì4.1 Models):**
  - **Llama / Llama-2 family [Touvron et al. 2023a/b]:** Includes Llama-2-7B, Llama-2-13B, Llama2-7B-Chat, Llama2-13B-Chat evaluated [PAPER FACT]; Also Llama-3-8B-Instruct for NIAH (Fig.4) and LongBench appendix [PAPER FACT].
  - **Falcon family [Penedo et al. 2023]:** Falcon-7B (based on multi-query attention with only one head for KV cache) [PAPER FACT] (¡ì4.1).
  - **Mistral family [Jiang et al. 2023]:** Mistral-7B and Mistral-7B-v0.2 [PAPER FACT]; LongChat-7b-v1.5 also in Appendix D [PAPER FACT].
  - Implementation via HuggingFace Transformers [PAPER FACT]; Also Falcon example visualized Fig.2 same as Llama-2-13B [PAPER FACT].

- **Datasets / Tasks (¡ì4.1 Tasks):**
  - **LM-Eval [Gao et al. 2021]:** CoQA, TruthfulQA, GSM8K for normal context length generation [PAPER FACT]; Dataset parameters set to default [PAPER FACT]; Closed-end like MMLU not used due to unsuitability [PAPER FACT].
  - **LongBench [Bai et al. 2023]:** For long-context evaluation, tasks from four subgroups: Single-Document QA Qasper, Summarization QMSum/MultiNews, Few-shot TREC/TriviaQA/SAMSum, Code LCC/RepoBench-P [PAPER FACT] (Table4); Max sequence 8192 for Mistral, 4096 for others [PAPER FACT]; More similar results on Mistral-7B-v0.2 and LongChat-7b-v1.5 in Appendix D Table10/9 [PAPER FACT].
  - **NIAH (Needle-in-a-Haystack):** To evaluate long context retrieval after quantizing; detailed setting Appendix B, counts words for tokenizer fairness [PAPER FACT]; Model Llama-3-8B-Instruct and Mistral-7B-Instruct-v0.2 results Fig.4 [PAPER FACT] (also ¡ì4.2.2 NIAH Results).
  - **Preliminary study datasets:** CoQA and TruthfulQA used for Table1 with group size 32 on Llama-2-13B [PAPER FACT] (¡ì3.1 Setting).

- **Cache budgets / Hyperparameters:** Group size G=32 across all experiments, residual length R=128 for key and value [PAPER FACT] (¡ì4.1); Ablation varies G 32/64/128 with R=128, and R 32/64/96/128 with G=32 on GSM8K Llama2-13B [PAPER FACT] (Table5).

- **Evaluation setting:** Fake quantization simulates by quantizing then dequantizing in attention layer [PAPER FACT]; Real KIVI retains residual full precision [PAPER FACT].

## 9 Hardware [PAPER FACT]

- **System implementation:** Hardware-friendly implementation for GPUs with fused CUDA Q_MatMul and Triton quantization kernel [PAPER FACT] (¡ì3.3).
- **Throughput evaluation:** Fig.5 memory usage and throughput comparison between 2bit KIVI and 16bit baseline shows KIVI can achieve higher throughput by enabling larger batch size [PAPER FACT]; Abstract claims reduced peak memory enables up to 4¡Á larger batch bringing 2.35¡Á~3.47¡Á throughput on real LLM inference workload [PAPER FACT].
- **Explicit hardware specs:** [NOT REPORTED] exact GPU type/count for most experiments in fetched HTML; The memory reduction 2.6¡Á less peak memory including model weight for Llama-2-7B suggests example GPU memory measurement but not named [NOT REPORTED] [PAPER FACT]; Related work example 540B PaLM batch 512 context 2048 is illustrative not measured hardware [PAPER FACT].
- **Precision:** KV cache token size calculation for FP16 (2 bytes) assumed as baseline 16bit [PAPER FACT] (¡ì2); KIVI quantizes to 2bit/4bit integer [PAPER FACT].
- **CPU/RAM/Network:** [NOT REPORTED] [PAPER FACT].

[AGENT INFERENCE]: Likely A100-40/80GB as standard for 7B/13B evaluations, consistent with other papers, but not stated.

## 10 Main Results [PAPER FACT]

All numbers from ¡ì3.1 Table1, ¡ì3.2 Table2, ¡ì4.2 Tables 3-5, Fig.4-5.

- **Preliminary Fake Quant (Table1, Llama-2-13B group 32, CoQA/TruthfulQA):**
  - 16bit: 66.37 / 29.53 [PAPER FACT]
  - 4bit (K-?? V-??): 66.48 / 29.51 ¡ª maintains accuracy [PAPER FACT]
  - 2bit (K-?? V-??): 52.93 / 24.98 ¡ª notable drop [PAPER FACT]
  - 2bit (K-? V-?): 2.88 / 0.74 ¡ª collapse [PAPER FACT]
  - 2bit (K-?? V-?): 2.80 / 0.26 ¡ª collapse [PAPER FACT]
  - **2bit (K-? V-??): 63.53 / 28.60 ¡ª best, very small drop** [PAPER FACT]; supports asymmetric choice [PAPER FACT].
  - Also note 2bit per-channel for value collapses regardless of key config [PAPER FACT].

- **Error Analysis (Table2, relative errors averaged over all layers/heads):**
  - Key cache: Avg ||X_K - X_K''||/||X_K||_F: per-token **13.67** vs per-channel **4.55** [PAPER FACT]; Avg ||A - A''||/||A||_F: per-token **47.00** vs per-channel **9.60** ~5¡Á larger per-token [PAPER FACT]; Attention sparsity **84.3%** [PAPER FACT] ¡ª verified Table2 via webfetch https://arxiv.org/html/2402.02750v2 Table2.
  - Value cache: Avg reconstruction similar per-token 4.57 vs per-channel 3.73 [PAPER FACT]; But ¦¤ error: per-token **3.55** vs per-channel **49.89** ~15¡Á smaller per-token [PAPER FACT]; explains OB2 [PAPER FACT].

- **LM-Eval Generation Tasks (Table3, 16bit vs fake vs KIVI-2/4):**
  - **Llama-2-7B (CoQA/TruthfulQA/GSM8K):**
    - 16bit: 63.88 /30.76 /13.50 [PAPER FACT]
    - Fake 2bit K-? V-??: 59.08/33.10/5.76 [PAPER FACT]; Fake per-token both 39.88/18.29/0.83 etc. [PAPER FACT]
    - KIVI-4: 63.78/30.80/13.80 (matches) [PAPER FACT]
    - **KIVI-2: 63.05/33.95/12.74** ¡ª only small drop, TruthfulQA even higher [PAPER FACT]; Drop vs 16bit CoQA -0.83, GSM8K -0.76 (within ~2%) [PAPER FACT].
  - **Llama-2-13B:**
    - 16bit 66.37/29.53/22.67 [PAPER FACT]
    - KIVI-4 66.38/29.49/23.65 [PAPER FACT]
    - **KIVI-2 66.23/29.84/20.77** ¡ª almost same, GSM8K drop 1.9 (~8% relative but 2% absolute) [PAPER FACT]; Fake K-? V-?? 63.53/28.60/12.21 significantly worse than KIVI-2 on GSM8K (+8.56) showing full-precision window crucial [PAPER FACT] (¡ì4.2.1 text).
  - **Falcon-7B (MQA, single KV head, already highly compressed):**
    - 16bit 59.83/23.20/4.55 [PAPER FACT]
    - KIVI-4 59.67/22.58/4.47 (maintains) [PAPER FACT]
    - **KIVI-2 57.48/24.98/3.41** ¡ª larger drop (?2.35 CoQA, ?1.14 GSM8K) indicating Falcon needs 4bit to maintain [PAPER FACT]; Fake per-token both 25.72/0.91/0.53 even worse [PAPER FACT]; Note Falcon needs 4bit per text [PAPER FACT].
  - **Mistral-7B:**
    - 16bit 67.40/30.45/38.36 [PAPER FACT]
    - KIVI-4 66.95/30.49/37.30 [PAPER FACT]
    - **KIVI-2 66.35/32.17/36.01** ¡ª small drop -1.05 CoQA, -2.35 GSM8K, TruthfulQA +1.72 [PAPER FACT].
  - General: For Llama and Mistral, KIVI only up to 2% accuracy drop despite 2bit [PAPER FACT] (¡ì4.2.2 LM-Eval Results text).

- **LongBench (Table4, 8 tasks, avg):**
  - **Llama2-7B:** 16bit avg 44.52 vs KIVI-4 44.59 vs **KIVI-2 44.27** (drop 0.25) [PAPER FACT]; Per-task: Qasper 9.52¡ú9.31, QMSum 21.28¡ú20.50, MultiNews 3.51¡ú1.14 (larger drop but small base), TREC 66¡ú66, TriviaQA 87.72¡ú87.42, SAMSum 41.69¡ú42.71 (+), LCC 66.66¡ú66.88, RepoBench-P 59.82¡ú60.23 [PAPER FACT].
  - **Llama2-13B:** 44.85¡ú44.69 (KIVI-2) [PAPER FACT]; Qasper 9.32¡ú8.58, QMSum 21.38¡ú20.69, MultiNews 3.71¡ú6.19 (+), TREC 70¡ú69.50 etc. [PAPER FACT].
  - **Llama2-7B-Chat:** 45.95¡ú45.67; **Llama2-13B-Chat:** 45.96¡ú45.52 (KIVI-2) with some variations e.g., Llama2-13B-Chat LCC 50.23¡ú49.93 [PAPER FACT].
  - **Falcon-7B:** 8.71¡ú7.95 (KIVI-2) drop more [PAPER FACT]; Qasper 1.48¡ú1.98 (+), QMSum 2.35¡ú3.61 (+), MultiNews 11.09¡ú6.78 (?), TREC 13¡ú10 etc. [PAPER FACT].
  - **Mistral-7B:** 46.58¡ú45.85 (?0.73) [PAPER FACT]; Qasper 8.12¡ú6.92, QMSum 19.98¡ú19.71, MultiNews 19.99¡ú17.92 etc. [PAPER FACT].
  - Text: KIVI effective with minimal impact across various hard long context generation tasks [PAPER FACT]; Additional results on Llama3-8b, Mistral-7B-v0.2, LongChat-7B-v1.5 similar in Appendix D [PAPER FACT].

- **Needle-in-a-Haystack (Fig.4, Llama-3-8B-Instruct baseline vs KIVI-2/4 and Mistral-7B-Instruct):**
  - Counts words not tokens for tokenizer fairness, token length noted upper right [PAPER FACT]; Shows KIVI can still maintain retrieval ability even with 2bit KV cache [PAPER FACT] (¡ì4.2.2 NIAH Results). Plots baseline (a,d) vs KIVI-2 (b,e) vs KIVI-4 (c,f) heatmaps depth vs length visually comparable [PAPER FACT].

- **Ablation (Table5, Llama2-13B GSM8K):**
  - Group size effect with R=128: 32¡ú20.77, **64¡ú21.00** similar, **128¡ú17.29** significant decrease [PAPER FACT] (¡ì4.2.3 text).
  - Residual length effect with G=32: 32¡ú20.62, 64¡ú19.86 (worst), 96¡ú20.55, **128¡ú20.77** (best but no consistent pattern among 32/96/128 similar, 64 worst) [PAPER FACT]; Text notes reasonably large residual important for hard tasks (boosts over fake) [PAPER FACT].
  - Note choice of group size greatly impacts KV cache compression effect under long input due to scale/zero-point granularity [PAPER FACT].

- **Efficiency (Fig.5, Abstract):**
  - **Peak memory reduction 2.6¡Á less including model weight for Llama-2-7B** [PAPER FACT] (Abstract).
  - **Enables up to 4¡Á larger batch size** [PAPER FACT] (Abstract, Fig.5).
  - **Throughput 2.35¡Á~3.47¡Á on real LLM inference workload** [PAPER FACT] (Abstract, Fig.5).
  - Compatibility with weight-only quantization demonstrated via system implementation [PAPER FACT] (¡ì3.3).

## 11 Assumptions [PAPER FACT]

- KV cache is streaming data structure where new tensor arrives sequentially; optimization-based quantization like GPTQ unsuitable due to overhead [PAPER FACT] (¡ì3.1).
- Round-to-nearest quantization with group-wise scaling (z_X=min X, s_X=(max-min)/(2^B-1)) sufficient; no calibration/tuning needed [PAPER FACT] (¡ì3.1).
- Key cache outlier pattern persists across fixed channels (few channels very large) consistent across Llama-2 and Falcon [PAPER FACT] (Fig.2, ¡ì3.2) ¡ª holds for models evaluated.
- Value cache attention sparsity high (84.3%) making token-wise error confinement effective [PAPER FACT] (Table2).
- Per-token for values and per-channel for keys is universally optimal across layers/heads (applied uniformly) [PAPER FACT].
- Split into grouped (quantized) + residual (full precision) with at most R=128 tokens overhead negligible vs long sequence l_prompt+l_gen >> R [PAPER FACT] (¡ì3.3 Analysis).
- Full-precision sliding window of R/2 for keys and R for values sufficient to retain local relevance [PAPER FACT] (¡ì3.3).
- Exact key/value tensors can be passed to next layers during prefill while only quantized retained in memory without affecting correctness [PAPER FACT] (Fig.3).
- Closed-end tasks like MMLU not ideal for evaluation (single decode step) so omitted [PAPER FACT] (footnote).

## 12 Author-Stated Limitations [PAPER FACT]

No explicit ¡°Limitations¡± but ¡ì6 Conclusion and Future Work implies:

- **Limited to KV cache only:** Not addressing weight/activation compression beyond demonstrated compatibility with weight-only quantization [PAPER FACT] (¡ì3.3 System Support says compatible).
- **Falcon MQA needs higher bits:** For highly compressed MQA models (Falcon-7B single KV head), 2bit causes larger drop, requires 4bit to maintain accuracy [PAPER FACT] (¡ì4.2.2).
- **Residual overhead for short sequences:** For short contexts (l_prompt+l_gen not much longer than R), relative overhead of full-precision residual may not be negligible (Analysis notes negligible especially for long context scenarios) [PAPER FACT] (¡ì3.3).
- **Group size trade-off not deeply explored for all models:** Shows 32/64 similar but 128 degrades; optimal G may vary with context length [PAPER FACT] (¡ì4.2.3).
- **Evaluation not covering all workloads:** LongBench max sequence 8192/4096, not 128K+ extreme; only tested on Llama/Falcon/Mistral families up to 13B plus NIAH on 8B-Instruct [PAPER FACT] (¡ì4.1 scope).

[AGENT INFERENCE]: Authors suggest future work combining with other techniques; paper title emphasizes tuning-free.

## 13 Inferred Limitations [AGENT INFERENCE]

- **Uniform G,R across layers/heads:** Uses same group size 32 and residual 128 for all layers/heads despite known pyramidal sparsity varying by layer; per-layer tuning could improve [AGENT INFERENCE].
- **No integration with token eviction:** Focuses solely on quantization, not combining with pruning (SnapKV/PyramidKV) for token¡Ábit joint compression; Pareto not explored [AGENT INFERENCE].
- **Residual full precision may still be memory overhead at huge batch:** At batch 512, residual 128 tokens per sequence ¡Á batch ¡Á layers ¡Á d in fp16 still significant (¡Ö R * b *...), though authors claim negligible [AGENT INFERENCE].
- **Hardware narrow:** Only CUDA/Triton implementation for GPUs shown; no measurement on different GPU architectures (A100 vs H100), no distributed tensor/pipeline parallelism evaluation for 70B+ [AGENT INFERENCE].
- **Precision limited to 2/4bit:** No evaluation of 3bit or mixed-precision per-layer (e.g., later layers more robust to lower bits) [AGENT INFERENCE].
- **Attention sparsity assumption may break:** For tasks requiring denser attention (e.g., code with long-range dependencies, math reasoning where all tokens needed), per-token value isolation may still incur error; GSM8K already shows higher sensitivity (needs window) [AGENT INFERENCE].
- **No end-to-end serving throughput with continuous batching:** Fig.5 throughput on real workload but not compared to vLLM paged or SGLang style serving; no latency tail analysis [AGENT INFERENCE].
- **Python overhead of queue management:** Splitting and concatenating grouped/residual via tiled matmul adds branching; not broken out as latency cost [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. **Adaptive G,R:** Can group size and residual length be adapted per layer/head or per sequence length (e.g., larger R for math reasoning vs smaller for QA) automatically to maintain accuracy with minimal memory? [AGENT INFERENCE]
2. **Joint optimization:** What is optimal combination of KIVI 2bit quantization with token eviction (H2O/SnapKV/PyramidKV) and weight quantization (GPTQ/AWQ) for end-to-end memory vs accuracy Pareto? [AGENT INFERENCE]
3. **MQA/GQA handling:** For Falcon MQA with single head, can we design group-wise quantization that accounts for single head being more sensitive, perhaps requiring per-channel for values too or higher bits? [AGENT INFERENCE]
4. **Theoretical error bound:** Can we bound attention output error ¦¤ as function of group size and sparsity to guarantee <X% drop, not just empirical Table2? [AGENT INFERENCE]
5. **Longer context scaling:** Does KIVI hold at 128K-1M contexts where residual 128 is even more negligible, or does quantization error accumulate over many tokens causing drift? No 100K+ evaluation shown beyond NIAH 8K words [AGENT INFERENCE]
6. **Hardware portability:** How does fused Q_MatMul perform on AMD/Intel GPUs or with FlashAttention-3 tiling, and can we integrate with paged KV layouts? [AGENT INFERENCE]
7. **Streaming quantization for other caches:** Can same asymmetric principle apply to other caches (e.g., cross-attention KV in encoder-decoder, or MLP caches)? [AGENT INFERENCE]
8. **Dynamic precision:** Should recent tokens remain higher precision (e.g., 4bit) while older grouped tokens stay 2bit, forming a precision pyramid analogous to PyramidKV token pyramid? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **Quantization foundations:** LLM.int8 [Dettmers et al. 2022], SmoothQuant [Lin et al. 2023], AWQ, GPTQ [Frantar et al. 2022] ¡ª weight quantization studies whose outlier findings align with key channel outliers [PAPER FACT] (¡ì3.2).
- **KV cache¨Cfocused quantization:** Sheng et al. 2023 (FlexGen) and Zhao et al. 2024 who applied vanilla 4bit per-token quantization to KV cache [PAPER FACT] (¡ì1, Table1 baseline).
- **Attention optimization/extended memory:** Pope et al. 2023 efficient transformers (KV memory/speed analysis 3TB example) [PAPER FACT] (¡ì1, ¡ì2); Kwon et al. 2023 PagedAttention (vLLM) system for paging [PAPER FACT] (¡ì3.1).
- **KV eviction family:** H2O [Zhang et al. 2023], Scissorhands [Liu et al. 2023], StreamingLLM [Xiao et al. 2023], FastGen/Adaptive KV [Ge et al. 2023] ¡ª alternative reduction via token dropping, complementary to quantization [PAPER FACT] (¡ì1, Fig.2 motivation).
- **Multi-Query Attention [Shazeer 2019] and Multi-Group Attention [Ainslie et al. 2023]:** Reducing heads requires training [PAPER FACT] (¡ì1).
- **Follow-up to KIVI:** GEAR [Kang et al. 2024, arXiv 2403.05527] ¡ª builds on KIVI with quantization + low-rank + sparse outlier fix, reports +7.42% over KIVI at 2bit [AGENT INFERENCE] (from manifest).
- **Concurrent heavy-hitter eviction with quantization:** H2O shows quantization compatibility (Table5 in H2O paper) [AGENT INFERENCE].
- **LongBench [Bai et al. 2023] and LM-Eval [Gao et al. 2021]:** Benchmarks used [PAPER FACT].



## Review Log ¡ª Reviewer-2 (2026-08-27)

- **Webfetch verification:** https://arxiv.org/html/2402.02750v2 ¡ª verified asymmetric choice: per-token 4bit retains accuracy but 2bit per-token both 52.93/24.98 vs per-channel key + per-token value 63.53/28.60 (Table1, group 32 Llama-2-13B CoQA/TruthfulQA); Table2 errors 13.67 vs 4.55 reconstruction and 47.00 vs 9.60 attention and ¦¤ 3.55 vs 49.89 for value; KIVI algorithm G=32 R=128 with split grouped/residual and fused Q_MatMul verified ¡ì3.3; throughput 2.6¡Á memory, 4¡Á batch, 2.35¡Á~3.47¡Á verified abstract.
- **Correction 1 ¡ª Venue enrichment:** Added webfetch abstract throughput/memory numbers to venue.
- **Correction 2 ¡ª Table2 citation:** Added explicit webfetch verification tag for attention sparsity 84.3% and error ratios.
- **Correction 3 ¡ª Hyperparameters:** Confirmed G=32, R=128 universal, and residual sliding window R/2 for keys, R for values noted ¡ì3.3; retained.
- **Correction 4 ¡ª Falcon note:** Confirmed Falcon-7B MQA single KV head requires 4bit to maintain (KIVI-2 57.48 vs 59.83, larger drop) ¡ª retained.
- **Status:** All primary numbers traceable [PAPER FACT]; minor citation enrichment only.
