# Paper Metadata

- **Title:** Efficient Streaming Language Models with Attention Sinks [PAPER FACT]
- **Authors:** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis [PAPER FACT] ¡ª MIT, Meta AI, Carnegie Mellon University, NVIDIA [PAPER FACT]
- **Venue:** arXiv:2309.17453 [cs.CL], Submitted 29 Sep 2023 v1, last revised 7 Apr 2024 v4; Published at ICLR 2024 [PAPER FACT] (Comments: ICLR 2024) [PAPER FACT]
- **DOI/URL:** https://doi.org/10.48550/arXiv.2309.17453 / https://arxiv.org/abs/2309.17453 [PAPER FACT]
- **Code:** https://github.com/mit-han-lab/streaming-llm [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/html/2309.17453v4 (arXiv HTML v4, CC BY 4.0) [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers must be traceable to text, else [NOT REPORTED]; no guessing.

## 1 Problem [PAPER FACT]

Deploy LLMs in **streaming applications** (e.g., multi-round dialogue, day-long conversations) where interactions are unbounded / infinite length is expected [PAPER FACT]. Two primary challenges [PAPER FACT]:
1. **KV cache memory & latency:** Transformer decodes cache Key/Value of all previous tokens, memory grows linearly and decoding latency increases [PAPER FACT] (Pope et al. 2022 cited).
2. **Length extrapolation failure:** Popular LLMs are trained with finite attention window (e.g., 4K for Llama-2 [Touvron et al. 2023b]) and their performance degrades when sequence length exceeds pre-training window [PAPER FACT] (Press et al. 2022; Chen et al. 2023).

Existing attempts (expanding window, FlashAttention, approximate attention, position interpolation fine-tuning) keep sequence length finite and do not allow persistent deployment [PAPER FACT]. The paper asks: *Can we deploy an LLM for infinite-length inputs without sacrificing efficiency and performance?* [PAPER FACT]

## 2 Motivation [PAPER FACT]

- LLMs (Radford 2018; Brown 2020; Zhang 2022; OpenAI 2023; Touvron 2023a/b) power chatbots, summarization, code completion, QA; ideal assistant should work stably over day-long conversations [PAPER FACT].
- Attention window during pre-training constrains generalized length; despite RoPE extension, FlashAttention, etc., acceptable length remains finite [PAPER FACT].
- **Window attention** (Beltagy et al. 2020) caching only most recent KVs offers constant memory but **collapses once cache evicts first token** ¡ª even removing just KV of token 0 spikes perplexity (Fig.3) [PAPER FACT].
- **Sliding window with recomputation** (rebuild KV of recent L tokens per token) retains performance but has **O(T¡¤L2) quadratic cost** within window and is impractical for streaming [PAPER FACT] (Fig.1b vs 1c).
- Need method that is both stable for infinite length **and** O(T¡¤L) efficient without fine-tuning existing models, and that decouples pre-training window from generation length [PAPER FACT].

## 3 Bottleneck [PAPER FACT]

1. **Window attention catastrophic failure on evicting initial tokens** [PAPER FACT]. Fig.3 shows dense attention fails beyond pre-training window, window attention fails beyond cache size; perplexity surges exactly when initial tokens evicted [PAPER FACT].
2. **Attention sink phenomenon dominates deeper layers** [PAPER FACT]. Visualization of Llama-2-7B average attention logits over 256 sentences ¡Á16 tokens shows: layers 0-1 local pattern, but beyond layer 2 model heavily attends to initial token across **all heads/layers** (Fig.2) [PAPER FACT]. Removing those KVs removes large denominator mass in Softmax Eq.1 ( Softmax(x)_i = e^{x_i}/(e^{x1}+¦²_{j=2} e^{x_j}), x1?x_j ) shifting distribution [PAPER FACT].
3. **Semantic vs positional ambiguity** [PAPER FACT]: Two hypotheses (semantics crucial vs absolute position bias). Substituting first 4 tokens with "\n" still attracts attention and restores perplexity, indicating **positional bias** matters more than semantics [PAPER FACT] (¡ì3.1).
4. **Softmax normalization forces sink:** Softmax must sum to 1 even when current query has no strong match, so model learns to dump redundant attention to globally visible initial tokens [PAPER FACT]; similar to quantization outlier observation, motivates Softmax-off-by-one [PAPER FACT].
5. **Positional encoding mismatch:** Na?vely keeping absolute positions (e.g., cache [0,1,2,3,6,7,8] with positions [0,1,2,3,6,7,8,9]) breaks RoPE/ALiBi relative distance; must reassign contiguous positions within cache [PAPER FACT].
6. **Pre-training inconsistency:** No stable sink token across samples (Llama-2 prefixes "<s>" before chunking ¡ú random token at position 0), causing model to spread sink over **multiple initial tokens** (¡Ö4) instead of one [PAPER FACT].

## 4 Core Idea [PAPER FACT]

**StreamingLLM = Attention sinks + Rolling KV Cache + cache-relative positional re-encoding; optional pre-training with dedicated sink token [PAPER FACT].**

- **Insight:** Attention sinks have high scores; preserving their KVs anchors attention distribution close to normal inference, restoring window attention performance with **just 4 initial tokens** [PAPER FACT]. Table 1: Llama-2-13B on PG19 65K tokens, window 0+1024 ¡ú PPL 5158.07, 4+1020 ¡ú 5.40, 4"\n"+1020 ¡ú5.60 [PAPER FACT]. Table 2: 1-2 sinks insufficient, 4 suffices, 8 gives diminishing returns across Falcon-7B, MPT-7B, Pythia-12B, Llama-2-7B [PAPER FACT].

- **Rolling KV Cache with Attention Sinks (¡ì3.2, Fig.4)** [PAPER FACT]: Split cache into:
  1. **Sink part:** fixed 4 initial tokens (always retained) stabilized attention [PAPER FACT].
  2. **Rolling part:** most recent tokens (sliding window) crucial for LM [PAPER FACT].
  With cache [0,1,2,3,6,7,8] decoding token 9, assigned positions are **[0,1,2,3,4,5,6,7] contiguous within cache**, not [0,1,2,3,6,7,8,9] original [PAPER FACT].

- **Positional encoding adaptation** [PAPER FACT]:
  - *RoPE:* Cache keys **before** rotary transform, apply rotation per decoding step with cache-relative positions [PAPER FACT].
  - *ALiBi:* Apply contiguous linear bias instead of jumping bias [PAPER FACT].
  Versatile for any autoregressive LM with RoPE or ALiBi [PAPER FACT].

- **Pre-training with dedicated sink token (¡ì3.3)** [PAPER FACT]:
  - Hypothesis: Adding a **learnable sink token** prepended to all training samples provides single committed repository for excess attention, so only sink + recent tokens needed [PAPER FACT].
  - Alternative *Zero Sink* = Softmax1(x)_i = e^{x_i}/(1+¦² e^{x_j}) equivalent to prepending zero KV token; partially helps but still needs other initials [PAPER FACT] (Table 3).
  - Training 160M models from scratch shows learnable sink yields stable streaming with **1+1023** config vs vanilla needing 4+ [PAPER FACT].

- Claim: Enables Llama-2 (7,13,70B), MPT (7,30B), Falcon (7,40B), Pythia (2.9/6.9/12B) to model **up to 4 million tokens stably** without fine-tuning, and up to **22.2¡Á speedup** vs recomputation [PAPER FACT].

## 5 System Changes [PAPER FACT]

- **Inference pipeline (Fig.1d):** Replace dense KV with bounded cache (sink + rolling). No model weight changes for deployed models; cache management intercepts attention inputs [PAPER FACT].
- **Cache management details (¡ì3.2)** [PAPER FACT]:
  - Cache capacity L = sink (4) + recent window (e.g., 2044 for total 2048) [PAPER FACT].
  - Eviction: when full, evict oldest non-sink token (rolling FIFO), keep sinks pinned [PAPER FACT].
  - Position reassignment: recompute RoPE rotations for all cached keys each step (cheap vs quadratic recompute) or apply ALiBi contiguous bias [PAPER FACT].
- **Implementation:** Built on **HuggingFace Transformers** library for evaluation, single-GPU decoding [PAPER FACT]; same scheme adopted by NVIDIA TensorRT-LLM, Intel Extension for Transformers, HuggingFace `SinkCache`, MLC LLM [PAPER FACT] (Impact Statement).
- **Pre-training modification (¡ì3.3, ¡ì4.2):** Prepend learnable sink token to every sample, train with same Pythia-160M recipe (Pile deduped, 8¡ÁA6000, batch 256, 143K steps, keep LR/schedule) [PAPER FACT]; alternatively swap Softmax for Softmax1 for Zero Sink [PAPER FACT].
- **Compatibility:** Complements context extension methods (LongChat-7b-32k, Llama-2-32K-Instruct) ¡ª streaming extension = increase max recent window size [PAPER FACT] (¡ì4.3).
- **No fine-tuning required for existing models** ¡ª drop-in [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Primary:**
  - **Language modeling perplexity (PPL lower better)** on long texts (PG19 20K¨C4M tokens, concatenated 100 books) [PAPER FACT]; plotted vs input length (Fig.3,5).
  - **Streaming QA accuracy (% exact match)** on concatenated ARC-Easy/Challenge and **StreamEval** (query every 10 lines, answer 20 lines prior, 100 samples avg) [PAPER FACT].
- **Secondary:**
  - **Per-token decoding latency** and **memory usage** vs cache size (Fig.10) [PAPER FACT].
  - **Zero-shot accuracy** on 7 NLP benchmarks (ARC-c/e, HellaSwag, LAMBADA, OpenbookQA, PIQA, Winogrande) for sink-pretrained model (¡ì4.2) [PAPER FACT].
  - **Speedup factor** vs sliding window with recomputation (22.2¡Á) [PAPER FACT].
  - **Pre-training loss convergence** (Fig.6) [PAPER FACT].
- **Not primary but reported:** Attention visualization patterns, perplexity vs #sinks, vs cache size [PAPER FACT].

## 7 Baselines [PAPER FACT]

- **Dense attention (full KV):** Caches all KVs, O(T2) time, growing cache; fails beyond pre-training window length [PAPER FACT] (¡ì1, Fig.1a, Fig.3).
- **Window attention (0+y):** Fixed sliding window of most recent KVs only, e.g., 0+1024, 0+2048, 0+4096; collapses after initial eviction [PAPER FACT] (Table 1,2, Fig.3).
- **Sliding window with recomputation (oracle):** Rebuild KV of recent L tokens per token, O(T¡¤L2); regarded as quality oracle with acceptable perplexity but slow [PAPER FACT] (Fig.1c, ¡ì4.1). StreamingLLM matches its PPL (¡ì4.1).
- **For pre-training sink (¡ì3.3):**
  - *Vanilla* (standard Softmax) [PAPER FACT]
  - *Zero Sink* (Softmax1) [PAPER FACT]
  - *Learnable Sink Token* [PAPER FACT]
- **For streaming QA (¡ì4.3):** One-shot sample-by-sample (ideal), Dense (OOM), Window [PAPER FACT].
- **Context-extended models:** LongChat-7b-v1.5-32k, Llama-2-7B-32K-Instruct as baselines showing complementarity [PAPER FACT].

## 8 Workloads [PAPER FACT]

- **Models (¡ì4):**
  - Llama-2 7B, 13B, 70B (RoPE) [PAPER FACT]
  - MPT 7B, 30B (ALiBi) [PAPER FACT]
  - Falcon 7B, 40B (RoPE) [PAPER FACT]
  - Pythia 2.9B/2.8B, 6.9B, 12B (RoPE) [PAPER FACT]
  - For pretraining: Pythia-160M codebase, 160M param models trained from scratch [PAPER FACT]
  - Instruction-tuned: Llama-2-7/13/70B-Chat [PAPER FACT]
  - Context-extended: LongChat-7b-v1.5-32k, Llama-2-7B-32K-Instruct [PAPER FACT]

- **Datasets:**
  - **PG19 test set:** 100 long books concatenated; evaluation slices: first book 65K tokens (Fig.1, Table1), 20K tokens (Fig.3), 400K tokens (Tables 2,6), 4M tokens (Fig.5) [PAPER FACT].
  - **StreamEval (proposed):** Inspired by LongEval, queries every 10 lines, answer 20 lines prior, up to 120K tokens, 100 samples [PAPER FACT] (Fig.8,9).
  - **ARC-Challenge/Easy:** Concatenated QA pairs streaming QA, exact match [PAPER FACT] (¡ì4.3).
  - **7 NLP benchmarks for sink pretraining:** ARC-c/e [Clark et al. 2018], HellaSwag [Zellers 2019], LAMBADA [Paperno 2016], OpenbookQA [Mihaylov 2018], PIQA [Bisk 2020], Winogrande [Sakaguchi 2019] [PAPER FACT].
  - **Attention visualization:** 256 sentences ¡Á16 tokens [PAPER FACT] (Fig.2,7).
  - **Pile deduped dataset** for pretraining 160M models [PAPER FACT].

- **Cache configs evaluated:** Default 4 sink + recent = total 2048 (Llama-2) or 1024 (MPT/Falcon/Pythia) = half pre-training window for clarity (¡ì4.1) [PAPER FACT]; sweeps 0+2048,1+2047,2+2046,4+2044,8+2040 and variants 4+252 to 4+4092 [PAPER FACT] (Tables 2,6).

## 9 Hardware [PAPER FACT]

- **Decoding latency/memory benchmark (¡ì4.5):** Single **NVIDIA A6000 GPU** using HuggingFace Transformers, Llama-2-7B and Llama-2-13B [PAPER FACT] (Fig.10).
- **Pre-training 160M sink models (¡ì4.2):** **8¡ÁA6000 NVIDIA GPU server**, Pile dataset, batch 256, 143K steps [PAPER FACT].
- **Latency/accuracy other experiments:** Not explicitly hardware-noted beyond A6000; no multi-GPU distribution details for 70B evaluation ¡ª but implies GPU server [NOT REPORTED] for 70B [PAPER FACT inferred as GPU].
- **Precision:** [NOT REPORTED] explicitly; likely FP16/bfloat16 as standard for Llama-2/MPT ¡ª paper does not state [NOT REPORTED].
- **CPU/RAM/Network:** [NOT REPORTED] [PAPER FACT].

## 10 Main Results [PAPER FACT]

All numbers from ¡ì3¨C4, Figs.3,5,9,10, Tables 1¨C6.

- **Attention sink validation (Table1, Llama-2-13B PG19 65K, 4+1020 vs 0+1024):**
  - Window 0+1024: **PPL 5158.07** (catastrophic) [PAPER FACT]
  - 4+1020: **5.40** [PAPER FACT]
  - 4"\n"+1020: **5.60** (similar, confirms positional) [PAPER FACT]

- **Number of sinks needed (Table2, PG19 400K):**
  - Falcon-7B: 0+2048 17.90 ¡ú 1+2047 12.12 ¡ú 4+2044 12.12 (stable) [PAPER FACT]
  - MPT-7B: 460.29 ¡ú14.99 ¡ú14.99 (4) [PAPER FACT]
  - Pythia-12B: 21.62 ¡ú11.95 (1) ¡ú12.09 (4) [PAPER FACT]
  - Llama-2-7B: 3359.95 ¡ú11.88 (1) ¡ú10.51 (2) ¡ú**9.59 (4)** ¡ú9.54 (8) [PAPER FACT] ¡ú 4 suffices, more than 4 diminishing [PAPER FACT].

- **Long-text LM stability (¡ì4.1):**
  - Fig.3: StreamingLLM PPL nearly matches recomputation oracle up to 20K, while dense/window diverge at pre-training window / cache size [PAPER FACT].
  - Fig.5: **Stable PPL across 4 million tokens** (100 PG19 books concatenated) for Llama-2 7/13/70B, Falcon 7/40B, Pythia 2.8/6.9/12B, MPT 7/30B; fluctuations due to book transitions [PAPER FACT].

- **Pre-training sink token (¡ì4.2, Table3, PG19 first sample):**
  - Vanilla: 27.87 (0+1024) ¡ú18.49 (1) ¡ú18.05 (4) [PAPER FACT]
  - Zero Sink: 29214 (0) ¡ú19.90 (1) ¡ú18.01 (4) [PAPER FACT]
  - Learnable Sink: 1235 (0) ¡ú**18.01 (1+1023)** ¡ú18.02 (4) [PAPER FACT] ¡ª only learnable sink achieves stable with single token [PAPER FACT].
  - Convergence: Fig.6 similar loss curves w/ and w/o sink [PAPER FACT].
  - Zero-shot 7 benchmarks (Table4): Vanilla vs +Sink similar: ARC-c 18.6¡ú19.6, ARC-e 45.2¡ú45.6, HS 29.4¡ú29.8, LBD 39.6¡ú39.9, OBQA 16.0¡ú16.6, PIQA 62.2¡ú62.6, WG 50.1¡ú50.8 [PAPER FACT] ¡ª no harm [PAPER FACT].

- **Streaming QA (¡ì4.3, Table5, cache 1024):**
  - Llama-2-Chat streaming ARC concatenated (1024 cache):
    - Dense = OOM for 7/13/70B [PAPER FACT]
    - Window accuracy: 7B 3.58%/1.39% (E/C), 13B 0.25%/0.34%, 70B 0.12%/0.32% (near random) [PAPER FACT]
    - StreamingLLM: **71.34/55.03 (7B), 80.89/65.61 (13B), 91.37/80.20 (70B)** matching One-shot 71.25/53.16, 78.16/63.31, 91.29/78.50 [PAPER FACT] (slightly above one-shot).
  - StreamEval (Fig.9): StreamingLLM maintains accuracy up to **120K tokens**, dense fails at pre-training length, window at cache size; complementing LongChat-32k extends effective recent window [PAPER FACT].

- **Cache size ablation (Table6, PG19 400K):** Increasing not monotonic ¡ª e.g., Falcon 4+252 13.61 ¡ú4+1020 12.34 ¡ú4+2044 12.84 (worsens); MPT 14.12¡ú14.99 worsens; Pythia/Llama similar ¡ª models may not fully utilize larger context [PAPER FACT].

- **Efficiency (¡ì4.5, Fig.10, A6000, Llama-2-7B/13B):**
  - Decoding latency: StreamingLLM linear growth vs recomputation quadratic; **up to 22.2¡Á speedup per token** as cache grows [PAPER FACT].
  - Memory footprint similar to recomputation baseline [PAPER FACT].

## 11 Assumptions [PAPER FACT]

- Autoregressive LLM with Softmax attention requiring scores sum to 1, creating need for sink dumping ground [PAPER FACT].
- Relative positional encoding (RoPE, ALiBi) is used; positional information is relative not absolute, allowing cache-relative reassignment [PAPER FACT].
- Initial tokens are **globally visible** to all later tokens during pre-training, thus readily learnable as sinks [PAPER FACT].
- Keeping only sinks + recent tokens suffices; middle tokens can be discarded for streaming without catastrophic drift [PAPER FACT].
- For RoPE, caching keys **before** rotary transformation and re-applying per step is equivalent to contiguous positions [PAPER FACT].
- Text is continuous stream; coherence can be generated from recent context only; long-range dependency beyond recent window is not required to be exact [PAPER FACT] (¡ì5 conclusion emphasizes not extending context length).
- Pre-training sink token assumption: prepending same learnable token to **all samples** will be learned as dedicated sink without harming convergence [PAPER FACT].
- Evaluation assumes infinite length = concatenated books with perplexity measured sequentially; book transitions cause fluctuations noted [PAPER FACT].

## 12 Author-Stated Limitations [PAPER FACT]

No explicit "Limitations" section; but discussed in ¡ì2, ¡ì4.4, ¡ì5, Appendices A-I, ¡ìA Discussions:

- **Not extending context window:** StreamingLLM does **not increase attendable context** or improve model memory/utilization of long text; it only stably generates from **recent tokens** within KV cache, without needing past data [PAPER FACT] (¡ì5: "without extending LLMs' context length... suits continuous operation needs with minimal past data reliance") [PAPER FACT].
- **Cache size does not always help:** Larger cache does not consistently lower PPL (Table6); authors note models may not maximize utility of entire context ¡ª limitation of underlying LM, not StreamingLLM alone [PAPER FACT] (¡ì4.4).
- **Need for sinks is artifact of pre-training:** Vanilla models still need **multiple (4) sinks** because no consistent starting token; single sink requires pre-training change [PAPER FACT] (¡ì3.1, ¡ì3.3).
- **Comparison scope:** Does not improve length extrapolation or context utilization per se; categorized as first category (Length Extrapolation to infinite) but orthogonal to context extension/usage categories [PAPER FACT] (¡ì2).
- **Future work hints:** Appendix I exploring more sink tokens, H encoder sinks, D long-range benchmark evaluation suggests limitations on broader benchmarks where long-range retrieval is needed [PAPER FACT] (Appendix list).

[AGENT INFERENCE]: Authors present StreamingLLM as complementary to context extension, not replacement.

## 13 Inferred Limitations [AGENT INFERENCE]

- **Loss of middle context:** Rolling eviction discards all middle tokens; tasks requiring retrieval of distant facts (e.g., needle-in-haystack beyond recent window) will fail even though perplexity looks stable [AGENT INFERENCE].
- **Fixed sink count (4) heuristic:** Optimal sink count may vary by model/family/layer; Llama-2 needs 4 but Falcon/MPT stable with 1 ¡ª fixed 4 may be wasteful or insufficient for future architectures [AGENT INFERENCE].
- **Positional reassignment may distort ROPE long-range:** Cache-relative positions discard true distance of sinks (they were 10K tokens ago but appear at position 0-3); may affect relative position semantics for models sensitive to long distances [AGENT INFERENCE].
- **No theoretical bound on drift:** Stable perplexity on PG19 (narrative) does not guarantee coherence for structured dialogues with many turns where middle context matters [AGENT INFERENCE].
- **Evaluation limited to perplexity/QA with recent answer (20 lines prior):** StreamEval answer 20 lines prior aligns with recent window; not testing retrieval at cache-evicted distance [AGENT INFERENCE].
- **No batch serving / throughput at scale:** Latency measured single-stream HuggingFace on A6000; no vLLM-style batched throughput, no paged memory interaction [AGENT INFERENCE].
- **Sink token pre-training cost:** Requires training from scratch (160M demo); retrofitting larger models incurs large cost, not evaluated at 7B+ scale [AGENT INFERENCE].
- **Hardware narrow:** Only A6000 single GPU; no A100/H100, no distributed tensor/pipeline parallelism [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. **Adaptive sink allocation:** Can we learn per-layer/head sink counts or dynamic sink selection (e.g., attention-score-based) instead of fixed 4 initial tokens? [AGENT INFERENCE]
2. **Middle-context retrieval without OOM:** How to combine StreamingLLM rolling window with infrequent offloading or sparse recall of evicted middle tokens for needle retrieval tasks? [AGENT INFERENCE]
3. **Optimal cache shape for different modalities:** Does persistence hold for code/math where distant definitions matter, and what recent window size is needed per domain? [AGENT INFERENCE]
4. **Interaction with quantization/paging:** How does cache-relative RoPE recomputation interact with paged KV (vLLM) and quantized KV? Can we page sink blocks permanently? [AGENT INFERENCE]
5. **Pre-training sink at scale:** Does single learnable sink scale to 70B+ and naturally emerge with modern pre-training recipes (e.g., with bos token every sample)? [AGENT INFERENCE]
6. **Theoretical characterization:** Can we bound perplexity drift as function of sink count and recent window size under power-law attention? [AGENT INFERENCE]
7. **Streaming evaluation realism:** How to design streaming benchmark that requires both long-term memory and streaming coherence beyond 20-line locality (e.g., long dialogues with callbacks)? [AGENT INFERENCE]
8. **Beyond decoder-only:** Do encoder Transformers (Appendix H) or encoder-decoder exhibit similar sinks and can StreamingLLM extend to them? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **Window/Rolling context:** Beltagy et al. 2020 Longformer (window attention) [PAPER FACT]; Transformer-XL [Dai et al. 2019] recurrence [AGENT INFERENCE].
- **Positional extrapolation:** RoPE (Su et al. 2021) [PAPER FACT]; ALiBi (Press et al. 2022) [PAPER FACT]; Chen et al. 2023 position interpolation [PAPER FACT]; kaiokendev 2023, Peng et al. 2023 [PAPER FACT]; bloc97 2023 [PAPER FACT].
- **Efficiency kernels:** FlashAttention (Dao et al. 2022; Dao 2023) [PAPER FACT]; Pope et al. 2022 efficient transformers [PAPER FACT].
- **Quantization outliers/Sinks connection:** Xiao et al. 2023 SmoothQuant, Bondarenko et al. 2023 outlier, Miller 2023 Softmax-off-by-one [PAPER FACT] (¡ì3.1).
- **Context extension vs streaming:** LongChat-7b-v1.5-32k (Li et al. 2023), Llama-2-7B-32K-Instruct (Together 2023) as complementary [PAPER FACT]; LongEval (Li et al. 2023), LongBench (Bai et al. 2023) [PAPER FACT].
- **Concurrent eviction work:** H2O (Zhang et al. 2023) Heavy-Hitter Oracle, Scissorhands (Liu et al. 2023) Persistence of Importance ¡ª both exploit attention sparsity but use heavy-hitter/recent retention rather than fixed sinks; comparison discussed in later literature [AGENT INFERENCE].
- **Follow-up to StreamingLLM [AGENT INFERENCE]:** SinkCache integrated into HuggingFace Transformers, TensorRT-LLM, MLC LLM implementations after paper.


## Review Log
Reviewer: Reviewer-1
Problems Found:
- Perplexity table 5158.07 (0+1024) vs 5.40 (4+1020) vs 5.60 (4 newline +1020) verified Table 1 Llama-2-13B PG19 65K ¡ª note correctly includes but previously missing dataset length qualifier (65K) now pinned.
- Sink count ablation Table 2 values (Falcon 17.90->12.12, MPT 460->14.99, Pythia 21.62->12.09, Llama-2 3359->9.59 with 4 sinks) verified; note correctly says 4 suffices, 8 diminishing.
- Stability up to 4M tokens (Fig.5, 100 books) and StreamEval 120K verified; speed 22.2x vs recomputation on A6000 Fig.10 verified.
- Hardware: A6000 single GPU for latency, 8xA6000 for 160M pretrain verified; precision NOT REPORTED correctly marked.
- No context extension claim: note correctly states does NOT increase attendable context, only recent window stable ¡ª verified against Sec.5 conclusion.
Corrections:
- Pinned dataset lengths to perplexity claims for clarity.
- No hallucination of batch serving (correctly notes single-stream HuggingFace, no vLLM paged interaction).
- Core phenomenon (attention sink due to softmax sum=1 dumping to initial tokens, need for cache-relative RoPE/ALiBi re-encoding, rolling KV with 4 pinned sinks) accurately summarized.
Confidence: High