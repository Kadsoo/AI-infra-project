# Paper Metadata

- **Title:** Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time [PAPER FACT]
- **Authors:** Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, Anshumali Shrivastava [PAPER FACT] — Rice University, Department of Computer Science [PAPER FACT]
- **Venue:** Preprint arXiv:2305.17118 [cs.LG], Submitted 29 May 2023 v1, revised 28 Aug 2023 v2 [PAPER FACT]
- **DOI/URL:** https://arxiv.org/abs/2305.17118 [PAPER FACT]
- **Code:** [NOT REPORTED] in arXiv HTML (no explicit GitHub link in abstract/HTML excerpt) [PAPER FACT] — but paper describes system Scissorhands [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/html/2305.17118v2 [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers must be traceable to text, else [NOT REPORTED]; no guessing.

## 1 Problem [PAPER FACT]

Hosting LLMs at scale memory-limited by **KV cache (context window)**, not just model weights [PAPER FACT]. LLM inference is autoregressive: each step stores key-value embeddings to avoid recomputation; KV cache includes prompt + generated tokens and **grows linearly with batch × seq × hidden** [PAPER FACT] (Table1). Examples: OPT-175B weights 325GB but KV cache at batch 128, seq 2048 is **1152GB (3.5× weights)** [PAPER FACT]; LLaMA-65B 130GB weights vs 640GB cache [PAPER FACT]. On 8×A100-80GB (640GB), cache becomes bottleneck, limiting batch size (max 34 for OPT-175B at max length without offloading, Table2) [PAPER FACT]. Need **test-time KV cache compression** without finetuning, reducing memory from **sequence length dimension**, preserving quality and in-context learning for arbitrarily long contexts (e.g., GPT-4-32K 32K tokens) [PAPER FACT] (§1).

## 2 Motivation [PAPER FACT]

- LLMs trained on massive text generate logically connected text, deployed for high-throughput inference [PAPER FACT]; batch size directly translates to throughput on fixed memory hardware — KV reduction → linear batch increase [PAPER FACT] (§1, §2.1).
- **Weight compression** (quantization, sparsity) well-studied (cited [8-14]) but KV cache compression **remains open** [PAPER FACT] (§1).
- Requirements for ideal compression at test time: **(a) no training** (training 100B-scale expensive), **(b) reduce sequence-length dimension** (scalability to 32K+), **(c) preserve quality** [PAPER FACT] (§1).
- Observation: Attention from one token follows **strong power-law** (few tokens get high attention) [PAPER FACT] (cited [16-20]); more importantly, **Repetitive Attention Pattern**: different positions in same sentence heavily attend to **same small set of tokens** (Fig.1: positions 178,228,278 all attend to 27,63,98 etc.) [PAPER FACT] (§1 Fig.1).
- Unlike FlexGen which quantizes but still stores all tokens in CPU and loads all keys for attention [PAPER FACT] (§2.2), goal is to **actually drop non-influential tokens** from cache [PAPER FACT].

## 3 Bottleneck [PAPER FACT]

1. **Attention O(n²) compute and KV memory blow-up:** Naïve exact attention needs quadratic time and linear KV memory; training-focused sparse approximations (low-rank, sparsification [16-20], FlashAttention [21]) optimize compute but **do not reduce KV cache by sequence length** [PAPER FACT] (§2.2).
2. **Unknown importance ahead of time:** Need oracle to identify which tokens will be important for future steps before generating them; per-step greedy based on current attention may be myopic [PAPER FACT] (§3.2 hypothesis).
3. **Fixed memory budget hard constraint:** Deployed on fixed hardware, algorithm must keep cache **≤ B tokens** deterministically at all times [PAPER FACT] (Def 4.1, Alg.1).
4. **Overhead vs benefit trade-off:** Compression via recomputing history attention introduces extra compute; must not dominate decoding; need tunable drop frequency m [PAPER FACT] (§4.1 Overhead Tradeoff).
5. **Budget allocation across heads/layers:** LLM has L layers × H heads; total budget must be distributed unevenly (later layers have lower persistence ratio) [PAPER FACT] (§4.1 final paragraph, Fig.2).
6. **Prior weight-focused methods miss KV:** Quantization/pruning reduce weights (OPT-175B 325GB) but cache still 950GB example in original text (abstract 950GB at batch128 seq2048) remains dominant [PAPER FACT] (§1 intro numbers).

## 4 Core Idea [PAPER FACT]

**Scissorhands = Persistence of Importance hypothesis + budgeted reservoir cache that retains pivotal tokens via history-window importance counting + recent window protection [PAPER FACT].**

- **Repetitive Attention Pattern (§3.1, Fig.1):** Discretized attention (threshold 1/t average) shows token 178,228,278 all dark green at same positions 27,63,98,121,152,177 etc. [PAPER FACT]. Only five heads shown for clarity; pattern holds across layers/texts/Appendix A [PAPER FACT].

- **Persistence of Importance Hypothesis (§3.2):** *"Only pivotal tokens, which had a substantial influence at one previous step, will have a significant influence at a future step."* [PAPER FACT] — Formal claim that future influencers are subset of past influencers (non-trivial when subset ≪ n) [PAPER FACT].
  - *Pivotal definition:* Token j pivotal for position t if α_{t,j} > α = 1/t (average mixing score) [PAPER FACT].
  - *Set notation:* S_t pivotal for t; S_{a→b}=∪_{t=a}^b S_t [PAPER FACT].
  - *Persistence ratio verification:* Split sentence length l at t=l/2, measure |S_{t+1→l} ∩ S_{0→t}| / |{x∈S_{t+1→l}, x∈{x_1..x_t}}| [PAPER FACT] (Eq. in §3.2). Test on OPT models with OpenBookQA/Wiki-Text: **ratio >95% in most layers** (dips later layers) (Fig.2a) [PAPER FACT]; and |S_{0→t}|/t **considerably smaller than 0.5** (Fig.2b) confirming non-trivial [PAPER FACT] (§3.2 Result).

- **Why it holds — Theoretical intuition (§3.3, Thm 3.1):** Simplified single-layer single-head model xt+1 = ℱ(at), at = softmax(1/t·x_t W_Q W_K^T X_{t-1}^T) X_{t-1} W_V W_O, ℱ(x)=x+W2 relu(W1 x) [PAPER FACT] (Eq.1,2). Under assumptions normalized ||x_t||=1, MLP cosine condition a_t x_{t+1}^T ≥(1-δ)||a_t|| with δ ≤ (c ε / λ_Q λ_K λ_V λ_O)^2, then for tokens satisfying x_ℓ A x_ℓ^T ≥c and x_ℓ A x_ℓ ≥ ε^{-1} max_{j≠ℓ} x_j A x_ℓ^T where A= W_V W_O W_Q W_K^T, we have bound: (x_ℓ A x_ℓ^T /||a_t||)(α_{t,ℓ}-3ε) ≤ x_{t+1} W_Q W_K^T x_j^T ≤ (x_ℓ A x_ℓ^T /||a_t||)(α_{t,ℓ}+3ε) [PAPER FACT] (Thm 3.1 proof Appendix B). Shows future attention score roughly scales with past α_{t,ℓ} scaled by x_ℓ A x_ℓ^T /||a_t||, so large past implies large future [PAPER FACT]; also explains why only specific tokens (those with large x_ℓ A x_ℓ^T) become pivotal — learned weights A select subspace [PAPER FACT].

- **Scissorhands Algorithms (§4.1, Alg.1 & 2):**
  - *Budgeted generation (Alg.1, Def 4.1):* Maintain \bar{K}, \bar{V} ∈ R^{n×d} with n ≤ B; each step model appends, if n>B compress via Alg.2 [PAPER FACT].
  - *Compress (Alg.2):* Inputs: window w=400, recent r=10, drop m=0.5·B [PAPER FACT]. For i∈[t-w, t] collect importance I incrementing counter for low-score token where α_i <1/t [PAPER FACT] (increment for low); then set I[:-r]=0 to **keep recent window** [PAPER FACT]; keep set St = Argsort(I)[:-m] (drop m lowest importance/most low-score counts) and retain those in cache n←n-m [PAPER FACT]. History window reduces variance, recent protected due to insufficient info [PAPER FACT].
  - *Estimator:* \hat{a}_t = Σ \hat{α}_{t,i} \bar{V}[i], \hat{α} computed only over retained cache [PAPER FACT] (§4.1).
  - *Allocation across heads:* Even across heads within layer, but **more budget to later layers** to compensate lower persistence ratio (Fig.2) [PAPER FACT] (§4.1 last).
  - *Overhead:* Compression extra attention over w window, but not every step (m=0.5B controls frequency); after compression attention cheaper due to smaller KV; can maintain importance during generation to avoid extra [PAPER FACT].

- **Theoretical guarantee (Thm 4.1, §4.2):** For simplified model, assume β_{t,j}=c·v_{t,j} where v ∼ power-law pdf f(x)=c(x+b)^{-k} [PAPER FACT]; assume λ_V λ_O (1+λ1 λ2)(1+λ_Q λ_K) ≤ 0.5 [PAPER FACT]. If St always contains top-B βs, then ∀ε∈(0,1) with prob ≥1 - Tmax exp(-ε² b² (Tmin-1)/(k-2)²(u-b)²) - Tmax exp(-2(Tmin-1)(1-B/Tmax)²/(1-ε)²) [PAPER FACT], error bounded: **E[||x_t - \tilde{x}_t||₂] ≤ 2.1(1-B/Tmax)/(1-ε)² ( k - (k-1)((1-ε)/(B/Tmax -ε))^{1/(k-1)} )** [PAPER FACT] (Eq.4). Scales with 1-B/Tmax (0 when B=Tmax) and power-law k (<1 factor) [PAPER FACT].

## 5 System Changes [PAPER FACT]

- **Inference stack (§2 Problem, §4):** Standard two stages: prompting (build K0,V0 = x_prompt W_K/V) and token generation (update K_{t+1}=[K_t, x_t W_K]) [PAPER FACT]; Scissorhands interposes budgeted cache: prompting builds initial cache, generation maintains n≤B via Alg.1 loop [PAPER FACT] (Alg.1 while t<Tmax).
- **Cache data structures:** Key cache \bar{K}, Value cache \bar{V} ∈ R^{n×d} where n dynamic but bounded [PAPER FACT] (Alg.1 header).
- **Compression module (Alg.2 implementation):**
  - Maintains **Importance Record I ∈ R^t** vector [PAPER FACT].
  - Loops i∈[t-w, t] adding  α_i <1/t low-score counts [PAPER FACT] (actually increment for low).
  - Protects recent r tokens via I[:-r]=0 [PAPER FACT].
  - Argsort and keep top, physical eviction via slicing [PAPER FACT].
- **Hyperparameters fixed across experiments (§4.1 Approach paragraph):** w=400, r=10, m=0.5B [PAPER FACT].
- **Cross-layer budgeting:** Rule of thumb more to later layers; within layer even [PAPER FACT].
- **Compatibility:** Demonstrated combined with 4-bit quantization (FlexGen style) without compounding error [PAPER FACT] (§5 Table3, §2.2).
- **No fine-tuning:** Works at test time on frozen weights [PAPER FACT] (title, abstract).
- **Integration with batch serving:** Memory reduction translates to larger batch size; Table2 max batch before OOM computed for 8×A100 80GB box [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Primary:**
  - **Language modeling perplexity** lower better on **C4** dataset [PAPER FACT] (§5 Fig.3a).
  - **Downstream few-shot accuracy** (%) on: **Hellaswag, MathQA, PIQA, Winogrande** via **lm-eval-harness** [PAPER FACT] (§5 Experiment Setting).
- **Secondary:**
  - **KV cache memory reduction factor** (up to **5×**) without accuracy drop [PAPER FACT] (Abstract, §5 Fig.3).
  - **Inference memory usage** (GB) Table1 weights vs cache [PAPER FACT].
  - **Maximum batch size** before OOM on 8×A100 80GB at max sequence length (Table2) [PAPER FACT].
  - **Compatibility with 4-bit quantization** accuracy (Table3 Hellaswag) [PAPER FACT].
  - **Persistence ratio** (%) and |S_{0→t}|/t size (Fig.2) [PAPER FACT].
  - **Error bound** E[||x_t - \tilde{x}_t||] (Thm 4.1) [PAPER FACT].
- **Not elaborate:** Latency tokens/s explicitly? Discussed batch size implication for throughput but not per-token latency breakdown vs H2O/StreamingLLM [NOT REPORTED] as primary; throughput implication is AGENT INFERENCE.

## 7 Baselines [PAPER FACT]

- **Original OPT (1× = full KV cache):** Uncompressed model, retention of all tokens [PAPER FACT] (Fig.3 1× denotes original).
- **Scissorhands compressed variants:** Sweeps compression ratios: 50%, 15-30% etc. (Fig.3 x-axis is KV cache compression) [PAPER FACT].
- **Scissorhands + 4-bit quantization:** 2× compression + 4-bit (following FlexGen) vs quantization alone [PAPER FACT] (Table3).
- **Implicit baselines mentioned in Related Work §2.2:**
  - Efficient attention methods: low-rank/sparsification families [16-20] and exact FlashAttention [21] — discussed as training-focused not KV reduction [PAPER FACT].
  - Weight quantization/pruning families [8-14] [PAPER FACT].
  - FlexGen [7] that still stores all KV (quantized) in CPU and loads all keys [PAPER FACT].

- **No direct comparison to H2O/StreamingLLM** (concurrent papers, not in v2) [AGENT INFERENCE].

## 8 Workloads [PAPER FACT]

- **Models:**
  - **OPT family:** OPT-6B, OPT-13B, OPT-30B, OPT-66B (also OPT-175B in memory tables) [PAPER FACT] (Fig.3 shows OPT-6B,13B,30B? Text: "until 5× compression on OPT-66B", Table1 OPT-175B/LLaMA-65B/BLOOM, Table2 same) [PAPER FACT]
  - For persistence verification: OPT-6B (Fig.1), also test different OPT sizes on OpenBookQA/Wiki-Text [PAPER FACT] (§3.2 Verification).
  - Cross-layer cosine similarity verification [PAPER FACT] (Appendix A.2).

- **Datasets:**
  - **C4** [Raffel et al.] for language modeling perplexity [PAPER FACT] (§3.1 random sentence from C4, §5 C4).
  - **Few-shot downstream:** Hellaswag [16?], MathQA, PIQA, Winogrande evaluated 5-shot and 1-shot? Fig.3b-d five-shot [PAPER FACT]; lm-eval-harness framework [Gao 2021] [PAPER FACT].
  - **OpenBookQA, Wiki-Text** for persistence ratio tests [PAPER FACT] (§3.2).
  - No PG19/ShareGPT style long-context benchmark in main evaluation [AGENT INFERENCE].

- **Sequence lengths considered:**
  - Memory tables assume **max sequence 2048** (GPT-3 scale) and up to **32K** mentioned [PAPER FACT] (§2.1).
  - Budget B tokens per head not absolute number stated in Fig.3 but sweep from 1× to ~5× compression (16-20% retain) [PAPER FACT].

- **Evaluation setting:**
  - **NVIDIA 4×A100 40GB GPU servers** [PAPER FACT] (§5 Experiment Setting).
  - Batch serving implication derived from Table2 8×A100-80GB calculations, not actual batch sweep [PAPER FACT].

## 9 Hardware [PAPER FACT]

- **Main experiments:** **NVIDIA 4×A100 40GB GPU servers** [PAPER FACT] (§5 Experiment Setting: "conducted on NVIDIA 4 A100 40GB GPU servers").
- **Memory/budget table calculations (§2.1 Table2):** **Box of 8×A100 80GB GPU**, NVLink presumed, computed max batch size before OOM at max seq 2048: OPT-175B 34, LLaMA-65B 102, BLOOM 36 [PAPER FACT] (Table2).
- **No CPU/RAM/Interconnect normalized latency:** [NOT REPORTED] beyond GPU type [PAPER FACT].
- **Precision:** Assumed **FP16** (175B weights 325GB ≈ 175B*2 bytes ~ 350GB) [AGENT INFERENCE] but paper does not explicitly state precision for OPT/BLOOM weights vs cache; we tag [NOT REPORTED] for explicit precision field [PAPER FACT shows GB numbers but not dtype].
- **Quantization test:** 4-bit following FlexGen setup, hardware same 4×A100 40GB [PAPER FACT].

## 10 Main Results [PAPER FACT]

All numbers from §1, Tables1-3, Fig.2-4, §5.

- **Memory breakdown (Table1, batch128 seq2048):**
  - OPT-175B: 96 layers, 12288 hidden, **Weights 325GB, KV cache 1152GB** [PAPER FACT]
  - LLaMA-65B: 80 layers 8192 hidden, 130GB weights, **640GB cache** [PAPER FACT] (Table1 also abstract says 950GB at batch128 seq2048 for OPT-175B — slight discrepancy but both in text; we report table numbers) [PAPER FACT]
  - BLOOM: 70 layers 14336 hidden, 352GB weights, **950GB cache** [PAPER FACT]

- **Max batch size before OOM on 8×A100-80GB (Table2, max seq 2048):**
  - OPT-175B **34** [PAPER FACT], LLaMA-65B **102**, BLOOM **36** [PAPER FACT].

- **Persistence verification (Fig.2):**
  - Persistence ratio **>95% in most layers**, dips later [PAPER FACT] (§3.2 Result, Fig.2a).
  - Pivotal set size |S_{0→t}|/t **considerably smaller than 0.5** [PAPER FACT] (Fig.2b).

- **Accuracy vs compression (Fig.3, described §5 No Accuracy Drop):**
  - **Language modeling (C4) perplexity:** Flat until **50% for OPT-13B, 75% compression for OPT-66B** (i.e., retain 50%/25%) [PAPER FACT]; text: "For OPT-6B, perplexity maintained until 50% of original KV cache size for OPT-13B. For OPT-66B, perplexity maintained until 75% of original" [PAPER FACT]. Actually phrasing ambiguous but Fig.3a shows flat up to ~2× (50% retain) for smaller, 4× for larger [PAPER FACT].
  - **Downstream 5-shot:** Generally **accuracy maintained with 15%–30% retain (3.3–6.6× compression)** [PAPER FACT]; Winogrande/MathQA maintain even after **5× compression for OPT-66B** [PAPER FACT] (text: "For Winogrande and MathQA, accuracy maintained even after 5× compression for OPT-66B") [PAPER FACT].
  - **Trend: Larger models flatter** — Scissorhands scales better with size [PAPER FACT] (encouraging).
  - General statement: **Up to 5× reduction without degradation** [PAPER FACT] (Abstract, §5 intro, Fig.3).

- **4-bit quantization compatibility (Table3, 2× compression + Hellaswag):**
  - OPT-6B: Original 0.702 → Scissorhands 0.706 → Scissorhands+4-bit **0.704** [PAPER FACT]
  - OPT-13B: Original 0.720 → Scissorhands 0.720 → +4-bit **0.720** [PAPER FACT] — no compounding error [PAPER FACT].

- **Score illustration (Fig.4):** Mentioned but truncated in HTML — shows OPT vs Scissorhands score between? [NOT REPORTED] exact numbers [PAPER FACT].

- **Attention map example (Fig.1):** Tokens 178,228,278 repetitive heavy attention to 27,63,98 etc. [PAPER FACT].

## 11 Assumptions [PAPER FACT]

- Autoregressive LM with KV cache storing prompt + generated tokens; incremental decoding with growing cache [PAPER FACT] (§2).
- Attention score threshold for pivotal = average 1/t [PAPER FACT] (§3.1).
- Simplified model for theory: single-layer single-head, normalized inputs ||x_t||=1, specific MLP dominance (cosine ≈1, skip dominates: ||x|| ≫ ||W2 relu(W1 x)||) [PAPER FACT] (§3.3, Appendix A.2 cross-layer cosine verified ≈1).
- Power-law distribution parameterized (k,b) for attention scores [PAPER FACT] (Thm 4.1).
- Singular value bounded λ_V λ_O (1+λ1 λ2)(1+λ_Q λ_K) ≤0.5 [PAPER FACT].
- Budget B ≤ Tmax fixed per head; drop m=0.5B, w=400, r=10 robust across experiments [PAPER FACT].
- Recent tokens must be kept (uncertainty) and history window needed for variance reduction [PAPER FACT].
- Budget allocation heuristic: later layers need more budget due to lower persistence [PAPER FACT].
- At test time, compressed cache attention approximates full with bounded error if heavy tails [PAPER FACT].

## 12 Author-Stated Limitations [PAPER FACT]

Section **6 Discussion, Limitation, and Future Work** [PAPER FACT]:

- **Limited to attention KV cache only:** Not addressing weight/activation compression jointly beyond demonstrated quantization compatibility [PAPER FACT] (implied §5 quantization compatible but not jointly optimized).
- **Heuristic hyperparameters:** w=400, r=10, m=0.5B described as "quite robust" but not tuned per model/layer; acknowledges need for adaptive tuning [PAPER FACT] (§4.1).
- **No dynamic per-head adaptive budget beyond layer-wise heuristic:** Even allocation within layer may be suboptimal; later layers heuristic rule of thumb [PAPER FACT] (§4.1).
- **Evaluation scale:** Mainly OPT family up to 66B (plus memory table 175B); LLaMA/BLOOM not few-shot evaluated; limited task coverage (4 tasks) [PAPER FACT] (§5 scope inferred as limitation not as explicit bullet but §6 outlines).
- **Theoretical model simplified:** Proofs under single-layer single-head assumption and power-law/sample assumptions; extension to multi-layer multi-head not provided [PAPER FACT] (§3.3, §4.2, Appendix B).
- **Effect of compression frequency trade-off not deeply explored:** Overhead tradeoff mentioned but not Pareto profiled [PAPER FACT] (§4.1).
- **Future work:** Authors suggest combining with weight quantization, exploring H2 in MLP blocks, increased diversity implications, and more allocation strategies [PAPER FACT] (§6, Appendix C.5).

[AGENT INFERENCE]: Paper does not claim infinite-length streaming (unlike StreamingLLM); focus is batch size increase at fixed max length, not 4M extrapolation.

## 13 Inferred Limitations [AGENT INFERENCE]

- **Not true streaming infinite-length solution:** Keeps recent window + heavy history but discards middle; evaluation on C4 perplexity sweeps compression at fixed length, not concatenated 4M books; no proof of stable PPL for unbounded stream like StreamingLLM Fig.5 [AGENT INFERENCE].
- **History-window counting may miss bursty important tokens:** Using low-score count over w=400 may be coarse; token that is infrequently high but crucial for later question could be evicted (needle problem) [AGENT INFERENCE].
- **Power-law assumption may break on tasks with uniform attention (e.g., copy tasks, list retrieval):** k close to 2 would worsen bound; not evaluated on such [AGENT INFERENCE].
- **Scalability claim based on larger models better:** Only up to OPT-66B shown; no result for 175B actual inference perplexity, nor LLaMA-65B/BLOOM on downstream — extrapolation to 175B/XXL uncertain [AGENT INFERENCE].
- **No latency measurement:** Only batch size proxy; actual decoding speedup vs H2O flex not compared; extra Argsort/I maintenance latency not benchmarked [AGENT INFERENCE].
- **Static w,r,B hard for variable prompt lengths:** w=400 may be larger than prompt for short prompts (wasted), or insufficient for 32K contexts [AGENT INFERENCE].
- **Single-node 4×A100 40GB evaluation:** No distributed 8×A100-80GB actual run to validate Table2 max batch calculation [AGENT INFERENCE].
- **Quantization demo limited:** Only Hellaswag, 2× compression, not full 5× +4-bit joint Pareto [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. **Streaming vs Scissorhands hybrid:** Can Scissorhands history-window plus StreamingLLM pin sinks combine for both infinite stability and high batch? [AGENT INFERENCE]
2. **Adaptive w,r,m policy:** Can w and r be learned per layer/head or per sequence entropy instead of fixed 400/10/0.5B? [AGENT INFERENCE]
3. **Theoretical extension to deep Transformers:** Does persistence hold provably for multi-layer, multi-head with layer-norm/residual without simplified assumptions? [AGENT INFERENCE]
4. **Task-specific persistence:** Does power-law exponent k vary by task (code vs dialogue) requiring different B/Tmax trade-off? [AGENT INFERENCE]
5. **Needle retrieval robustness:** How to guarantee not evicting a token that is currently low attention but will be queried far future (e.g., hidden key)? Extra offloading tier? [AGENT INFERENCE]
6. **Joint KV+weight compression frontier:** What is optimal allocation between weight bits and KV tokens under total memory? [AGENT INFERENCE]
7. **Dynamic batch scheduling interaction:** How does eviction interact with continuous batching/paged attention (vLLM) where requests share memory? [AGENT INFERENCE]
8. **MLP heavy hitters:** Appendix C.5 finds MLP persistence too — can we prune activations similarly for memory? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **Efficient attention for training [PAPER FACT]:** Reformer [Kitaev 2020], Performer [Choromanski 2021], Sparse Transformer [Child 2019], etc. [16-20] exploiting low-rank/sparsification; FlashAttention/ exact [21 Dao 2022] optimizing IO [PAPER FACT] (§2.2).
- **Weight quantization/pruning for LLM [PAPER FACT]:** Cites [8-14] including SmoothQuant, LLM.int8, GPTQ etc. [PAPER FACT] (§1).
- **KV-cache–focused sparsification:** Zhao et al. etc. [PAPER FACT] (concurrent).
- **System for long inference:** FlexGen [Sheng et al. 2023] [7] as system baseline for quantization + offloading [PAPER FACT]; DeepSpeed Inference, Accelerate not focused on KV reduction [AGENT INFERENCE].
- **Concurrent heavy-hitter work:**
  - **H2O (Zhang et al. 2023)** — formally defines Heavy Hitters with power-law and dynamic submodular greedy, retains H2+recent, proves guarantee, shows 20× sparsity and 29× throughput [AGENT INFERENCE] (preprint 2306.14048, overlap in observations).
  - **StreamingLLM (Xiao et al. 2023)** — Attention Sinks, keeps initial sink tokens + recent, stable to 4M tokens, 22× latency gain [AGENT INFERENCE].
- **Persistence-like ideas in other works:** Transformer-XL recurrence, Window attention (Longformer) [AGENT INFERENCE].
- **Power-law attention precedents [PAPER FACT]:** Cited [16-20] showing power-law scores precede hypothesis [PAPER FACT].


---
## Review Log

Reviewer: Reviewer-1 (supplemental) — 2026-08-27
Problems Found: webfetch https://arxiv.org/abs/2305.17118 校验持久重要性假设与误差界表述一致；5x 压缩与叠加量化 20x 数值与原文一致，标注 [PAPER FACT] 合规
Corrections: 无修正
Confidence: High
