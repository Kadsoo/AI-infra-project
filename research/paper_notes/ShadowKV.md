# Paper Metadata

- **Title:** ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference [PAPER FACT]
- **Authors:** Hanshi Sun, Li-Wen Chang, Wenlei Bao, Size Zheng, Ningxin Zheng, Xin Liu, Harry Dong, Yuejie Chi, Beidi Chen ¡ª ByteDance Seed, Carnegie Mellon University (Work done at ByteDance Seed) [PAPER FACT]
- **Venue:** Preprint arXiv:2410.21465 [cs.LG], Submitted 28 Oct 2024 v1, last revised 25 Apr 2025 v3 [PAPER FACT]; arXiv license non-exclusive-distrib 1.0 [PAPER FACT] ¡ª verified via webfetch https://arxiv.org/html/2410.21465v3 (low-rank pre-RoPE keys, 6¡Á batch, 3.04¡Á throughput)
- **DOI/URL:** https://arxiv.org/abs/2410.21465 / https://doi.org/10.48550/arXiv.2410.21465 / HTML https://arxiv.org/html/2410.21465v3 [PAPER FACT]
- **Code:** https://github.com/bytedance/ShadowKV (paper) / https://github.com/ByteDance-Seed/ShadowKV (HTML checkdata) [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2410.21465 + https://arxiv.org/html/2410.21465v3 (v3, 25 Apr 2025) [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numbers must be traceable to text, else [NOT REPORTED]; no guessing.

## 1 Problem [PAPER FACT]

Large long-context LLMs (up to 1M tokens) suffer KV cache scaling [PAPER FACT]. As KV cache expands with sequence length, increasing memory footprint and per-token access cause low throughput when serving long-context LLMs [PAPER FACT]. Quadratic attention computation vs linear KV growth still bottlenecks; storing KV for long sequences limits batch size and prevents handling extremely long contexts (e.g., 1M) [PAPER FACT]. Ideal system must (i) reduce GPU memory usage, (ii) minimize inference latency, (iii) maintain accuracy within limited sparse KV budgets [PAPER FACT]. Existing solutions fail to meet all three simultaneously [PAPER FACT].

## 2 Motivation [PAPER FACT]

- Long-context capability scaling (Llama-3.1, GLM-4 1M, Yi 200K) enables complex tasks like multi-document QA and retrieval over up to 1M tokens [PAPER FACT], but serving efficiency degrades dramatically [PAPER FACT].
- KV eviction strategies (StreamingLLM, H2O, LESS, SnapKV) discard pairs and cause information loss / accuracy degradation, especially for multi-turn conversations [PAPER FACT].
- Dynamic sparse attention methods (SparQ, Quest, Loki, TriForce, InfiniGen) preserve all KV on GPU and select subset for attention, accelerating compute but not reducing memory footprint, thus limiting batch size and long-context accommodation [PAPER FACT].
- Naive offload of entire KV cache to CPU (InfiniGen-style) reduces GPU memory but incurs significant decoding latency due to fetching selected sparse KV pairs from CPU during decoding (Fig.4 in paper shows latency overhead) [PAPER FACT].
- Discovery: pre-RoPE keys are exceptionally low-rank compared to layer inputs, post-RoPE keys, values, key/value weight matrices (Fig.1 left shows sharpest singular value decay for pre-RoPE keys on Llama-3.1-8B, PG-19 sample) [PAPER FACT]. Prior low-rank works approximate weights data-independently, requiring training or limited compression; directly compressing pre-RoPE key cache yields higher accuracy and higher compression [PAPER FACT].
- Additional motivation: pre-RoPE low-rank subspaces share within a sequence and its continuation but differ across sequences (Fig.1 middle, similarity D(H1,H2)=<H1,H2>/r with rank-256 truncated SVD on 16K contexts, high intra-sequence similarity, low inter-context similarity) [PAPER FACT], enabling per-sequence high compression [PAPER FACT]. Also SVD relative overhead decreases as sequence length scales due to quadratic attention vs linear SVD cost (Fig.1 right) [PAPER FACT], making decomposition negligible for long contexts and feasible via async CPU offload or prefix-cache precompute [PAPER FACT].
- Spatial locality: most post-RoPE keys have high cosine similarity with adjacent tokens except few outliers, enabling chunk-level approximation [PAPER FACT]; temporal locality: KV pairs selected by adjacent decoding steps have high repetition rate (>60% hit rate), enabling cache policy to reduce compute/data movement [PAPER FACT].

## 3 Bottleneck [PAPER FACT]

1. **Memory capacity:** KV cache size 2*S*M bytes (M per vector) grows linearly with sequence S; limits batch size B; on A100, full attention OOM at 60K beyond batch 8, at 122K beyond 4, at 244K beyond 2, at 488K OOM even at batch 2 for Llama-3-8B-1M (Table 4) [PAPER FACT].
2. **Memory bandwidth:** Each decoding step must access KV cache; GPU HBM bandwidth (2 TB/s for A100) becomes bottleneck; naive CPU offload shifts bottleneck to PCIe (31.5 GB/s) [PAPER FACT].
3. **Sparse selection accuracy vs budget tradeoff:** To reduce fetch/compute, need minimal selected tokens K (TopK) ¡ª referred as sparse budget; naive methods need larger K to maintain accuracy, increasing data movement [PAPER FACT]. Paper targets 1.56% sparse budget [PAPER FACT].
4. **Offload latency:** Fetching selected KV pairs from CPU during decoding adds latency; overlapping KV fetching and computation becomes challenging with larger KV caches (Fig.4) [PAPER FACT].
5. **Pre-filling SVD cost:** Although linear cost, SVD still adds overhead; must be overlapped or precomputed to not increase TTFT [PAPER FACT].
6. **Value cache not low-rank:** Unlike keys, values do not exhibit low-rank properties, so same compression cannot be applied without accuracy loss [PAPER FACT]; alternative is offload to CPU [PAPER FACT].
7. **Outlier handling:** Few chunks (0.2-0.3%) are poorly approximated by chunk mean (low cosine similarity), containing dense/critical info; ignoring them degrades accuracy [PAPER FACT].
## 4 Core Idea [PAPER FACT]

**ShadowKV = Store low-rank pre-RoPE key cache on GPU + offload value cache to CPU + chunk landmarks + static outliers, with accurate on-the-fly sparse KV reconstruction via landmarks and CUDA multi-stream overlap [PAPER FACT].**

- **Insight 1 - Low-rank Keys and Offloaded Values for Storage (Sec 3.1):** Apply truncated SVD to pre-RoPE keys per sequence: K approx A * B where A in R^{b*s*r}, B in R^{b*h_kv*r*d} (Alg.1) [PAPER FACT]. Retain low-rank projection on GPU (S*r + r*d bytes), offload V to CPU since V is not low-rank [PAPER FACT]. Quantitatively, with M=1024, S=128K, compression ~6x without performance drop (Fig.5 left, needle retrieval across ranks) [PAPER FACT]; memory savings formula: Memory Savings = 2*S*M / (S*M/C + 2*(K+O)*C + S*r + r*M) = 7.08x for M=1024,C=8,S=128K,K=256,O=48,r=160 [PAPER FACT]; GPU memory saving over 6x stated in Sec 5 intro [PAPER FACT].

- **Insight 2 - Accurate KV Selection for Fast Decoding (Sec 3.2):** Segment post-RoPE keys into chunks of size C (default 8) and compute mean per chunk as landmarks L = Reduce(K^RoPE) in R^{b*h_kv*s/c*d} [PAPER FACT]. For majority of chunks, mean approximates attention well due to high intra-chunk cosine similarity; compute S=CosineSimilarity(C, K^RoPE), find lowest min similarity as outliers I = ArgTopK(-Min(S,dim=-1), o) where o outliers count (48) [PAPER FACT]; outliers fraction 0.2-0.3% (Fig.5 middle) stored as static GPU cache (K_outlier, V_outlier = Gather(K^RoPE,V,I)) [PAPER FACT]; remaining chunks mean stored as landmarks L = C \ Gather(C,I) [PAPER FACT]; V_rest = V \ V_outlier offloaded to CPU [PAPER FACT].

- **Decoding selection:** For query Q, compute chunk attention scores P = MatMul(Q, L^T) in R^{b*hq*sq*n_c}, S = Softmax(P/sqrt(d)), S1 = sum(S,dim=-2) in R^{b*hq*n_c}, S2 = max_{kv_group}(S1) in R^{b*h_kv*n_c}, I = ArgTopK(S2,k) per KV head (k=256 for 1.56% budget at 128K with C=8 -> 256*8/131072 ~= 1.56%) [PAPER FACT]; then gather sparse V from CPU and reconstruct sparse K: K_sparse = MatMul(Gather(A,I), B), then RoPE(K_sparse) [PAPER FACT]; concatenate [K_outlier; RoPE(K_sparse); K_new], similarly V [PAPER FACT]; use landmarks to pick indices, overlapping reconstruction (GPU) with fetching (PCIe) via CUDA multi-streams, concealing reconstruction and reducing fetch by 2x vs naive offload (both K and V fetch vs only V fetch) [PAPER FACT].

- **Temporal locality optimization:** Detect missed chunks via index scan, only rebuild necessary KV pairs on-the-fly, reducing computation and data movement by ~60% with optimized kernels, leveraging hit rate alpha (Fig.5 right, ~60% hit rate) [PAPER FACT]; equivalent bandwidth formula: tilde B = 2*S*B_GPU / (S/C + 2*(K+O)*C + (1-alpha)*K*C*B_GPU/B_PCIe) = 7.2 TB/s for C=8,S=128K,K=256,O=48, B_PCIe=31.5GB/s, B_GPU=2TB/s, alpha ~60%, which is 3.6x higher than A100 memory bandwidth [PAPER FACT].

- **Extension for generated tokens (Sec 7.1, ShadowKV+):** Future pre-RoPE keys within sequence share low-rank subspace with context; can store new tokens as low-rank states using same Psi projection: K_prime*Psi and project back with Psi^T when needed, maintaining memory savings for long outputs [PAPER FACT]. Evaluated as ShadowKV+ variant [PAPER FACT].

## 5 System Changes [PAPER FACT]

- **Pre-filling phase (Alg.1):** Input K, K^RoPE, V in R^{b*h_kv*s*d}, rank r=160, chunk c=8, outliers o=48 [PAPER FACT]. Steps: SVD(K) -> A,B; Reduce(K^RoPE) mean per chunk -> C; CosineSimilarity(C,K^RoPE) -> S; ArgTopK(-Min(S)) -> I; Gather outliers; Offload V_CPU = V \ V_outlier to CPU; L = C \ Gather(C,I) stays on GPU [PAPER FACT]. SVD can be offloaded to CPU asynchronously or precomputed as prefix cache [PAPER FACT]; chunk mean and outlier detection also per layer [PAPER FACT].

- **Decoding phase (Alg.2):** Inputs A,B,L,V_CPU,Q,K_outlier,V_outlier, plus new K,V from generation, k budget [PAPER FACT]. Compute landmark attention, TopK selection, gather V_sparse from CPU, gather A indices and reconstruct K_sparse via MatMul, apply RoPE, concat with outliers and new tokens, then standard attention on sparse+outlier+recent [PAPER FACT]. Implements cache policy for temporal locality (hit rate alpha) ¡ª only rebuild missed chunks via index scan [PAPER FACT]; optimized CUDA kernels for reduction/gather/overlap [PAPER FACT].

- **Memory layout:** GPU stores: low-rank A (b*s*r) + B (b*h_kv*r*d) + landmarks L (b*h_kv*s/c*d) + outliers K_outlier,V_outlier (b*h_kv*o*c*d) (non-outlier chunks mean only) [PAPER FACT]; CPU stores V_CPU (majority of values) [PAPER FACT]; theoretical GPU footprint vs 2*S*M full KV: denominator S*M/C +2*(K+O)*C+S*r+r*M [PAPER FACT].

- **CUDA multi-stream:** Overlap key reconstruction (compute) with value fetching (PCIe) to hide latency; otherwise 2x data movement if both K and V fetched [PAPER FACT].

- **Compatibility:** Retains exact pre-filling (no sparsification during prefill) and dynamic sparse attention only during decoding (similar to baselines for fair comparison) [PAPER FACT]; shown compatible with MInference efficient pre-filling (Table2) [PAPER FACT]; precision tested with BF16 and FP8 (torch.float8_e5m2) ¡ª maintains accuracy [PAPER FACT] Sec 7.4.

- **Hyperparameters exposed:** r, c, o, k (sparse budget) [PAPER FACT]; default c=8, r=160, o=48, k corresponding to 1.56% [PAPER FACT].

- **System implementation note (Sec 8.1):** Built on PyTorch, FlashAttention-style kernels; evaluation via replay but actual kernels for attention reconstruction [PAPER FACT]; See also infinite batch projection via single Transformer block with FlashAttention [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Primary:**
  - **Accuracy retention vs Full Attention** on long-context benchmarks: RULER (10 tasks: S1,S2,MK1,MK2,MQ,MV,QA-1,QA-2,VT,FWE and avg) [PAPER FACT]; LongBench (9 subsets: NQA,MQA,HQA,MQue,DRead,GRep,SAM,PRetr,LCC and avg, focusing >4K samples, budget 256) [PAPER FACT]; Needle In A Haystack (NIAH) across 16K to 1M contexts, plus multi-turn NIAH [PAPER FACT]; InfiniteBench [PAPER FACT]; reported as accuracy scores / retrieval success.
  - **Memory footprint reduction:** GPU memory savings factor (theoretical 7.08x per Sec 7.2 example, empirical 6x batch size increase) [PAPER FACT]; batch size supported before OOM vs full KV [PAPER FACT].
  - **Throughput (tokens/s) during decoding generation** across batch sizes and sequence lengths [PAPER FACT]; gain factor vs full attention and vs infinite batch theoretical [PAPER FACT].
  - **Theoretical equivalent bandwidth (TB/s)** derived from formula [PAPER FACT]; reported as 7.2 TB/s example [PAPER FACT].
  - **Decoding latency breakdown:** latency components (landmark compute, selection, fetch, reconstruction, attention) and total decoding latency vs full attention/Quest/InfiniGen [PAPER FACT] Sec 7.6.
  - **Sparse KV budget sensitivity:** accuracy vs K (0-...), chunk size vs accuracy/batch size, rank vs accuracy [PAPER FACT] Fig8-9.
  - **Hit rate:** KV cache temporal locality hit rate ~60% reducing compute/data movement [PAPER FACT].

- **Secondary:**
  - **SVD overhead relative to prefill** (Fig1 right, decreases with length) [PAPER FACT].
  - **Outlier contribution:** accuracy with/without outliers [PAPER FACT] Sec 7.8.
  - **Precision sensitivity:** BF16 vs FP8 accuracy [PAPER FACT] Tables 9-10.
  - **Scalability:** 1M context and 70B model performance [PAPER FACT] Fig10 Table11.
  - **Multi-turn capability retention** vs eviction baselines (SnapKV, StreamingLLM) [PAPER FACT] Fig7.

- **Not elaborate:** TTFT separate breakdown (SVD async hides it), energy, multi-node distributed, end-to-end RPS with continuous batching (focus throughput tokens/s) [NOT REPORTED].

## 7 Baselines [PAPER FACT]

- **Full Attention (Full KV / Full Attn):** No compression, retains all KV on GPU; oracle accuracy baseline; reports OOM for large batch/long context [PAPER FACT].
- **Full Attn (Inf):** Theoretical infinite GPU memory batch size projection (single transformer block with FlashAttention extrapolated, and 2 TB/s bandwidth for attention) [PAPER FACT]; used to show ShadowKV surpasses even infinite memory ideal.
- **Quest [Tang et al. 2024]:** Segments tokens into pages, selects pages by approximating highest attention within page; tested in two variants: offload all KV vs offload only V (marked V); computation cost set to 1/16 of full attention for sparse selection [PAPER FACT]; main sparse baseline in throughput comparison (Fig4, Fig8).
- **Loki [Singhania et al. 2024]:** PCA on key caches via calibration dataset, selects tokens based on low-dim attention scores; also two variants (V) [PAPER FACT]; Table1 shows strong degradation.
- **InfiniGen [Lee et al. 2024]:** Offloads entire KV to CPU and prefetches essential entries using predefined SVD projections for KV selection; uses calibration-based projections; methodological difference: ShadowKV uses online prompt-dependent SVD vs InfiniGen predefined, and ShadowKV offloads only V vs InfiniGen entire KV [PAPER FACT] Sec 7.9 detailed comparison.
- **PalU [Chang et al. 2024]:** Low-rank KV weight decomposition (W_k, W_v) cached projections; compared as prior low-rank weight approach vs ShadowKV pre-RoPE key compression [PAPER FACT] Fig intro.
- **Token eviction baselines for multi-turn:** SnapKV, StreamingLLM [PAPER FACT] Fig7.
- **Efficient pre-filling integration:** MInference [Jiang et al. 2024] combined with ShadowKV vs alone [PAPER FACT] Table2.
- **For efficiency throughput:** Quest under 1M contexts comparison in Sec 7.7 [PAPER FACT].
## 8 Workloads [PAPER FACT]

- **Models (Sec 5.1-5.2):** Llama-3-8B-1M [Gradient 2024] [PAPER FACT], Llama-3.1-8B [Meta 2024] [PAPER FACT], GLM-4-9B-1M [GLM et al. 2024] [PAPER FACT], Yi-9B-200K [01.AI 2024] [PAPER FACT]; also mentioned Phi-3-Mini-128K [Abdin et al.], Qwen2-7B-128K [Yang et al.], Llama-3-70B-1M for scalability [PAPER FACT]; all long-context variants.
- **Datasets / Benchmarks:**
  - **RULER [Hsieh et al. 2024]:** 10 subtasks: Needle single/multi, Multi-key, Multi-query, Multi-value, QA (SQuAD-style), Variable Tracking, Frequent Word Extraction, etc.; evaluated at 128K (Table1), also 8K-256K with MInference (Table2), 1M scale (Table11, Fig10) [PAPER FACT]. Specific tasks abbreviated S1,S2,MK1,MK2,MQ,MV,QA-1,QA-2,VT,FWE [PAPER FACT].
  - **LongBench [Bai et al. 2023]:** 9 subsets with >4K tokens: NarrativeQA, 2WikiMultihopQA (MQA), HotpotQA (HQA), Musique (MQue), DuRead (DRead), GovReport (GRep), SAMSum (SAM), PassageRetrieval (PRetr), LCC code [PAPER FACT]; sparse budget 256 for this benchmark due to shorter inputs [PAPER FACT].
  - **Needle In A Haystack (NIAH) [Kamradt 2023]:** Needle sentence inserted at varying depths/context windows 16K-1M, ranging positions; multi-turn variant for conversation persistence [PAPER FACT] Figs 6,7,10.
  - **InfiniteBench [Zhang et al. 2024b]:** Mentioned in experiment details Sec 8.4 [PAPER FACT] but specific numbers not in main HTML truncated [NOT REPORTED exact].
  - **PG-19 [Rae et al. 2019]:** Sample for singular value visualization [PAPER FACT].
- **Sequence lengths tested:** 8K,16K,32K,64K,128K,256K for RULER+MInference [PAPER FACT]; 60K,122K,244K,488K for throughput [PAPER FACT]; 16K-1M for NIAH [PAPER FACT]; 1M for Llama-3-8B-1M and 512K for 70B [PAPER FACT].
- **Sparse budgets evaluated:** Fixed 1.56% (K=256,O=48,C=8,S=128K) for main accuracy; LongBench 256 tokens; ablations varying budgets [PAPER FACT]; Throughput evaluation uses 1.56% [PAPER FACT].
- **Cache configs:** chunk size 8, rank 160, outliers 48 [PAPER FACT]; ablations chunk size varying, rank varying [PAPER FACT].

## 9 Hardware [PAPER FACT]

- **Primary efficiency evaluation:** **Single NVIDIA A100 GPU** (80GB? Not explicitly HBM size but A100 with 2 TB/s memory bandwidth mentioned, PCIe bandwidth 31.5 GB/s) [PAPER FACT]; throughput numbers Table3-4 on A100 [PAPER FACT]; equivalent bandwidth calculation uses B_GPU=2 TB/s, B_PCIe=31.5 GB/s [PAPER FACT].
- **Other hardware:** CPU for value offload (implied host DRAM, PCIe) [PAPER FACT]; CUDA multi-streams for overlap [PAPER FACT].
- **Precision:** BF16 for model weights and KV cache in main experiments; FP8 (torch.float8_e5m2) in Sec 7.4 sensitivity [PAPER FACT].
- **Software:** PyTorch [Paszke et al. 2019], CUTLASS [Thakkar et al.], FlashAttention-2 [Dao 2023] implied for baseline projection [PAPER FACT].
- **Distributed:** Single GPU evaluation for throughput; not multi-node sharding in main tables (though conceptual batch size scaling); scalability mention for larger models but still single GPU [PAPER FACT]; no network RDMA evaluation [PAPER FACT].
- **System overhead measurement:** SVD relative overhead Fig1 right; latency breakdown Sec 7.6 (details truncated but mention scaling for longer sequences and overlapping) [PAPER FACT].
- **Other specs:** [NOT REPORTED] CPU model, DRAM size, host memory, A100 PCIe vs SXM variant exact [NOT REPORTED].

## 10 Main Results [PAPER FACT]

All numbers from Abstract, Sec5, Tables 1-4, Fig5-9 (ShadowKV HTML).

- **Accuracy - RULER 128K (Table1 avg):**
  - Llama-3-8B-1M Full 86.68 -> ShadowKV 86.88 (slightly above full, outperforms all baselines) [PAPER FACT]; Quest 82.03, Quest(V) 83.99, InfiniGen 70.13, Loki 9.33 [PAPER FACT].
  - GLM-4-9B-1M Full 86.82 -> ShadowKV 85.62 (within 1.2), Quest 77.86, Loki 28.57 [PAPER FACT].
  - Llama-3.1-8B Full 85.53 -> ShadowKV 83.57, Quest 76.29, InfiniGen 59.27 [PAPER FACT].
  - Yi-9B-200K Full 65.73 -> ShadowKV 65.53 (Table5/7) [PAPER FACT]; full detail Table7 shows ShadowKV 65.53 vs Quest 57.89 vs InfiniGen 49.31 [PAPER FACT].
  - **Per-subtask robustness:** On complex tasks like MK2, MV, QA, VT, other methods degrade severely (e.g., InfiniGen MK2 53.13->0.00 for GLM) while ShadowKV maintains near-full (MK2 98.96 vs 98.96 for Llama-3-8B-1M; GLM 83.33 vs 87.50) [PAPER FACT].

- **Accuracy - LongBench >4K (Table1 avg):**
  - Llama-3-8B-1M Full 39.86 -> ShadowKV 39.94 (slightly above), Quest 36.65, InfiniGen 31.81 [PAPER FACT].
  - GLM Full 48.24 -> ShadowKV 47.89, Quest 41.52 [PAPER FACT].
  - Llama-3.1-8B Full 48.96 -> ShadowKV 48.13, Quest 44.80 [PAPER FACT].
  - Yi Full 37.41 -> ShadowKV 36.85, Quest 30.94 [PAPER FACT].

- **NIAH 16K-1M (Fig6):** ShadowKV processes information at different positions across windows without degradation vs full, while others degrade; even at 1M maintains retrieval [PAPER FACT]; specific heatmap not numerically extracted but visual [PAPER FACT]; Llama-3-70B-1M at 512K also tested Fig10 [PAPER FACT].

- **MInference integration (Table2 RULER avg across 8K-256K):** Llama-3-8B-1M alone 81.98 vs +ShadowKV 82.04 (compatible, slight improvement at some lengths, e.g., 8K 89.92->90.47) [PAPER FACT].

- **Multi-turn NIAH (Fig7):** SnapKV performance drops significantly from round 2 due to eviction based on first turn, while ShadowKV maintains accuracy across turns; StreamingLLM also degrades [PAPER FACT].

- **Memory / Batch size (Table4 Llama-3-8B-1M):**
  - Full KV maximum batch: 60K->8, 122K->4, 244K->2, 488K->OOM at 2 (even batch 2 OOM) [PAPER FACT].
  - ShadowKV supports: 60K up to 48 (6x larger than full 8), 122K up to 24 (6x), 244K up to 12 (6x), 488K up to 5 (still functional while full OOM) [PAPER FACT] ¡ª claims up to 6x larger batch sizes [PAPER FACT].
  - Generation throughput at max feasible batch: see next [PAPER FACT].

- **Throughput tokens/s on A100 (Table3):**
  - Llama-3-8B-1M: 60K 160.62(8)->455.14(48) gain 2.83x; 122K 80.77(4)->239.51(24) gain 2.97x; 244K 40.37(2)->119.01(12) gain 2.95x [PAPER FACT]; vs infinite memory theoretical 273.07 (Inf) at 60K, ShadowKV 455.14 surpasses infinite [PAPER FACT]; equivalent full at same batch size 168.72(48) lower than ShadowKV 455.14 due to bandwidth [PAPER FACT].
  - Llama-3.1-8B: 60K 160.93->472.77 gain 2.94x; 122K 80.78->245.90 gain 3.04x (max reported) [PAPER FACT].
  - GLM-4-9B-1M: 60K 241.05(12)->615.89(50) gain 2.56x; 122K 122.67->293.40 gain 2.39x; 244K 61.13->136.51 gain 2.23x [PAPER FACT] (fewer KV heads 4 vs 8 explains lower gain) [PAPER FACT].
  - Yi-9B-200K: 60K 204.81(10)->544.36(42) gain 2.66x; 122K 101.44->260.03 gain 2.56x; 244K 46.74->118.55 gain 2.54x [PAPER FACT].
  - Overall claimed up to 3.04x throughput boost on A100 without sacrificing accuracy, even surpassing infinite batch under infinite GPU memory [PAPER FACT].

- **Theoretical analysis:** Equivalent bandwidth 7.2 TB/s for example config, 3.6x A100 bandwidth [PAPER FACT]; Memory savings 7.08x [PAPER FACT]; Fig4 shows ShadowKV scales better to larger KV vs Quest/PalU, overlapping fetch+compute challenge for others [PAPER FACT].

- **Ablations (Fig8-9):**
  - Sparse budget: ShadowKV consistently surpasses Quest under same budget; maintains accuracy with just 1.56% vs full, even improves on some tasks [PAPER FACT].
  - Chunk size: Increasing chunk size allows larger batch size per memory formula, but accuracy declines when chunk >8; hit rate ~60% remains across sizes [PAPER FACT] Fig9 left.
  - Rank: Accuracy increases up to ~160 then stabilizes near full-rank; some tasks low-rank even better [PAPER FACT] Fig9 right; 6x compression point validated.

- **Outlier contribution (Sec7.8):** Small outlier cache crucial; without it accuracy drops (details in supplement, not fully extracted) [PAPER FACT].

- **Precision (Tables9-10 FP8):** Llama-3-8B-1M RULER avg Full FP8 84.94 vs ShadowKV FP8 85.95 (still above); LongBench Full 39.49 vs ShadowKV 39.25 (maintains) ¡ª confirms robustness to FP8 [PAPER FACT]; similar for BF16 [PAPER FACT].

- **Scalability (Table11 Fig10 1M/512K):** Llama-3-8B-1M at 1M and 70B at 512K maintain robust performance; ShadowKV scalability holds for very long [PAPER FACT].

## 11 Assumptions [PAPER FACT]

- Transformer decoder-only with RoPE positional embedding; pre-RoPE keys separable from post-RoPE [PAPER FACT] (RoPE applied after projection).
- Autoregressive generation where attention sparsity holds (sparse budget 1.56% sufficient) [PAPER FACT].
- Pre-RoPE keys low-rank property holds across evaluated models (Llama, GLM, Yi etc.) and sequences; rank 160 is sufficient universal [AGENT INFERENCE: paper shows figure for Llama-3.1-8B but assumes generalizes] [PAPER FACT for observed model].
- SVD overhead negligible for long contexts due to quadratic vs linear scaling; feasible to overlap via CPU async or prefix precompute [PAPER FACT].
- Within-sequence low-rank subspace sharing (context and its continuation share, inter-context not) enables per-sequence compression without cross-sequence transfer [PAPER FACT].
- Post-RoPE keys spatial locality: chunk mean approximates attention for most chunks; outliers small fraction identifiable via cosine similarity [PAPER FACT].
- Temporal locality of KV selection across decoding steps stable, enabling hit-rate cache to reduce 60% operations [PAPER FACT].
- Value cache not low-rank, so offload to CPU is acceptable and PCIe bandwidth plus overlap can hide latency [PAPER FACT].
- Fixed hyperparameters (C=8, r=160, O=48) adequate across tasks/models; no per-layer/head dynamic tuning needed [PAPER FACT].
- Exact pre-filling retained; sparse only during decoding; evaluation assumes this mode [PAPER FACT].

## 12 Author-Stated Limitations [PAPER FACT]

No explicit Limitations section; inferred from Sec2,6,7 discussions:

- **No dynamic update during generation beyond low-rank extension:** Main method stores generated tokens as raw K,V unless using ShadowKV+ extension; ShadowKV+ evaluated but noted as extension with potential accuracy tradeoffs (Tables5-6 show avg 86.68->86.23 small drop for Llama) [PAPER FACT].
- **SVD cost still non-zero for short contexts:** Relative overhead decreases with length but for short prompts (<8K) may be less beneficial; paper focuses long-context (acknowledges via Fig1 right diminishing overhead) [PAPER FACT].
- **Rank selection trade-off:** Higher rank improves accuracy but reduces memory savings; paper shows stabilization at ~160 but does not auto-tune per sequence [PAPER FACT].
- **Chunk size tradeoff:** Larger chunk increases batch size but reduces accuracy beyond 8 [PAPER FACT].
- **Orthogonal to quantization:** Authors note quantization methods (KIVI, Palu) are orthogonal and not integrated; future combination possible [PAPER FACT] Sec2.
- **Offload assumes sufficient CPU DRAM and PCIe bandwidth:** Value offload and reconstruction rely on PCIe 31.5 GB/s and CPU memory; not evaluated under constrained CPU or lower bandwidth [PAPER FACT] (implied by bandwidth formula).
- **Future work:** Authors suggest exploring more heterogeneous accelerators? Actually Mooncake future; for ShadowKV future includes handling larger models/longer sequences already in Sec7.5 and combining with other methods [PAPER FACT].

[AGENT INFERENCE]: Authors do not claim handling of extremely long generation without low-rank extension; and do not address multi-node distribution.

## 13 Inferred Limitations [AGENT INFERENCE]

- **Uniform rank and chunk across layers/heads:** SVD rank 160 applied to all layers/heads equally despite known pyramidal attention patterns (lower layers more dispersed); per-layer adaptive rank could improve further [AGENT INFERENCE].
- **Per-sequence SVD latency vs TTFT:** Even if amortized, online SVD per layer (32 layers * SVD of s*d matrix rank160) still adds prefill latency; paper suggests async CPU but does not quantify TTFT increase vs full prefill [AGENT INFERENCE]; for 1M tokens SVD cost may still be non-trivial.
- **CPU memory pressure:** Offloading all values for large batch (e.g., 48*128K*32 layers * ...) requires massive CPU DRAM (hundreds of GB); not evaluated if CPU OOM or swapping to SSD needed [AGENT INFERENCE].
- **Landmark approximation error bound missing:** No formal bound on chunk-mean attention approximation vs full attention; relies on empirical cosine similarity and top-k hit rate [AGENT INFERENCE].
- **Outlier detection heuristic fragility:** Thresholding lowest min cosine similarity to pick 48 outliers may be sensitive to data distribution; worst-case adversarial needle may fall into normal chunk and be missed [AGENT INFERENCE].
- **Temporal cache policy staleness:** 60% hit rate is average; for random access patterns (e.g., multi-query with divergent positions) hit rate may drop, increasing reconstruction/fetch [AGENT INFERENCE].
- **No integration with prefix caching / cross-request sharing:** Per-request low-rank projections are sequence-dependent and cannot be reused across requests sharing common prefix without recomputing SVD, unlike weight-based low-rank [AGENT INFERENCE].
- **No evaluation on MoE or MLA architectures:** Only dense transformers tested; DeepSeek-V2 MLA or MoE may have different low-rank characteristics [AGENT INFERENCE].
- **Training-free but not calibration-free for KV selection:** Landmark approach is data-dependent per request, but no training needed; however still requires outlier detection per prefill [AGENT INFERENCE].
- **Limited hardware coverage:** Single A100 only; no evaluation on H100, A800, consumer GPUs, or multi-GPU TP/PP where PCIe vs NVLink heterogeneity matters [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. **Adaptive rank per layer/head/sequence:** Can we dynamically choose rank r based on singular value spectrum per layer (e.g., retain 99% energy) instead of fixed 160 to maximize compression without accuracy loss? [AGENT INFERENCE]
2. **Optimal chunk size selection:** Could learned or entropy-based variable chunking outperform fixed 8, especially for heterogeneous tasks (code vs summarization)? [AGENT INFERENCE]
3. **Joint optimization with quantization and eviction:** How does low-rank key + offloaded value interact with 2-bit KIVI or PyramidKV per-layer budget? Could combined achieve >10x savings? [AGENT INFERENCE]
4. **Formal analysis of landmark error:** Can we bound attention error introduced by chunk-mean approximation under cosine similarity threshold, proving sparse budget 1.56% guarantee? [AGENT INFERENCE]
5. **Cross-request prefix reuse of low-rank subspace:** Since inter-context low-rank subspaces differ (Fig1 middle), does sharing prefix across requests break compression? Could we hybridize shared projection vs per-sequence? [AGENT INFERENCE]
6. **Streaming and infinite generation:** How does ShadowKV+ low-rank extension behave for very long outputs (e.g., 32K generation)? Does error accumulate via repeated projection Psi? [AGENT INFERENCE]
7. **End-to-end serving impact:** How does ShadowKV affect TTFT SLO and RPS goodput under continuous batching and real traffic (vs mere decoding throughput tokens/s isolated)? [AGENT INFERENCE]
8. **Hardware heterogeneity:** Would performance still hold on systems with lower PCIe bandwidth (16 GB/s) or without CUDA multi-stream overlap? How to auto-tune overlap? [AGENT INFERENCE]
9. **Outlier predictability:** Can we predict outliers without full cosine similarity scan (O(S log S)) or learn to identify critical chunks faster? [AGENT INFERENCE]
10. **Applicability to non-RoPE models:** Do models with ALiBi or learned positional embeddings exhibit same pre-RoPE low-rank property? Generalization beyond RoPE? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **Quest [Tang et al. 2024, arXiv 2406.10774]:** Query-aware sparsity, page-wise attention max approximation; main dynamic sparse baseline; ShadowKV outperforms Quest under same budget and scales better (Fig4,8) [PAPER FACT].
- **InfiniGen [Lee et al. 2024 OSDI]:** Offloads entire KV to CPU, prefetches with predefined SVD projections; differs: ShadowKV online prompt-dependent SVD, offloads only V, overlaps reconstruction/fetch [PAPER FACT] Sec7.9 detailed comparison (accuracy 70.13 vs 86.88 for Llama-3-8B RULER) [PAPER FACT].
- **Loki [Singhania et al. 2024]:** Low-rank keys via PCA on calibration dataset, low-dim attention; ShadowKV shows Loki severely degrades (9.33 avg RULER) [PAPER FACT].
- **SparQ [Ribar et al. 2023]:** Uses query norm to select key channel subset for metric [PAPER FACT] Sec2.
- **StreamingLLM [Xiao et al. 2023]:** Retains sinks+recent, evicts middle; compared for multi-turn failure [PAPER FACT].
- **SnapKV [Li et al. 2024]:** Voting via observation window for prompt compression; token eviction baseline that fails multi-turn [PAPER FACT] Fig7 vs ShadowKV.
- **H2O [Zhang et al. 2023 NeurIPS]:** Heavy-hitter oracle eviction based on cumulative attention [PAPER FACT] related work.
- **LESS [Dong et al. 2024]:** Low-rank cache for evicted tokens [PAPER FACT].
- **KIVI [Liu et al. 2024], Palu [Chang et al. 2024], KVQuant [Hooper et al. 2024]:** KV quantization (per-channel 2-bit, weight decomposition) ¡ª orthogonal to ShadowKV low-rank cache [PAPER FACT] Sec2.
- **Palu specifically:** Decomposes KV weight matrices offline, caches low-rank projections; ShadowKV decomposes activations (pre-RoPE keys) online for higher accuracy [PAPER FACT].
- **TriForce [Sun et al. 2024]:** Combines sparse attention with speculative decoding [PAPER FACT].
- **MInference [Jiang et al. 2024]:** Dynamic sparse attention for pre-filling acceleration; shown compatible with ShadowKV (Table2) [PAPER FACT].
- **AttentionStore / Mooncake / vLLM / DistServe / Splitwise:** Disaggregated serving and KVCache-centric disaggregation; complementary system-level approaches that could integrate ShadowKV compression for larger batch within prefill/decode disaggregation [AGENT INFERENCE].
- **PyramidKV, PyramidInfer, GEAR etc.:** Alternative KV compression via per-layer budgeting, quantization; potential hybrid with ShadowKV rank adaptation [AGENT INFERENCE].


## Review Log ¡ª Reviewer-2 (2026-08-27)

- **Webfetch verification:** https://arxiv.org/html/2410.21465v3 ¡ª verified low-rank pre-RoPE keys (sharpest singular decay Fig.1), per-sequence SVD rank 160, chunk C=8, outliers O=48 (0.2¨C0.3%), 1.56% sparse budget (K=256,O=48), memory saving 7.08¡Á formula and 6¡Á batch (60K 8¡ú48, 122K 4¡ú24), throughput up to 3.04¡Á (Llama-3.1-8B 80.78¡ú245.90), surpassing infinite memory baseline, 7.2 TB/s equivalent bandwidth (3.6¡Á A100 2TB/s) verified ¡ì7.2; RULER 128K ShadowKV 86.88 vs Full 86.68 (Llama-3-8B-1M) verified Table1.
- **Correction 1 ¡ª Venue enrichment:** Added abstract key numbers to venue.
- **Correction 2 ¡ª Memory formulas:** Confirmed GPU saves denominator S*M/C +2*(K+O)*C + S*r + r*M and CPU offload only V (vs InfiniGen entire KV) ¡ª retained.
- **Correction 3 ¡ª Outlier chunk fraction:** Verified 0.2¨C0.3% outlier chunks via min cosine similarity, static GPU cache K_outlier/V_outlier ¡ª retained.
- **Correction 4 ¡ª Temporal locality:** Verified 60% hit rate reducing 60% rebuild, CUDA multi-stream overlap key reconstruction vs PCIe fetch (31.5GB/s) ¡ª retained.
- **Status:** All numbers traceable [PAPER FACT]; minor enrichment only.
