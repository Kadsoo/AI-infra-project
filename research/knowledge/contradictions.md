# Contradictions / Tensions Map

> **Workdir:** `F:\AIinfraResearch` | **Input:** `research/paper_notes/*.md` (43) + `research/manifests/papers.md` | **Output:** `research/knowledge/contradictions.md`
> **Date:** 2026-08-27 | **Reviewer:** Contradiction Finder Agent (Stage 2A Step F)
> **Method:** Pairwise conflict scan across 8+ tension pairs; claims traced as `ShortName Year + metric`; resolution via workload/hardware/metric/implementation/scale/assumption analysis. Do NOT force contradictions.

## Summary of Checks

At least 8 potential tension pairs were systematically checked. Of those, **6 tensions** show apparent contradictions resolvable via differing conditions, and **2 checks show no strong contradiction** (consistent or orthogonal). All 43 paper_notes + manifests were sampled; quantitative metrics verified against paper_notes tagged [PAPER FACT].

| # | Pair Checked | Papers (representative) | Domain | Result |
|---|---|---|---|---|
| 1 | Disaggregation vs Chunked-Prefill Hybrid | Splitwise 2023-24 + DistServe 2024 + TetriInfer 2024 vs Sarathi-Serve 2024 | Prefill/decode scheduling | **Tension #1** |
| 2 | Static Disaggregation vs Elastic Sequence Parallelism | DistServe 2024 (4P+4D) + Mooncake 2024-25 vs LoongServe 2024 ESP | Distributed / long-context | **Tension #2** |
| 3 | KV-Transfer Overhead Negligible vs Significant | Splitwise 2023-24 (<7% prompt) + Mooncake/DistServe vs FlowKV 2025 | System implementation | **Tension #3** |
| 4 | Persistent / Uniform Eviction vs Pyramidal / Phase-Separated | H2O 2023 + Scissorhands 2023 + StreamingLLM 2023 vs SnapKV 2024 + PyramidKV 2024 vs SCOPE 2025 | Eviction / compression | **Tension #4** |
| 5 | 2-bit Quantization Sufficiency | KIVI 2024 vs GEAR 2024 | Compression (quant) | **Tension #5** |
| 6 | Prefix-Only vs Arbitrary-Chunk vs Trainable Link Reuse | RAGCache 2024 vs CacheBlend 2024 + Cache-Craft 2025 vs KVLink 2025 | RAG / prefix reuse | **Tension #6** |
| 7 | Full Offload vs Selective vs Low-Rank vs CXL Pool | FlexGen 2023 vs InfiniGen 2024 vs ShadowKV 2024 vs Beluga 2025 | Hierarchical / heterogeneous | **Tension #7** |
| 8 | Single-Round Goodput vs Multi-Round Interleaved | DistServe 2024 / Mooncake 2024 vs AMPD 2026 + KVFlow 2025 + Continuum 2025 | Agent / multi-turn | **No Strong Contradiction** |

**Stopping criterion:** File exists with >=4 tensions (6 provided) + 8 checks satisfied.
---

## Contradiction / Tension #1: Disaggregation Improves Goodput vs Chunked Hybrid Maximizes Capacity

### Papers involved
- **Splitwise 2023-24 (ISCA24)** — Patel et al.
- **DistServe 2024 (OSDI24)** — Zhong et al.
- **TetriInfer 2024 (arXiv 2401.11181)** — Hu et al.
- **Mooncake 2024-25 (FAST25 Best)** — Qin et al.
- **vs Sarathi-Serve 2024 (OSDI24)** — Agrawal et al.

### What each claims
- **Splitwise 2023-24 + metric:** Up to **2.35x throughput at same cost+power** and **1.4x at 20% lower cost** via HA (H100 prefill / A100 token) with layer-wise KV transfer overlapped to **~5ms H100 / 8ms A100 (<7% prompt, 0.8% E2E vs 3% serialized)** (Fig14-15). Iso-power conversation 2.15x vs Baseline-A100 (Fig18a).
- **DistServe 2024 + metric:** **2.0-4.6x higher request rate vs vLLM** at >90% SLO on ShareGPT, **12.6x tighter SLO** on LongBench summarization (Fig8-9). KV transfer **<0.1% total, 95% <30ms** even at OPT-175B via Alg2 NVLink co-location (Fig10), 25 Gbps cross-node or 800 Gbps high-affinity.
- **TetriInfer 2024 + metric:** **38% less resources, -97% avg TTFT, -47% JCT** vs vLLM on mixed workloads (Fig16) via chunked prefill (ChunkSize 512) + disaggregation + power-of-two predicted decode scheduling (74.9% predictor).
- **Mooncake 2024-25 + metric:** **Up to 525% simulated, 40% on L-Eval (>80% cache), 75% more requests under real Kimi** (avg 7590 in /182 out, 50% cache) vs vLLM-20M, using CPP + disaggregated KV pool with RDMA 800 Gbps.
- **Sarathi-Serve 2024 + metric (counter):** **Up to 2.6x (Mistral-7B 1xA100) / 3.7x (Yi-34B 2xA100) / 5.6x (Falcon-180B PP) higher capacity vs vLLM** without disaggregation, via **stall-free chunked-prefills + token-budget hybrid (tau=512 strict /2048 relaxed)**. Explicitly argues disaggregation underutilizes prefill GPU memory and requires KV migration (Sec6). Hybrid full prefill spikes TBT **28.3x vs decode-only** (Fig9); chunked overhead **25% at 512** (Fig14).

### Why they appear to conflict
Both families claim dominant solution to prefill-decode interference with multi-x gains over vLLM, but prescribe opposite architectures: separate pools + KV transfer vs single pool + uniform token budget. If disaggregation were strictly optimal per Splitwise/DistServe goodput curves, Sarathi 3.7x without migration should be impossible. Conversely, if stall-free hybrid is optimal per Sarathi PP bubble analysis, DistServe 12.6x tighter SLO should not hold. TetriInfer heavy-heavy marginal gain vs Sarathi 25% overhead shows neither universally wins.

### Resolution analysis
- **Metric different:** Splitwise/DistServe optimize per-GPU goodput under stringent TTFT+TPOT SLO attainment (90%/99%); Sarathi optimizes capacity at P99 TBT with median scheduling delay <2s. Goodput rewards strict SLO via independent parallelism search; capacity rewards max sustainable Poisson QPS before queue blow-up.
- **Workload different:** Sarathi on openchat_sharegpt4 (median prompt 1730, P90 5696) and arxiv_summarization (7059, P90 12985) total <=8K/16K; DistServe adds HumanEval code (short, TTFT 0.125s) and LongBench 2K capped; Splitwise on Azure coding (1500/13) vs conversation (1020/129) bimodal. Short-output coding favors HA disaggregation; long-prompt arxiv favors chunked hybrid. TetriInfer heavy prefill+heavy decode corner shows disaggregation overhead not offset — aligning with Sarathi regime.
- **Hardware different:** Splitwise 2x DGX-A100 + 2x DGX-H100 IB 200/400 Gbps and assumes IB H100->A100; DistServe 4 nodes x8 A100 25 Gbps (low-affinity forces Alg2 NVLink co-location); Sarathi Azure 4xA100 pairwise NVLink + 100 Gbps inter-node + A40 for TP4-PP2 focusing on PP bubbles not cross-node KV. Transfer cost thus 0.1% (co-located) to 25% (FlowKV NCCL 13K) — hardware + PagedAttention implementation determines viability.
- **Implementation different:** DistServe/Splitwise use layer-wise async MSCCL++ one-sided put zero-copy + semaphore; TetriInfer mocks 200-300 Gbps and flips roles; Sarathi uses FlashAttention v2 + FlashInfer + PagedAttention chunked fused reshape. FlowKV later shows naive NCCL PagedAttention indeed costs 23,469 calls ->1 via (B,L,2,H) reshape + segment alignment, confirming Sarathi skepticism held for naive NCCL but not optimized transfer.
- **Scale different:** DistServe 32 GPUs, Mooncake 20 nodes (160 GPUs) with CPU DRAM pool, Sarathi <=8 GPUs 2 nodes. Placement search yields benefit only when many GPUs allow tailored intra/inter-op (OPT-175B inter3 intra3 prefill vs inter3 intra4 decode). With <=4 GPUs, heterogeneous tuning space collapses.
- **Assumption different:** Both assume Poisson predictable length distribution and no cross-request prefix sharing; Sarathi assumes token budget tau profiled via Vidur; both break under bursty multi-modal mixes.

### Confidence
**Medium-High.** Gains reproduced but mutually exclusive optimal claims dissolve when conditioning on prompt distribution, SLO strictness, and interconnect. No fundamental contradiction — hybrid and disaggregated are Pareto-optimal at different points.
---

## Contradiction / Tension #2: Static Prefill/Decode Partition vs Elastic Sequence Parallelism

### Papers involved
- **DistServe 2024** — static 4P+4D on 8 GPUs via simulator search
- **Mooncake 2024-25** — static 10P+10D on 20 nodes, CPP
- **vs LoongServe 2024 (SOSP24)** — ESP elastic DoP

### What each claims
- **DistServe 2024 + metric:** With 4P+4D on 8 GPUs, 4.3x lower TTFT and 12.6x tighter SLO on LongBench vs vLLM, using tailored intra/inter-op per phase. Placement search exhaustive, minutes, claims optimal per-GPU goodput.
- **Mooncake 2024-25 + metric:** Fixed CPP groups (X, prefill_chunk >1000) claims 525% simulated 128K, 20% arxiv /40% L-Eval on 4 nodes vs vLLM-4M. Acknowledges imbalance: 2P+2D TTFT worse than 3P+1D despite more decode nodes.
- **LoongServe 2024 + metric:** Up to **5.81x total /3.58x input over DistServe (4+4), 3.85x over chunked prefill, 4.64x over vLLM** on 8xA800 single-node (Fig10), and **1.86x multi-node 16 GPUs** vs vLLM, using ESP with zero-overhead scaling: proactive scale-down via ring KV reuse (<2% overhead) and multi-master decode (2x at large batch, <10% otherwise). Also **2.33x/1.98x P90 goodput over best static** (TP2 SP4 / replication) on Zipf.

### Why they appear to conflict
DistServe/Mooncake present static partition + simulator search as optimal, while LoongServe shows same static 4+4 OOMs on L-Eval/LV-Eval/Mixed because min-group capacity limits max length <1M (488 GB KV at 1M) whereas unified pool with ESP succeeds. If DistServe search were globally optimal, LoongServe 5.81x should not exist on same hardware (8 GPUs) and same model (LWM-1M Text 7B).

### Resolution analysis
- **Scale / workload different:** LoongServe targets **1M context (LWM 500K token KV =488 GB)** where single request exceeds any 4-GPU group memory. DistServe caps at 2048 tokens (OPT positional limit); Mooncake simulated 128K with dummy 70B. At <=32K static partition suffices; at 100K-1M elasticity dominates — no contradiction, just operating point beyond DistServe max length.
- **Hardware different:** DistServe low-affinity 25 Gbps forces stage colocation (Alg2), fragmenting pool; LoongServe assumes NVLink 400 GB/s intra-node, 4x200 Gbps IB, reuses ring communication for scaling — hides migration cost.
- **Metric different:** DistServe measures per-GPU goodput under TTFT 0.25s / TPOT 0.1s for chatbot; LoongServe measures normalized latency with SLO 25x base and P90 goodput under Zipf. Goodput definitions diverge; LoongServe win driven by fragmentation and FFN bottleneck, not just isolation.
- **Assumption different:** DistServe assumes workload stable for hours/days to amortize minutes search; LoongServe assumes iteration-granular (tens of ms) variance and user-provided max seq len. Under stable 8-16K workloads static search wins; under 1K-500K mixed with 105.97x prefill variance elastic wins.

### Confidence
**High.** Verified across tables: DistServe Table3 4P+4D vs LoongServe Fig10-12 OOM points. Contradiction resolved by 1M scale and unified distributed KV pool.

---

## Contradiction / Tension #3: KV-Cache Transfer Is Negligible vs Dominates E2E

### Papers involved
- **Splitwise 2023-24** — <7% prompt, constant 8ms A100/5ms H100, 0.8% E2E
- **DistServe 2024** — <0.1% total, 95% <30ms even at 175B
- **Mooncake 2024-25** — RDMA 800 Gbps overlapped
- **vs FlowKV 2025 (arXiv 2504.03775)** — Alibaba

### What each claims
- **Splitwise/DistServe/Mooncake:** Transfer negligible if optimized (layer-wise async, NVLink intra-node, 87GB/s RDMA).
- **FlowKV 2025 + metric:** On LLaMA-3.1-8B/70B with PagedAttention, vanilla NCCL accounts for **~1/4 E2E at 13K in +100 out** (Fig1), with **23,469 calls per request ->1 via (B,L,2,H) reshape + bidirectional segment alignment** (Sec4). Results: **0.3010s vs 0.0044s single 500/100 (55.2x vs Mooncake RDMA), avg 0.944s->0.053s -96% (31.5x single, 12.6x multi-heterogeneous)** (Table3). Throughput vs DistServe/Mooncake/vLLM-Disagg **+95%/+40%/+35%** (Table1-2). Concludes PagedAttention fragmentation causes call storm and SM contention.

### Why they appear to conflict
If transfer were truly <0.1%, FlowKV 25% and 96.8% reduction impossible on similar models. Either earlier papers under-measured or FlowKV workload is pathological.

### Resolution analysis
- **Implementation different (core):** Splitwise uses MSCCL++ one-sided put zero-copy, contiguous blocks sharing semaphore, threshold 512; DistServe constrains to NVLink co-location; Mooncake uses CPU DRAM pool + Messenger GPUDirect RDMA + layer-wise prefetch. FlowKV diagnoses vanilla vLLM-disaggregated NCCL that does not reshape (L,2,B,H) — each non-contiguous block needs Lx2 calls, plus random segment allocation prevents merging, plus per-layer multiple small tensors cause kernel launch + GEMM contention. After FlowKV (B,L,2,H) transpose + segment heap + bidirectional alignment, cost drops to same ~96% overlapped regime — FlowKV confirms earlier optimized path but shows many systems used naive baseline.
- **Hardware different:** Splitwise H100 400 Gbps IB, Mooncake 800 Gbps aggregate, FlowKV heterogeneous L20 48GB + H20 96GB via ENI (limited) and A100 8-GPU NVLink with auto IPC for single-node. Limited ENI amplifies NCCL fragmentation (2.1250s vs 0.0993s multi at 8K/100).
- **Workload different:** FlowKV 13K input longer than Splitwise median 1020-1500 and DistServe 512 capped — transfer size scales linearly, so % grows from 0.8% at 1.5K to 25% at 13K if not overlapped.
- **Metric different:** Splitwise measures second-token latency (+16.5% vs +64% serialized) and prompt-relative <7%; FlowKV measures absolute E2E and throughput (tokens/s) at RPS 0.1-2.0.

### Confidence
**High.** Verified via paper_notes Sec4-5: Splitwise Fig14 constant vs serialized linear, DistServe Fig10 <30ms CDF, FlowKV Table3 23,469->1. No fundamental contradiction — disaggregation cost is negligible iff implementation includes reshape+segment coalescing+NVLink/RDMA, significant otherwise.

---

## Contradiction / Tension #4: Token Importance Is Persistent / Sink-Based vs Pyramidal / Phase-Separated

### Papers involved
- **H2O 2023 (NeurIPS23)** — power-law Heavy Hitters + recent, 20% budget
- **Scissorhands 2023 (NeurIPS23)** — persistence hypothesis, w=400 r=10
- **StreamingLLM 2023 (ICLR24)** — 4 attention sinks + recent, 4M stable
- **SnapKV 2024 (NeurIPS24)** — observation window voting, pooling, 380x
- **PyramidKV 2024 (arXiv 2406.02069)** — pyramidal decreasing per layer
- **vs SCOPE 2025 (ACL25)** — phase-aware prefill+decoding separation

### What each claims
- **H2O 2023 + metric:** 95% sparsity at 1% threshold, power-law H2, retaining H2+recent via greedy accumulated attention (submodular 1-1/e) yields 5x reduction (20% budget) without degradation (e.g., OPT-66B RTE +0.73%, OpenBookQA 43.0 vs 43.2 Full at 20%) and 29x over DeepSpeed /3x over FlexGen on T4.
- **Scissorhands 2023 + metric:** Persistence >95% in most layers, |S|/t <<0.5, retaining pivotal tokens via history-window counter (w=400) + recent r=10, drop m=0.5B gives up to 5x reduction, even + quantization 20x (C4 perplexity flat until 50% for OPT-13B, 75% for OPT-66B).
- **StreamingLLM 2023 + metric:** 4 sink tokens (initial) + recent window restores window attention from catastrophic PPL 5158 ->5.40 (LLaMA-2-13B 65K, 4+1020 vs 0+1024) and stable 4M tokens across models, with 22.2x speedup vs recompute.
- **SnapKV 2024 + metric:** LLM knows before generation: last window queries identify future important keys. Using L_obs window (16) vote + pooling (kernel 5-13) + TopK per head compresses prompt KV to 1024 (92% reduction avg 13K) with negligible drop, 380K context on single A100-80GB (1024 retained, 380x ratio, OOM baseline at 33K) and 11/16 LongBench better than H2O-4096 with only 1024.
- **PyramidKV 2024 + metric:** Observes pyramidal funneling: lower layers broad, upper concentrated (Fig2 layer 0 uniform -> 24-30 spike). Arithmetic decreasing k^l = k^0 - (k^0-k^{m-1})/(m-1)*l with alpha=8 beta=20 yields 12% cache matches Full (41.49 vs 41.46 LLaMA-3-8B 2048) and 0.7% (KV64) up to +20.5 TREC vs SnapKV/H2O. Claims uniform per-layer (H2O/SnapKV/StreamingLLM) is suboptimal.
- **SCOPE 2025 + metric (counter):** Shows prefill-only compression (SnapKV/PyramidKV) collapses on decoding-heavy tasks: GSM8K+ 20% prefill compression ->95% accuracy drop vs PassageRetrieval near-lossless under same 20%. Decoding heavy hitters drift to decoding phase (Top-15% across steps 1/300/500), unified greedy Top-K suffers bias to recent. Phase-separated SCOPE with prefill alpha1+alpha2=2048 (60%, alpha2=8) invariant + decoding Slide/Adaptive/Discontinuous (alpha2=256, alpha1+alpha2=512/1024 =25%/12.5%) achieves near Full at 35% total (Slide 56.21 vs Full 59.78 4K; 52.17 vs 53.26 with SnapKV prefill+SCOPE decode) vs H2O 50.88 / PyramidInfer 53.09, while Discontinuous 25.92 tok/s (37.1% mem) vs Full 36.57.

### Why they appear to conflict
- H2O/Scissorhands claim fixed 20% budget with H2+recent is near-lossless universally; SnapKV claims prompt 92% compression before generation suffices; PyramidKV claims uniform is worst and 12% matches Full; SCOPE claims same 20% on GSM8K+ destroys accuracy but not on retrieval — cannot all be true for same budget.
- StreamingLLM pins 4 initial tokens; H2O may evict them if not heavy; Scissorhands keeps recent 10 but not sinks — which tokens are indispensable?

### Resolution analysis
- **Workload different (primary):** SCOPE isolates reasoning generation (GSM8K+, CoT 30 steps) vs retrieval (PassageRetrieval-en, HotpotQA). Reasoning needs entire prompt chain in decoding (state evolves), so decoding-phase KV matters (~7.4K prefill + 0.5K decoding avg). LongBench/SnapKV avg 13K retrieval QA tolerates heavy prefill pruning because answer depends on local heavy hitters. PyramidKV funneling observed on multi-doc QA — broad lower layers need more budget, upper need less — explains why uniform SnapKV 1024 underperforms PyramidKV 64 on Qasper/TREC but similar on PRe (saturated).
- **Metric / budget different:** H2O 5x at 20% budget (0.2n); SnapKV 380x at 1024 fixed cap for up to 380K (0.26% at 380K, avg 13K ->92%); PyramidKV 0.7% (KV64) and 12% (KV2048) are average across layers with pyramidal shape, not per-layer 20%. SCOPE 35% total =2048 prefill (60% input) +512 decode (25% decode). Direct % comparisons misleading without input-length normalization.
- **Assumption different:** H2O assumes power-law + submodular, local approx global holds for streaming; Scissorhands assumes persistence >95% from first half to second half of same sentence; StreamingLLM assumes softmax dumping to initial tokens (positional bias > semantics, \n still works); SnapKV assumes last L_obs window queries proxy future generation (validated hit rate H). SCOPE falsifies last assumption for long decoding: heavy hitters migrate during decoding (step 300 vs 1), so pre-decoding TopK biases to p-prefill. Its Adaptive formula hat beta1 = (t-alpha2)beta1/(T-alpha2) and Discontinuous every (T-alpha2)/beta1 steps explicitly corrects this — not contradiction but temporal evolution missing in prior static TopK.
- **Hardware / implementation different:** StreamingLLM needs cache-relative RoPE re-encoding on A6000 single GPU HuggingFace; H2O FlexGen backend vs SnapKV pooling kernel vs PyramidKV alpha=8/beta=20 sequence. Small choices shift measured drop.

### Confidence
**High.** Verified hit rates, PPL tables, and SCOPE Table1. Apparent contradictions resolve as workload-dependent optimal budget and phase: retrieval -> prefill-only + pyramidal suffices; reasoning -> need phase-separated + recent preservation; infinite streaming -> sinks essential.

---

## Contradiction / Tension #5: Asymmetric 2-bit Quantization Is Near-Lossless vs Requires Low-Rank+Sparse

### Papers involved
- **KIVI 2024 (ICML24)** — Liu et al.
- **GEAR 2024 (arXiv 2403.05527)** — Kang et al.

### What each claims
- **KIVI 2024 + metric:** Asymmetric per-channel Key (G=32) + per-token Value with residual R=128 is optimal: key per-token reconstruction 13.67 vs 4.55 per-channel, attention error 47.00 vs 9.60, while value Delta 3.55 per-token vs 49.89 per-channel (84.3% sparsity). Claims KIVI-2 (2bit) near-lossless: Llama-2-7B CoQA 63.05 vs 63.88 Full (-0.83), GSM8K 12.74 vs 13.50 (-0.76), LongBench avg 44.27 vs 44.52 (-0.25) and 2.6x less peak memory, 4x batch, 2.35-3.47x throughput.
- **GEAR 2024 + metric:** Claims KIVI collapses on hard CoT: Llama-3-8B GSM8K-CoT 8-shot FP16 54.21/38.19/53.66 vs KIVI 30.17/25.36/30.92 (avg 28.82) vs GEAR (KIVI g64 + low-rank r=4/2 + sparse s=2%) 54.59/38.19/50.30 (avg 47.69 ~ FP16 48.69). Overall +14.95 avg over best baseline at 2bit, up to 24.42% (Abstract). Efficiency: 2.39x memory, 5.07x throughput (V100 16GB batch3->18) with small buffer nb=20, r=4 prefill.

### Why they appear to conflict
Both evaluate on GSM8K and LongBench but opposite conclusions at 2bit: KIVI reports -0.76 on GSM8K (normal) while GEAR reports -25 pts on GSM8K-CoT. If KIVI were truly near-lossless, GEAR extra 2% sparse + rank-4 should be unnecessary.

### Resolution analysis
- **Workload / metric different (decisive):** KIVI GSM8K is 5-shot standard (prefill 672, gen 96) without CoT; LongBench avg 3642 prefill, 256 gen, mix QA/summarization/code where 4bit per-token already near-lossless at 2bit (27.69 vs 27.83 even without GEAR). GEAR GSM8K-CoT 8-shot prefill 900, gen 256 with chain-of-thought multi-step reasoning — dense correlated info, error compounds per step (Fig1 logit diff). KIVI itself notes GSM8K hard: Fake 63.53 vs KIVI 20.77 with window vs 12.21 without — residual window helps but still 2% drop on normal; on CoT drop magnifies to ~40%. Thus KIVI is near-lossless on easy NLU/short-gen, GEAR needed on CoT/reasoning.
- **Implementation different:** KIVI uses group G=32, residual R=128 kept full-precision, fused Q_MatMul, Triton quant — maintains full-precision sliding window R/2 for keys, R for values. GEAR uses KCVT coarse for 4bit but switches to KIVI g=64 nb=64 for 2bit, plus s=2% outlier sparse + head-wise low-rank r=4/2 via power iteration + QR. GEAR-Lite (only low-rank) already 38.34 vs KIVI 25.25, showing coherent residual not captured by KIVI grouping.
- **Hardware / overhead:** KIVI reports 2.6x memory, 4x batch on A100-class; GEAR reports 2.39x memory, 5.07x throughput on V100 16GB with 8-bit weights + CUDA fused kernels. Group-size tradeoff: KIVI G=32 many FP16 scales + residual multiple of G, GEAR KCVT large group reduces overhead. Both hardware-friendly but GEAR streaming buffer nb=20 vs KIVI nb=64/128 changes latency.
- **Assumption different:** KIVI assumes channel outlier pattern persists fixed channels and value sparsity 84.3% makes per-token isolation optimal universally. GEAR assumes residual after quantization has low-rank coherent structure + sparse outliers, and autoregressive compounding requires error < threshold scaling with generation length.

### Confidence
**High.** Verified Table1 per-token 52.93 vs per-channel 63.53 at 2bit and GEAR Table1 40.52->40.20 vs KIVI 25.25 on 2bit CoT. Tension is workload-conditioned: Easy/short -> KIVI sufficient; CoT/long -> need GEAR.

---

## Contradiction / Tension #6: RAG KV Reuse — Prefix-Only Scales vs Arbitrary Reuse Needs 15-30% Recompute vs Trainable Links

### Papers involved
- **RAGCache 2024 (arXiv 2404.12457)** — Jin et al.
- **CacheBlend 2024-25 (EuroSys25 Best)** — Yao et al.
- **Cache-Craft 2025 (SIGMOD25)** — Agarwal et al.
- **KVLink 2025 (arXiv 2502.16002)** — Yang et al.

### What each claims
- **RAGCache 2024 + metric:** Knowledge tree (prefix tree) with PGDSF (Clock+Freq*Cost/Size) storing KV blocks via vLLM PagedAttention. Claims top 3% docs receive 60% requests (20x uniform), cached prefix 11.5x faster than full prefill (3.9x with Host load). TTFT 1.2-4x vs vLLM+Faiss, 1.1-3.5x vs SGLang+Faiss, throughput +30-110% (2.1x) on MMLU/NQ with Mistral-7B (0.125 MiB/token) on A10G 24GB + 192 GiB host cache (host 8->128 GiB PGDSF +2-32% over GDSF, +6-62% over LRU).
- **CacheBlend 2024-25 + metric:** Claims prefix caching limited: only first chunk is prefix, non-prefix cannot reuse. Full reuse ignoring cross-attention degrades F1 0.1-0.2 QA /0.03-0.25 summarization. Proposes selective recompute of 10-15% high-KV-deviation tokens (HKVD) with layer-wise Spearman correlation and pipelined load vs recompute (Trecompute 3ms vs Tload 16ms per layer for 7B 4K on NVMe 4.8 GB/s hideable) -> TTFT 2.2-3.3x vs full recompute/prefix, 5-18% recompute loss <=0.015, throughput 2.8-5x (extended 4.1-6.6x) on A40 2x GPUs, 128 GB RAM, 1 TB NVMe Runpod.
- **Cache-Craft 2025 + metric:** Claims even CacheBlend-style reuse across 3+ history requests fails: >50% requests with 5 chunks scattered across >=3 histories; naive 5-way reuse from 5 histories drops F1 50% even with RoPE correction (Fig8-10). Exact prefix caching hits only 8% requests, 18% prefill tokens (production Sys-X/Y). Proposes CCI (a_bar/b_bar sigmoid), beta (prefix overlap) + gamma (Kendall Tau order penalty) -> CFO, and selective recompute of externally contextualized tokens + relevance-aware early termination. Results: vs prefix -51% redundant compute, vs full -75%, throughput +1.6x, delay -2.1x on LLaMA-3-8B/70B (30% recompute for 90% ROUGE, 45-60% for ~1% of Full) on p4de.24xlarge 8xA100 80GB, 1152 GB host, 8 TB NVMe 16 GB/s.
- **KVLink 2025 + metric:** Claims position-independent reuse via RoPE stripping (store W_{k,v}*x, re-apply global Ri) + K=5 trainable link tokens per document (custom attention: docs attend same-doc + prior link tokens) recovers cross-doc attention without per-token recompute. Results: TTFT -85% at 1K -> -96% at 5K (10 docs x100-500 tokens) vs standard, QA accuracy ~+4% over SOTA (PromptCache 18.6% NQ, CacheBlend 25.7% -> KVLink5 45.0% Llama-3.2-1B; NQ 72.5 vs 70.8 BlockAttention at 8B) and within 1.9% of Finetuned Upperbound (46.9% vs 45.0%). Training on 8xH100, 6000 steps, batch 64; storage 131 MB per 1K tokens for Llama-3-8B.

### Why they appear to conflict
All claim to solve arbitrary-position chunk reuse but report incompatible reuse fraction and quality cost: RAGCache says 11.5x faster prefix is sufficient with tree; CacheBlend says prefix insufficient, need 15% recompute; Cache-Craft says 15% insufficient when mixing across multiple histories, need 30% + CCI-aware; KVLink says no recompute needed, 5 link tokens trained suffice and outperform 18% recompute.

### Resolution analysis
- **Workload different:** RAGCache workload Wikipedia 0.3M docs avg 3718 tokens, top-k=2 (MMLU) /6 (synthetic 512-token chunks): skew 20x uniform, but top-k small, so prefix hit higher via tree (O(h) lookup). CacheBlend synthetic 6 chunks x512 (3K total) random order top-6 per query via L2 — stuff mode where non-prefix chunks dominate. Cache-Craft enterprise Sys-X/Y avg 30K prefill vs 600 decode, top-k=5 dispersed across >=3 histories and long contexts 1K-20K — mixed-source contamination stronger, explaining 50% drop when naive stitching. KVLink Contriever retrieval 10 docs per QA with independent precompute — each doc independent, not mixed histories.
- **Hardware / storage assumption:** RAGCache Host 192 GiB cache PCIe 4.0 x16 A10G; CacheBlend 1 TB NVMe 4.8 GB/s RAM pipelined per-layer load 16ms vs recompute 3ms for 7B hideable; Cache-Craft 8 TB NVMe 16 GB/s Host 1152 GB PCIe 64 GB/s layer-wise preloading + RoPE one-time correction; KVLink CPU->GPU load + RoPE re-encode + link tokens compute negligible (85-96% TTFT still). Storage cost: RAGCache 0.125-0.5 MiB/token, KVLink 131 MB/K tokens — similar, but KVLink also pays training cost 8xH100 6000 steps.
- **Metric different:** RAGCache TTFT includes learning via T(alpha,beta) bilinear cost model: Prefill = MissRate*Full + (1-Miss)*Hit. CacheBlend F1 <=0.02 drop at 2.2-3.3x TTFT, throughput 2.8-5x at same TTFT bound. Cache-Craft ROUGE >=0.6 good, >=0.8 indistinguishable per user study (81%/93% Yes), claims 1.6x throughput at same ROUGE. KVLink Accuracy % + TTFT 85-96% but requires fine-tuning — trades training cost vs tuning-free.
- **Assumption different:** RAGCache assumes document order matters but prefix tree can capture common prefixes — works when queries share same order prefix. CacheBlend assumes HKVD correlation high and sparsity -> few tokens dominate cross-attention. Cache-Craft assumes CCI + beta penalized overlap predicts pollution; naive mixing across histories has high CCI -> CFO high -> needs more recompute. KVLink assumes RoPE can be decoupled and link tokens can learn cross-doc attention via document-internal causal + link-attends-all-prior mask — validated but assumes fine-tuning permissible and KB stable.

### Confidence
**High.** Verified PGDSF priority, HKVD 15% vs CFO 30%, CCI sigmoid, and link K=5 attention mask. Tensions resolve as operating point on reuse-fidelity vs system cost: prefix tree cheapest but only for ordered prefixes; selective recompute tuning-free but needs 15-30% and pipelining; trainable links best quality/TTFT when fine-tuning allowed.

---

## Contradiction / Tension #7: Hierarchical Memory — Full Offload Is Optimal vs Selective / Low-Rank / CXL Needed

### Papers involved
- **FlexGen 2023 (ICML23)** — Sheng et al.
- **InfiniGen 2024 (OSDI24)** — Lee et al.
- **ShadowKV 2024-25 (arXiv 2410.21465)** — Sun et al.
- **Beluga 2025-26 (SIGMOD26)** — Yang et al. (Alibaba)

### What each claims
- **FlexGen 2023 + metric:** On single T4 16GB + 208 GB DRAM + 1.5 TB SSD (2 GB/s read), OPT-175B 69x (0.69 token/s) via 4-bit group (g64) with effective batch 256, 112x (1.12 token/s) with batch 144 vs DeepSpeed/Accelerate batch2 baseline 0.01 (Table2, s512). Using zig-zag block schedule + LP search + 6-way overlap, claims near 2x optimal I/O (Theorem 4.1) and pipeline 4-GPU superlinear decode 201->764 tokens/s. Weight 325 GB, KV 1.2 TB at b512 s512 n32 =3.8x weights.
- **InfiniGen 2024 + metric:** On A6000 48GB + Xeon 96GB DDR4 PCIe 3.0 x16, claims FlexGen full KV fetch wastes 96.9% transfer time (91.8% H2O, 1.52x vs ideal) (Fig18). With offline SVD skew + speculative prefetch at layer i-1 using partial Q_i (30% columns) + dynamic threshold max-alpha (alpha4 OPT/5 LLaMA) -> <10% KV avg cap 20%, achieves up to 3.00x over FlexGen and 32.6 pts accuracy (Abstract, Fig14-16) and Counter eviction = LRU under 80% CPU limit (Table2 Wiki 11.68 vs 19.64 FIFO).
- **ShadowKV 2024-25 + metric:** On A100 2 TB/s HBM PCIe 31.5 GB/s, claims naive offload of entire KV still OOM at 60K batch8, 122K batch4, 488K OOM batch2 for Llama-3-8B-1M (Table4). Using low-rank pre-RoPE keys (rank160, 6x compression) + offload only V to CPU + chunk mean landmarks C=8 + outliers o=48 (0.2-0.3%) + temporal hit 60%, yields 6x batch (8->48 at 60K, 4->24 at 122K), throughput 2.83-3.04x (160->455 at 60K) even surpassing infinite memory baseline 273, 7.08x mem saving, equivalent BW 7.2 TB/s (3.6x A100).
- **Beluga 2025-26 + metric:** On 2 servers x8 H20 96GB Xeon Platinum 2 TB DRAM 8 TB CXL pool via XConn XC50256 (256 lanes 2 TB/s per chip 750ns 64B 1 TB/s to 16 servers), claims RDMA pools (Dynamo, MoonCake) suffer 75% sync overhead (8s of 10.55s for 16KB) sglist 30 limit vs 128 chunks (Qwen-32B GQA 128x20KB) CPU bounce SM polling waste. Using CXL Direct P2P DAX mmap native load/store + custom gather/scatter kernels + CXL RPC 2.11us vs 8.39us RDMA-RC + cache-oblivious scheduling, achieves 89.6% TTFT down (13.00s->1.36s) 7.35x QPS (1.54->11.32) vs MoonCake on LV-Eval Qwen-32B 16 instances (Table5, 14.6% HBM hit) 7.0x write /6.3x read latency vs RDMA, and 16-token block native vs 256-token RDMAsuperblock 76.8s->13.0s.

### Why they appear to conflict
All aim at same bottleneck: KV exceeds GPU HBM (30B 180GB at batch128 s2048, 175B 1152GB) but prescribe opposite hierarchies: FlexGen says disk/CPU offload with 4-bit quantization is sufficient and near-optimal; InfiniGen says disk is not enough, must be selective (<10%); ShadowKV says selective still OOM at 488K need low-rank + only-V offload; Beluga says any RDMA pool is 75% sync waste need CXL. If FlexGen zig-zag were 2x optimal, InfiniGen 3x over FlexGen should be impossible on same PCIe.

### Resolution analysis
- **Workload / scale different:** FlexGen workload batch 512 seq 512-1024 gen 32 throughput-oriented latency-insensitive (t up to 12,000s per block) dummy weights for throughput vs real for accuracy OPT 30/175B. InfiniGen workload batch 4-20 seq 1920+128 (2048) 2000 tokens PG-19 random OPT-13B/30B + Llama-2, metric wall-clock latency with explicit CPU->GPU per-layer fetch. ShadowKV workload long-context 60K-488K (up to 1M) batch 8-48 Llama-3-8B-1M / GLM-4 / Yi, metric throughput under 1.56% sparse budget. At 488K even Infinity batch2 OOM — FlexGen 1.5TB SSD cannot hold 5x48x488Kx layers KV requiring low-rank factor 6x. Beluga workload LV-Eval >15K Qwen-32B GQA 128x20KB non-contiguous 16 vLLM centralized scheduler 28.3 GB HBM KV per 60GB model. RDMA bottleneck is control path (work-request, CQ, QP ordering, sglist 30) and host-staged bounce — negligible for FlexGen single-GPU but dominant for 16-instance scheduler with 74% non-contiguous 16-token loads.
- **Hardware different:** FlexGen T4 16GB (low) + PCIe + 2 GB/s SSD — disk intentionally used, zig-zag amortizes slow read; 13% decode utilization I/O bound -> disk viable when latency not critical. InfiniGen A6000 48GB PCIe3.0 x16 (16 GB/s) — PCIe CPU->GPU 2 orders faster than SSD so selective fetch (<10%) beats full fetch 96.9%. ShadowKV A100 2 TB/s HBM PCIe 31.5 GB/s — HBM 63x PCIe so fetching only V (half bytes) + overlapping reconstruction vs fetch via multi-stream hides cost; low-rank SVD overhead linear vs quadratic attention negligible at 128K. Beluga H20 96GB PCIe5.0 x16 CXL 750ns 64B 1 TB/s pool — CXL direct load/store eliminates bounce + sglists + QP ordering and CUDA-stream integrated control eliminates 8s sync overhead. Cost: $210 adapter vs $1745 NIC $5800 XConn vs $16000 Mellanox $218 vs $800 per 64 GB/s. Thus FlexGen optimal on single-GPU offline throughput low-cost, Beluga optimal on rack-scale low-latency — not contradictory.
- **Assumption about KV sharing:** FlexGen assumes no cross-request sharing per batch isolation. InfiniGen assumes consecutive block inputs highly similar (cosine 0.95-0.97 for OPT, 0.89 for Llama-2) due to outliers + LayerNorm — enabling cross-layer speculation. ShadowKV assumes pre-RoPE keys low-rank within sequence but not across sequences — per-sequence SVD needed not reusable. Beluga assumes single-writer multi-reader pool software coherence via ntstore/CLFLUSH/UC + DSA and 16-token native block is desired — optimal for vLLM PagedAttention.

### Confidence
**High.** Verified block sizes, sync fractions, throughput numbers. Tensions are scale/hardware-conditioned: single-GPU offline -> FlexGen; latency-sensitive 2-32K -> InfiniGen selective; 60K-1M -> ShadowKV low-rank; rack-share 15K+ with 16 instances -> Beluga CXL.

---

## Additional Checks: No Strong Contradiction Found (2 checks)

### Check #8: Single-Round Goodput vs Multi-Round Interleaved Scheduling

**Papers:** DistServe 2024 / Mooncake 2024 (goodput static) vs AMPD 2026 + KVFlow 2025 + Continuum 2025 (+ Dynamo/Llumnix)

**Why checked:** AMPD claims 67-339% SLO attainment gain over colocated and SOTA disaggregated baselines for multi-round (ReAct agents iterative RAG) with interleaved incremental prefill, via adaptive routing (local decode vs routed to prefill) + prefill reordering + ILP deployment planning (alpha-beta T_pre/T_dec/T_kv). This appears to contradict DistServe claim that its placement search maximizes goodput.

**Resolution:** No contradiction — workloads are disjoint. DistServe/Mooncake evaluate single-round Poisson (ShareGPT/HumanEval/LongBench) where each request is one prefill + one decode. AMPD/KVFlow/Continuum target interleaved prefill-decode where environment output (tool call/retrieved doc) becomes next-round incremental input (l_incr) requiring where (local batch pause vs transfer) and how (reorder) decisions plus windowed TTFT/ITL (past 10s) and binding to decode worker. DistServe goodput = max RPS meeting 90% TTFT+TPOT is under-specified for multi-round (TTFT now includes incremental prefill ITL interleaved). KVFlow shows LRU mis-evicts in agent workflows (Agent Step Graph) Continuum TTL decides retention across tool gaps (8x JCT) Llumnix live migration pipeline + virtual usage for load — all orthogonal to DistServe M/D/1 queue. When evaluated on same single-round trace AMPD reduces to DistServe-like.

**Confidence:** High. No data overlap; AMPD explicitly states co-located multi-round optimizations (InferCept discard/swap/preserve vLLM-Continuum TTL MARS/AugServe) are co-located and not disaggregated-aware — gap is extension not contradiction.

### Check #9 (extra): Online Scheduling Theoretical vs Practical

**Papers:** Online Scheduling 2025 (Jaillet et al. competitive ratio) vs FastServe 2023 (31.4x SLO) vs Sarathi-Serve

**Finding:** No contradiction. Theoretical work provides competitive ratio under KV cache constraints with batch scheduling guarantee but assumes adversarial arrivals and clairvoyant lengths; practical systems use FCFS + adaptive batching + prediction and show 31.4x over vLLM on Azure traces with skip-join MLFQ token-level preemption. Theory bounds worst-case practice measures average-case — complementary.

---

## Overall Assessment & Recommendations

**No fundamental contradictions remain after conditioning.** All apparent conflicts are workload-, hardware-, metric-, implementation-, or scale-conditioned Pareto tradeoffs not logical inconsistencies. Key sensitivities:

1. **Transfer cost is implementation-sensitive:** Future disaggregated systems must implement FlowKV-style reshape (B,L,2,H) + segment alignment and layer-wise overlap; otherwise Sarathi skepticism holds.
2. **Eviction optimal is workload-phase-sensitive:** Use StreamingLLM sinks for infinite streaming, H2O/Scissorhands persistence for short prompts, SnapKV/PyramidKV pyramidal for retrieval-heavy 13K avg, SCOPE phase-separated for reasoning-heavy long decoding (GSM8K+). Unified Top-K without phase split will drift.
3. **Quantization level is generation-length sensitive:** KIVI 2bit suffices for NLU/short (LongBench normal GSM8K); reasoning CoT >=256 gen needs GEAR (s2%+r4). Evaluate with CoT benchmark not just NLU.
4. **RAG reuse is order- and history-sensitive:** Prefix tree (RAGCache) for small k=2 (MMLU); selective recompute 15% (CacheBlend) for 6-chunk random-order reuse; CCI-aware 30% + early stopping (Cache-Craft) for multi-history stitching; trainable links (KVLink) only if fine-tuning allowed and KB stable.
5. **Hierarchy is scale-sensitive:** Single-GPU offline -> FlexGen disk zig-zag; latency-sensitive 2-32K -> InfiniGen <10% selective; 60K-1M -> ShadowKV low-rank (rank160); rack-share 15K+ with 16 instances -> Beluga CXL (not RDMA).

**Gaps for Stage 2A:**

- Need unified evaluation harness that sweeps prompt length (1K -> 1M) output length (32 -> 512) batch (1->64) SLO (TTFT 0.1s vs 30s) hardware (T4 vs A100 vs H20 25 Gbps vs 400 Gbps vs CXL) on same traces reporting both goodput and capacity.
- Need ablation that isolates paging layout (PagedAttention) RoPE handling placement search and transfer overlap — to attribute FlowKV vs Splitwise gap.
- Need joint token x bit Pareto (PyramidKV + KIVI/GEAR + SCOPE) instead of isolated curves.

---

## Traceability & Methodology Notes

- **Sources:** All 43 research/paper_notes/*.md + research/manifests/papers.md (38 ->43 after Coverage Auditor). Numbers above tagged [PAPER FACT] in notes; qualitative judgments tagged [AGENT INFERENCE] where noted as such in notes.
- **Reading:** Default.webfetch + local read for short papers; cross-checked via Review Logs (e.g., HotPrefix placeholder noted as incomplete not used for quantitative contradiction).
- **Confidence rubric:** High = >=2 papers with consistent numbers across sections + hardware reported; Medium = single paper or truncated hardware; Low = imputed.
- **Limitations:** Not all 43 papers have quantitative vs vLLM head-to-head (e.g., OnlineScheduling theory FromAttention survey). No fabricated tests; only reported metrics used. Short papers (HotPrefix 5624 bytes OnlineScheduling 5495) have limited data — marked accordingly not forced into contradictions.

*Contradiction Finder Agent — Stage 2A Step F — 2026-08-27 — Workdir F:\AIinfraResearch — Output F:\AIinfraResearch\research\knowledge\contradictions.md (6 tensions + 2 no-strong + 8 checks) — Only this file modified.*
