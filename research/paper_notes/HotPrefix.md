# Paper Metadata

- **Title:** HotPrefix: Hotness-Aware KV Cache Scheduling for Efficient Prefix Sharing in LLM Inference Systems [PAPER FACT]
- **Authors:** Yuhang Li, Rong Gu, Chengying Huan, Zhibin Wang, Renjie Yao, Chen Tian, Guihai Chen [PAPER FACT]
- **Venue:** Proc. ACM Manag. Data (SIGMOD) Vol.3 No.4 Article 250, September 2025, 27 pages, DOI 10.1145/3749168 [PAPER FACT]
- **Code:** [NOT REPORTED] implemented on SGLang v0.4.1.post1 [PAPER FACT]
- **Reading Source:** webfetch https://cs.nju.edu.cn/tianchen/lunwen/2026/sigmod26-liyuhang.pdf [PAPER FACT]

> Authenticity rule: [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED] numeric traced else [NOT REPORTED]

## 1 Problem [PAPER FACT]

Prompt engineering introduces shared prefix redundancy; system prompts >1000 tokens shared; prefix sharing needed [PAPER FACT].

## 2 Motivation [PAPER FACT]

- Reuse opportunity large in RAG/few-shot; GPT-3 4.5MB/token, 73GB for 1024 ctx batch16 [PAPER FACT]
- SGLang LRU degrades after shuffle to vLLM level [PAPER FACT]
- Naive offload 2138s with 1200s offload 56 percent [PAPER FACT]

## 3 Bottleneck [PAPER FACT]

1. Dynamic hotness tracking vs overhead [PAPER FACT]
2. Host offload I/O bottleneck [PAPER FACT]
3. GPU eviction misplacement [PAPER FACT]

## 4 Core Idea [PAPER FACT]

HotPrefix = Dynamic Hotness Tracking + Selective Admission + Hotness Promotion pipelined [PAPER FACT]

## 5 System Changes [PAPER FACT]

- Prefix tree per node; Cuckoo filter n buckets 4 entries {fingerprint,clock,frequency,depth} 8-bit max_age 255 MurmurHash Eq2-4 [PAPER FACT]
- Update: reuse frequency+1 clock=max_age, periodic aging clock-1 [PAPER FACT]
- Eviction Eq5 priority=(frequency+clock)/length leaf only [PAPER FACT]
- Admission Eq6 hotness=frequency*clock threshold 10 [PAPER FACT]
- Promotion Alg2 host desc GPU asc pipeline with decode via CUDA stream [PAPER FACT]

## 6 Target Metrics [PAPER FACT]

- Throughput k tokens/s, Latency total time, Offload time GPU-CPU, Cache Hit Ratio [PAPER FACT]

## 7 Baselines [PAPER FACT]

- vLLM 0.6.4.post1, SGLang-LRU 0.4.1.post1, LFU, FIFO, 2Q [PAPER FACT]

## 8 Workloads [PAPER FACT]

- Models LLaMA-2 13B, Gemma 9B, LLaMA-3 8B, Qwen-2 72B [PAPER FACT]
- Datasets 5-shot MMLU 567 tokens, Hellaswag 753, BBH 723, CEVAL 1673 shuffled [PAPER FACT]
- Batch 64 MMLU/BBH 32 CEVAL/Hellaswag [PAPER FACT]

## 9 Hardware [PAPER FACT]

- 256GB DRAM 3.84TB NVMe PCIe Gen3 [PAPER FACT]
- Single A6000 for 13B, 4xA6000 DP/TP, RTX3090/A30/A6000/A100 for 8B [PAPER FACT]
- bfloat16 Top-p 0.95 [PAPER FACT]

## 10 Main Results [PAPER FACT]

- vs SGLang-LRU 1.55-2.00x latency, vs vLLM 1.54-2.25x latency 1.91x throughput, vs SGLang 2x latency 1.64x throughput [PAPER FACT]
- Hit ratio +1.17-2.38x LRU, 1.11-3.27x vLLM [PAPER FACT]
- Offload 25.66s MMLU 24.8 BBH 68.28 CEVAL 27.03 Hellaswag [PAPER FACT]
- Scalability TP 1.57-2.42x SGLang 1.60-2.61x vLLM, DP linear [PAPER FACT]
- Ablation selective admission 10-65x offload reduction [PAPER FACT]

## 11 Assumptions [PAPER FACT]

- Hotness estimable via frequency+clock, parent hotter than child, prefill dominates reuse [PAPER FACT]

## 12 Author-Stated Limitations [PAPER FACT]

- Future multimodal extension only stated [PAPER FACT]

## 13 Inferred Limitations [AGENT INFERENCE]

- Hyperparam sensitivity, offline shuffle, PCIe Gen3 single env, no fairness/privacy [AGENT INFERENCE]

## 14 Open Questions [AGENT INFERENCE]

1. Adaptive Eq5/6 weights? 2. Combine with quantization? 3. Hierarchical storage? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- vLLM, SGLang, CachedAttention, PromptCache, RAGCache, LMCache 2510.09665 [PAPER FACT]

---
### Details Supplement [PAPER FACT]

- Eq5 priority=(frequency+clock)/length penalizes large nodes, Eq6 hotness=frequency*clock [PAPER FACT]
- Cuckoo filter 8-bit fields, host memory size = GPU KV size, admission threshold 10 [PAPER FACT]
- Promotion selects coldest GPU leaves vs hottest host roots, token capacity check, skip if parent evicted [PAPER FACT]
- Pipeline: prefill completes then promotion overlaps decode where reuse rare [PAPER FACT]
- Gemma 9B 1.45-1.64x over SGLang-LRU 1.58-1.91x over vLLM [PAPER FACT]
- CEVAL longest 1673 tokens gives max gain 2.6x [PAPER FACT]
- Ablation E+A+P vs A+P 1.07x hit 7.77x offload 1.38x latency [PAPER FACT]
- Testbed 256GB DRAM 3.84TB NVMe, LLaMA-2 13B Gemma 9B single A6000, Qwen-2 72B 4xA6000 TP, LLaMA-3 8B cross-GPU RTX3090/A30/A6000/A100 [PAPER FACT]
- Workloads avg tokens MMLU 567 Hellaswag 753 BBH 723 CEVAL 1673, batch 64/32 shuffled [PAPER FACT]
- Baselines SGLang variants LRU LFU FIFO 2Q + vLLM PagedAttention coarse [PAPER FACT]
- Related: vLLM SOSP23, SGLang RadixAttention, CachedAttention USENIX24, PromptCache MLSys24, LMCache 2510.09665 [PAPER FACT]

---
## Review Log

Reviewer: Reviewer-3 (RAG/Agent/Heterogeneous) — 2026-08-27
Scope: 读取全文 -> webfetch抽查关键数值 -> 标注核验
Webfetch抽查: CacheBlend arXiv:2405.16444v3 (TTFT 2.2-3.3x/2.8-5x verified), Cache-Craft arXiv:2502.15734v1 (51%/75% verified), KVLink arXiv:2502.16002v4 (TTFT 85-96% verified), KVFlow 1.83x/2.19x, Continuum JCT 1.12-3.66x, Beluga 7.35x, InfiniGen 3.00x, FlowKV 96.8% NCCL reduction, FastServe 31.4x, FlexGen 69x/112x verified
Problems Found: 数值层面无重大错误；HotPrefix 内容简略建议补全 Cuckoo filter 参数及 hardware 细节，其余标注合规
Corrections: 建议 HotPrefix 补全 Host=GPU KV size / CUDA stream promotion / vLLM 0.6.4 等细节；其余无修正
Confidence: High
