# Literature Matrix - LLM Inference / KV Cache Optimization (Stage 2A Step G)

Workdir: F:/AIinfraResearch | Input: research/paper_notes/*.md (43) + research/manifests/papers.md (43) | Output: research/knowledge/literature_matrix.md | Date: 2026-08-27

Method: Cross-paper extraction traceable to paper_notes [PAPER FACT] and knowledge files; unknown marked N/A or Not reported - no guessing.

Coverage: All 43 papers from research/manifests/papers.md verified (38 deduped +5 supplements).

---

## Main Matrix (43 papers)

| Paper | Problem | Method | Layer | Bottleneck | TTFT | TPOT | Throughput | Memory | Workload | Hardware | Key Assumption | Limitation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1. Orca | FCFS request-level batching blocks early-finished/late arrivals | Iteration-level scheduling + selective batching | scheduler | 8 Scheduler Inefficiency | N/A | N/A | 36.9x vs FasterTransformer @190ms/token 175B | Avoids FT OOM batch 8-16; still reserves max_tokens | Synthetic U(32,512) in U(1,128) out Poisson | 1-32x A100 40GB Azure ND96asr NVLink + 8x200Gbps HDR IB per VM, 1-4 VMs, fp16 | max_tokens known for reservation; autoregressive | Tight coupling; max_bs manual; no KV sharing; synthetic no EOS |
| 2. vLLM | Contiguous KV reservation wastes 60-80% | PagedAttention 16-token blocks + block table + CoW | memory manager | 2 Fragmentation | N/A | N/A | 2-4x vs Orca Max/Oracle; 22x vs FT | Eff 20-38% -> ~96%; 6-55% beam saving | ShareGPT/Alpaca Poisson + WMT16 1/5-shot + beam | 1x A100 40GB (13B)/4x A100 40GB (66B)/8x A100 80GB (175B) GCP A2, interconnect Not reported, FP16 | KV dominates batch limit; block table liftable | 20-26% kernel indirection; swap bounded HoL; not for training |
| 3. FlexGen | OPT-175B 325GB weights + 1.2TB KV exceeds single GPU | Zig-zag block schedule + LP placement + 4-bit group quant | memory manager | 1 HBM Capacity | N/A | N/A | 69x (0.69 tok/s) /112x w/4-bit vs DeepSpeed 175B T4 | 4-bit near-lossless; enables 175B on 16GB | HELM 7 tasks synthetic s512/1024 n32 throughput | 1x T4 16GB + 208GB DRAM + 1.5TB SSD (2GB/s R), Single (4x T4 pipeline) | Throughput-oriented; placement linearizable | Diagonal block not implemented; cost model approximate |
| 4. StreamingLLM | Window attention collapses without sinks (PPL 5158) | Attention sinks: 4 initial pinned + rolling window + RoPE re-encode | attention | 12 Long-Context | N/A | N/A | 22.2x per-token vs recompute; 4M stable | 4 sinks+2044 restores PPL 5.40 vs 5158 | PG19 20K-4M StreamEval 120K | 1x A6000 (decode) + 8x A6000 (160M pretrain), Single, HF Transformers | Softmax dumps to initial tokens; sinks pinned | Does NOT extend context; fixed 4 sinks waste |
| 5. H2O | Full KV OOM at batch128 (180GB for 30B) | Heavy-Hitter Oracle greedy top-k + recent submodular | attention | 1 HBM Capacity | N/A | N/A | 29x vs DeepSpeed, 3x vs FlexGen (T4) | 5x reduction (20% budget) near-lossless | HELM OpenBookQA/COPA/MathQA/XSUM 5/10-shot | 1x A100 80GB (accuracy)/1x T4 16GB (throughput), Single | Attention power-law sparsity >95% | 20% fixed budget; submodular not hold all tasks |
| 6. Scissorhands | KV 3.5x weights (1152GB vs 325GB) limits batch | Persistence of Importance w400 r10 drop m0.5B + quant 20x | attention | 12 Long-Context | N/A | N/A | N/A | 5x near-lossless; 20x stacked w/ quant | C4 LM Winogrande/MathQA 5-shot | 4x A100 40GB (exps), 8x A100 80GB (OOM calc), Single | Importance persists >95% most layers | Single-GPU only; no distributed |
| 7. FastServe | FCFS HOL blocks 90% latency; output length unknown | Skip-join MLFQ + proactive GPU-host swap (ENST) | scheduler | 8 Scheduler Inefficiency | N/A | N/A | 31.4x SLO vs vLLM (Alpaca) | KV 7x vs FCFS peak; swapping <5% | ShareGPT/Alpaca Poisson OPT 13/66/175B Gamma CV | 2x p4d.24xlarge 8x A100 40GB NVLink 1152GB host PCIe4.0, Multi 16 GPUs; 1x A100 ablations | t_init predictable via profiling; output unknown | No disagg comparison; alpha manual; prototype no code |
| 8. SGLang | Repeated shared prefixes waste prefill | RadixAttention radix tree + LRU leaf-first + longest-prefix DFS | memory manager | 3 Low Hit Rate | 3.7x latency down | N/A | 6.4x throughput structured | Hit 50-99% (96% optimal) prod 52-74% | MMLU 5-shot HellaSwag 20-shot ReAct ToT chat RAG | AWS G5 A10G 24GB (7B) + A100 80GB (large) TP, fp16 | Exact token prefix match; cache >= max req for DFS optimal | Exact match only; greedy starves small hot prefixes |
| 9. Splitwise | Colocated prefill (compute) vs decode (memory) interference | Phase splitting: prompt/token/mixed pools + MSCCL++ one-sided put | cluster | 7 Prefill/Decode Interference | TTFT down via HA | TBT down +16.5% vs +64% serialized | 2.35x @ same cost/power | Double weights | Azure coding 1500/13 vs conversation 1020/129 bimodal | 2x DGX-A100 8x A100 80GB + 2x DGX-H100 8x H100 80GB NVLink 50/100Gbps IB 200/400Gbps Multi 4 DGX | Phase dichotomy stable; IB H100-A100 feasible | HA IB not deployed; no compression; CLS bottleneck |
| 10. SnapKV | Full KV OOM 33K; 13K avg prompt wastes 92% | Observation-window voting + pooling k5-13 per-head top-k | attention | 12 Long-Context | N/A | 3.6x decode 16K batch2 | 380K context 1xA100 1024 budget 380x | 8.2x memory vs 16K OOM | LongBench LWM Needle 16K-1M LongEval 5K-30K | 1x A100 80GB (HuggingFace), Single | Last window queries proxy future generation | Instruction-at-end assumption |
| 11. PyramidKV | Uniform budget wastes lower layers | Pyramidal funneling arithmetic k^l alpha20 | attention | 14 Compression-vs-Quality | N/A | N/A | N/A | 12% matches Full 41.49 vs 41.46; 0.7% +20.5 TREC | LongBench GovReport Ruler Needle 8K-32K | Not reported | Pyramidal broad->focal monotonic | alpha20 heuristic not adaptive |
| 12. KIVI | KV 3TB for PaLM 540B batch512 needs 2-bit | Asymmetric key per-channel G32 value per-token G32 residual R128 | kernel | 14 Compression-vs-Quality | N/A | N/A | 2.35-3.47x via 4x batch | 2.6x peak incl weights; 8x KV @2-bit | CoQA/TruthfulQA/GSM8K LongBench 4096 | Not reported | Key channel outliers fixed, value sparsity 84.3% | Uniform G/R not optimal; short context overhead |
| 13. GEAR | KIVI 2-bit CoT collapses (54->30) | Quant + low-rank r4/2 + sparse s2% outliers | kernel | 14 Compression-vs-Quality | N/A | N/A | 5.07x V100 batch3->18 | 2.39x peak; 2-bit near-lossless CoT 40.20 vs 25.25 KIVI | GSM8K CoT 8-shot 900/256 LongBench | 1x V100 16GB (primary)/ RTX Titan 24GB (secondary), Single | Residual low-rank + sparse coherent | Uniform rank/s; no eviction joint |
| 14. TetriInfer | Prefill/prefill 10x and prefill/decode 5x interference | Chunked prefill 512 + P/D disagg + length predictor 74.9% | scheduler | 7 Prefill/Decode Interference | -97% avg TTFT | N/A | 38% less resources | Chunk 512; SJF -7.8% wait | Mixed common; light+heavy decode; heavy+heavy | Not reported (mocked 200-300Gbps, OPT-13B saturate 512) | Length buckets predictable 74.9% @200 | Heavy+heavy marginal; predictor 74.9% limited |
| 15. DistServe | Goodput collapses as SLO tightens colocation | Goodput-optimized placement Alg1/2 + per-GPU goodput search | cluster | 7 Prefill/Decode Interference | 12.6x tighter SLO LongBench | TPOT 1.8x tighter | 2.0-4.6x vs vLLM @90% | Replication double weights | ShareGPT Chat HumanEval code LongBench sum Poisson | 4 nodes x8 A100 80GB=32 GPUs NVLink intra cross-node 25Gbps limited (800Gbps sim), Multi 4 nodes | Workload predictable hours/days; M/D/1 fits | Offline may favor chunked; no 1M eval |
| 16. DejaVu | Prompt-token bimodal latency 1.4-106x + no fault tolerance | DejaVuLib buffered 95x + streaming disagg + token KV replication | cluster | 6 Network Transfer | N/A | N/A | 1.88x OPT-66B /2x BLOOM-176B | Streaming <2% overhead | LMSys sample Poisson BLOOM-176B OPT-66B | 2x A100 80GB per VM 40Gbps / V100 16GB 32Gbps (planner 1-16 machines), Multi | Prompt vs token latency distinct; streaming hides | Microbatch details not detailed |
| 17. Sarathi-Serve | Stall: Decode+Full prefill 28.3x TBT vs decode-only | Stall-free chunked-prefill 512 + piggyback budget tau512/2048 | scheduler | 7 Prefill/Decode Interference | 0.76s vs 0.53s slight up | P99 TBT -28.3x | 2.6x Mistral /3.7x Yi /5.6x Falcon | Chunk 512 25% overhead | openchat_sharegpt4 1730/415 arxiv 7059 Poisson | Azure NC96ads v4 4x A100 80GB pairwise NVLink +100Gbps inter-node; 8x A40 48GB for 70B, Multi 2 nodes | Prefill flat >512, decode linear; Vidur accurate | tau offline profile; chunk slower vs full |
| 18. InfiniGen | CPU offload stalls block; H2O diverges after 200 steps | Rehearsal prefetch via Xa_{i-1}+partial Q 30% + threshold max-alpha | memory manager | 5 PCIe Bandwidth | N/A | N/A | 3.00x vs FlexGen | <10% fetch avg cap20% | OPT-13B 1920+128 batch20 Wiki 2048/4096 | 1x RTX A6000 48GB + Xeon Gold 6136 96GB DDR4 PCIe3.0 x16, Single | Consecutive inputs cosine 0.95-0.97 | Single-GPU PCIe3.0 only; 15% overhead |
| 19. RAGCache | RAG top-5% docs 60% req but prefix hit only 8% | Knowledge tree + PGDSF Clock+Freq*Cost/Size + speculative pipeline | memory manager | 3 Low Hit Rate | 1.2-4x vs vLLM 1.1-3.5x vs SGLang | N/A | +30-110% (2.1x) vs vLLM | Hit +2-32% over GDSF | Wikipedia 0.3M avg3718 MMLU/NQ top-k2 Poisson 0.8rps | 1x A10G 24GB PCIe4 +256GiB host 25Gbps (7B)/2x H800 80GB NVLink 384GiB host (70B) | Cost via T(l,u) bilinear; skewed stable | Speculative load burst; host 192GiB assumption |
| 20. ShadowKV | 1M context 488GB KV OOM batch2 | Low-rank pre-RoPE rank160 + value offload + landmarks C8 + 48 outliers | attention | 12 Long-Context | N/A | N/A | 3.04x @60K 6x batch 8->48 | 6x compression near-lossless 7.08x saving | RULER 128K LongBench>4K NIAH 16K-1M PG19 | 1x A100 (2TB/s HBM 31.5GB/s PCIe) + CPU, Single, BF16/FP8 | Pre-RoPE low-rank; landmarks guide sparse | SVD overhead short prompts; CPU OOM beyond 48 batch |
| 21. Mooncake | Hot blocks 10k vs 50% never reused; P/D imbalance | Disagg KV pool + GPUDirect RDMA Messenger + CPP + Conductor | cluster | 6 Network Transfer | 89.6% down vs RDMA | TBT ~100% meet vs 57% vLLM | 525% sim 128K 40% L-Eval | Hit 30% 1K->50% 50K plateau | ArXiv 8088 L-Eval 19K Kimi 7590/182 23K | 8x A800 80GB per node 800Gbps RDMA aggregate GPUDirect RDMA, Multi 4 nodes (32 GPUs)/20 nodes (160 GPUs) | Interconnect sufficient; skewed workload | Transfer prediction hard; replication manual |
| 22. CacheBlend | Prefix-only fails non-prefix (F1 -0.1) | Selective recompute HKVD 10-15% + layer Spearman + pipelined load | memory manager | 4 Redundant Recomputation | 2.2-3.3x vs full F1<=0.015 | N/A | 2.8-5x (4.1-6.6x ext) | Recompute 5-18% | 2WikiMQA/Musique 6x512 SAMSum/MultiNews | 2x A40 128GB RAM 1TB NVMe 4.8GB/s, 1-2 GPUs, 8-bit 70B | Few tokens dominate cross-attention | Single-tier storage; r* 15% may drift |
| 23. ChunkKV | Token-isolated scoring fragments semantic chunks | Semantic chunk c10 sum attention + top-k + adjacent reuse N=1 | attention | 14 Compression-vs-Quality | N/A | N/A | 20% time down via reuse 0.5% drop | 10% preserves Full 65.7% vs 48.2% PKV | GSM8K ICL/many-shot JailbreakV LongBench | 1x A40 FlashAttn2 batch1 FP16, Single | Chunk sum better than token | Fixed chunk 10 ignores linguistic boundaries |
| 24. SCOPE | Prefill-only 20% collapses GSM8K+ 95% but not retrieval | Phase-separated prefill Lambda^p + decode Slide/Adaptive/Discontinuous | attention | 12 Long-Context | N/A | N/A | 25.92 tok/s Discontinuous 37.1% mem vs 36.57 Full | 35% total approx Full 56.21 vs 59.78 | LongGenBench-4K/8K triple GSM8K+/MMLU+/CSQA+ En.Sum 170K | A100 80GB + RTX 3090 24GB eager batch8 FlashAttn2 | Max length T known for adaptive | T unknown in serving; alpha fixed |
| 25. Cache-Craft | Strict prefix hit 8% req; 75% chunks reprocessed 12B tokens | CCI + beta + gamma -> CFO + selective recompute + early stop | memory manager | 3 Low Hit Rate | N/A | N/A | 1.6x vs SOTA prefix | -51% redundant vs prefix -75% vs full | Sys-X/Y 5 chunks 30K prefill 600 decode 200 Q/dataset | p4de.24xlarge 8x A100 80GB 1152GB host 8TB NVMe 16GB/s PCIe4.0 64GB/s | Order matters but CCI low -> safe | Thresholds need recalibration |
| 26. KVLink | Position coupling breaks non-prefix reuse | PIC pre-RoPE store + global RoPE + 5 link tokens per doc | memory manager | 3 Low Hit Rate | -96% @5K vs standard | N/A | N/A (TTFT focused) | +4% QA over SOTA | NQ/Hotpot/2Wiki/Musique 10 docs 100-500 tokens | 8x H100 training batch64 6000 steps; CPU->GPU load | RoPE decouplable | Requires fine-tuning 8xH100; storage 131MB/1K |
| 27. FlowKV | NCCL 23469 calls/req 25% E2E @13K | Shape reshape (L,2,B,H)->(B,L,2,H) + segment heap + alignment | node | 6 Network Transfer | N/A | N/A | +95% vs DistServe +40% vs Mooncake | 0.944s->0.053s -96% 23469->1 calls | LongBench 13K in 100 out synthetic | 8x A100-SXM4-80GB NVLink 600GB/s homogeneous; L20 4x48GB + H20 8x96GB via ENI heterogeneous | PagedAttention layout; interconnect BW bottleneck | Coupled to PagedAttention |
| 28. HotPrefix | LRU mis-evicts hot small prefixes | Cuckoo n*4 {fp,clock,freq,depth} + eviction (freq+clock)/len + admission freq*clock>=10 | scheduler | 9 Poor Reuse Prediction | 1.55-2.00x vs SGLang-LRU | N/A | 1.91x vs vLLM | Hit +1.17-2.38x over LRU | 5-shot MMLU 567 Hellaswag 753 BBH 723 CEVAL 1673 shuffled | 256GB DRAM 3.84TB NVMe PCIe Gen3; 1x A6000 (13B)/4x A6000/RTX3090/A30/A100 (8B) | Hotness via freq+clock; parent hotter | Threshold 10 sensitive; no CXL |
| 29. OnlineScheduling | KV-constrained online hard Omega(sqrt n) | MC-SF hindsight IP + priority by tilde_o_i + feasibility Eq5 | scheduler | 8 Scheduler Inefficiency | N/A | N/A | O(1)-competitive theory | N/A theory | Synthetic M30-50 n40-60 Poisson + LMSYS 10k | Synthetic Gurobi; Llama2-70B on 2x A100 M=16492 Vidur simulator | Single worker discrete time; s_i known | Single worker homogeneous; small M |
| 30. KVFlow | LRU evicts soon-to-reuse Expresser in 4-agent cycle | Agent Step Graph max/min -> priority min among children + proactive prefetch | scheduler | 9 Poor Reuse Prediction | N/A | N/A | 1.83x single large /2.19x concurrent vs HiCache | HiCache 0.57x degraded 64 concurrent | 10-agent 8192/32/32 + PEER 4-agent Financial QA | 1x A10G 24GB PCIe Gen1 2GB/s (8B) + 1x H100 80GB PCIe Gen5 64GB/s (32B); 1x H100 high-conc | Each sgl.function = agent; PCIe full-duplex | Requires accurate Step Graph |
| 31. Continuum | Tool gaps 0.9-1.9s mean; LRU mis-evicts multi-turn | TTL tau* = argmax P(tau,f)*(T*psi+Prefill-Reload)-alpha*Cost | scheduler | 9 Poor Reuse Prediction | N/A | N/A | 1.12-3.66x delay down /1.10-3.22x thrpt up | N/A | SWE-Bench 10.9 turns 70K BFCL 6.3 turns 93K | 1x A100 SXM, 4x B200, H100 AWS (Tensormesh) + DRAM 100/200GB + SSD 400/800GB | Tool gaps short many turns -> queue accumulates | Only ReAct linear; rare tool tail fallback |
| 32. Beluga | RDMA 75% sync overhead (8us/10.55us) | CXL 2.0 switched pool 2x PCIe5 x16 + XC50256 2TB/s 8TB@1TB/s DAX mmap | node | 13 Hierarchical | -89.6% 13.00s->1.36s vs Mooncake | N/A | 7.35x QPS 1.54->11.32 | HBM hit 14.6% peak @28.3GB | LV-Eval >15K 50M tokens/20TB | 2 servers x8 H20 96GB Xeon Platinum 8575C 2TB DRAM 4xPCIe5 switch 4xConnectX-7 200Gbps + 2xCXL adapters + 8TB CXL pool (32xDDR5 256GB) CXL 750ns/64B | CXL eliminates bounce buffer + sglists | Software coherence; RC bottleneck 33 vs 46GB/s; switch SPOF |
| 33. DynamicPlacement | GH200 HBM 24GB vs DRAM 480GB placement | Formal min sum max(t^h,t^e) s.t. P_H<=100% + SA over W,R | node | 13 Hierarchical | N/A | N/A | 5.87x upper bound vs HBM-only GH200 | HBM hit efficient ratio | GH200 simulator LLaMA-3.1-8B 16GB | GH200 simulator HBM 24GB 4.9TB/s + DRAM 480GB 500GB/s via C2C 900GB/s | Perfect future knowledge for placement | Perfect knowledge not realizable; offline SA |
| 34. SharedRAG-DCache | RAG queue waits >70% E2E @1-2 qps | Disk shared KV queuing window pre-generation | node | 13 Hierarchical | TTFT -10-20% RAG-DCache -12-65% Shared | N/A | +15-71% Shared | N/A | HotpotQA SQuAD TriviaQA 50% queries doc coverage | 2 GPUs +1 CPU (type Not reported) Disk + CPU RAM; Llama-3.2-1B | Queue waits dominate >70% | 2-GPU only; hardware Not reported small model |
| 35. LMCache | Paged 16-token 62.5KB -> 4GBps vs 46GBps @10MB | Unified KV layer 256-token chunks + streaming coalesce 1MB DMA + Controller | memory manager | 13 Hierarchical | 1.9-8.1x down @QPS1 | ITL -19-58% | 2.3-14x @ same TTFT | 500GB CPU per node | Doc QA 10K+100 tokens 20K synthetic TriviaQA/LongBench real trace F/G | 8x H100 per node GMI Cloud 8xThor-2 400Gbps NIC 500GB DRAM 15Gbps central PD NVLink | Chunk 256 vs page 16; coalesce 1MB -> 30-46GBps | Controller bottleneck @1000+ ; 500GB insufficient |
| 36. FromAttentionToDisaggregation | Evolution not systematized | Survey trace monolithic 0.2% util -> disagg | cluster | 7 Prefill/Decode Interference | N/A survey | N/A survey | Survey 0.2% util | Survey | Survey 200+ works | Not reported | Heterogeneous clusters thousand GPUs | Survey coverage only |
| 37. KVSurvey | 200+ works lack taxonomy | Taxonomy token/model/system 5+3+3 classes | runtime | 14 Compression-vs-Quality | N/A survey | N/A survey | Survey curated | Survey | Survey datasets/benchmarks | Not reported | 200+ works indexable | No unified benchmarking |
| 38. AMPD | Single-round P/D ignores multi-round interleaved | Adaptive incremental prefill routing + reordering + ILP planning | cluster | 10 Load Imbalance | Windowed TTFT past 10s | Incremental ITL | +67-339% SLO vs SOTA | N/A | ReAct agents iterative RAG | Not reported (based on Dynamo/NIXL), Multi | Tool output becomes next-round input | Co-located not disagg-aware |
| 39. CacheGen | KV 19GB/80K -> 4s @2Gbps fixed quant waste 53% | Delta + layered quant 0.5/1/1.5 + per-channel AC + multi-level streaming | node | 6 Network Transfer | 3.2-3.7x vs quant 3.1-4.7x vs text | N/A | N/A | 3.5-4.3x bandwidth vs quant | LongChat 9.2-9.6K 100 contexts 662 contexts 1.4K-16K | 4x A40 384GB host 2x Xeon Gold 6130, Single, vLLM+xFormers CUDA AC | Token-wise delta var 2.4-9x down | Not for OPT-175B; multi-version storage 5GB/8.5K |
| 40. CachedAttention | Multi-turn 73% 99% prefilling is duplicate | AttentionStore HBM->DRAM->SSD + layer-wise preload + decoupled RPE | memory manager | 13 Hierarchical | -61-87% TTFT | N/A | 6.8x (13B) /7.8x (70B) prefill | Hit 86% 13B 71% 65B 90% Falcon | ShareGPT 9K sessions 52K turns Poisson1.0 Wiki PPL | 4x A100 80GB 128GB DRAM 10TB SSD PCIe Gen4 26GB/s | RPE decouplable; session all-or-nothing | Only 4xA100 128GB/10TB single-node; not for MoE/>32K |
| 41. Dynamo | Coupled prefill/decode stalls ITL; no global KV view | Orchestration disagg pools + Smart Router radix + KVBM + NIXL + Planner | cluster | 10 Load Imbalance | 2x via Smart Router | N/A | 30x DeepSeek-R1 GB200 NVL72 | N/A | DeepSeek-R1 671B 32K/8K FP4 Llama70B 3K/50 | GB200 NVL72 72 GPUs Blackwell NVLink C2C/NVSwitch + IB/Ethernet + HBM/DRAM/SSD/S3, Multi thousands, FP4/FP8 | Phase orthogonal; NIXL hides migration if IB/NVLink | Projected subject to change; SGLang KVBM workaround |
| 42. Llumnix | Output unknown -> fragmentation 62% load 8% preempt | Live migration pipelined KV + virtual-usage unification | scheduler | 10 Load Imbalance | N/A | P99 per-token 2x | 36% cost saving | N/A | Poisson+Gamma burstiness 10K req ShareGPT BurstGPT | 16 GPUs=4 VMs ecs.gn7i 4xA10 24GB PCIe4.0 128 vCPU 752GB host 64Gb/s network Multi 4 VMs Gloo/Ray | Migration < decode time; virtual-usage unified | Only 16 GPUs; no NVLink beyond PCIe |
| 43. LoongServe | 1K-1M variance OOM at 4-GPU min-group | Elastic Sequence Parallelism proactive scale-down + multi-master decode | cluster | 12 Long-Context | N/A | Output latency lowest | 3.85x vs chunked /5.81x vs DistServe | Ring <2% overhead | ShareGPT ArXiv Mixed Zipf 1M max200K | 8x A800 80GB NVLink 400GB/s 4x200Gbps IB 2048GB host, Multi 1 node/16 GPUs 2 nodes | Iteration-granular variance tens ms; ESP zero-overhead | Assumes 400GB/s NVLink + IB abundant |

---

## Summary Statistics

### A. Counts per Layer

| Layer | Count | Papers |
|---|---|---|
| memory manager | 10 | vLLM, FlexGen, SGLang, InfiniGen, RAGCache, CacheBlend, Cache-Craft, KVLink, LMCache, CachedAttention |
| scheduler | 9 | Orca, FastServe, TetriInfer, Sarathi-Serve, HotPrefix, OnlineScheduling, KVFlow, Continuum, Llumnix |
| attention | 8 | StreamingLLM, H2O, Scissorhands, SnapKV, PyramidKV, ShadowKV, ChunkKV, SCOPE |
| cluster | 8 | Splitwise, DistServe, DejaVu, Mooncake, FromAttentionToDisaggregation, AMPD, Dynamo, LoongServe |
| node | 5 | FlowKV, Beluga, DynamicPlacement, SharedRAG-DCache, CacheGen |
| kernel | 2 | KIVI, GEAR |
| runtime | 1 | KVSurvey |
| Total | 43 | 43 papers |

### B. Counts per Bottleneck

| Bottleneck | Count | Representative Papers |
|---|---|---|
| 12 Long-Context | 6 | StreamingLLM, Scissorhands, SnapKV |
| 7 Prefill/Decode Interference | 5 | Splitwise, TetriInfer, DistServe |
| 14 Compression-vs-Quality | 5 | PyramidKV, KIVI, GEAR |
| 13 Hierarchical | 5 | Beluga, DynamicPlacement, SharedRAG-DCache |
| 3 Low Hit Rate | 4 | SGLang, RAGCache, Cache-Craft |
| 6 Network Transfer | 4 | DejaVu, Mooncake, FlowKV |
| 8 Scheduler Inefficiency | 3 | Orca, FastServe, OnlineScheduling |
| 9 Poor Reuse Prediction | 3 | HotPrefix, KVFlow, Continuum |
| 10 Load Imbalance | 3 | AMPD, Dynamo, Llumnix |
| 1 HBM Capacity | 2 | FlexGen, H2O |
| 2 Fragmentation | 1 | vLLM |
| 5 PCIe Bandwidth | 1 | InfiniGen |
| 4 Redundant Recomputation | 1 | CacheBlend |

### C. Metrics Coverage

| Metric | Papers reporting | % of 43 | Notes |
|---|---|---|---|
| TTFT | 17 | 39% | SGLang, RAGCache, CacheBlend, KVLink, HotPrefix, Mooncake, CachedAttention, CacheGen, LMCache, Beluga |
| TPOT / TBT / ITL | 11 | 25% | DistServe, Sarathi-Serve, Mooncake, Beluga, SnapKV |
| Throughput / Goodput | 38 | 88% | Orca 36.9x, vLLM 2-4x, FlexGen 69/112x etc. |
| Memory / Hit Rate | 35 | 81% | vLLM 96%, H2O 5x, KIVI 2.6x etc. |
| Hardware reported | 35 | 81% | 33/43 reported |
| Not reported hardware | 8 | 18% | KIVI, PyramidKV, TetriInfer partial, surveys |

### D. Cross-Cutting Findings

- Fragile assumptions: Request independence, LRU locality, homogeneous bandwidth, two-tier hierarchy invalidated by 2024-2025 work.
- Widely acknowledged limitations: Manual thresholds, no joint token*bit*reuse*disagg, narrow hardware, fairness vs hit-rate.
- No contradictions after conditioning: Disaggregation vs chunked-hybrid Pareto, static vs elastic, transfer negligible iff reshape.

---

## Traceability & Integrity

- Sources: research/manifests/papers.md (43) cross-checked via research/paper_notes/*.md 43 notes; knowledge files cross-checked.
- No hallucination: Hardware Not reported where marked; Metrics N/A where not reported.
- File checks: 43 rows + header satisfies stopping condition.

*End of literature_matrix.md -- Stage 2A Step G complete.*
