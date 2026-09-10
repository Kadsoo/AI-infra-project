# Paper Metadata

- **Title:** Beluga: A CXL-Based Memory Architecture for Scalable and Efficient LLM KVCache Management [PAPER FACT]
- **Authors:** Xinjun Yang, Qingda Hu, Junru Li, Feifei Li, Yicong Zhu, Yuqi Zhou, Qiuru Lin, Jian Dai, Yang Kong, Jiayu Zhang, Guoqiang Xu, Qiang Liu [PAPER FACT] ? All Alibaba Cloud Computing (Sunnyvale, Hangzhou, Beijing, Shanghai, Shenzhen) [PAPER FACT]; corresponding: Qingda Hu (qingda.hqd@alibaba-inc.com), Junru Li (rusuo.ljr@alibaba-inc.com) [PAPER FACT]
- **Venue:** arXiv:2511.20172 [cs.DC] v1 25 Nov 2025 (1,958 KB), v2 27 Nov 2025 [PAPER FACT]; 13 pages, accepted by SIGMOD'26 (International Conference on Management of Data, May 31?Jun 05, 2026, Bengaluru, India) [PAPER FACT]; DOI https://doi.org/10.48550/arXiv.2511.20172 [PAPER FACT]; arXiv non-exclusive license [PAPER FACT]; ISBN 978-1-4503-XXXX-X/2026/05 [PAPER FACT]
- **Code:** [NOT REPORTED] ? no public repository linked in paper [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2511.20172 + https://arxiv.org/html/2511.20172v2 [PAPER FACT]

> Authenticity rule: Tagged [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]; numeric values traced to paper, else [NOT REPORTED].

## 1 Problem [PAPER FACT]

Rapid LLM model size growth and long-context inference (e.g., 50M tokens in Kimi requires ~20 TB DRAM for maximal hit ratio [Qin et al. 2025] [PAPER FACT]) make memory the critical bottleneck in GPU-accelerated serving. HBM on GPUs is fast but limited (?96 GB per H20, 60 GB model + 28.3 GB KVCache in evaluation [PAPER FACT]); host DRAM per socket limited by memory channels. Current systems aggregate remote DRAM via **RDMA-based disaggregated memory pools** (Dynamo, MoonCake, LMCache) pooling ~2 TB per server to 4 TB pool via 400 Gbps NICs + Mellanox RoCE switches [PAPER FACT]. RDMA, designed as networking protocol not memory bus, introduces: extra data copies via host bounce buffers, high access latency, complex communication protocols (QP ordering, sglists limited to 30 entries on ConnectX-7), synchronization overhead (CPU-GPU or SM polling) and complex cache-aware scheduling [PAPER FACT]. Emerging **CXL (Compute Express Link) 2.0/3.0** promises native load/store memory semantics over fabric with near-local latency, but prior work limited to CXL 1.1 single-device or FPGA simulations, lacking commercial multi-host evaluation; first production CXL 2.0 switch XConn XC50256 (256 lanes, 2 TB/s per chip, 750 ns 64B minimal latency) now enables large-scale study [PAPER FACT]. Problem: design GPU-accessible shared large-scale memory pool via CXL switches with performance analysis and KVCache management.

## 2 Motivation [PAPER FACT]

- In-database LLM inference rapidly integrated by vendors (MindsDB, Aurora, PolarDB, GaussDB) for RAG QA and NL2SQL, demanding efficient long-context processing on GPU-CPU platforms [PAPER FACT].
- KVCache is essential space-for-time optimization storing KV activations per token per layer, enabling reuse across multi-turn dialogues, shared system prompts, long-document RAG, avoiding recomputation [PAPER FACT] (Fig. 1a RDMA pool vs 1b Beluga).
- Requirements for KVCache storage (?2.1): scalable capacity (gigabytes per request, linear growth), efficient sharing across multi-server GPU clusters, low-latency access (< recomputation to improve TTFT), high aggregate throughput (multi-GPU parallelism) [PAPER FACT].
- Remote pooling needed because local GPU clusters financially/technically infeasible for 20 TB-scale caches; Dynamo/MoonCake adopt RDMA pooling but suffer inefficiencies [PAPER FACT].
- RDMA control vs data path split: CPU-driven (vLLM/MoonCake/LMCache) requires bounce buffer GPU?Host?Remote and CPU issuing RDMA commands; GPU-driven via GPUDirect RDMA bypasses CPU but needs dedicated polling kernel occupying SMs, only on datacenter GPUs (A100/H100, not RTX 4090) [PAPER FACT] (Fig. 2 architecture).
- CXL.mem 2.0 switch-based pooling elevates beyond CXL 1.1 single-node expansion; XConn XC50256 enables 16 servers to 8 TB pool at 1 TB/s, split lanes between memory devices and servers, providing viable alternative with 750 ns 64B latency [PAPER FACT].
- Opportunity: leverage CXL load/store and DSA/Direct P2P to eliminate copies, unify control in CUDA stream, provide unified DAX-mapped address space via mmap() for simple memory management and lower hardware cost [PAPER FACT] (Table 1 cost: CX-7 $1,745 vs Adapter $210; Mellanox switch $16k vs XConn $5.8k; $800 vs $218.75 per 64GB/s [PAPER FACT]).

## 3 Bottleneck [PAPER FACT]

1. **Indirect host-staged data path (RDMA CPU-driven):** Forces all data through bounce buffer in host DRAM (GPU?Host?Remote for writes, reverse for reads) introducing substantial latency [PAPER FACT].
2. **Complex inefficient control path:** Requires costly cross-component synchronization (CPU-GPU coordination or polling/compute SM sync) ? microbenchmark H20 16 KB transfer total 10.55 ?s, actual data movement 2.68 ?s, **~75% (8 ?s) synchronization overhead**, 3? data transfer time [PAPER FACT]; KVCache fragmentation worsens control overhead (single block Qwen-32B GQA requires 128?20 KB non-contiguous transfers; sparse 160-byte ?1024 chunks) [PAPER FACT].
3. **Inefficient resource utilization:** CPU-driven polls waste entire CPU cores; GPU-driven occupies valuable SMs, contending with inference [PAPER FACT].
4. **System complexity:** [PAPER FACT] Non-trivial RDMA programming (work-request prep, CQ polling), manual batching and QP ordering for consistency, tight cache-aware scheduling (locality-driven routing leads to skew, load imbalance, maintenance overhead) [PAPER FACT] (cited Dynamo/MoonCube/Zuo 2025).
5. **Non-uniform memory hierarchy:** Local vs remote latency gap forces scheduler to balance compute vs locality [PAPER FACT].
6. **Hardware constraints:** sglist limit 30 entries vs 128 non-contiguous chunks per Qwen-32B block forces splitting into multiple RDMA requests; consumer GPU lack of GDR [PAPER FACT].
7. **CXL 2.0 lacks multi-host coherence:** CXL.cache/CXL.mem support host-device coherence but not host-host; each CPU has isolated L1/L2/L3 (640 MiB L3, 320 per socket), write stays local, read may see stale cached copy [PAPER FACT]; requires software-managed coherence.

## 4 Core Idea [PAPER FACT]

**Beluga = CXL 2.0 switch-based shared memory architecture + systematic characterization + Beluga-KVCache system (efficient transfers, CXL RPC, cache-oblivious scheduling) [PAPER FACT].**

- **Architecture (?4.1):** Replace 4 RDMA NICs with 2 PCIe/CXL adapters per server; each of 2 CPU sockets (NUMA) connects via PCIe 5.0 x16 adapter to CXL switch (two XC50256 chips, 256 lanes each, 2 TB/s forwarding each) connecting up to 16 servers to 8 TB pool (32?DDR5 4800 MT/s 256GB) at 1 TB/s total bandwidth, supporting concurrent multi-host via address mapping/forwarding (Fig. 2b, Fig. 3 deployment) [PAPER FACT].
- **Inherent advantages (?4.2, Fig. 4):** [PAPER FACT]
  - *Data path:* GPU accesses global pool directly via load/store, P2P cudaMemcpy, custom kernels, eliminating bounce buffers ? latency ? [PAPER FACT].
  - *Control path:* Transfers integrated seamlessly into GPU native CUDA stream, eliminating external CPU coordination / GPU polling sync overhead [PAPER FACT].
  - *Programming:* Same as local DRAM (load/store, DSA), freeing developers from low-level network stack and QP ordering [PAPER FACT].
  - *Memory management:* BIOS reserves contiguous physical address for CXL devices; hosts manage in **Direct Access (DAX) mode** exposing CXL as block device, mmap() mapping entire region into virtual address space ? enables logical partitioning via offsets or sharing same region with unified view [PAPER FACT].
  - *Hardware cost:* CXL components cheaper and not over-provisioned vs 400 Gbps NICs; separates memory from CPU for better cloud utilization [PAPER FACT] (Table 1).
- **Characterization & Optimizations (?5, Table 3 O1-O9):** Evaluate three coherence methods, latency vs size, bandwidth bottleneck (Root Complex) [PAPER FACT] (see ?5).
- **Beluga-KVCache management (?6, Fig. 9):** [PAPER FACT]
  - *Data transfers (?6.1):* Fine-grained custom copy kernel handles unlimited gather writes (many GPU non-contiguous ? one CXL block) / scatter reads (one CXL ? many GPU) for dense (layer?K/V) and sparse (per-head per-layer) KP, avoiding sglist limit and batching [PAPER FACT].
  - *CXL-based RPC (?6.2):* Shared-memory producer-consumer via reserved CXL slots: client writes request + REQ_READY flag via ntstore, server spin-polls flags, writes reply + RESP_READY; optimizations: ntstore to avoid cache pollution, CLFLUSH before read for visibility, batched mfence, cacheline alignment, all user-space (no kernel transitions) [PAPER FACT].
  - *Cache-oblivious scheduling (?6.3):* Near-local latency makes offload to CXL competitive with local memory ? scheduler can ignore KVCache locality, use standard load balancing, add/remove nodes without rebalancing partitions, decoupled from hierarchy [PAPER FACT].

## 5 System Changes [PAPER FACT]

- **Hardware:** Dual-socket Intel Xeon Platinum 8575C per server, 2 TB DRAM (32?DDR5 64GB), 4 PCIe 5.0 switches, 8? H20 96GB GPUs, 4? ConnectX-7 dual-port 200Gbps (for RDMA baseline), 2? PCIe 5.0 x16 CXL adapters per server; switch node + memory box with 32 DDR5 devices [PAPER FACT] (Table 2). OS Ubuntu 22.04 kernel 6.2.0-1015 [PAPER FACT].
- **CXL memory pool:** Exposed as DAX block devices, mmap() into user space; interleaving at 2 MB granularity (software) across 32 devices + 2 adapters; future Granite Rapids hardware interleaving 256B?8-way [PAPER FACT].
- **Software coherence handling (optimizations Table 3):** [PAPER FACT]
  - O1 CPU load/store: ntstore for writes, CLFLUSH before reads (write 2.41 ?s vs CLFLUSH 8.50 vs UC 281.56; read CLFLUSH 5.98 vs UC 166.49 for 16KB) [PAPER FACT] (Table 4 Exp #1)
  - O2 CPU DSA: set memory as Uncacheable (write 1.69 ?s, read 2.12 ?s) [PAPER FACT]
  - O3 GPU: set memory Uncacheable + disable DDIO (D2H 9.14 vs 11.06 with flush, H2D 10.55 vs 16.81) [PAPER FACT]
- **Latency optimizations:** O4 CPU <4KB use direct load/store, >4KB DSA (crossover 16KB DSA wins) [PAPER FACT]; O5 launch kernels async via CUDA streams to hide launch latency (launch dominates 10.55 vs 2.68 data) [PAPER FACT]; O6 for GPU transfers <<24KB on Uncacheable memory use custom kernel (cudaMemcpy on UC takes ~1.23 ms for <24KB) [PAPER FACT] (Fig. 5 Exp #2: CXL?GPU 11.73 ?s vs CPU?GPU 10.32 ?s at 64KB competitive [PAPER FACT]).
- **Bandwidth optimizations:** O7 future direct GPU-to-CXL switch bypass Root Complex (identified as bottleneck: CPU RC limits write 33 vs expected 46.2 GB/s read, GPU?CXL 26 vs 55.4 PCIe, paired NIC?GPU P2P matches); O8 more adapters for scalable BW (2 pairs 46 GB/s vs 1 pair 23-33); O9 interleave across devices (per-device 22.5 GB/s limit) [PAPER FACT] (Fig. 6 Exp).
- **KVCache integration:** vLLM V1 (v0.8.5) prefix caching + HBM KV storage, extended to use Beluga as external pool; replaces LMCache RDMA path with CXL custom kernels for gather/scatter; block handling supports native 16-token blocks without super-block batching [PAPER FACT].
- **RPC service:** Pre-allocate fixed-size slots for request/reply buffering in CXL pool; server internal spin-wait loops polling flags [PAPER FACT].
- **Scheduler:** Centralized scheduler for 16 vLLM instances; eliminates cache-aware routing logic [PAPER FACT].

## 6 Target Metrics [PAPER FACT]

- **Primary:**
  - **End-to-end LLM inference:** Time-To-First-Token (TTFT) avg & P99 seconds [PAPER FACT]; Time-Per-Output-Token (TPOT) avg & P99 seconds [PAPER FACT]; Queries-Per-Second (QPS) / throughput [PAPER FACT]. Reported for cache-populate (first run 30% hit) vs cache-hit (second run pre-populated) [PAPER FACT].
  - **Latency reductions & throughput speedups:** % reduction and ? improvement vs RDMA baselines [PAPER FACT]; claims 89.6% TTFT reduction and 7.35? throughput vs RDMA/MoonCake (abstract & Table 5 second run), also 7.0? lower write latency /6.3? lower read latency in microbenchmarks vs RDMA [PAPER FACT].
- **Secondary:**
  - Microbenchmark latency (?s) for 16KB/64B read/write across CPU/GPU paths vs local DRAM & RDMA (Fig. 5) [PAPER FACT]
  - Bandwidth GB/s for different CXL access paths (Fig. 6) with bottleneck analysis [PAPER FACT]
  - Bandwidth/median/P99 latency under concurrent skewed accesses (zipf 0.99, 16 threads, 64B/16K) (Fig. 7 Exp #3) [PAPER FACT]
  - Latency under background workloads (Fig. 8 Exp #4) [PAPER FACT]
  - Dense KVCache transfer write/read latency breakdown (Fig. 14 Exp #9) [PAPER FACT]
  - Sparse KVCache sparsity analysis (non-contiguous fraction 74% for Qwen32B top256 tokens) and 16-token load latency 97/211 ?s CXL vs 2670/5260 ?s RDMA (Table 6 Exp #10) [PAPER FACT]
  - RPC round-trip latency & throughput (Fig. 15 Exp #11: CXL 2.11 ?s vs RDMA-RC 8.39/UD 8.83 at QD1, 12.13 vs 4.5/6.65 Mops at QD128) [PAPER FACT]
  - Sensitivity to request rates (0.3?9.0 QPS, Fig. 11), input context lengths (2K/4K/8K, Fig. 12), P-D ratio and block sizes (16 vs 256 tokens, Fig. 13) [PAPER FACT]
  - Memory interleaving gain (11.32 vs 8.49 QPS) [PAPER FACT]
  - Cache hit ratio in GPU HBM (peak 14.6% with 28.3 GB KVCache allocation) [PAPER FACT]

## 7 Baselines [PAPER FACT]

- **Local DRAM:** Local memory baseline for latency characterization (Fig. 5) [PAPER FACT]
- **RDMA MemPool:** [PAPER FACT]
  - **MoonCake v3.2** (highly-optimized RDMA baseline on vLLM + LMCache v0.3.1) [PAPER FACT]; cache capacity limited to 2 TB for fairness vs Beluga [PAPER FACT]; CPU-driven with bounce buffer path [PAPER FACT].
  - **Dynamo v0.4.1** (NVIDIA RDMA baseline, v0.8.5? actually Dynamo 0.4.1) [PAPER FACT]
  - **RDMA-RC (Reliable Connection) and RDMA-UD (Unreliable Datagram)** for RPC evaluation with busy-poll CQ [PAPER FACT]
  - **vLLM alone** (no external pool, prefix caching + HBM) as lower bound [PAPER FACT]
- **Variants:**
  - Non-interleaved Beluga vs interleaved [PAPER FACT]
  - Block size 256 vs 16 for RDMA LMCache [PAPER FACT]
- **Versions at submission time** caveat [PAPER FACT].

## 8 Workloads [PAPER FACT]

- **Models:** [PAPER FACT]
  - **Qwen-32B unquantized** default [PAPER FACT]; also Qwen3-32B, Llama-3.1-8B, Qwen3-32B-FP8 for dense/sparse microbenchmarks (Fig. 14) [PAPER FACT]
  - Each model has distinct KVCache block layout: Qwen-32B GQA (n_heads=8) ? 128 non-contiguous sub-blocks (64 layers ?2) per 16-token block, each 20KB; Llama-3.1-8B 64 sub-blocks [PAPER FACT]; Qwen3-32B similar.
  - Sparse models: Qwen-32B / Llama-3-8B with attention-score sparsification, selecting top 256 tokens per head per layer for sequence 7942 tokens ? over 74% non-contiguous in Qwen32B [PAPER FACT]
- **Workloads:**
  - **LV-Eval** long-context QA traces all inputs >15K tokens [PAPER FACT]; variants LV-Eval-2K / -4K / -8K via limiting input context to 2K/4K/8K for sensitivity (Fig. 12) [PAPER FACT]
  - Example 50M tokens Kimi needing 20TB DRAM [PAPER FACT] motivational.
- **Setup:**
  - Cluster **2 servers ?8 H20 96GB =16 vLLM instances** with centralized scheduler [PAPER FACT]; model occupies 60 GB, 92% memory allocated ? **28.3 GB KVCache in HBM** [PAPER FACT]; evaluation closed-loop client model for peak throughput [PAPER FACT]
  - Two scenarios: **Cache-populate (first run)** 30% hit while computing/storing KVCache into Beluga; **Cache-hit (second run)** all KVCache pre-populated, prefill accelerated via reuse [PAPER FACT]
  - Request rates 0.3?9.0 QPS (second run fully cache-hit) [PAPER FACT]
  - Microbenchmark configs: QD=1 for single-request latency; concurrent skewed with 64GB space 16 threads zipf 0.99 (Fig. 7); background workload 0?15 GB/s on same device (Fig. 8) [PAPER FACT]
  - For dense transfer: vLLM default 16-token blocks constant; sparse: loading 16 tokens with top-256 sparsity [PAPER FACT]

## 9 Hardware [PAPER FACT]

- **Table 2 Experimental setup:** [PAPER FACT]
  - OS Ubuntu 22.04 kernel 6.2.0-1015 [PAPER FACT]
  - CPU 2? Intel Xeon Platinum 8575C [PAPER FACT]; L3 640 MiB (320 per CPU) [PAPER FACT]
  - DRAM 2 TB (32?DDR5 4800 MT/s 64GB) [PAPER FACT]
  - PCIe 4? PCIe Switch 5.0 [PAPER FACT]
  - GPU 8? H20 96GB [PAPER FACT]
  - NICs/Server 4? ConnectX-7 dual-port 200Gbps [PAPER FACT]
  - RDMA MemPool 4 TB (2? GPU Servers) [PAPER FACT]
  - PCIe/CXL Adapter/Server 2? PCIe 5.0 x16 [PAPER FACT]
  - CXL MemPool 8 TB (32?DDR5 4800 256GB) [PAPER FACT]
  - Interconnect: NVLink? [NOT REPORTED]; RDMA up to 1.6 Tbps per server via 4 NICs [PAPER FACT]; CXL minimal 64B I/O ~750 ns, switching capacity 2 TB/s per chip (XConn) [PAPER FACT]
- **XConn XC50256 CXL 2.0 switch:** 256 PCIe 5.0 lanes, 2 chips per switch node + memory box split lanes evenly between memory devices and servers; 1 TB/s total BW for 16 servers to 8 TB [PAPER FACT].
- **Beyond characterization:** Future direct GPU-to-CXL fabric (Fig. 16) to bypass Root Complex not yet implemented [PAPER FACT].

## 10 Main Results [PAPER FACT]

- **Abstract headline:** 89.6% TTFT reduction and **7.35? throughput improvement** in vLLM vs RDMA-based solutions [PAPER FACT]; also 7.0? lower write, 6.3? lower read latency vs RDMA in micro [PAPER FACT].
- **Table 5 LV-Eval end-to-end (Exp #5, Qwen-32B, 16 vLLM):**
  - *Cache-populate (30% hit):* Avg TTFT Beluga **17.22 s** vs MoonCake 19.66 vs Dynamo 17.96 vs vLLM 18.76 (12.4% reduction vs MoonCake) [PAPER FACT]; QPS **1.24** vs 1.02 MoonCake /1.15 Dynamo/0.96 vLLM (21.5% ? vs MoonCake) [PAPER FACT]; P99 TTFT 44.6 vs 41.65 MoonCake vs 54.53 Dynamo etc; Avg TPOT 1.54 vs 1.97 MoonCake etc [PAPER FACT]
  - *Cache-hit (pre-populated):* Avg TTFT **1.36 s** vs MoonCake **13.00 s** (89.6% ?) vs Dynamo 15.69 vs vLLM 18.23 [PAPER FACT]; P99 TTFT 5.02 vs 39.91; Avg TPOT **0.15 s** vs 1.10 vs 1.38 vs 2.82; P99 TPOT 1.34 vs 10.58; **QPS 11.32 vs 1.54 (MoonCake 7.35?), vs 1.32 Dynamo (8.57?), vs 0.96 vLLM (11.79?)** [PAPER FACT]; paper highlights up to 4.79? throughput over MoonCake in contributions section for cache-hit scenario [PAPER FACT].
  - Peak GPU HBM cache hit ratio **14.6%** under 28.3 GB allocation [PAPER FACT].
- **Fig. 11 Request-rate sensitivity (0.3?9.0 QPS, second run):** Beluga consistently lower TTFT/TPOT, gap because read becomes bottleneck and CXL more efficient [PAPER FACT].
- **Fig. 12 Context length sensitivity (2K/4K/8K):** Benefit grows with length (KVCache transfer larger fraction); e.g., at 8K improvement most significant [PAPER FACT].
- **Fig. 13 Software config sensitivity:**
  - P-D disaggregated: Beluga **3.41??9.47? higher QPS** vs MoonCake depending on P/D ratio [PAPER FACT] (Fig.13a/b).
  - Block size: MoonCake 256-token block 13.0 s TTFT but **76.8 s with 16-token (exceeds recompute first run)**; Beluga efficient with native 16-token, eliminating batching [PAPER FACT] (Fig.13c).
  - Interleaving: **11.32 vs 8.49 QPS (+33.2%)** with software interleaving across 2 adapters + 32 devices [PAPER FACT].
- **Microbenchmarks (Fig.5 Exp #2 QD1):** [PAPER FACT] CXL?GPU at 64KB 11.73 ?s vs CPU?GPU 10.32 ?s competitive; CPU <4KB direct store faster than DSA, DSA faster >4KB (crossover 16KB) [PAPER FACT].
- **Concurrent skewed (Fig.7 Exp #3):** Median latency CXL **10.2%?13.3% of RDMA for 64B**, 39.5%?56.2% for 16KB; write latency comparable to local DRAM at 16KB [PAPER FACT]; without interleaving 16KB skewed BW lower / latency higher due to first device bottleneck [PAPER FACT].
- **Background workload (Fig.8 Exp #4):** Median stable regardless of 0?15 GB/s background, P99 increases when same-direction pressure, demonstrating bidirectional capability [PAPER FACT].
- **Dense transfer (Fig.14 Exp #9):** vs MoonCake CPU-centric two-step, Beluga reduces **write 36.2% / read 38.7%** latency across Qwen3-32B/Llama-3.1-8B/Qwen3-32B-FP8 [PAPER FACT].
- **Sparse transfer (Table 6 Exp #10):** Non-contiguous tokens 131330 vs contiguous 130814 (Llama8B) and 782774 vs 265802 (Qwen32B) per head+per layer top256; load 16 sparse tokens latency CXL **97 vs RDMA 2670 ?s (Llama)**, **211 vs 5260 ?s (Qwen)** ? **95.9% reduction for Qwen** due to single kernel vs numerous RDMA requests [PAPER FACT].
- **RPC (Fig.15 Exp #11):** QD1 round-trip CXL **2.11 ?s** vs RDMA-RC **8.39 ?s** (4?) vs RDMA-UD 8.83 ?s (4 operations + queuing) [PAPER FACT]; QD128 throughput **12.13 Mops** vs 4.5 (2.7?) vs 6.65 (1.8?) [PAPER FACT]; notes lower reliability vs RDMA, relies on upper-layer mechanisms [PAPER FACT].
- **Coherence Table 4 (Exp #1 16KB):** Detailed numbers confirm O1-O3 optima as in ?5 (see System Changes) [PAPER FACT].
- **Bandwidth bottleneck (Fig.6 Exp):** Read 46.2 GB/s vs write 33 GB/s CPU?CXL single adapter; GPU?CXL 26 vs 55.4 PCIe; two pairs scale to 46 GB/s [PAPER FACT].

## 11 Assumptions [PAPER FACT]

- CXL.mem load/store semantics sufficient for KVCache sharing; single-writer multiple-reader model eliminates concurrent write contention beyond single insertion [PAPER FACT].
- Hosts can configure CXL regions as Uncacheable via MTRRs and disable DDIO to achieve software coherence; software flush/bypass instructions (ntstore, CLFLUSH*, DSA bypass flag) correctly enforce visibility [PAPER FACT].
- PCIe full-duplex and bidirectional CXL capability allows overlapping background traffic with measured P99 stability [PAPER FACT].
- KVCache can be uniformly interleaved at 2 MB granularity without excessive metadata overhead [PAPER FACT].
- GPU HBM hit ratio peak 14.6% representative of production with 28.3 GB allocation [PAPER FACT].
- Network stack overhead dominates RDMA control path, justifying load/store replacement [PAPER FACT].
- Reliability of CXL RPC lower than RDMA is acceptable for rack-scale deployments with upper-layer guarantees [PAPER FACT].
- Precision is FP? [NOT REPORTED] but unquantized Qwen-32B implies FP16/BF16 typical [AGENT INFERENCE].
- Block size 16 tokens is optimal for GPU memory management, independent of transfer efficiency [PAPER FACT].

## 12 Author-Stated Limitations [PAPER FACT]

- Paper notes CXL 2.0 switch does **not support host-to-host hardware cache coherence** [PAPER FACT]; software-managed coherence required, adding fallback complexity (though simple to set) [PAPER FACT].
- CXL 3.0 hardware coherence is still evolving, region size/guarantees unclear/cost-limited; software designs necessary for non-coherent regions [PAPER FACT] (end ?5.1).
- **Future architecture** needed: direct GPU-to-CXL switch connections to bypass Root Complex bottleneck ? current RC limits bandwidth, not yet available in Beluga (O7) [PAPER FACT].
- RPC provides lower reliability than RDMA; upper-layer mechanisms needed, strengthening guarantees is future research, not complete RDMA replacement [PAPER FACT] (?7.3 end).
- Evaluation versions caveat: based on versions available at submission (vLLM 0.8.5, MoonCake 3.2, Dynamo 0.4.1, LMCache 0.3.1) continuously improved by community (extra copies/allocations overhead in MoonCake) [PAPER FACT].
- Evaluation scope limited to rack-scale single-rack memory pool; cost analysis shows CXL favorable at rack scale, not evaluated beyond [PAPER FACT].
- Future work on CXL switches, software resource pooling schedulers, and coherence protocols remains open: need directory-based coherence, application-semantic relaxed coherence, hybrid hardware-coherent metadata region for larger space, vector/graph DB applications [PAPER FACT] (?8).

## 13 Inferred Limitations [AGENT INFERENCE]

- **Capacity vs bandwidth trade:** 8 TB pool at 1 TB/s shared among 16 servers ? average 64 GB/s per server peak, but RC bottleneck per adapter limits to ~46 GB/s with two adapters; scaling beyond 16 servers will saturate switching capacity (2 TB/s per chip) not measured [AGENT INFERENCE].
- **Inter-socket NUMA effects:** Dual-socket NUMA with 2 adapters (one per socket) may introduce cross-NUMA latency not characterized; evaluations on single socket? [AGENT INFERENCE]
- **Power/thermal not evaluated:** 32 DDR5 devices + switch power vs RDMA NIC power not quantified beyond purchase cost [AGENT INFERENCE].
- **Failure domain:** CXL switch becomes single point of failure for 8 TB; no evaluation of failover / redundancy vs RDMA distributed DRAM (4 TB across servers can survive partial failure) [AGENT INFERENCE].
- **Dynamic workload not tested:** LV-Eval is static long-context QA; no multi-agent tool-interleaved or RAG chunk variance, prefill/decode ratios fixed [AGENT INFERENCE].
- **Consumer GPU gap:** Optimizations rely on H20 (Hopper) P2P and DSA (Sapphire Rapids); consumer GPUs without these features would see different bottlenecks [AGENT INFERENCE].
- **Security/multi-tenancy:** mmap() DAX shared address space across hosts lacks isolation analysis; cross-host stale read due to coherence bug could leak private KV [AGENT INFERENCE].
- **Cost non-recurring:** XConn B1 sample price $5,800 subject to change; volume production economics not stable [AGENT INFERENCE].

## 14 Open Questions [AGENT INFERENCE]

1. How to achieve scalable directory-based hardware coherence for CXL 3.0 that covers TB-scale without excessive snoop traffic ? can switch resources host directory? [AGENT INFERENCE]
2. Can flow-aware prefetch (KVFlow) and TTL retention (Continuum) be layered over Beluga to orchestrate which KV blocks reside in HBM vs CXL vs DRAM, using Beluga?s near-local latency to simplify policy? [AGENT INFERENCE]
3. What is optimal interleaving granularity adaptively (256B to 2 MB) per workload ? can runtime auto-tune based on access zipf skewness? [AGENT INFERENCE]
4. How to design GPU-direct CXL fabric bypassing RC (O7) in practice ? what PCIe/CXL topology changes and driver support needed? [AGENT INFERENCE]
5. Could Beluga?s shared-memory RPC replace gRPC/TCP for all control plane, and how to add reliability (retry, ordering) without losing 4? latency win? [AGENT INFERENCE]
6. How to extend Beluga to heterogeneous memory (CXL-attached HBM, remote NUMA, disk tiers) for hierarchical KVCache with automatic tiering like FlowKV scheduling? [AGENT INFERENCE]
7. What is the consistency model for multi-writer KVCache updates (e.g., speculative decoding multiple candidates) over non-coherent CXL ? need software transactional protocol? [AGENT INFERENCE]
8. How to evaluate cost-performance at cluster scale (?100 servers, multiple racks) with multiple CXL switches ? does 1 TB/s fabric become bottleneck vs RDMA fabric scale-out? [AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **MoonCake (Qin et al. 2025 FAST?25) & Dynamo (NVIDIA 2025):** State-of-art RDMA disaggregated KVCache pools, Conductor / KV-aware routing, RDMA 1.6 Tbps per server [PAPER FACT].
- **LMCache (Liu et al. 2024d, 2510.09665):** KV cache layer for enterprise inference, CPU offload framework compared as RDMA backend [PAPER FACT].
- **vLLM PagedAttention (Kwon et al. SOSP23):** Base inference engine (V1 0.8.5) with paged KV management Beluga integrates [PAPER FACT].
- **RDMA characterization (Kalia 2016, Ziegler 2023, Del Monte 2022):** QP ordering, batching, one-sided primitives [PAPER FACT].
- **CXL characterization early work (Wang 2024, Zhong 2025, Liu 2025, Gouk 2022, Li 2023a):** FPGA / CXL 1.1 prototypes, performance analysis methods Beluga extends to commercial switch [PAPER FACT].
- **XConn XC50256 switch [Technologies, Yang 2025b]:** First commercial CXL 2.0 switch enabling this study [PAPER FACT].
- **In-DB LLM (Giannakouris SIGMOD25, Lu 2025, Jo 2025):** Motivation for large-scale shared memory for retrieval/RAG [PAPER FACT].
- **KVCache compression/sparsity (MQA Shazeer 2019, GQA Ainslie 2023, H2O Zhang 2023, FastGen Ge 2024, SnapKV Li 2024, KIVI, GEAR):** Algorithmic reduction complementary to system pooling [PAPER FACT] (cited ?9).
- **CloudMatrix (Zuo et al. 2025 Huawei UB):** Large-scale supernode CXL-like fabric but vendor-specific vs CXL standard [PAPER FACT].
- **[AGENT INFERENCE] KVFlow (Pan et al. 2507.07400) & Continuum (Li et al. 2511.02230):** Workflow-aware eviction / TTL retention that could use Beluga as secondary tier instead of CPU DRAM, exploiting near-local latency to hide even more.
- **[AGENT INFERENCE] CachedAttention / RAGCache / CacheBlend:** Long-context KV reuse techniques that benefit from low-latency shared pool like Beluga for cross-instance sharing.
- **[AGENT INFERENCE] CXL 3.0 coherence proposals & Directory protocols:** Future hardware direction to remove software flush overhead.


---
## Review Log

Reviewer: Reviewer-3 (RAG/Agent/Heterogeneous) — 2026-08-27
Scope: 读取全文 -> webfetch抽查关键数值 -> 标注核验
Webfetch抽查: CacheBlend arXiv:2405.16444v3 (TTFT 2.2-3.3x/2.8-5x verified), Cache-Craft arXiv:2502.15734v1 (51%/75% verified), KVLink arXiv:2502.16002v4 (TTFT 85-96% verified), KVFlow 1.83x/2.19x, Continuum JCT 1.12-3.66x, Beluga 7.35x, InfiniGen 3.00x, FlowKV 96.8% NCCL reduction, FastServe 31.4x, FlexGen 69x/112x verified
Problems Found: 数值层面无重大错误；HotPrefix 内容简略建议补全 Cuckoo filter 参数及 hardware 细节，其余标注合规
Corrections: 建议 HotPrefix 补全 Host=GPU KV size / CUDA stream promotion / vLLM 0.6.4 等细节；其余无修正
Confidence: High
