# Paper Metadata

- **Title:** Accelerating LLM Inference via Dynamic KV Cache Placement in Heterogeneous Memory System [PAPER FACT]
- **Authors:** Yunhua Fang, Rui Xie, Asad Ul Haq, Linsen Ma, Kaoutar El Maghraoui, Naigang Wang, Meng Wang, Liu Liu, Tong Zhang [PAPER FACT] — Rensselaer Polytechnic Institute (Fang, Xie, Ul Haq, Ma, Wang, Liu, Zhang) and IBM T.J. Watson Research Center (El Maghraoui, Wang) [PAPER FACT]
- **Venue:** IEEE Computer Architecture Letter (Comments: IEEE Computer Architecture Letter) ; arXiv:2508.13231 [cs.AR] v1 17 Aug 2025, v2 15 Sep 2025 [PAPER FACT]
- **DOI/URL:** https://doi.org/10.48550/arXiv.2508.13231 / https://arxiv.org/abs/2508.13231 / HTML https://arxiv.org/html/2508.13231v2 [PAPER FACT]
- **Code:** [NOT REPORTED] [PAPER FACT]
- **Reading Source:** default.webfetch https://arxiv.org/abs/2508.13231 + https://arxiv.org/html/2508.13231v2 (v2, 15 Sep 2025) [PAPER FACT]

> Authenticity rule: 全文标注 [PAPER FACT]/[AGENT INFERENCE]/[UNVERIFIED]；数值必回正文否则 [NOT REPORTED]；不猜测。

## 1 Problem [PAPER FACT]

- LLM 推理受限于显存带宽，decode 阶段对 KV cache 的频繁访问主导数据搬运 [PAPER FACT]；attention sparsity 可减少部分流量，但 token 重要性随时间动态变化，需要全量 KV 仍可访问以应对相关性回升，持续施压带宽与容量 [PAPER FACT]。
- 带宽提升滞后于算力增长，memory wall 成为推理吞吐根本约束 [PAPER FACT]；HBM 带宽最高但容量受限于堆叠/封装密度且成本功耗高，不适合作为唯一大容量推理内存 [PAPER FACT]。
- 异构内存（HBM + off-package DRAM via NVLink/LPDDR5X/UALink/PCIe）已成为实际方案，但如何利用聚合带宽并在容量约束下做动态放置仍未被形式化研究 [PAPER FACT]；现有调度缺少理论上限刻画，难以判断运行时优化空间 [PAPER FACT]。

## 2 Motivation [PAPER FACT]

- 解码是 read-intensive，KV 访问占主导成为瓶颈；prefill 并行，decode 自回归逐 token 依赖 KV [PAPER FACT]。
- Dynamic token bypassing（按注意力选重要 token 拉取）可降带宽但不降容量需求——因为重要性会回升，必须保留全量 KV 在内存 [PAPER FACT]；Fig.2 用 LLaMA-3.1-8B 在 LongBench 上层8对 token 2777/4286 的 attention scores 展示重要性大幅波动、峰谷交替 [PAPER FACT]。
- 随着模型增大与 CoT 延长序列，KV footprint 快速膨胀，而 HBM 容量提升缓慢，差距拉大 [PAPER FACT]。
- 技术基础成熟：LPDDR5X 达 8533 MT/s，NVLink 4.0 每链 100 GB/s，GH200 Grace Hopper Superchip 中 Hopper GPU 96GB HBM3 4TB/s + Grace CPU 512GB LPDDR5X，via NVLink-C2C 900 GB/s 使 GPU 可高速直访 CPU 侧内存，带宽差距已缩至一个数量级内 [PAPER FACT]；聚合带宽利用成为关键 [PAPER FACT]。
- 放置决策显著影响性能：频繁访问 token 若放在慢内存则高带宽资源欠利用 [PAPER FACT]；动态数据放置调度在此异构层次下变得关键但尚未被探索 [PAPER FACT]。

## 3 Bottleneck [PAPER FACT]

1. **Token importance 时变且不可预知：** 重要 token 在解码过程中持续变化，需全量可访问 [PAPER FACT]。
2. **带宽 vs 容量双重压力：** sparsity 降带宽但不降容量；长上下文下 KV 超出 HBM 容量 [PAPER FACT]。
3. **异构带宽不均：** HBM 高带宽与 off-package 较低带宽共存，放置不当导致聚合带宽欠利用 [PAPER FACT]。
4. **迁移开销权衡：** 预见性放置可提升命中率，但迁移本身消耗带宽/时间，若工作集 > HBM 则频繁迁移与推理访问竞争 [PAPER FACT]（Sec IV-B 低稀疏下 Reactive/Page 粒度仅边际收益即为例）。
5. **稀疏度与变化率耦合：** 低稀疏（<60%）活跃集大、迁移多；高变化率需频繁跟踪，动态放置收益下降 [PAPER FACT]。
6. **缺乏形式化模型与上限：** 此前无对动态 KV 调度的数学建模与理论最优探索 [PAPER FACT]；论文声称首个 formal treatment [PAPER FACT]。

## 4 Core Idea [PAPER FACT]

**不提出具体可部署调度器，而是形式化异构内存上的动态 KV 放置问题，并以 Simulated Annealing 启发式在完美预知注意力访问模式下逼近最优，量化理论上限以揭示 headroom 并催生后续自适应调度研究 [PAPER FACT]。**

- 数学建模推理时延，仅聚焦 decode 带宽瓶颈；假设模型权重常驻 HBM，仅 KV 放置为调度变量；MoE 下 KV 占比更高，强化调度动机 [PAPER FACT]。
- 单步时延建模为 max{t^h_{(n,l)}, t^e_{(n,l)}}，其中 t^h 覆盖 HBM 侧访问、权重/中间结果等，t^e 覆盖 external DRAM 侧，基于数据量/带宽比 [PAPER FACT]；总时延 T = sum_{n,l} max{...} [PAPER FACT]。
- 优化目标：min_S sum max{t^h,t^e} s.t. P_H <=100%（HBM 占用百分比），off-package 假设容量足以容纳全量 KV [PAPER FACT]。
- 上限探索：SA 搜索 W（前瞻窗口，评估未来 decode token 数）与 R in [0,1]（可迁移量中实际迁移比例）二维控制；每步按未来 W 内访问频次排序优先队列，仅迁移 top-R 部分以权衡迁移开销 [PAPER FACT]。
- SA 操作子：window move ΔW in {±1,±2} 固定 R，ratio move ΔR = ±0.1 固定 W，diagonal move 同时扰动两者，采样概率 (0.4,0.4,0.2) [PAPER FACT]；接受准则 Metropolis P(accept)=exp[-ΔT/C]，初温按 p0=0.8 高接受、冷却率 alpha=0.9，终止条件为最优时延在连续温层改进 <0.1% 或温降至下限或达迭代预算 [PAPER FACT]。
- 因预知未来访问，SA 结果为任何实时无预知策略不可超越的理论上限 [PAPER FACT]。

## 5 System Changes [PAPER FACT]

- **非可部署系统，仅行为级 simulator：** 建模 decode 阶段异构层次下的访存，时延按 Sec III-A 公式估算，假设带宽为主要瓶颈 [PAPER FACT]。
- **SA 调度的决策粒度：** 以 token 级 KV entry 为单位排序迁移，非页粒度；Page Granularity Scheduling 作为对比基线仅页级迁移（页大小 16， emulation Quest，完美预知但引入页内不重要 token 开销）[PAPER FACT]。
- **状态维护：** 每步收集未来 W 个 token 将访问的 KV，频次优先队列 + 选择 top-R [PAPER FACT]。
- **基线对比框架：** 5 种策略：Unlimited HBM（理想无限 HBM 全放 HBM）、Static Placement（写一次不迁移，填满 HBM 后放 external）、Reactive Scheduling（miss 时晋升 HBM，满则 LRU 驱逐）、Page Granularity、SA-Guided（上限）[PAPER FACT]。
- **未实现真实运行时系统/迁移引擎、预取器或内存控制器改动 [PAPER FACT]；** 论文明确 not proposing a specific scheduling policy [PAPER FACT]。

## 6 Target Metrics [PAPER FACT]

- **Primary：**
  - **总解码时延 T**（ sum max{t^h,t^e} ）及由其导出的 **throughput（tokens/s，归一化 vs Static 或 vs Unlimited HBM）** [PAPER FACT]；Fig.3 归一化 tokens/s vs Static，Fig.4 归一化 vs Unlimited HBM [PAPER FACT]
  - **HBM hit rate（注意力稀疏度 60% 时）** Fig.5 [PAPER FACT]
- **Secondary：**
  - 敏感度：不同 attention sparsity（<60% 低稀疏，>80% 高稀疏）与 token importance variation（低/高变化合成 trace）下的归一化 tokens/s [PAPER FACT]
  - 迁移开销隐含影响：SA 通过 R 控制迁移量 [PAPER FACT]
- **Not reported：** 能耗、成本、TTFT/TBT 细粒度、端到端 E2E 延迟数字表（仅归一化图示，无具体 ms 表） [NOT REPORTED]

## 7 Baselines [PAPER FACT]

- **Unlimited HBM：** 假设 HBM 无限，所有数据含权重+KV 全在 HBM，最优但不现实 [PAPER FACT]
- **Static Placement：** 首次写入定放，HBM 填满后落 external，无动态迁移 [PAPER FACT]
- **Reactive Scheduling：** 访问 miss 时提至 HBM，满则 LRU 驱逐至 external，基于观测复用 [PAPER FACT]
- **Page Granularity Scheduling：** 模拟 Quest，页大小 16，整页迁移且完美预知 token 重要度，但页内包含不重要 token 带来开销 [PAPER FACT]
- **SA-Guided Scheduling：** 本文上限，基于先验访问统计的 SA 优化放置 [PAPER FACT]
- 另以 GH200 配置作为硬件参考 [PAPER FACT]

## 8 Workloads [PAPER FACT]

- **模型：** LLaMA-3.1-8B，模型大小约 16GB [PAPER FACT]；其余模型 [NOT REPORTED]
- **数据集/输入：** LongBench 的 NarrativeQA 数据集，约 30k tokens 的 prompt 喂给模型，自回归解码 10K tokens [PAPER FACT]；记录解码阶段逐层 attention scores 作为访存模式 [PAPER FACT]
- **Token importance 观测：** 层8的 token 2777 与 4286 案例 [PAPER FACT]
- **合成 trace：** 为评估 variation，综合两份合成 trace 模拟低/高变化场景（60% 稀疏下 vs Unlimited HBM 对比）[PAPER FACT]
- **序列长度：** prompt ~30k，decode 10k；在多稀疏度下扫参 [PAPER FACT]

## 9 Hardware [PAPER FACT]

- **建模平台：** 行为 simulator，内存系统配置基于 NVIDIA GH200 Grace Hopper Superchip Table I [PAPER FACT]
  - HBM: Bandwidth 4.9 TB/s, Capacity 24 GB [PAPER FACT]
  - Off-package DRAM: Link bandwidth 900 GB/s (NVLink-C2C), DRAM bandwidth 500 GB/s, Capacity 480 GB [PAPER FACT]
- **背景提及 GH200 变体：** H100 96GB HBM3 达 4 TB/s 带宽，Grace CPU 512GB LPDDR5X via NVLink-C2C 900 GB/s（此处数字与 Table I 的 24GB/480GB 为仿真具体取值，前者为产品最大配置描述，二者共存）[PAPER FACT]
- **模型占用示例：** LLaMA-3.1-8B 16GB 占 HBM，余约 8GB 可放 KV，需异构扩展 [PAPER FACT]
- **其他硬件（CPU 型号/单机 GPU 数/存储） [NOT REPORTED]**

## 10 Main Results [PAPER FACT]

- **上限 vs Static：** SA 上限相较 Static Placement 达到最高 **5.87x 更高吞吐**（tokens/s）[PAPER FACT]（Abstract/Conclusion）——数值在正文中以归一化图展示，具体 5.87x 为 headline 数字 [PAPER FACT]
- **敏感度 Fig.3（归一 vs Static，不同稀疏度）：**
  - 低稀疏（<60%）时 Reactive 与 Page Granularity 仅边际增益，因工作集 > HBM 容量导致频繁迁移与推理访存竞争、带宽效率下降 [PAPER FACT]
  - 高稀疏（>80%）时活跃 KV footprint 缩小，迁移成本降低，SA 与其他策略差距收窄 [PAPER FACT]
  - 总体 SA 持续领先基线 **4x 到 5x** [PAPER FACT]（Sec IV-B 末句）
- **Variation 敏感度 Fig.4（60% 稀疏，归一 vs Unlimited HBM，低/高变化合成 trace）：**
  - 低 variation 时 Page 与 SA 接近最优，稳定重要性需极少迁移 [PAPER FACT]
  - 高 variation 时二者均退化，需频繁迁移跟踪模式；结论：动态放置对低 variation 任务收益更大 [PAPER FACT]
- **HBM hit rate Fig.5：** SA 在 60% 稀疏下 hit rate 控制在高效比例（图示数值未以表格给出）[PAPER FACT]，具体数值 [NOT REPORTED] 文本仅称 efficient ratio [PAPER FACT]
- **未以 ms 表格给出绝对时延/吞吐数值，仅归一化曲线 [PAPER FACT]**

## 11 Assumptions [PAPER FACT]

- 仅 decode 阶段为带宽瓶颈，prefill 不纳入优化 [PAPER FACT]
- 推理性能主要受内存带宽约束，可由公式估算总时延 [PAPER FACT]
- 模型权重在整个推理期间常驻 HBM，仅 KV 为可调度变量 [PAPER FACT]
- Off-package DRAM 容量足够容纳全量 KV [PAPER FACT]
- 理论上限允许完美预知每步注意力访问模式，可在计算前将 KV 置于最优位置（实际不可得）[PAPER FACT]
- 迁移开销可用 R 比例统一控制，W 窗口扫未来可捕获重要度排序 [PAPER FACT]
- Single-host GH200 样式异构系统，双层内存、带宽参数如 Table I [PAPER FACT]

## 12 Author-Stated Limitations [PAPER FACT]

- 仅推导理论上限，未提出可部署的实时调度算法或系统实现 [PAPER FACT]
- 假设注意力访问模式先验已知，用于 bound 探索 [PAPER FACT]
- Simulator 行为级，未在真实硬件上端到端验证；缺真实迁移引擎与一致性开销 [PAPER FACT]
- 评估限于 LLaMA-3.1-8B + NarrativeQA 长上下文；MoE 等架构仅论及潜力未评测 [PAPER FACT]
- 未考虑权重放置优化（固定 HBM）与多实例/多租户干扰 [PAPER FACT]
- 结论中呼吁未来研究 predictive modeling、强化学习或在线学习来逼近上限 [PAPER FACT]

## 13 Inferred Limitations [AGENT INFERENCE]

- 完美预知假设过强，上限与可实现策略间差距可能因预测误差更大 [AGENT INFERENCE]
- SA 搜索需枚举 W/R 且依赖未来频次排序，真实在线开销与搜索实时性未评估 [AGENT INFERENCE]
- 行为模型简化为带宽比值的 max 并行，未建模 bank 冲突、访存排队、写回与一致性、NUMA 延迟等 [AGENT INFERENCE]
- 仅 token 粒度，未考虑页/块对齐、碎片与元数据开销；Page 16 基线的 overhead 可能被低/高估 [AGENT INFERENCE]
- 单模型单数据集（30k+10k）可能不代表短上下文、高并发服务或混合工作负载 [AGENT INFERENCE]
- 未对比异构感知缓存逐出/预取（类似 H2O/SnapKV）或量化压缩对带宽-容量的联合影响 [AGENT INFERENCE]
- 24GB/480GB 的 HBM/DRAM 容量与 4.9TB/s/900GB/s 带宽为仿真取值，真实 GH200 产品为 96GB/512GB 与 4TB/s，泛化性需再校准 [AGENT INFERENCE]

## 14 Open Questions [AGENT INFERENCE]

1. 如何在无未来预知下用轻量预测器（attention predictor / 学习型分类器）逼近 SA 上限的 W* 与 R*？[AGENT INFERENCE]
2. 能否将 placement 与 KV 压缩/稀疏化（量化、低秩、逐出）联合优化以同时缓解带宽与容量？[AGENT INFERENCE]
3. 异构三层（HBM-DRAM-SSD/CXL）下多级迁移策略与一致性协议如何设计？[AGENT INFERENCE]
4. 对于低 variation 长上下文（如 summarization），最优 W 是否趋短？高 variation 对话的自适应 R 调度如何在线调参？[AGENT INFERENCE]
5. 在 PD 解耦与分布式推理中，跨节点 KV 迁移与放置如何协同？[AGENT INFERENCE]
6. MoE 场景下专家权重与 KV 的联合放置是否改变目标函数与约束？[AGENT INFERENCE]
7. 能否形式化迁移开销 vs 命中率的 Pareto 前沿并给出在线近似比保证？[AGENT INFERENCE]
8. 真实硬件上迁移带宽与计算的干扰如何精确建模与隐藏？[AGENT INFERENCE]

## 15 Related Papers [PAPER FACT + AGENT INFERENCE]

- **QUEST: Query-aware sparsity for efficient long-context LLM inference (Tang et al. ICML'24)**：页级稀疏估计与迁移思想来源，页大小 16 的粒度被本文作为 Page Granularity 基线 [PAPER FACT]
- **H2O: Heavy-Hitter Oracle (Zhang et al. NeurIPS'23)**：注意力稀疏性利用，通过 heavy-hitter 驱逐降低访存，本文的 token importance 动态性与之呼应 [PAPER FACT]
- **NVIDIA GH200 Grace Hopper Superchip 架构**：异构内存硬件原型，提供 4TB/s HBM3 + 900GB/s NVLink-C2C 扩展 [PAPER FACT]
- **Taming Throughput-Latency tradeoff with Sarathi-Serve (Agrawal et al. OSDI'24)**：带宽/时延权衡背景中被引用 [PAPER FACT]
- **AI and memory wall (Gholami et al. IEEE Micro'24)**：memory wall 背景 [PAPER FACT]
- **后续可结合：** heterogeneous-aware PagedAttention / FlexGen / HMA-LLM / CXL 内存扩展等异构调度，以及 InfiniGen 预取与 learned cache 策略 [AGENT INFERENCE]
