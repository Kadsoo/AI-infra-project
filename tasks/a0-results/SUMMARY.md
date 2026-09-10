# 实验2 (Benchmark 2 + vLLM) 复现报告 — A0

> 任务来源：`tasks/我的任务A0.md` 实验2（Benchmark 2 Agent workflow + vLLM）
> 运行日期：2026-09-09；远端：Ubuntu 22.04 + 2×V100-32GB（已按约定删除全部安装并验证）

## 1. 实际环境（与任务书的偏差）

| 项目 | 任务书 | 实际 | 原因 |
|---|---|---|---|
| 模型 | 3B-GPTQ-Int4 | Qwen2.5-3B-Instruct **FP16** | V100(sm70)无Marlin核，GPTQ走legacy又慢又有bug；3B-FP16仅5.8GB，32GB卡随便装 |
| 精度 | bf16 | **fp16** | V100无bf16硬件，bf16直接起不来 |
| 框架 | vLLM(未定版) | **vLLM 0.11.0** + transformers 4.55.4 | 0.8.5无原生offload；0.11是支持sm70的最高可用版（需`VLLM_ENABLE_CUDA_COMPATIBILITY=1`+`--enforce-eager`，V1 engine默认FA2在V100上不可用但可回退） |
| workload | BFCL v4直接跑 | agent风味合成workload（共享tools system prefix + tool任务） | BFCL是精度榜 Calls；本任务要的是serving侧TTFT/ITL/hit率/offload。结构对标BFCL multi-turn（system+tools前缀+用户轮次） |

## 2. Baseline矩阵（`--gpu-memory-utilization 0.5` → 约8.6GiB GPU KV）

- V0：无prefix cache（显式`--no-enable-prefix-caching`，0.11默认是开的）
- V1：`--enable-prefix-caching`
- V2-LRU：原生`OffloadingConnector/CPUOffloadingSpec`（0.11.0 schema：`num_cpu_blocks=14000, block_size=16` ≈ 8GiB CPU tier，写死LRU）
- V2-ARC：**BLOCKED**——vLLM 0.11与LMCache 0.5.4均无ARC实现；新版vLLM有ARC但无sm70预编译核
- V3：`LMCacheConnectorV1` + `LMCACHE_LOCAL_CPU=True, LMCACHE_MAX_LOCAL_CPU_SIZE=8`

## 3. 结果：常规workload（200请求，并发4，短prompt）

| Baseline | TTFT p50/p95 | ITL p50 | req/s | out tok/s | prefix命中率 |
|---|---|---|---|---|---|
| V0 | 0.256 / 0.362s | 0.084s | 0.39 | 45.1 | 0% |
| V1 | **0.147** / 0.198s | 0.075s | 0.46 | 53.4 | **90.4%** |
| V2-LRU | 0.148 / 0.171s | 0.075s | 0.46 | 53.4 | 90.4% |
| V3 | 0.148 / 0.219s | 0.074s | 0.45 | 52.9 | 90.4% |

结论：prefix cache把TTFT p50砍掉**43%**；工作集能装进GPU时，V1=V2=V3（offload无事可做，符合预期）。

## 4. 结果：压力workload（100请求，并发2，每请求~2.8k token unique doc，工作集≈20GB > 8.6GB GPU）

| Baseline | TTFT p50/p95 | ITL p50 | req/s | 命中率 |
|---|---|---|---|---|
| V1-P | 4.64 / 8.25s | 0.345s | 0.069 | 8.6% |
| V2LRU-P | 4.61 / 8.15s | 0.344s | 0.069 | 8.6% |

结论（诚实的阴性结果）：两者**完全一致**（命中数精确到个位相同）。原因是unique doc只用一次、跨请求零复用——被驱逐的块永远不会被再请求，offload救不了。这是给Benchmark 3/OPT的关键输入：**offload的价值 ∝ 被驱逐内容的复用率**，后续应构造“超GPU的多轮复用会话”（如重复访问的长RAG语料/agent多episode回看）才能拉开差距。
注：第一版V1P因与被kill的短prompt跑混过server导致命中率虚高46%，已废弃，重跑的干净版V1P2才用于对比。

## 5. 遥测缺口（任务书要的offload字节/带宽）

vLLM 0.11.0的`/metrics`里**没有**`kv_offload_total_bytes/time/size`这类counter（任务书的前瞻指标在该版本不存在），也无`external_prefix_cache_*`。本次带宽只能定性；定量需升级vLLM（要Ampere+卡）或给connector加instrumentation。另：`request_queue_time`总量可忽略（≈0.01s/req），说明瓶颈在计算而非排队。

## 6. 文件清单（`tasks/a0-results/`）

`bench_V*.json`（per-req TTFT/ITL/E2E + metrics前后快照）、`bench_*pressure.json`、
`fig_ttft_cdf.png / fig_itl_cdf.png / fig_bars.png / fig_pressure_ttft_cdf.png / fig_pressure_bars.png`、
`smoke_v0.json`。复现脚本逻辑见本报告§2（serve参数）与bench方法（streaming SSE计时+`/metrics`差值）。

## 7. 清理验证（远端 210.28.133.13）

已删：`~/a0-work`、`~/a0-venv`、`~/a0-venv2`（共~23.5GB）、`~/.cache/uv`（17GB）、`/tmp/pip-*`残留、tmux `a0/a0bench`、全部vLLM/bench进程。
验证：双卡0MiB、无残留进程、家目录只剩原有`conteb-v100-20260826`+`remote-init.sh`、别人的`qamcr_full`未动。
遗留（已告知）：`~/.cache/pip`涨约2.8GB（共享缓存，可`pip cache purge`，未动他人文件）；盘使用率74%（158G可用，跑前后基本一致）。
