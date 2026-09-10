# AI Infra Research — KV Cache Management Optimization for LLM Inference

> Option 5: Reproduce and profile KV cache offloading across multiple storage tiers
> (GPU HBM / CPU DRAM / SSD / peer GPUs), optimize offloading for specialized
> workloads (e.g. agentic workloads), improve TTFT / throughput.

## Structure

- `tasks/` — task specs (`tasks/我的任务A0.md`) and experiment outputs.

## Experiment A0: Benchmark 2 (agent workflow) + vLLM — DONE

复现报告：[`tasks/a0-results/SUMMARY.md`](tasks/a0-results/SUMMARY.md)（数据、图表、结论、与任务书的偏差说明都在里面）。

Setup: Qwen2.5-3B-Instruct FP16, vLLM 0.11.0, 2× Tesla V100-32GB.
Baselines: V0 (no cache) / V1 (prefix cache) / V2-LRU (native CPU offload) /
V3 (LMCache CPU). V2-ARC blocked (no ARC implementation on this stack).

Headline results (agent workload, N=200, conc=4):

| Baseline | TTFT p50 | Prefix hit rate |
|---|---|---|
| V0 | 0.256s | 0% |
| V1 | 0.147s | 90.4% |
| V2-LRU | 0.148s | 90.4% |
| V3 | 0.148s | 90.4% |

Under memory pressure (working set > GPU KV): V1 and V2-LRU identical
(TTFT p50 ≈ 4.6s, hit 8.6%) — offload only helps when evicted blocks are
re-requested; zero-reuse pressure traffic gets no benefit. See SUMMARY for details.
