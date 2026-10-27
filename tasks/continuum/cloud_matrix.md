# 云机一次集中烧卡清单（Phase B，前提：用户开机 + 本地 A3 就绪）

预计 2–4 小时。请在开机后按序执行，全部只读 except 标明处。

## B0. 探测（只读，5 分钟）
- `nvidia-smi`（确认 4090 24G 空闲 ~0MiB）、`df -h /`（需 ≥20G：Qwen3-8B 约16.4G）、`free -g`。
- 沿用 `/root/241880423/`（模型/results/code），GPU 已清零状态继承。

## B1. 环境（约 30–60 分钟，首次）
- 按 Kairos `requirements-server.txt` 装服务端栈（torch 2.13+cu132 / vLLM 0.28.0 / LMCache 0.5.5rc4 / transformers 5.16.1 / httpx / numpy），`uv venv + uv pip install`。
- 拉模型 `Qwen/Qwen3-8B`（Apache-2.0，免审批；若卡住回退已在云机的 Qwen2.5-3B，并在报告注明与 Kairos 基准模型不一致）。
- `pip freeze > /root/241880423/results/env_freeze.txt`（环境锁）。

## B2. 三方回放（主体，约 1–2 小时）
- `python -m kairos.run --method continuum --model <ckpt> --output runs/cont-paper`（TTLPolicy paper=Eq.2）。
- 同命令 `--method kairos` → `runs/kairos`。
- 消融（零代码，环境变量即可）：`CONTINUUM_TTL_MODE=fixed CONTINUUM_TTL_SECONDS=2` → static-2s；
  `CONTINUUM_TTL_MODE=zero` → 不 pin。两者各跑一次。
- 每轮 `scripts/report.py`，原始 JSON 全部保留；`CONTINUUM_LOG_PATH` 等决策日志（pin/TTL/等待/重载）是边界分析的输入。

## B3. 回收（10 分钟）
- 拉回：`runs/*`、`env_freeze.txt`、决策日志 → 本地 `tasks/continuum/results_cloud/`。
- 云机 kill 服务进程，`nvidia-smi` 确认 0MiB（上次已验证流程）。
- 关机（用户操作）以停费。

## 风险与备注
- Kairos 服务端栈（vLLM 0.28/torch 2.13）与我们之前 0.10.2 实验不同构；数字只做方法内比较，不跨栈比绝对值。
- 继承 Kairos 已声明局限：单次回放、无多 seed 置信区间、teacher-forcing 回放、P95 TTFT/E2E 非全面改善——结论措辞按此限定。
