# Kairos 实际跑通版本交接（Task 1 Phase B，2026-10-27）

本文记录对方复跑所需的**精确版本、补丁、启动命令与服务器路径**。
结论先行：Kairos 树本身**零修改**（除新增一个测试文件）；跑通的关键是环境变量与依赖版本。

## 1. Kairos 版本

- 上游：`https://github.com/MaxizYB/Kairos`，commit `7971fa40323f16e8e4ce8f12dfd5db9755a98cf1`
  （2026-10-26，`docs: use dollar-delimited math formulas`），`--depth 1` clone。
- 唯一新增文件：`tests/test_ttl_policy.py`（6 个 Continuum Eq.2 回归测试：
  冷启动 `ln(B)`、K=100 分界、per-tool/global ECDF 回退、τ=0 枚举、fixed/zero 模式）。
  已复制一份到本仓库 `tasks/continuum/test_ttl_policy_kairos.py`，在 Kairos checkout 里跑：
  `python -m unittest tests.test_ttl_policy -v`。
- 其余文件与上游逐字节一致（`verify` 179 文件全过可证）。

## 2. 服务器环境与路径（bayesdl，Ubuntu 22.04，RTX 4090D 24G，251G RAM）

| 内容 | 路径 |
|---|---|
| Kairos checkout | `/root/241880423/code/kairos` |
| Python venv（3.12） | `/root/241880423/code/kairos/.venv`（symlink → `/root/bayes-tmp/241880423/venv`，overlay 空间不足时的权宜做法） |
| 模型 | `/root/241880423/models/hub2/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218`（Qwen3-8B pin revision，snapshot 目录结构，`verify --model` 全 shard SHA 通过） |
| 回放输出 | `/root/241880423/code/kairos/runs/{continuum,continuum_r2,kairos,kairos_r2}` |
| 决策/审计日志 | 各 run 目录下 `scheduler.jsonl`、`transfers.jsonl`、`pin_decision` 事件 |
| 缓存目录 | `UV_CACHE_DIR`/`XDG_CACHE_HOME` 均指向 `/root/bayes-tmp/241880423/cache`（overlay 只有 52G） |

## 3. 依赖（`requirements-server.txt` 实测可用组合）

- 先装 `torch==2.13.0+cu132`（`--index-url https://download.pytorch.org/whl/cu132`），
  再装 `requirements-server.txt`：vLLM 0.28.0 / LMCache 0.5.5rc4 / transformers 5.16.1 /
  httpx 0.28.1 / numpy 2.2.6。完整 freeze 见 `results_cloud/kairos_3way/env_freeze_kairos.txt`。
- 另需 `torchvision==0.28.0`（默认源 cu130 build 即可；vLLM warmup 会 import，缺了 EngineCore 起不来）。
- **不要装 torchaudio**：cu132 源没有 cp312 wheel；本路径无人 import 它。
- 模型下载：`HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`（直连 HF 会被重置）。

## 4. 启动命令（逐字）

父进程环境必须先 export（缺一不可，原因见注）：

```bash
export CONTINUUM_PREFIX_ONLY_CORRECTNESS=1   # 否则 validate 跑全套 gapped fixture，撞 pilot 门控 fatal
export CONTINUUM_WORK_MESSAGES=$ROOT/data/workload/visible_messages.json
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONPATH=$ROOT   # $ROOT=/root/241880423/code/kairos
```

核对（零 GPU）：

```bash
./.venv/bin/python -m kairos.verify --model "$KAIROS_CHECKPOINT"
./.venv/bin/python -m unittest discover -s tests
./.venv/bin/python -m kairos.run --method continuum --model "$KAIROS_CHECKPOINT" --output runs/dry --dry-run
```

回放（独占 GPU；端口 18765–18768 须空闲；串行一次一个）：

```bash
./.venv/bin/python -m kairos.run --method continuum --model "$KAIROS_CHECKPOINT" \
  --gpu 0 --engine-cpus 4-5,68-69 --helper-cpus 6-7,70-71 --output runs/continuum
./.venv/bin/python -m kairos.run --method kairos --model "$KAIROS_CHECKPOINT" \
  --gpu 0 --engine-cpus 4-5,68-69 --helper-cpus 6-7,70-71 --output runs/kairos
```

其中 `KAIROS_CHECKPOINT=<hub2 路径>/snapshots/b968826d9c46dd6066d109eabc6255188de91218`。
本次实际产出：`runs/{continuum,continuum_r2,kairos,kairos_r2}`，各 178/178、0 失败，
summary 已拉回 `results_cloud/kairos_3way/`。

## 5. `cloud_repro/run_relaxed.py`（本目录）

共享卡专用 wrapper，**只在 GPU 非独占时用**；独占卡直接用原生 `python -m kairos.run`。
作用：把 `gpu_process_scope` 包一层——跨容器邻居进程（nvidia-smi 可见、/proc 不可见）
只记 `/tmp/foreign_*.log`（时间戳 + pid），不 abort；另 `RELAX_ABLATION=1` 时跳过
`verify_checkout`（仅用于非密封消融，标准跑不要开）。
仓库文件零改动；seal 恢复后重跑过 `verify`（179/178 全过）。
注意：TTL 模式是**实验名路由**（`scheduler.py:411`），`CONTINUUM_TTL_MODE` 环境变量
对回放无效——不要据此做 fixed/zero 消融（一次无效尝试已记录在 SUMMARY §7）。

## 6. 已知坑（复跑前必看）

1. 父进程缺 `CONTINUUM_PREFIX_ONLY_CORRECTNESS=1` → validate 撞墙（见 §4）。
2. torchvision 必须装（0.28.0），torchaudio 必须不装（§3）。
3. 首次启动慢（torch.compile + warmup），`start_timeout` 默认 2400s 足够；不要因前几分钟无响应而 kill。
4. 非独占卡上 `resources.json` 的 `foreign_gpu_pids` 即邻居干扰证据；验收 run 时检查
   其时间戳是否落在 `summary.json` 的回放窗内。
