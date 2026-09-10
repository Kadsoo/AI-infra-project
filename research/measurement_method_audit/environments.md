# Measurement Method Audit — Environments (2026-08-30)

> Source: probes 2026-08-30 + frozen `F:\AIinfraResearch\research\environment_linux.md` (2026-08-29), `measurement/environment.md` (2026-08-28), `environment_migration/*`, and `F:\papers\reproduction\v100/*`. No guessing; gaps marked UNKNOWN / NOT DOCUMENTED.

## Summary

| Environment | Label | Status | Use |
|---|---|---|---|
| W | Windows 11 + RTX 4060 Laptop 8GB WDDM (LAPTOP-1PB54QSI) | **AVAILABLE** | current host — NOT research-grade for TTFT (L0 FAIL) |
| V | 2xV100 Linux (conteb-v100-20260826, /home/yaoyunchao) | **BLOCKED** | PRIMARY CANDIDATE if reachable, but credential/tunnel missing |
| WSL2-kernel | WSL2 kernel 6.6.87.2-microsoft-standard-WSL2 (docker-desktop only) | **PARTIALLY AVAILABLE (kernel only)** | fallback attempted, no CUDA, L0 FAIL |

A_01 stays **SUSPENDED — ENVIRONMENT / MEASUREMENT BLOCKED**.

---

## Environment W — Windows RTX 4060 (current host)

### Provenance
- Probe 2026-08-30: `nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader` => `0, NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 5756 MiB, 596.21`
- Frozen: `research/environment_linux.md` §2-§5, `measurement/environment.md`, `environment_migration/l0_stationarity.md` PID 44756
- Harness: `hf_server.py` hash `39cc7cd90bac` (FIXED_EXECUTOR 32, torch threads 16/16), `harness.py` `aae22af6712f`, minimal_trace `c9b465cca1e8`, pooled `httpx 0.28.1`

| Item | Value | Source |
|---|---|---|
| OS | Windows 11 Home 10.0.26200 Build 26200.9168 64-bit AMD64 | frozen env |
| Kernel | 10.0.26200 SP0 | env_linux §2 |
| WSL | WSL 2.5.10.0, kernel 6.6.87.2-microsoft-standard-WSL2 | `wsl -d docker-desktop -- uname -a` |
| Distros | docker-desktop Stopped only; Ubuntu-24.04 not installed (hangs) | wsl --list |
| Hostname | LAPTOP-1PB54QSI | frozen |
| CPU | Intel i7-14650HX 16C/24T (8P+8E), L1d 576KiB L2 24MiB L3 30MiB | lscpu in WSL2 |
| RAM | 31.78 GB DDR5, ~14.9 GB free | env_linux §3 |
| Storage | NTFS F: F:\AIinfraResearch, ~480 GB free | Get-PSDrive |
| GPU | 1x RTX 4060 Laptop GPU Ada CC 8.9 8188 MiB Bus 01:00.0 PCIe | nvidia-smi |
| Driver | 596.21, CUDA driver API 13.2, CUDA runtime NOT INSTALLED | nvidia-smi |
| Power | 125W cap idle P8 3-5W SM 210 MHz idle -> 1800+ boost temp 43-47C | nvidia-smi -q |
| CPU governor | Windows Balanced SCHEME_BALANCED 381b4222-f694-41f0-9685-ff5bb260df2e freq 1466-2200 MHz | powercfg |
| GPU model | WDDM (shared display, gpu_util noisy threshold 70%) | env_linux §8 |
| NVLink | None (single GPU) | topo |
| Python | 3.13.5 MSC v1943 64-bit pip 26.0.1 | python --version |
| .venv | F:\papers\.venv exists, F:\AIinfraResearch NOT a git repo (hashes) | probe |
| PyTorch | 2.13.0+cpu torch.cuda.is_available()==False torch.version.cuda==None threads 16 | pip show |
| Transformers | 5.16.1 | env_linux §5 |
| Serving | hf_server.py naive (no PagedAttention) fastapi 0.141.1 uvicorn 0.52.4 anyio 4.9.0 httpx 0.28.1 tiktoken 0.12.0 psutil 7.2.2 nvidia-ml-py 13.610.43; vLLM/SGLang NOT INSTALLED (Windows wheel blocked) | frozen |
| Model | sshleifer/tiny-gpt2 124M FP32 CPU n_positions=1024 cached ~/.cache/huggingface/hub/models--sshleifer--tiny-gpt2 | env_linux §6 |
| Precision | FP32 CPU (intended FP16 on Linux GPU) | harness |
| Parallelism | TP=1 PP=1 | frozen |
| Workload | synthetic 512/64 n=40 (2 warmup excluded +38 measured) closed arrival prefix_reuse=0 temp 0.0 stream True timeout 180 | env_linux §6 |
| Concurrencies | c=1, c=4 (gate) | locked |
| Repetitions | 5 per conc (10 total) interleaved c1,c4,c4,c1,c1,c4,c1,c4,c4,c1 seeds 6101-6105/6401-6405 | l0_stationarity §1 |
| Warmup | session 1x c=4 n=40 pooled OFF seed 8000 + per-run 2 sequential excluded | env_linux §6 |
| Client | pooled httpx.AsyncClient(max_connections=max_keepalive=max(8,concurrency)) | harness |
| Known failures | c1 median 3.28% PASS but p95 19.78% FAIL; c4 median 87.29% FAIL p95 32.46% FAIL; thr c4 6.21% FAIL; replicates 3M-B 67% /3M-D 20% pattern; throughput CV 1.36% c1 /2.90% c4 stable vs TTFT CV 36.67% c4 | l0_stationarity §3, migration_summary §4 |
| Status | **AVAILABLE — NOT RESEARCH-GRADE FOR TTFT** | L0 FAIL 2026-08-29 |
---

## Environment V — 2xV100

### Provenance
- Found: `F:\papers\reproduction\v100\author-v100-training-config.yml`, `bootstrap-v100-env.sh`, `remote-init.sh`. No file in `F:\AIinfraResearch` documents V100 serving host; `research/environment_linux.md` §1 marks Native Linux GPU server BLOCKED — no IP, no SSH.
- Exhaustive grep (`Select-String`/`rg`) across `F:\papers` + `F:\AIinfraResearch` found V100 only in `reproduction/v100` configs and paper-notes (GEAR etc.), not in serving SSH docs. No `~/.ssh/config`, no host alias.
- Evidence below is only literal file statements; else UNKNOWN.

| Item | Value | Evidence |
|---|---|---|
| 2xV100 | YES — n_gpus:2, assert device_count >=2 | author-v100-training-config.yml n_gpus:2, bootstrap assert |
| Per-GPU VRAM | 32 GiB | comment "32 GiB V100s" |
| PCIe / SXM | UNKNOWN / NOT DOCUMENTED | no lspci/topology file |
| NVLink | UNKNOWN / NOT DOCUMENTED | no NVLink mention |
| CPU | UNKNOWN / NOT DOCUMENTED | — |
| RAM | UNKNOWN / NOT DOCUMENTED | — |
| Linux distro | UNKNOWN / NOT DOCUMENTED | bootstrap uses bash/python3 but no distro file; remote-init path /home/yaoyunchao implies Linux but not documented => UNKNOWN per rule |
| Kernel | UNKNOWN / NOT DOCUMENTED | — |
| NVIDIA driver | UNKNOWN — supports CUDA 12.2 (comment) | bootstrap comment: server has NVIDIA driver CUDA 12.2 support |
| CUDA | 12.2 driver + 12.1 torch wheel cu121 | torch==2.2.2 --index-url .../cu121 + driver comment |
| PyTorch | 2.2.2+cu121 (V100 env) | bootstrap |
| Python | 3.10 (server supplies 3.10 not 3.11) | README: server supplies Python 3.10 |
| Transformers | 4.48.0 (V100 run) | bootstrap pins 4.48.0 |
| Other pkgs | accelerate 1.7.0 sentence-transformers 3.3.1 configue 5.0.0 datasets 3.2.0 pylate 1.0.0 | bootstrap |
| Serving framework | NOT INSTALLED / NOT DOCUMENTED for serving — V100 env is for contextual-embeddings training (ModernBERT), not hf_server/vLLM/SGLang | file absence |
| Model V100 | ModernBERT-embed-large SDPA FP32 weights FP16 autocast 4096 tokens bf16 disabled gradient_checkpointing reentrant=false | yml |
| Batch | per_device 2 => global 4 (2 GPUs) | yml |
| Attn | sdpa (FlashAttention2 not buildable) | yml + README |
| Path | /home/yaoyunchao/conteb-v100-20260826 | remote-init.sh ROOT |
| Network | UNKNOWN / NOT DOCUMENTED | — |
| Storage | UNKNOWN / NOT DOCUMENTED | — |
| GPU topology | UNKNOWN / NOT DOCUMENTED — TP=2 via accelerate launch --multi_gpu --num_processes 2 --main_process_port 29610, NCCL implicit but not recorded | run-author-accelerate.sh |
| Can use both GPUs | YES (DDP 2-process) | accelerate command |
| Shared server | UNKNOWN / NOT DOCUMENTED | home dir suggests shared host, not proven |
| Scheduler | UNKNOWN / NOT DOCUMENTED | — |
| Other users contention | UNKNOWN / NOT DOCUMENTED | — |
| Access method | UNKNOWN / NOT DOCUMENTED — no SSH alias/Host/VPN doc found; only local path /home/yaoyunchao/... requires operator credential | exhaustive grep zero SSH docs; env_linux §1 says no IP/SSH |
| Currently available | BLOCKED — credential/VPN/campus net/operator approval required; probing this workstation cannot reach /home/yaoyunchao | STATE F |

**Priority per §3.2:** IF reachable via authorized SSH, V would be PRIMARY CANDIDATE (native Linux + NVIDIA GPU). Per §3.3 it remains candidate until L0 PASS. Single-V100 first per §9.

---

## Other environments

| Env | Status | Notes |
|---|---|---|
| WSL2 fallback (kernel only) | PARTIALLY AVAILABLE | wsl 2.5.10.0 kernel 6.6.87.2 via docker-desktop; no Ubuntu-24.04 userland, python3/nvidia-smi not found, wsl --install hangs; L0 simple-warmup 1x c=4 n=40 FAIL c4 87% median — same pathology as Windows |
| Native Linux workstation | BLOCKED | host boots Windows 11 only |
| H100/A100 dedicated | NOT USABLE | no hardware |

## Availability tags (§7)

- W: AVAILABLE — but NOT research-grade for TTFT (L0 FAIL)
- V: BLOCKED (hardware documented but access requires operator-owned credential; no serving stack verified)
- WSL2: PARTIALLY AVAILABLE (kernel) / BLOCKED (userland+CUDA)

## Re-probe before use (house rules: history != availability)

```
ssh <alias> -- nvidia-smi --query-gpu=index,name,memory.total --format=csv
ssh <alias> -- python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
ssh <alias> -- uname -a && cat /etc/os-release && nvidia-smi -q | head
```

Store byte-faithfully in raw/<probe>.log before claiming AVAILABLE.

*Generated 2026-08-30 for §7. Do not mix absolute performance across envs; stationarity only.*
