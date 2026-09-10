# Research-Grade Linux Environment Record — Stage 3E

> **Date:** 2026-08-29 17:30 Asia/Shanghai  
> **Purpose:** Frozen environment record for Stage 3E migration gate. Every L0 result under `research/environment_migration/` must be traceable to this file. Any run on different hardware/OS must be flagged and excluded from the primary L0 baseline.  
> **Status:** **MIGRATION ATTEMPTED — NATIVE LINUX UNAVAILABLE — WSL2 FALLBACK PARTIALLY AVAILABLE**  
> **Predecessor:** `research/measurement/environment.md` (2026-08-28 Windows) and `research/measurement_repair/stage3mb_summary.md` (FAIL) + `research/measurement_repair/independent_validation/stage3md_summary.md` (FAIL — NOT REPRODUCED, 20% TTFT variance at c4).  
> **A_01:** **SUSPENDED — ENVIRONMENT BLOCKED** (unchanged).

---

## 1. Migration Priority Verification

| Priority | Target | Available on this site | Verdict |
|---|---|---|---|
| 1 | Native Linux GPU server (bare-metal, datacenter, A100/H100, TCC, CUDA toolkit, vLLM) | **NO** — no IP, no SSH, no second host on LAPTOP-1PB54QSI; `F:\AIinfraResearch` is single Windows laptop | **BLOCKED** |
| 2 | Native Linux workstation (bare-metal Ubuntu, single GPU, CUDA) | **NO** — host boots Windows 11 only; dual-boot Ubuntu not installed | **BLOCKED** |
| 3 | WSL2 Ubuntu (fallback per Stage 3E §1) | **KERNEL present, DISTRO install BLOCKED** — `wsl --version` 2.5.10 kernel `6.6.87.2-microsoft-standard-WSL2` confirmed via `wsl -d docker-desktop -- uname -a`; `wsl --list --verbose` shows only `docker-desktop` (Stopped, v2), no Ubuntu distro; `wsl --install Ubuntu-24.04` hangs >120 s and exits without installing (Microsoft Store download blocked/permission, same on retry with PID 33608); `docker-desktop` distro has no `python3`/`nvidia-smi` | **PARTIALLY AVAILABLE (kernel only)** |

**Conclusion per §1:** *Do not treat Windows/WDDM as formal paper benchmark platform.* The only research-grade Linux available today is **WSL2 kernel level**; a full Ubuntu userland + CUDA WSL + PyTorch CUDA is **NOT READY** on this machine without manual `wsl --import` of a rootfs tarball + CUDA toolkit install (requires network + elevation). This record documents the attempted migration and freezes the fallback.

> **Native Linux check:** FAILED. This file therefore records the **intended fallback WSL2 environment** plus the **current Windows baseline** for traceability. No result in `research/environment_migration/` may be claimed as native Linux without updating this file to a real `uname -a` Ubuntu + `nvidia-smi` + `torch.cuda.is_available()==True` run.

---

## 2. Machine & OS (current real host)

| Item | Value | Source |
|---|---|---|
| Hostname | `LAPTOP-1PB54QSI` | `hostname` |
| Primary OS | Microsoft Windows 11 Home (Chinese), 10.0.26200, Build 26200.9168, 64-bit, AMD64 | `Win32_OperatingSystem` |
| WSL2 kernel | `6.6.87.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Thu Jun 5 18:30:46 UTC 2025 x86_64 Linux` | `wsl -d docker-desktop -- uname -a` |
| WSL version | 2.5.10.0, WSLg 1.0.66, MSRDC 1.2.6074, Direct3D 1.611.1-81528511, DXCore 10.0.26100.1 | `wsl --version` |
| Intended distro | `Ubuntu-24.04` | `wsl --list --online` (available, not installed) |
| Installed distros | `docker-desktop` (Stopped, v2) only | `wsl --list --verbose` |
| Power plan | Balanced `381b4222-f694-41f0-9685-ff5bb260df2e` (Active), `SCHEME_BALANCED` | `powercfg /getactivescheme` |
| CPU frequency observed | 1466 MHz (current) / 2200 MHz (max) via `psutil.cpu_freq()` | sampling at idle |
| Disk | NTFS, project root `F:\AIinfraResearch`, free ~480 GB on F: | `Get-PSDrive F` |

**Repro note:** Native Linux GPU server priority not met. Record will be updated to `Ubuntu 22.04/24.04, kernel 6.8+, x86_64` when real Linux host provisioned. Until then, WSL2 kernel is the only Linux evidence.

---

## 3. CPU / Memory

| Item | Value |
|---|---|
| CPU model | Intel Core i7-14650HX (Raptor Lake-HX) |
| Topology | 16 cores / 24 threads (8 P-cores + 8 E-cores), 2 threads per core, 1 socket, BogoMIPS 4838 per `lscpu` in WSL2 |
| Caches (from `lscpu` in WSL2) | L1d 576 KiB (12×), L1i 384 KiB (12×), L2 24 MiB (12×), L3 30 MiB (1×) |
| Virtualization | VT-x, Microsoft Hyper-V hypervisor (WSL2) |
| RAM | 31.78 GB (34124718080 bytes), ~14.9 GB free at capture, DDR5 |
| WSL2 CPU view | 24 vCPUs (matches host), NUMA 1 node, node0 0–23 |
| CPU power | Balanced; no locked performance governor; `cpu_freq` varies 1466–2200 MHz; no `cpupower` record on Windows |

---

## 4. GPU & Accelerator Topology

| Item | Value |
|---|---|
| GPU count | 1 |
| GPU model | NVIDIA GeForce RTX 4060 Laptop GPU, Ada Lovelace, CC 8.9 |
| VRAM | 8188 MiB (8585740288 bytes), 6398–6419 MiB free at idle per `nvidia-smi` |
| Bus | `00000000:01:00.0`, PCIe, WDDM driver model (not TCC) |
| Driver | 596.21 |
| CUDA driver API | 13.2 (reported by `nvidia-smi`) |
| CUDA runtime (`nvcc`) | **NOT INSTALLED** on Windows; **not found** in `docker-desktop` WSL (`/bin/sh: nvidia-smi: not found`) |
| NVLink | None (single laptop GPU) |
| GPU clocks (idle) | SM 210 MHz, Mem 405 MHz |
| GPU temp (idle) | 46–47 °C, P8, 3–5 W / 125 W cap |
| Power sampling caveat | WDDM mode: `gpu_util` via NVML is approximate (Stage 3M-B raised gate to 70% to avoid 39% WDDM noise) |
| GPU topology note | Single device, no tensor/pipeline parallelism (>1) possible; vLLM TP>1, SGLang DP, P/D disaggregation **BLOCKED** on this hardware in any OS |

**CUDA WSL status:** Host driver 596.21 supports CUDA WSL (≥510), but without Ubuntu userland + toolkit, `nvidia-smi` inside WSL is unavailable. A native Linux would expose `/proc/driver/nvidia/version` and `torch.cuda.is_available()==True`. This will be re-recorded when Linux provisioned.

---

## 5. Software Stack (frozen at 2026-08-29)

| Component | Version | Location / Source |
|---|---|---|
| Python (Windows host) | 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, MSC v.1943 64-bit) | `python --version` |
| pip | 26.0.1 | `pip --version` |
| numpy | 2.3.5 | `pip show numpy` |
| pandas | 3.0.1 | |
| scipy | 1.17.1 | |
| scikit-learn | 1.8.0 | |
| matplotlib | 3.10.7 | |
| tiktoken | 0.12.0 | workload length estimation |
| psutil | 7.2.2 | system metrics |
| nvidia-ml-py | 13.610.43 | `pynvml` alias |
| openai (client) | 2.32.0 | harness HTTP client |
| httpx | 0.28.1 | pooled `AsyncClient` |
| fastapi | 0.141.1 | serving shim |
| uvicorn | 0.52.4 | ASGI server |
| sse-starlette | 3.4.8 | streaming |
| **PyTorch** | **2.13.0+cpu** (CPU only) | `pip show torch`: `torch.cuda.is_available()==False`, `torch.version.cuda==None` |
| transformers | 5.16.1 | HF |
| accelerate | 1.14.0 | |
| tokenizers | 0.23.1 | |
| **vLLM** | **NOT INSTALLED** — no wheel on Windows; Linux build requires Bazel + CUDA toolkit + `cu124` | `pip show vllm` → not found |
| **SGLang** | **NOT INSTALLED** — Linux-only | |
| WSL Python | **NOT FOUND** in `docker-desktop` (`/bin/sh: python3: not found`) | would be `python3 3.12` on Ubuntu-24.04 when installed |

**Intended Linux stack (for next gate, NOT YET):**

```
OS: Ubuntu 22.04 LTS or 24.04 LTS (Native or WSL2 Ubuntu-24.04)
Kernel: 6.8+ (native) / 6.6.87.2-microsoft-standard-WSL2 (WSL2 fallback already present)
Driver: 535+ / 550+ / 565+ (Linux TCC equivalent, not WDDM)
CUDA runtime: 12.4 (cu124) or 12.6
PyTorch: 2.4.0+cu124 or 2.5+cu124 (CUDA, not +cpu), torch.cuda.is_available()==True
Transformers: same 5.16.1 (hash pinned)
Serving: hf_server.py 39cc7cd90bac (naive, pinned) OR vllm 0.28.0+ (if Linux server provides A100)
```

Until the above `+cu124` and `nvidia-smi` inside WSL/Ubuntu succeed, this file remains **WSL2-kernel-only**, not a full research-grade GPU environment.

---

## 6. Serving Framework Decision (Stage 3E §3 — Preserve Infrastructure)

**Principle (Stage 3E §2):** reuse workload generator, pooled client, benchmark harness, result schema, host-state logging, minimal tracing interface, tests; do not rewrite whole system.

| Item | Locked value for Stage 3E L0 gate |
|---|---|
| Harness | `research/measurement/harness/harness.py` hash `aae22af6712f`, `run_stage3e.py` (new) |
| Workload generator | `research/measurement/workloads/generator.py` hash `5ef1dcbf575c` |
| Serving shim | `research/measurement/harness/hf_server.py` hash `39cc7cd90bac`, fixed executor `max_workers=32`, `torch.set_num_threads(16)` / `set_num_interop_threads(16)` |
| Minimal trace | `research/measurement/harness/minimal_trace.py` hash `c9b465cca1e8`, **L0 OFF only** for gate (L1 forbidden) |
| Client | pooled `httpx.AsyncClient(max_connections=max_keepalive=max(8,concurrency))`, per §2 reuse |
| Model | `sshleifer/tiny-gpt2` (0.5B-class, 124M, fits 8 GB even FP32, `n_positions=1024` ≥512) cached at `~/.cache/huggingface/hub/models--sshleifer--tiny-gpt2` |
| Precision | FP32 on CPU (torch CPU); intended Linux FP16 on GPU (or BF16) — recorded per run manifest `dtype` |
| Parallelism | TP=1, PP=1 (single GPU) |
| Workload semantics | synthetic 512 input / 64 output, closed (`arrival_distribution=closed`, `arrival_rate=0`), `prefix_reuse=0`, `seed` varied, `temperature=0.0`, `stream=True`, `timeout=180` |
| Request count | 40 per run = 2 warmup sequential (excluded) + 38 measured (same as A_01) |
| Warmup (Stage 3E §5 simple) | **1× c=4 n=40 pooled OFF `seed=8000+chrono`** before session + per-run 2 warmup; **not** the data-driven Protocol D (1×c4+GC+sleep) from 3M-C/D — fixed, uniform, reasonable per spec |
| Sampling | 0.3 s host sampler every OFF run |
| Concurrencies | c=1, c=4 only (L0 gate), c=8 not measured |
| Repetitions | ≥5 per concurrency (10 total), interleaved `c1,c4,c4,c1,c1,c4,c1,c4,c4,c1` seeded 6101–6105 / 6401–6405 |
| Tracing | **L1 FORBIDDEN** for this gate — only `POST /stage3/trace/level {"level":0}` |

**Git commit:** `F:\AIinfraResearch` is **NOT a git repository** (`git status` → `fatal: not a git repository`). For reproducibility we record SHA-256 per file in manifests (same as Stage 3M-B/D). When migrated to a Linux server with git, `git rev-parse HEAD` will be appended.

---

## 7. Environment Variables (frozen, no overrides)

```
CUDA_VISIBLE_DEVICES=0 (implicit single GPU, not set explicitly)
HF_HOME=%USERPROFILE%\.cache\huggingface
HF_HUB_OFFLINE=0
PYTHONUTF8=1
No VLLM_* / SGLANG_* / NCCL_* overrides
No PYTHONMALLOC / OMP_NUM_THREADS overrides (torch threads fixed via code, not env)
```

On native Linux, add `CUDA_VISIBLE_DEVICES`, `NCCL_DEBUG`, `VLLM_*` only if vLLM used; currently `N/A`.

---

## 8. GPU Topology & Power/Performance Configuration

| Item | Current (Windows/WDDM) | Intended Linux (target) |
|---|---|---|
| Topology | Single 4060 Laptop, no NVLink, PCIe 01:00.0, WDDM | Single 4060 Laptop under Linux/TCC or A100-40/80GB if server available |
| Driver model | WDDM (shared with display, `gpu_util` noisy) | Linux `nvidia` (TCC-equivalent, exclusive) |
| Power limit | 125 W, idle 3–5 W P8 | Same hardware, but `nvidia-smi -q -d POWER` will be recorded |
| Clocks | SM 210 MHz idle (will boost to 1800+ under load) | Will record `nvidia-smi -q -d CLOCK` + `clocks_throttle_reasons` |
| CPU governor | Windows Balanced, idle 1466 MHz, max 2200 MHz; no `performance` lock | Linux `cpupower frequency-info` + `performance` or `schedutil` will be recorded |
| Thermal | GPU 47°C idle, no throttling; CPU not throttled | Will add `sensors` / `nvidia-smi -q -d TEMPERATURE,PERFORMANCE` |

**Why not performance-locked Windows?** Stage 3M-B/D showed CPU not saturated (10–28% at c4) and GPU temp 43–45°C stable, so thermal throttling was not the primary candidate. Locking to High Performance will be done on Linux (`powercfg /setactive SCHEME_MIN` equivalent `cpupower frequency-set -g performance` + `nvidia-smi -pm 1 -pl 125`).

---

## 9. Reproducibility & Provenance

- Environment captured: 2026-08-29 17:30 Asia/Shanghai (UTC+8), after Stage 3M-B (FAIL, c4 67% TTFT) and 3M-D (FAIL NOT REPRODUCED, c4 20% TTFT, 12 ms absolute).
- No git commit; SHA-256 hashes recorded per run manifest: `hf_server.py 39cc7cd90bac`, `harness.py aae22af6712f`, `minimal_trace.py c9b465cca1e8`, `generator.py 5ef1dcbf575c`, `run_stage3e.py` (new, hash logged per session).
- All random seeds fixed; `describe_workload` logs token stats per run (input mean 510.8–511.2, <0.08% variance).
- WSL2 kernel hash is not available (Microsoft prebuilt), but `uname -a` and `wsl --version` logged above.
- When Linux provisioned, append `git rev-parse HEAD`, `pip freeze`, `nvidia-smi -q`, `torch.__version__+cu`, `/proc/cmdline`, `lspci -vv`, `powercfg` or `cpupower` dump.

---

## 10. Hardware Limitation Disclosure (unchanged from Stage 3R-A)

- 8 GB laptop GPU cannot host 7B FP16 (~14 GB) for RQ-1 E3/E4/E5, RQ-2 E2–E5, etc. — those remain **BLOCKED**, recorded as not-runnable, not simulated on smaller model.
- Single node, no 25 Gbps cross-node, no NVLink pair — distributed bottlenecks not testable here.
- Tiny models (`tiny-gpt2` 124M) with `n_positions=1024` safely cover 512/64 but not 13K long-context; long-context gate will require A100.

---

## 11. Checklist (Stage 3E §3)

- [x] OS / kernel / WSL version recorded (Windows + WSL2 kernel)
- [x] GPU model / count / memory / driver / CUDA driver / CUDA runtime recorded
- [x] CPU / RAM / topology recorded
- [x] Python / pip / key packages / torch (+cpu) recorded
- [x] Serving framework decision and blocking reason (vLLM/SGLang blocked on Windows, hf naive primary) recorded
- [x] Model / precision / parallelism recorded
- [x] Env vars recorded
- [x] Topology & power config recorded
- [x] Date recorded
- [ ] Framework git commit — pending git repo on Linux rig; file hashes instead
- [ ] Native Linux validation — **PENDING** (this file is WSL2-kernel-only until Ubuntu userland + CUDA succeeds)

*End of linux environment record — must be updated before any native-Linux PASS may be claimed.*
