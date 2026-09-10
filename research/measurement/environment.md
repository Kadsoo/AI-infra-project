# Stage 3R-A Environment Record — Real Serving Baseline

> **Date:** 2026-08-28
> **Purpose:** Frozen environment for all 3R-A baseline runs. Every result in `research/measurement/` must be traceable to this file. Any run on different hardware must be flagged and excluded from the primary baseline.

## 1. Machine & OS

| Item | Value |
|---|---|
| Hostname | LAPTOP-1PB54QSI |
| OS | Microsoft Windows 11 Home (Chinese), 10.0.26200, 64-bit |
| Kernel / Build | Windows 11 10.0.26200 SP0, AMD64 |
| CPU | Intel Core i7-14650HX (16 cores / 24 threads, Raptor Lake-HX) |
| RAM | 31.78 GB (34124718080 bytes), ~15.4 GB available at idle |
| Disk | NTFS, project root `F:\AIinfraResearch` |
| WSL | `docker-desktop` (WSL2 v2, state Stopped), no Ubuntu distro installed |
| Docker | Docker Desktop 28.3.2 (engine not running at baseline capture) |

## 2. GPU & Accelerator Topology

| Item | Value |
|---|---|
| GPU count | 1 |
| GPU model | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8188 MiB (8585740288 bytes), 6398 MiB free at idle |
| Compute Capability | 8.9 (Ada Lovelace) |
| Driver Version | 596.21 |
| CUDA Driver (reported by nvidia-smi) | CUDA 13.2 |
| CUDA Runtime (nvcc) | NOT INSTALLED (no toolkit on host) |
| NVLink | None (single laptop GPU) |
| PCIe Topology | Single device at `00000000:01:00.0`, WDDM driver model |

**Implication:** No multi-GPU tensor/pipeline parallelism on this machine; all distributed-framework features (vLLM TP>1, SGLang DP, P/D disaggregation) are blocked. Host DRAM tier exists (32 GB RAM) but PCIe bandwidth is shared laptop link, not datacenter HBM tier.

## 3. Software Stack (exact versions at 2026-08-28)

| Component | Version | Source |
|---|---|---|
| Python | 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, MSC v.1943 64-bit) | `python --version` |
| pip | 26.0.1 | `pip --version` |
| numpy | 2.3.5 | `pip show numpy` |
| pandas | 3.0.1 | |
| scipy | 1.17.1 | |
| scikit-learn | 1.8.0 | |
| matplotlib | 3.10.7 | |
| tiktoken | 0.12.0 | tokenizer for workload length estimation |
| psutil | 7.2.2 | system metrics |
| nvidia-ml-py | 13.610.43 | GPU metrics via NVML (import name `pynvml`) |
| pynvml (alias) | Provided by nvidia-ml-py | `pynvml.nvmlSystemGetDriverVersion() == 596.21` |
| openai (client) | 2.32.0 | harness HTTP client |
| fastapi | 0.141.1 | serving shim |
| uvicorn | 0.52.4 | ASGI server |
| sse-starlette | 3.4.8 | streaming support |
| PyTorch | **NOT INSTALLED** | Blocked: `cu124` wheel is 2.53 GB, download retries time out on this network; `cu121` has no cp313 wheel. CPU wheel not yet installed in baseline. Will be installed as Step 2. |
| transformers | **NOT INSTALLED** | Requires torch; pending |
| accelerate | **NOT INSTALLED** | Pending |
| vLLM | **NOT INSTALLED** | Decision: Windows has no prebuilt wheel (source tarball 0.28.0, requires Linux + Bazel + CUDA toolkit). See §5. |
| SGLang | **NOT INSTALLED** | Same Linux-only constraint |
| Ollama / llama.cpp | **NOT INSTALLED** | Candidate for Windows-native real engine (optional) |

## 4. Serving Framework Decision (Stage 3R-A §3)

**Principle:** "优先使用项目当前最容易可靠运行的真实 serving framework；先保证一个系统测得可靠，不要为了系统数量牺牲实验质量。"

| Framework | Availability on this host | Verdict |
|---|---|---|
| vLLM 0.28.0 | No wheel, must compile from source on Linux, needs CUDA toolkit + MSVC incompatible | **BLOCKED on Windows** — documented as `NOT AVAILABLE` but harness is vLLM-compatible via OpenAI API |
| SGLang | Same Linux-only, Rust + Python build | **BLOCKED on Windows** |
| HF Transformers + torch (naive but real) | Achievable after torch install; runs real forward passes on GPU/CPU, exposes prefill/decode | **PRIMARY baseline engine for 3R-A on this laptop** — real inference, PagedAttention disabled, but measures true TTFT/TPOT/memory |
| Mock deterministic server | Always available | **Harness-validation only** — not counted as real serving, used for instrumentation tests |

**Model selection for 8 GB VRAM:**

- Target: `Qwen/Qwen2-0.5B-Instruct` (~0.5B, ~1.0 GB FP16, fits with KV cache headroom) or `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (~2.2 GB FP16). Both support chat template and streaming.
- Fallback: `gpt2` (124M) for harness smoke test (fastest download, ~500 MB).
- Quantization: NOT USED in baseline (FP16/BF16 default if GPU available, FP32 on CPU). Recorded as `precision: FP16 (GPU) / FP32 (CPU fallback)`.
- Tensor parallelism: 1 (single GPU). Pipeline parallelism: 1.

**Relevant environment variables (baseline, no overrides):**

```
CUDA_VISIBLE_DEVICES=0 (implicit single GPU, not set explicitly)
HF_HOME=%USERPROFILE%\.cache\huggingface
HF_HUB_OFFLINE=0
PYTHONUTF8=1
No VLLM_* / SGLANG_* / NCCL_* overrides
```

## 5. Framework Version Tracking (for future Linux rig)

When the same harness runs on a Linux A100 machine, the following must be recorded and appended here:

```
framework: vllm
version: 0.28.0 (or newer)
git commit: <git rev-parse HEAD>
install: pip install vllm --extra-index-url https://...
model: Qwen/Qwen2.5-7B-Instruct (or chosen)
precision: FP16 / BF16 / INT8 / AWQ
tp: 1, pp: 1
cuda: 12.4, driver: 535.x, torch: 2.4.0+cu124
```

The current Windows baseline is explicitly tagged `engine: hf_transformers_naive` to distinguish from future `engine: vllm` baselines. Results are NOT mixed across engines without label.

## 6. Experiment Date & Provenance

- Environment captured: 2026-08-28 13:XX Asia/Shanghai
- No git repository in `F:\AIinfraResearch` (`git status` → `fatal: not a git repository`). For our own experiment code we record SHA-256 hashes per run (see `research/measurement/logs/`).
- All subsequent runs must log: command, config JSON, file hashes, `environment.md` hash, raw CSV, processed CSV.

## 7. Hardware Limitation Disclosure (for Gate)

- Single 8 GB GPU cannot host 7B FP16 weights (~14 GB). Long-context 13K avg (RQ-2 E2) and batch-8 decode (RQ-2 E3) are impossible here.
- No second node, no 25 Gbps / NVLink pair, no 3×/4× A100 topology. Distributed bottleneck hypotheses cannot be tested on this machine and must be marked BLOCKED, not simulated.
- Laptop GPU is WDDM mode, not TCC; GPU utilization reported via NVML is approximate (see instrumentation notes).

## 8. Reproducibility Checklist (Stage 3R-A §4)

- [x] OS / CPU / RAM / GPU model / count / memory / driver / CUDA recorded
- [x] Python / pip / key packages recorded
- [x] Serving framework decision and blocking reason recorded
- [x] Model / precision / parallelism recorded
- [x] Env vars recorded
- [x] Topology recorded
- [x] Date recorded
- [ ] Framework git commit — pending torch/vLLM install on Linux rig; current engine hash recorded per run instead
