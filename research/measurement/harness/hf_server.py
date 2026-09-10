"""
Minimal real LLM serving shim — HF Transformers (naive, no PagedAttention)
Implements OpenAI-compatible endpoints:
  POST /v1/chat/completions  (streaming SSE + non-streaming)
  POST /v1/completions       (legacy)
  GET  /health
  GET  /metrics  (basic)

Design goals (§6 Instrumentation principle):
- Minimal instrumentation, does not change serving behavior
- Uses existing framework metrics (HF generate) + logging hooks
- Supports streaming for TTFT/TPOT measurement
- Exposes GPU/CPU stats via NVML/psutil without polluting generate path
"""
import time
import json
import asyncio
import uuid
import os
import hashlib
import threading
from typing import AsyncGenerator, Optional
from pathlib import Path

import concurrent.futures
import psutil
try:
    import pynvml
    _NVML = True
except ImportError:
    try:
        import nvidia_ml_py as pynvml
        _NVML = True
    except ImportError:
        _NVML = False
        pynvml = None

# Fixed stabilization config — Stage 3M-B (see LOCKED_STATIONARITY_PLAN.md §2)
FIXED_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=32, thread_name_prefix="hf-fixed-3mb")
FIXED_TORCH_THREADS = 16
FIXED_TORCH_INTEROP = 16

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# Lazy imports — torch/transformers may be absent; server still starts in mock mode
try:
    import torch
    import transformers as transformers_module
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None
    transformers_module = None

try:
    from .causal_trace import RequestTrace, TraceStore, new_request_trace
except ImportError:
    from causal_trace import RequestTrace, TraceStore, new_request_trace

try:
    from .minimal_trace import (
        MinimalRequestTrace,
        MinimalTraceStore,
        CounterStore,
        new_minimal_trace,
        L1_HANDLER_ENTER,
        L1_EXECUTOR_SUBMIT,
        L1_EXECUTOR_START,
        L1_FIRST_FORWARD_BEGIN,
        L1_FIRST_CONTENT_YIELD,
        L1_SERVER_DONE,
    )
except ImportError:
    from minimal_trace import (
        MinimalRequestTrace,
        MinimalTraceStore,
        CounterStore,
        new_minimal_trace,
        L1_HANDLER_ENTER,
        L1_EXECUTOR_SUBMIT,
        L1_EXECUTOR_START,
        L1_FIRST_FORWARD_BEGIN,
        L1_FIRST_CONTENT_YIELD,
        L1_SERVER_DONE,
    )

app = FastAPI(title="Stage3R-A HF Serving Shim")

@app.on_event("startup")
async def _stage3mb_fixed_init():
    # Lock torch threads (Stage 3M-B §2)
    if _HAS_TORCH and torch is not None:
        try:
            torch.set_num_threads(FIXED_TORCH_THREADS)
            try:
                torch.set_num_interop_threads(FIXED_TORCH_INTEROP)
            except Exception:
                pass
            # also set intra-op via set_num_threads already
        except Exception as e:
            print(f"[hf_server] WARNING: torch thread fix failed: {e}")
    # Fix executor for asyncio.to_thread (Stage 3M-B ThreadPool 32)
    try:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(FIXED_EXECUTOR)
        print(f"[hf_server] fixed executor max_workers=32 active={FIXED_EXECUTOR._max_workers} torch_threads={FIXED_TORCH_THREADS}/{FIXED_TORCH_INTEROP}")
    except Exception as e:
        print(f"[hf_server] WARNING: fixed executor setup failed: {e}")

# Global state
STATE = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "model_id": None,
    "dtype": None,
    "loaded": False,
    "mock": False,
    "mock_latency_ms": 20,  # per-token mock latency for harness validation when torch absent
    "request_count": 0,
    "start_time": time.time(),
    "trace_enabled": False,
    "trace_level": 0,  # 0=OFF, 1=L1 minimal (6 pts), 2=L2 full, 10=L1a, 11=L1b, 12=L1c
    "trace_store": TraceStore(),
    "trace_store_minimal": MinimalTraceStore(capacity=50000),
    "trace_counter_store": CounterStore(),
    "trace_sample_ratio": 1.0,
}


def _source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _process_stats():
    process = psutil.Process(os.getpid())
    try:
        rss_mb = process.memory_info().rss / (1024 * 1024)
    except Exception:
        rss_mb = None
    try:
        process_threads = process.num_threads()
    except Exception:
        process_threads = None
    return {
        "rss_mb": rss_mb,
        "process_threads": process_threads,
        "python_threads": threading.active_count(),
    }


def _runtime_details():
    if not _HAS_TORCH:
        return {
            "torch_version": None,
            "transformers_version": None,
            "torch_num_threads": None,
            "torch_num_interop_threads": None,
            "fixed_torch_threads": FIXED_TORCH_THREADS,
            "fixed_torch_interop": FIXED_TORCH_INTEROP,
            "fixed_executor_max_workers": 32,
        }
    return {
        "torch_version": getattr(torch, "__version__", None),
        "transformers_version": getattr(transformers_module, "__version__", None),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "fixed_torch_threads": FIXED_TORCH_THREADS,
        "fixed_torch_interop": FIXED_TORCH_INTEROP,
        "fixed_executor_max_workers": 32,
    }

def _get_gpu_stats():
    if not _NVML:
        return {"gpu_util": None, "gpu_mem_used_mb": None, "gpu_mem_total_mb": None}
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {"gpu_util": float(util.gpu), "gpu_mem_used_mb": mem.used/(1024*1024), "gpu_mem_total_mb": mem.total/(1024*1024)}
    except Exception:
        return {"gpu_util": None, "gpu_mem_used_mb": None, "gpu_mem_total_mb": None}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pid": os.getpid(),
        "loaded": STATE["loaded"],
        "mock": STATE["mock"],
        "model_id": STATE["model_id"],
        "device": str(STATE["device"]),
        "has_torch": _HAS_TORCH,
        "request_count": STATE["request_count"],
        "uptime_s": time.time() - STATE["start_time"],
        "trace_enabled": STATE["trace_enabled"],
        "trace_level": STATE["trace_level"],
        "trace_sample_ratio": STATE["trace_sample_ratio"],
        "source_hash": _source_hash(),
        "runtime": _runtime_details(),
        "process": _process_stats(),
        **_get_gpu_stats(),
    }

@app.get("/metrics")
async def metrics():
    # Expose what we can; mark unavailable per spec §5
    gpu = _get_gpu_stats()
    vm = psutil.virtual_memory()
    return {
        "throughput": {"request_count": STATE["request_count"]},
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_percent": vm.percent,
            "ram_used_gb": vm.used/(1024**3),
            **gpu,
        },
        "not_available": {
            "kv_cache_usage": "NOT AVAILABLE (hf naive has no paged KV exposure)",
            "scheduler_queue": "NOT AVAILABLE",
            "batch_size": "NOT AVAILABLE (no continuous batching)",
            "active_requests": "NOT AVAILABLE",
            "preemptions": "NOT AVAILABLE",
            "cache_eviction": "NOT AVAILABLE",
            "cache_hit_reuse": "NOT AVAILABLE (no prefix cache)",
        }
    }


@app.post("/stage3/trace/mode")
async def trace_mode(req: Request):
    body = await req.json()
    enabled = body.get("enabled")
    if not isinstance(enabled, bool):
        return JSONResponse({"error": "enabled must be a boolean"}, status_code=400)
    STATE["trace_enabled"] = enabled
    # legacy boolean mode maps to level 2 (full) when enabled, 0 when disabled
    if enabled and STATE["trace_level"] == 0:
        STATE["trace_level"] = 2
    elif not enabled:
        STATE["trace_level"] = 0
    return {"trace_enabled": STATE["trace_enabled"], "trace_level": STATE["trace_level"]}


@app.post("/stage3/trace/level")
async def trace_level(req: Request):
    body = await req.json()
    level = body.get("level")
    if level not in (0, 1, 2, 10, 11, 12):
        return JSONResponse({"error": "level must be 0,1,2,10,11,12"}, status_code=400)
    sample_ratio = body.get("sample_ratio", 1.0)
    if not isinstance(sample_ratio, (int, float)) or not (0 < sample_ratio <= 1.0):
        sample_ratio = 1.0
    STATE["trace_level"] = int(level)
    STATE["trace_enabled"] = level != 0
    STATE["trace_sample_ratio"] = float(sample_ratio)
    return {"trace_level": STATE["trace_level"], "trace_enabled": STATE["trace_enabled"], "sample_ratio": STATE["trace_sample_ratio"]}


@app.get("/stage3/trace/{run_id}")
async def take_trace(run_id: str):
    level = STATE["trace_level"]
    if level == 1 or level == 10 or level == 11:
        return {
            "run_id": run_id,
            "traces": STATE["trace_store_minimal"].take_run(run_id),
            "level": level,
        }
    elif level == 12:
        return {
            "run_id": run_id,
            "counters": STATE["trace_counter_store"].take_run(run_id),
            "level": level,
        }
    else:
        return {
            "run_id": run_id,
            "traces": STATE["trace_store"].take_run(run_id),
            "level": level,
        }

def load_model(model_id: str = "gpt2", device: str = "auto", dtype: str = "auto", mock: bool = False):
    STATE["mock"] = mock or (not _HAS_TORCH)
    STATE["model_id"] = model_id
    if STATE["mock"]:
        print(f"[hf_server] MOCK mode — no real model (mock={mock}, has_torch={_HAS_TORCH})")
        STATE["loaded"] = True
        STATE["device"] = "mock"
        STATE["dtype"] = "mock"
        return
    # real load
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    STATE["device"] = device
    if dtype == "auto":
        dtype = torch.float16 if device == "cuda" else torch.float32
        STATE["dtype"] = str(dtype)
    else:
        STATE["dtype"] = dtype
        if dtype == "float16":
            dtype = torch.float16
        elif dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    print(f"[hf_server] loading {model_id} on {device} dtype={dtype} ...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Use low_cpu_mem_usage and device_map for large models; for small models simple
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"[hf_server] load failed: {e}, falling back to mock")
        STATE["mock"] = True
        STATE["loaded"] = True
        STATE["device"] = "mock"
        return
    STATE["tokenizer"] = tok
    STATE["model"] = model
    STATE["loaded"] = True
    print(f"[hf_server] loaded {model_id} — ready")

# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def _mock_generate(prompt: str, max_tokens: int) -> str:
    """Deterministic mock output for harness validation (not counted as real serving)."""
    # Return prompt echo + filler
    base = f" [mock response to: {prompt[:80]}] "
    filler = " ".join(["token"] * max_tokens)
    return (base + filler)[: max_tokens*6]

async def _stream_mock(prompt: str, max_tokens: int) -> AsyncGenerator[str, None]:
    text = _mock_generate(prompt, max_tokens)
    # split into token-ish chunks
    words = text.split()
    for w in words:
        await asyncio.sleep(STATE["mock_latency_ms"]/1000)
        yield w + " "

async def _stream_real(
    prompt: str,
    max_tokens: int,
    temperature: float,
    trace = None,
) -> AsyncGenerator[str, None]:
    """Naive streaming via incremental generate (not true continuous batching, but measures real forward).
    Handles position overflow for small models (e.g., tiny-gpt2 n_positions=1024) by truncating input.
    Runs blocking torch work in threadpool to avoid starving asyncio event loop.

    Supports both L2 full (RequestTrace) and L1 minimal (MinimalRequestTrace).
    L1 hot path: only 6 marks, no per-token dict, no get_ident, no sorting.
    """
    tok = STATE["tokenizer"]
    model = STATE["model"]
    device = STATE["device"]
    max_pos = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", None) or 2048

    is_minimal = isinstance(trace, MinimalRequestTrace)
    is_full = isinstance(trace, RequestTrace)

    # L1 minimal: executor submit (idx 1)
    if is_minimal:
        trace.mark(L1_EXECUTOR_SUBMIT)
    elif is_full:
        trace.mark("t_prepare_submit_ns")

    def _prepare():
        if is_minimal:
            trace.mark(L1_EXECUTOR_START)
        elif is_full:
            trace.mark("t_prepare_worker_start_ns")
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max(16, max_pos - max_tokens - 2))
        input_ids = inputs["input_ids"].to(device)
        if is_full:
            trace.mark("t_prepare_worker_done_ns")
        return input_ids

    input_ids = await asyncio.to_thread(_prepare)
    if is_full:
        trace.mark("t_prepare_resume_ns")
    import torch as _torch
    generated = input_ids
    past = None
    for step_index in range(max_tokens):
        # Full trace per-token overhead
        if is_full:
            step = {"index": step_index}
            submitted = time.perf_counter_ns()
            step["t_submit_ns"] = submitted
            if step_index == 0:
                trace.mark("t_first_step_submit_ns", submitted)
        else:
            step = None

        def _one_step(gen, past_cache):
            if is_full:
                worker_start = time.perf_counter_ns()
                step["t_worker_start_ns"] = worker_start
                step["worker_thread_id"] = threading.get_ident()
                if step_index == 0:
                    trace.mark("t_first_step_worker_start_ns", worker_start)
            with _torch.no_grad():
                if is_minimal and step_index == 0:
                    # first forward begin for L1 (idx 3) — inside worker, captured before model
                    fb = time.perf_counter_ns()
                    trace.mark(L1_FIRST_FORWARD_BEGIN, fb)
                elif is_full:
                    forward_begin = time.perf_counter_ns()
                    step["t_forward_begin_ns"] = forward_begin
                    if step_index == 0:
                        trace.mark("t_first_forward_begin_ns", forward_begin)
                if past_cache is None:
                    out = model(input_ids=gen, use_cache=True)
                else:
                    out = model(input_ids=gen[:, -1:], past_key_values=past_cache, use_cache=True)
                if is_full:
                    forward_end = time.perf_counter_ns()
                    step["t_forward_end_ns"] = forward_end
                    if step_index == 0:
                        trace.mark("t_first_forward_end_ns", forward_end)
                logits = out.logits
                past_next = out.past_key_values
                next_token_logits = logits[:, -1, :]
                if temperature and temperature > 0:
                    probs = _torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = _torch.multinomial(probs, num_samples=1)
                else:
                    next_token = _torch.argmax(next_token_logits, dim=-1, keepdim=True)
                if is_full:
                    token_sampled = time.perf_counter_ns()
                    step["t_token_sampled_ns"] = token_sampled
                    if step_index == 0:
                        trace.mark("t_first_token_sampled_ns", token_sampled)
                gen_next = _torch.cat([gen, next_token], dim=-1)
                txt = tok.decode(next_token[0], skip_special_tokens=True)
                eos = next_token.item() == tok.eos_token_id
                return gen_next, past_next, txt, eos
        try:
            generated, past, txt, is_eos = await asyncio.to_thread(_one_step, generated, past)
        except Exception as e:
            if is_full:
                step["error"] = f"{type(e).__name__}: {e}"
                trace.add_step(step)
            print(f"[hf_server] _stream_real step failed: {e}")
            break
        if is_full:
            resumed = time.perf_counter_ns()
            step["t_resume_ns"] = resumed
            if step_index == 0:
                trace.mark("t_first_step_resume_ns", resumed)
            trace.add_step(step)
        if is_eos:
            break
        if txt:
            if is_minimal:
                # first content yield (idx 4) — only once, check ts slot zero
                if trace.ts[L1_FIRST_CONTENT_YIELD] == 0:
                    trace.mark(L1_FIRST_CONTENT_YIELD)
            elif is_full and "t_first_content_yield_ns" not in trace.timestamps:
                trace.mark("t_first_content_yield_ns")
            yield txt
        await asyncio.sleep(0)
        # Guard against exceeding max position
        if generated.shape[1] >= max_pos:
            break

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
def _sse_format(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    trace_run_id = req.headers.get("X-Stage3-Run-Id")
    trace_request_id = req.headers.get("X-Stage3-Request-Id")
    # Select trace type based on level
    level = STATE["trace_level"]
    trace = None
    if level == 1 or level == 10 or level == 11:
        trace = new_minimal_trace(
            enabled=bool(STATE["trace_enabled"] and trace_run_id and trace_request_id),
            level=level,
            run_id=trace_run_id or "",
            request_id=trace_request_id or "",
            sample_ratio=STATE["trace_sample_ratio"],
        )
    elif level == 2:
        trace = new_request_trace(
            enabled=bool(STATE["trace_enabled"] and trace_run_id and trace_request_id),
            run_id=trace_run_id or "",
            request_id=trace_request_id or "",
        )
    elif level == 12:
        # L1c aggregate only — count, no per-request trace
        if trace_run_id and trace_request_id:
            STATE["trace_counter_store"].inc(trace_run_id)
        trace = None
    else:
        trace = None

    if isinstance(trace, MinimalRequestTrace):
        trace.mark(L1_HANDLER_ENTER)
    elif isinstance(trace, RequestTrace):
        trace.mark("t_handler_enter_ns")
    body = await req.json()
    if isinstance(trace, MinimalRequestTrace):
        # L1 reuses handler_enter as receive; no separate body_parsed to save one mark
        pass
    elif isinstance(trace, RequestTrace):
        trace.mark("t_body_parsed_ns")
    messages = body.get("messages", [])
    prompt = ""
    for m in messages:
        if m.get("role") in ("user", "system"):
            prompt += m.get("content", "") + "\n"
    prompt = prompt.strip() or "Hello"
    max_tokens = int(body.get("max_tokens", 64))
    temperature = float(body.get("temperature", 0.0))
    stream = bool(body.get("stream", False))
    model_id = body.get("model", STATE["model_id"] or "default")
    trace_response_headers = None
    if trace_run_id and trace_request_id:
        trace_response_headers = {
            "X-Stage3-Run-Id": trace_run_id,
            "X-Stage3-Request-Id": trace_request_id,
        }
    STATE["request_count"] += 1

    if not STATE["loaded"]:
        return JSONResponse({"error": "model not loaded"}, status_code=503)

    if STATE["mock"]:
        if not stream:
            text = _mock_generate(prompt, max_tokens)
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}],
                "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": max_tokens, "total_tokens": len(prompt.split())+max_tokens}
            }
        else:
            async def gen():
                async for chunk in _stream_mock(prompt, max_tokens):
                    payload = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [{"index":0,"delta":{"content":chunk},"finish_reason":None}]
                    }
                    yield _sse_format(payload)
                # final
                payload = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{"index":0,"delta":{},"finish_reason":"stop"}]
                }
                yield _sse_format(payload)
                yield "data: [DONE]\n\n"
            return StreamingResponse(
                gen(),
                media_type="text/event-stream",
                headers=trace_response_headers,
            )
    else:
        # real
        if not stream:
            # non-streaming: generate full then return
            tok = STATE["tokenizer"]
            model = STATE["model"]
            device = STATE["device"]
            max_pos = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", None) or 2048
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max(16, max_pos - max_tokens - 2))
            input_ids = inputs["input_ids"].to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=temperature>0,
                    temperature=temperature if temperature>0 else None,
                    pad_token_id=tok.eos_token_id,
                    use_cache=True,
                )
            # decode only new tokens
            new_ids = out_ids[0][input_ids.shape[1]:]
            text = tok.decode(new_ids, skip_special_tokens=True)
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}],
                "usage": {"prompt_tokens": input_ids.shape[1], "completion_tokens": len(new_ids), "total_tokens": len(out_ids[0])}
            }
        else:
            async def gen_real():
                try:
                    if isinstance(trace, MinimalRequestTrace):
                        # L1: no stream_runtime_enter separate, executor_submit already recorded
                        pass
                    elif isinstance(trace, RequestTrace):
                        trace.mark("t_stream_runtime_enter_ns")
                    async for chunk in _stream_real(
                        prompt,
                        max_tokens,
                        temperature,
                        trace=trace,
                    ):
                        payload = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_id,
                            "choices": [{"index":0,"delta":{"content":chunk},"finish_reason":None}]
                        }
                        yield _sse_format(payload)
                    payload = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [{"index":0,"delta":{},"finish_reason":"stop"}]
                    }
                    yield _sse_format(payload)
                    yield "data: [DONE]\n\n"
                finally:
                    if isinstance(trace, MinimalRequestTrace):
                        if trace.ts[L1_SERVER_DONE] == 0:
                            trace.mark(L1_SERVER_DONE)
                        STATE["trace_store_minimal"].add(trace)
                    elif isinstance(trace, RequestTrace):
                        if "t_server_done_ns" not in trace.timestamps:
                            trace.mark("t_server_done_ns")
                        STATE["trace_store"].add(trace)
            return StreamingResponse(
                gen_real(),
                media_type="text/event-stream",
                headers=trace_response_headers,
            )

@app.post("/v1/completions")
async def completions(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    max_tokens = int(body.get("max_tokens", 64))
    temperature = float(body.get("temperature", 0.0))
    stream = bool(body.get("stream", False))
    model_id = body.get("model", STATE["model_id"] or "default")
    STATE["request_count"] += 1
    if not STATE["loaded"]:
        return JSONResponse({"error": "model not loaded"}, status_code=503)
    if STATE["mock"]:
        if not stream:
            text = _mock_generate(prompt, max_tokens)
            return {
                "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"text": text, "index":0, "finish_reason":"stop"}],
            }
        else:
            async def gen():
                async for chunk in _stream_mock(prompt, max_tokens):
                    payload = {
                        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [{"text": chunk, "index":0, "finish_reason":None}]
                    }
                    yield _sse_format(payload)
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
    else:
        tok = STATE["tokenizer"]
        model = STATE["model"]
        device = STATE["device"]
        if not stream:
            max_pos = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", None) or 2048
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max(16, max_pos - max_tokens - 2))
            input_ids = inputs["input_ids"].to(device)
            with torch.no_grad():
                out_ids = model.generate(input_ids, max_new_tokens=max_tokens, do_sample=temperature>0, temperature=temperature if temperature>0 else None, pad_token_id=tok.eos_token_id, use_cache=True)
            new_ids = out_ids[0][input_ids.shape[1]:]
            text = tok.decode(new_ids, skip_special_tokens=True)
            return {
                "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"text": text, "index":0, "finish_reason":"stop"}],
            }
        else:
            async def gen_real():
                async for chunk in _stream_real(prompt, max_tokens, temperature):
                    payload = {
                        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [{"text": chunk, "index":0, "finish_reason":None}]
                    }
                    yield _sse_format(payload)
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen_real(), media_type="text/event-stream")

# CLI
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2", help="HF model id")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--mock", action="store_true", help="force mock mode (for harness validation without torch)")
    p.add_argument("--mock-latency-ms", type=int, default=20)
    p.add_argument("--trace", action="store_true", help="enable Stage 3A request tracing at startup")
    p.add_argument("--executor-workers", type=int, default=32, help="fixed ThreadPoolExecutor max_workers (3M-B locked=32)")
    p.add_argument("--torch-threads", type=int, default=16, help="torch num_threads (3M-B locked=16)")
    args = p.parse_args()
    # Apply locked stabilization before load
    FIXED_EXECUTOR._max_workers = args.executor_workers  # type: ignore
    # Note: ThreadPoolExecutor max_workers cannot be changed after creation; recreate if needed
    if args.executor_workers != 32:
        import concurrent.futures as _cf
        globals()["FIXED_EXECUTOR"] = _cf.ThreadPoolExecutor(max_workers=args.executor_workers, thread_name_prefix="hf-fixed-3mb")
    globals()["FIXED_TORCH_THREADS"] = args.torch_threads
    if _HAS_TORCH and torch is not None:
        try:
            torch.set_num_threads(args.torch_threads)
            torch.set_num_interop_threads(args.torch_threads)
        except Exception as e:
            print(f"[hf_server] WARNING pre-load torch thread set failed: {e}")
    STATE["mock_latency_ms"] = args.mock_latency_ms
    STATE["trace_enabled"] = args.trace
    STATE["trace_level"] = 2 if args.trace else 0
    load_model(args.model, device=args.device, dtype=args.dtype, mock=args.mock)
    print(f"[hf_server] serving on http://{args.host}:{args.port} (mock={STATE['mock']}, level={STATE['trace_level']}, executor={args.executor_workers}, torch_threads={args.torch_threads})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
