"""
Stage 3R-A Unified Benchmark Harness
- Engine-agnostic: talks to any OpenAI-compatible HTTP endpoint (vLLM, SGLang, hf_server, mock)
- Records Request-level + System-level metrics per spec §5
- Supports warmup, repeated runs, structured raw/processed output
- No simulation: every latency comes from measured wall-clock + SSE streaming
"""
import asyncio
import time
import json
import csv
import hashlib
import uuid
import statistics
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import psutil
import httpx
import tiktoken

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

ENC = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RequestRecord:
    request_id: str
    workload_type: str
    arrival_time: float  # unix epoch, client-side schedule time
    dispatch_time: float  # when coroutine actually started
    queue_time: float  # dispatch - arrival (>=0, client-side queue)
    ttft: Optional[float] = None  # seconds from dispatch to first token (streaming)
    tpot: Optional[float] = None  # (total_decode_time) / (output_tokens-1) if streaming
    inter_token_latencies: List[float] = field(default_factory=list)
    total_latency: Optional[float] = None  # dispatch -> completion
    completion_time: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    prompt_text: str = ""
    output_text: str = ""
    success: bool = False
    error: str = ""
    status_code: Optional[int] = None
    trace_run_id: Optional[str] = None
    client_slot_acquired_time: Optional[float] = None
    client_slot_acquired_perf_ns: Optional[int] = None
    client_send_perf_ns: Optional[int] = None
    client_headers_perf_ns: Optional[int] = None
    client_first_content_perf_ns: Optional[int] = None
    client_done_perf_ns: Optional[int] = None
    # server-side hints if available
    server_queue_time: Optional[float] = None
    server_prefill_time: Optional[float] = None

@dataclass
class SystemSample:
    timestamp: float
    cpu_percent: float
    ram_percent: float
    ram_used_gb: float
    gpu_util: Optional[float]  # NOT AVAILABLE if NVML absent
    gpu_mem_used_mb: Optional[float]
    gpu_mem_total_mb: Optional[float]
    gpu_mem_percent: Optional[float]
    cpu_freq_current_mhz: Optional[float] = None
    server_process_cpu_percent: Optional[float] = None
    server_process_rss_mb: Optional[float] = None
    server_process_threads: Optional[int] = None

@dataclass
class RunResult:
    run_id: str
    config: Dict[str, Any]
    environment_hash: str
    start_time: float
    end_time: float
    requests: List[RequestRecord]
    system_samples: List[SystemSample]
    # aggregates
    throughput_rps: Optional[float] = None
    token_throughput: Optional[float] = None
    median_latency: Optional[float] = None
    mean_latency: Optional[float] = None
    p50_latency: Optional[float] = None
    p95_latency: Optional[float] = None
    p99_latency: Optional[float] = None
    median_ttft: Optional[float] = None
    p95_ttft: Optional[float] = None
    median_tpot: Optional[float] = None
    # NOT AVAILABLE markers
    kv_cache_usage: str = "NOT AVAILABLE"  # hf naive has no paged KV exposure; vLLM would expose via /metrics
    scheduler_queue: str = "NOT AVAILABLE"
    batch_size: str = "NOT AVAILABLE"
    active_requests: str = "NOT AVAILABLE"
    preemptions: str = "NOT AVAILABLE"
    cache_eviction: str = "NOT AVAILABLE"
    cache_hit_reuse: str = "NOT AVAILABLE"

# ---------------------------------------------------------------------------
# System sampler
# ---------------------------------------------------------------------------
class SystemSampler:
    def __init__(self, interval: float = 0.5, server_pid: Optional[int] = None):
        self.interval = interval
        self.samples: List[SystemSample] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._nvml_inited = False
        self._handle = None
        self._server_process = None
        if server_pid is not None:
            try:
                self._server_process = psutil.Process(int(server_pid))
                self._server_process.cpu_percent(interval=None)
            except Exception:
                self._server_process = None
        if _NVML:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_inited = True
            except Exception:
                self._nvml_inited = False

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except asyncio.TimeoutError:
                self._task.cancel()

    async def _loop(self):
        while self._running:
            try:
                s = self._sample_once()
                self.samples.append(s)
            except Exception:
                pass
            await asyncio.sleep(self.interval)

    def _sample_once(self) -> SystemSample:
        cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        freq = psutil.cpu_freq()
        gpu_util = None
        gpu_mem_used = None
        gpu_mem_total = None
        gpu_mem_pct = None
        if self._nvml_inited and self._handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                gpu_util = float(util.gpu)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                gpu_mem_used = mem.used / (1024*1024)
                gpu_mem_total = mem.total / (1024*1024)
                gpu_mem_pct = 100.0 * mem.used / mem.total
            except Exception:
                pass
        server_cpu = None
        server_rss_mb = None
        server_threads = None
        if self._server_process is not None:
            try:
                server_cpu = self._server_process.cpu_percent(interval=None)
                server_rss_mb = self._server_process.memory_info().rss / (1024 * 1024)
                server_threads = self._server_process.num_threads()
            except Exception:
                pass
        return SystemSample(
            timestamp=time.time(),
            cpu_percent=cpu,
            ram_percent=vm.percent,
            ram_used_gb=vm.used / (1024**3),
            gpu_util=gpu_util,
            gpu_mem_used_mb=gpu_mem_used,
            gpu_mem_total_mb=gpu_mem_total,
            gpu_mem_percent=gpu_mem_pct,
            cpu_freq_current_mhz=float(freq.current) if freq is not None else None,
            server_process_cpu_percent=server_cpu,
            server_process_rss_mb=server_rss_mb,
            server_process_threads=server_threads,
        )

# ---------------------------------------------------------------------------
# Token counting helper
# ---------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    try:
        return len(ENC.encode(text))
    except Exception:
        return len(text.split())

# ---------------------------------------------------------------------------
# Single request execution (streaming-aware)
# ---------------------------------------------------------------------------
async def execute_request(
    client: httpx.AsyncClient,
    base_url: str,
    rec: RequestRecord,
    max_tokens: int,
    temperature: float,
    stream: bool = True,
    timeout: float = 120.0,
    trace_run_id: Optional[str] = None,
) -> RequestRecord:
    """
    Hits POST {base_url}/v1/completions or /v1/chat/completions depending on server.
    Supports both OpenAI chat and legacy completions.
    For streaming, measures TTFT as time to first SSE token.
    """
    # Use chat completions as primary (hf_server and vLLM both support it)
    url_chat = base_url.rstrip("/") + "/v1/chat/completions"
    url_legacy = base_url.rstrip("/") + "/v1/completions"
    # Prefer chat
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": rec.prompt_text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    rec.input_tokens = count_tokens(rec.prompt_text)
    rec.trace_run_id = trace_run_id
    rec.dispatch_time = time.time()
    rec.queue_time = max(0.0, rec.dispatch_time - rec.arrival_time)
    rec.client_send_perf_ns = time.perf_counter_ns()
    headers = None
    if trace_run_id:
        headers = {
            "X-Stage3-Run-Id": trace_run_id,
            "X-Stage3-Request-Id": rec.request_id,
        }
    first_token_time = None
    token_times: List[float] = []
    output_chunks: List[str] = []
    try:
        # try chat first
        url = url_chat
        # We need streaming read
        async with client.stream(
            "POST",
            url,
            json=payload,
            timeout=timeout,
            headers=headers,
        ) as resp:
            rec.status_code = resp.status_code
            rec.client_headers_perf_ns = time.perf_counter_ns()
            if resp.status_code != 200:
                # fallback to legacy or record error
                body = await resp.aread()
                # try legacy if chat not supported
                if resp.status_code == 404:
                    # retry legacy
                    payload2 = {
                        "model": "default",
                        "prompt": rec.prompt_text,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": stream,
                    }
                    url = url_legacy
                    async with client.stream(
                        "POST",
                        url,
                        json=payload2,
                        timeout=timeout,
                        headers=headers,
                    ) as resp2:
                        rec.status_code = resp2.status_code
                        rec.client_headers_perf_ns = time.perf_counter_ns()
                        if resp2.status_code != 200:
                            body2 = await resp2.aread()
                            rec.success = False
                            rec.error = f"HTTP {resp2.status_code}: {body2[:500]}"
                            rec.completion_time = time.time()
                            rec.client_done_perf_ns = time.perf_counter_ns()
                            rec.total_latency = rec.completion_time - rec.dispatch_time
                            return rec
                        # stream read legacy
                        async for line in resp2.aiter_lines():
                            if not line:
                                continue
                            # SSE: data: {...}
                            if line.startswith("data:"):
                                data = line[5:].strip()
                                if data == "[DONE]":
                                    break
                                try:
                                    j = json.loads(data)
                                    # legacy completions: choices[0].text
                                    txt = j.get("choices", [{}])[0].get("text", "")
                                    if txt is None:
                                        txt = ""
                                    if txt:
                                        now_ns = time.perf_counter_ns()
                                        now = now_ns / 1_000_000_000
                                        if first_token_time is None:
                                            first_token_time = now
                                            rec.client_first_content_perf_ns = now_ns
                                            rec.ttft = (now_ns - rec.client_send_perf_ns) / 1_000_000_000
                                        token_times.append(now)
                                        output_chunks.append(txt)
                                except Exception:
                                    continue
                            # non-SSE JSON (non-streaming fallback)
                            elif line.strip().startswith("{"):
                                try:
                                    j = json.loads(line)
                                    txt = j.get("choices", [{}])[0].get("text", "") or j.get("choices", [{}])[0].get("message", {}).get("content", "")
                                    output_chunks.append(txt)
                                except Exception:
                                    pass
                else:
                    rec.success = False
                    rec.error = f"HTTP {resp.status_code}: {body[:500]}"
                    rec.completion_time = time.time()
                    rec.client_done_perf_ns = time.perf_counter_ns()
                    rec.total_latency = rec.completion_time - rec.dispatch_time
                    return rec
            else:
                # chat success, parse SSE
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            j = json.loads(data)
                            choice = j.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            txt = delta.get("content", "")
                            if txt is None:
                                txt = ""
                            # also handle non-delta (some servers send message)
                            if not txt:
                                txt = choice.get("message", {}).get("content", "") or ""
                                # avoid duplicating full message each chunk; only first
                                if txt and len(output_chunks) == 0 and choice.get("finish_reason") is None:
                                    pass
                                else:
                                    if txt and len(txt) > 200:
                                        # likely full message, truncate to avoid double count
                                        txt = ""
                            if txt:
                                now_ns = time.perf_counter_ns()
                                now = now_ns / 1_000_000_000
                                if first_token_time is None:
                                    first_token_time = now
                                    rec.client_first_content_perf_ns = now_ns
                                    rec.ttft = (now_ns - rec.client_send_perf_ns) / 1_000_000_000
                                token_times.append(now)
                                output_chunks.append(txt)
                        except Exception:
                            continue
                    elif line.strip().startswith("{"):
                        try:
                            j = json.loads(line)
                            # non-streaming JSON
                            txt = j.get("choices", [{}])[0].get("message", {}).get("content", "") or j.get("choices", [{}])[0].get("text", "")
                            if txt:
                                output_chunks.append(txt)
                                if first_token_time is None:
                                    now_ns = time.perf_counter_ns()
                                    first_token_time = now_ns / 1_000_000_000
                                    rec.client_first_content_perf_ns = now_ns
                                    rec.ttft = (now_ns - rec.client_send_perf_ns) / 1_000_000_000
                                token_times.append(first_token_time)
                        except Exception:
                            pass
                # if streaming but no token arrived but success, fallback to read body (non-stream)
                if not output_chunks and first_token_time is None:
                    # try to read as single JSON (happens if server ignored stream)
                    pass

        # After streaming, if still no output but status 200, try non-streaming fetch as fallback?
        # We already captured chunks; compute metrics
        rec.output_text = "".join(output_chunks)
        # If streaming disabled, output will be empty above; try non-stream path
        if not rec.output_text and not stream:
            # Already handled via SSE fallback; attempt direct POST
            pass
        if stream and not rec.output_text:
            # last resort: try non-streaming request
            payload_ns = dict(payload)
            payload_ns["stream"] = False
            resp2 = await client.post(url, json=payload_ns, timeout=timeout, headers=headers)
            if resp2.status_code == 200:
                j = resp2.json()
                txt = j.get("choices", [{}])[0].get("message", {}).get("content", "") or j.get("choices", [{}])[0].get("text", "") or ""
                rec.output_text = txt
                if rec.ttft is None and txt:
                    now_ns = time.perf_counter_ns()
                    rec.client_first_content_perf_ns = now_ns
                    rec.ttft = (now_ns - rec.client_send_perf_ns) / 1_000_000_000
                    token_times = [now_ns / 1_000_000_000]

        rec.output_tokens = count_tokens(rec.output_text) if rec.output_text else 0
        # TPOT: time between tokens, decode window
        if len(token_times) >= 2 and first_token_time is not None:
            total_decode = token_times[-1] - first_token_time
            # need at least 1 inter-token interval
            rec.inter_token_latencies = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
            if rec.output_tokens and rec.output_tokens > 1:
                rec.tpot = total_decode / max(1, rec.output_tokens - 1)
            elif len(token_times) > 1:
                rec.tpot = statistics.mean(rec.inter_token_latencies) if rec.inter_token_latencies else None
        elif len(token_times) == 1:
            rec.tpot = None
        rec.completion_time = time.time()
        rec.client_done_perf_ns = time.perf_counter_ns()
        rec.total_latency = rec.completion_time - rec.dispatch_time
        rec.success = True
    except Exception as e:
        rec.completion_time = time.time()
        rec.client_done_perf_ns = time.perf_counter_ns()
        rec.total_latency = rec.completion_time - rec.dispatch_time if rec.dispatch_time else None
        rec.success = False
        rec.error = f"{type(e).__name__}: {e}"
    return rec

# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------
async def run_benchmark(
    base_url: str,
    workload: List[Dict[str, Any]],  # each item: prompt, max_tokens, temperature, arrival_offset
    config: Dict[str, Any],
    concurrency: int = 1,
    warmup_requests: int = 0,
    stream: bool = True,
    sampler_interval: float = 0.5,
    timeout: float = 120.0,
    arrival_distribution: str = "closed",  # closed = next request starts when slot free; poisson/open handled via offsets
    client_mode: str = "per_request",
) -> RunResult:
    """
    workload: list of dicts with keys prompt, max_tokens, temperature, arrival_offset (seconds from run start)
    concurrency: max concurrent in-flight requests
    warmup_requests: number of initial requests to discard from metrics (still executed)
    """
    run_id = config.get("run_id") or str(uuid.uuid4())[:8]
    env_hash = config.get("environment_hash", "unknown")
    if client_mode not in {"per_request", "pooled"}:
        raise ValueError(f"unsupported client_mode: {client_mode}")
    shared_client = None
    if client_mode == "pooled":
        connection_limit = max(8, concurrency)
        shared_client = httpx.AsyncClient(
            http2=False,
            limits=httpx.Limits(
                max_connections=connection_limit,
                max_keepalive_connections=connection_limit,
            ),
        )
    start_time = time.time()
    sampler = SystemSampler(
        interval=sampler_interval,
        server_pid=config.get("server_pid"),
    )
    await sampler.start()

    # Warmup phase if needed (sequential, not measured)
    if warmup_requests > 0:
        warmup = workload[:warmup_requests]
        if shared_client is not None:
            for item in warmup:
                rec = RequestRecord(
                    request_id=f"warmup-{uuid.uuid4().hex[:6]}",
                    workload_type=item.get("workload_type", "warmup"),
                    arrival_time=time.time(),
                    dispatch_time=0,
                    queue_time=0,
                    prompt_text=item["prompt"],
                )
                await execute_request(shared_client, base_url, rec, item.get("max_tokens", 64), item.get("temperature", 0.0), stream=stream, timeout=timeout)
                await asyncio.sleep(0.05)
        else:
            async with httpx.AsyncClient() as client:
                for item in warmup:
                    rec = RequestRecord(
                        request_id=f"warmup-{uuid.uuid4().hex[:6]}",
                        workload_type=item.get("workload_type", "warmup"),
                        arrival_time=time.time(),
                        dispatch_time=0,
                        queue_time=0,
                        prompt_text=item["prompt"],
                    )
                    await execute_request(client, base_url, rec, item.get("max_tokens", 64), item.get("temperature", 0.0), stream=stream, timeout=timeout)
                    await asyncio.sleep(0.05)

    # Main phase: schedule according to arrival_offset if open-loop, else closed-loop via semaphore
    sem = asyncio.Semaphore(concurrency)
    records: List[RequestRecord] = []
    run_start = time.time()

    # Build arrival times
    for idx, item in enumerate(workload[warmup_requests:] if warmup_requests else workload):
        offset = item.get("arrival_offset", 0.0)
        # arrival_time is run_start + offset
        item["_arrival_time"] = run_start + offset
        item["_idx"] = idx

    # Sort by arrival time
    sorted_workload = sorted(
        (workload[warmup_requests:] if warmup_requests else workload),
        key=lambda x: x["_arrival_time"]
    )

    async def _run_one(item) -> RequestRecord:
        # wait until arrival time (open-loop pacing)
        now = time.time()
        wait = item["_arrival_time"] - now
        if wait > 0:
            await asyncio.sleep(wait)
        async with sem:
            slot_acquired_time = time.time()
            slot_acquired_perf_ns = time.perf_counter_ns()
            rec = RequestRecord(
                request_id=f"req-{item['_idx']:04d}-{uuid.uuid4().hex[:6]}",
                workload_type=item.get("workload_type", "unknown"),
                arrival_time=item["_arrival_time"],
                dispatch_time=0,
                queue_time=0,
                prompt_text=item["prompt"],
                client_slot_acquired_time=slot_acquired_time,
                client_slot_acquired_perf_ns=slot_acquired_perf_ns,
            )
            if shared_client is not None:
                await execute_request(
                    shared_client,
                    base_url,
                    rec,
                    item.get("max_tokens", 64),
                    item.get("temperature", 0.0),
                    stream=stream,
                    timeout=timeout,
                    trace_run_id=run_id,
                )
            else:
                async with httpx.AsyncClient() as client:
                    # Per-request clients reproduce the original observation.
                    await execute_request(
                        client,
                        base_url,
                        rec,
                        item.get("max_tokens", 64),
                        item.get("temperature", 0.0),
                        stream=stream,
                        timeout=timeout,
                        trace_run_id=run_id,
                    )
            return rec

    # Limit concurrency but also respect arrival pacing: create tasks staggered
    tasks = []
    for item in sorted_workload:
        # pacing already done inside _run_one via sleep, but we need to not block scheduling loop
        # So we launch tasks with small stagger
        tasks.append(asyncio.create_task(_run_one(item)))
        # small delay to avoid thundering herd on tight arrival offsets; not needed for correctness
        if arrival_distribution == "closed":
            # closed-loop: no extra pacing, tasks will queue on semaphore
            pass
        else:
            await asyncio.sleep(0.001)

    # Wait for all
    done = await asyncio.gather(*tasks, return_exceptions=True)
    for r in done:
        if isinstance(r, Exception):
            # synthesize failed record
            rec = RequestRecord(
                request_id=f"failed-{uuid.uuid4().hex[:6]}",
                workload_type="unknown",
                arrival_time=time.time(),
                dispatch_time=time.time(),
                queue_time=0,
                prompt_text="",
            )
            rec.success = False
            rec.error = f"task exception: {r}"
            rec.completion_time = time.time()
            rec.total_latency = 0
            records.append(rec)
        else:
            records.append(r)

    await sampler.stop()
    end_time = time.time()

    # Aggregates (only successful)
    succ = [r for r in records if r.success and r.total_latency is not None]
    lat_sorted = sorted([r.total_latency for r in succ])
    ttfts = sorted([r.ttft for r in succ if r.ttft is not None])
    tpots = [r.tpot for r in succ if r.tpot is not None]

    def pct(data, p):
        if not data:
            return None
        k = (len(data)-1) * p/100
        f = int(k)
        c = min(f+1, len(data)-1)
        if f == c:
            return data[f]
        d0 = k - f
        return data[f]*(1-d0) + data[c]*d0

    result = RunResult(
        run_id=run_id,
        config=config,
        environment_hash=env_hash,
        start_time=start_time,
        end_time=end_time,
        requests=records,
        system_samples=sampler.samples,
    )
    if succ:
        result.throughput_rps = len(succ) / max(1e-6, end_time - run_start)
        total_tokens = sum(r.output_tokens or 0 for r in succ)
        result.token_throughput = total_tokens / max(1e-6, end_time - run_start)
        result.median_latency = statistics.median(lat_sorted) if lat_sorted else None
        result.mean_latency = statistics.mean(lat_sorted) if lat_sorted else None
        result.p50_latency = pct(lat_sorted, 50)
        result.p95_latency = pct(lat_sorted, 95)
        result.p99_latency = pct(lat_sorted, 99)
        result.median_ttft = statistics.median(ttfts) if ttfts else None
        result.p95_ttft = pct(ttfts, 95) if ttfts else None
        result.median_tpot = statistics.median(tpots) if tpots else None
    if shared_client is not None:
        await shared_client.aclose()
    return result

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_raw(result: RunResult, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # requests CSV
    req_path = out_dir / f"{result.run_id}_requests.csv"
    with open(req_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "request_id","workload_type","arrival_time","dispatch_time","queue_time",
            "ttft","tpot","inter_token_latencies","total_latency","completion_time",
            "input_tokens","output_tokens","success","error","status_code","prompt_text","output_text",
            "trace_run_id","client_slot_acquired_time","client_slot_acquired_perf_ns",
            "client_send_perf_ns","client_headers_perf_ns","client_first_content_perf_ns",
            "client_done_perf_ns",
            "server_queue_time","server_prefill_time"
        ])
        w.writeheader()
        for r in result.requests:
            d = asdict(r)
            d["inter_token_latencies"] = json.dumps(d["inter_token_latencies"])
            # truncate long text for CSV readability (full kept in json)
            d["prompt_text"] = d["prompt_text"][:2000]
            d["output_text"] = d["output_text"][:4000]
            w.writerow(d)
    # system CSV
    sys_path = out_dir / f"{result.run_id}_system.csv"
    with open(sys_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "cpu_percent",
                "ram_percent",
                "ram_used_gb",
                "gpu_util",
                "gpu_mem_used_mb",
                "gpu_mem_total_mb",
                "gpu_mem_percent",
                "cpu_freq_current_mhz",
                "server_process_cpu_percent",
                "server_process_rss_mb",
                "server_process_threads",
            ],
        )
        w.writeheader()
        for s in result.system_samples:
            w.writerow(asdict(s))
    # full JSON
    json_path = out_dir / f"{result.run_id}_raw.json"
    with open(json_path, "w", encoding="utf-8") as f:
        # serialize: convert dataclasses to dict
        payload = {
            "run_id": result.run_id,
            "config": result.config,
            "environment_hash": result.environment_hash,
            "start_time": result.start_time,
            "end_time": result.end_time,
            "aggregates": {
                "throughput_rps": result.throughput_rps,
                "token_throughput": result.token_throughput,
                "median_latency": result.median_latency,
                "mean_latency": result.mean_latency,
                "p50_latency": result.p50_latency,
                "p95_latency": result.p95_latency,
                "p99_latency": result.p99_latency,
                "median_ttft": result.median_ttft,
                "p95_ttft": result.p95_ttft,
                "median_tpot": result.median_tpot,
            },
            "not_available": {
                "kv_cache_usage": result.kv_cache_usage,
                "scheduler_queue": result.scheduler_queue,
                "batch_size": result.batch_size,
                "active_requests": result.active_requests,
                "preemptions": result.preemptions,
                "cache_eviction": result.cache_eviction,
                "cache_hit_reuse": result.cache_hit_reuse,
            },
            "requests": [asdict(r) for r in result.requests],
            "system_samples": [asdict(s) for s in result.system_samples],
        }
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return req_path, sys_path, json_path

def compute_processed(result: RunResult) -> Dict[str, Any]:
    succ = [r for r in result.requests if r.success]
    fail = len(result.requests) - len(succ)
    lat = [r.total_latency for r in succ if r.total_latency is not None]
    ttft = [r.ttft for r in succ if r.ttft is not None]
    tpot = [r.tpot for r in succ if r.tpot is not None]
    def stats(arr):
        if not arr:
            return {"count":0,"mean":None,"median":None,"std":None,"min":None,"max":None,"p50":None,"p95":None,"p99":None}
        arr_sorted = sorted(arr)
        def pct(p):
            k = (len(arr_sorted)-1)*p/100
            f=int(k); c=min(f+1,len(arr_sorted)-1)
            if f==c: return arr_sorted[f]
            d0=k-f; return arr_sorted[f]*(1-d0)+arr_sorted[c]*d0
        return {
            "count": len(arr),
            "mean": statistics.mean(arr),
            "median": statistics.median(arr),
            "std": statistics.stdev(arr) if len(arr)>1 else 0.0,
            "min": min(arr),
            "max": max(arr),
            "p50": pct(50),
            "p95": pct(95),
            "p99": pct(99),
        }
    sys_cpu = [s.cpu_percent for s in result.system_samples if s.cpu_percent is not None]
    sys_cpu_freq = [s.cpu_freq_current_mhz for s in result.system_samples if s.cpu_freq_current_mhz is not None]
    sys_server_cpu = [s.server_process_cpu_percent for s in result.system_samples if s.server_process_cpu_percent is not None]
    sys_server_rss = [s.server_process_rss_mb for s in result.system_samples if s.server_process_rss_mb is not None]
    sys_server_threads = [s.server_process_threads for s in result.system_samples if s.server_process_threads is not None]
    sys_gpu = [s.gpu_util for s in result.system_samples if s.gpu_util is not None]
    sys_gmem = [s.gpu_mem_percent for s in result.system_samples if s.gpu_mem_percent is not None]
    return {
        "run_id": result.run_id,
        "config": result.config,
        "success": len(succ),
        "failed": fail,
        "total_requests": len(result.requests),
        "duration_s": result.end_time - result.start_time,
        "throughput_rps": result.throughput_rps,
        "token_throughput": result.token_throughput,
        "latency": stats(lat),
        "ttft": stats(ttft),
        "tpot": stats(tpot),
        "system": {
            "cpu": stats(sys_cpu),
            "cpu_freq_current_mhz": stats(sys_cpu_freq) if sys_cpu_freq else {"note":"NOT AVAILABLE"},
            "server_process_cpu": stats(sys_server_cpu) if sys_server_cpu else {"note":"NOT AVAILABLE"},
            "server_process_rss_mb": stats(sys_server_rss) if sys_server_rss else {"note":"NOT AVAILABLE"},
            "server_process_threads": stats(sys_server_threads) if sys_server_threads else {"note":"NOT AVAILABLE"},
            "gpu_util": stats(sys_gpu) if sys_gpu else {"note":"NOT AVAILABLE"},
            "gpu_mem_percent": stats(sys_gmem) if sys_gmem else {"note":"NOT AVAILABLE"},
            "ram_max_gb": max([s.ram_used_gb for s in result.system_samples], default=None),
        },
        "not_available": {
            "kv_cache_usage": result.kv_cache_usage,
            "scheduler_queue": result.scheduler_queue,
            "batch_size": result.batch_size,
        }
    }
