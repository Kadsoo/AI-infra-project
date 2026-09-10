"""
Stage 3R-A Workload Generator (§7)
Supports controlling:
- request count, input length, output length, concurrency, arrival rate/distribution,
  prefix reuse, repeated prefixes, request groups, burstiness

Six workload families:
1. SyntheticControlled — precise length control, no reuse, for isolation
2. Chat-like — short-medium natural prompts
3. Prefix-Reuse — shared system prefix across requests
4. Long-Context — 2k-8k tokens input
5. RAG-like — context+question pattern
6. Agent-like — multi-turn-ish tool traces (without extreme fabrication)

All workloads emit List[Dict] compatible with harness.run_benchmark:
  {prompt, max_tokens, temperature, arrival_offset, workload_type}
"""
import random
import math
from typing import List, Dict, Any, Optional
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")

# Small prompt bank for realistic chat prefixes (not synthetic extremes)
CHAT_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to reverse a linked list.",
    "Summarize the following article in three bullet points:",
    "What are the trade-offs between REST and GraphQL?",
    "Give me a 5-day itinerary for Tokyo on a budget.",
    "Debug this code: ",
    "Translate 'Hello, how are you today?' into French and Japanese.",
    "Compare and contrast supervised and unsupervised learning.",
    "Write a haiku about autumn in Kyoto.",
    "What is the difference between a thread and a process?",
    "Generate a regex to validate email addresses and explain it.",
    "How does a transformer attention mechanism work?",
]

RAG_CONTEXT_TEMPLATE = """Context:
{context}

Question: {question}
Answer concisely using only the context above."""

AGENT_TOOL_TRACE = """You are an AI assistant with tool access.
User: {user_query}
Available tools: search(query), calculator(expr), read_doc(id)
Trace:
- Thought: {thought}
- Action: {action}
- Observation: {observation}
Continue the trace to answer the user."""

# Common shared prefix for reuse experiments (e.g., system prompt + doc)
SHARED_PREFIX = (
    "You are a helpful assistant. Follow these instructions carefully.\n"
    "System rules: Be concise, cite sources, refuse disallowed content.\n"
    "Document corpus (shared prefix, ~512 tokens when repeated):\n"
)

def _estimate_tokens(text: str) -> int:
    return len(ENC.encode(text))

def _make_filler(target_tokens: int, seed: int = 0) -> str:
    """Deterministic filler to reach target token length."""
    rnd = random.Random(seed)
    base = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    # repeat+shuffle to avoid trivial compression benefits
    toks = _estimate_tokens(base)
    reps = math.ceil(target_tokens / max(1, toks))
    # vary word order slightly per seed to avoid exact dedup
    words = base.split()
    out_parts = []
    for i in range(reps):
        rnd.shuffle(words)
        out_parts.append(" ".join(words))
    text = " ".join(out_parts)
    # trim to target
    ids = ENC.encode(text)
    if len(ids) > target_tokens:
        text = ENC.decode(ids[:target_tokens])
    return text

def _arrival_offsets(n: int, arrival_rate: float, distribution: str, seed: int) -> List[float]:
    """Generate arrival offsets (seconds) for open-loop."""
    rnd = random.Random(seed)
    if arrival_rate <= 0 or distribution == "closed":
        return [0.0]*n
    if distribution == "poisson":
        # exponential inter-arrival
        offsets = [0.0]
        for _ in range(1, n):
            ia = rnd.expovariate(arrival_rate)
            offsets.append(offsets[-1] + ia)
        return offsets
    elif distribution == "gamma":
        # gamma shape=2 (less bursty than exp with same mean, but controllable)
        # use shape k=2, theta = 1/(k*rate) ? mean = k*theta => theta=1/(k*rate)? Actually mean=k*theta, so theta=1/(k*rate) approx.
        # We expose burstiness via gamma; for simplicity shape=2
        k = 2.0
        theta = 1.0/(k*arrival_rate)
        offsets = [0.0]
        for _ in range(1, n):
            ia = rnd.gammavariate(k, theta)
            offsets.append(offsets[-1] + ia)
        return offsets
    elif distribution == "bursty":
        # 2-state: burst with high rate, idle with low rate (on/off)
        offsets = [0.0]
        burst_rate = arrival_rate * 3
        idle_rate = arrival_rate * 0.3
        in_burst = True
        burst_remaining = rnd.randint(3, 8)
        for _ in range(1, n):
            rate = burst_rate if in_burst else idle_rate
            ia = rnd.expovariate(rate)
            offsets.append(offsets[-1] + ia)
            burst_remaining -= 1
            if burst_remaining <= 0:
                in_burst = not in_burst
                burst_remaining = rnd.randint(2, 6) if in_burst else rnd.randint(4, 10)
        return offsets
    elif distribution == "uniform":
        # uniform inter-arrival around mean
        mean_ia = 1.0/arrival_rate
        offsets = [0.0]
        for _ in range(1, n):
            ia = rnd.uniform(mean_ia*0.5, mean_ia*1.5)
            offsets.append(offsets[-1] + ia)
        return offsets
    else:
        return [0.0]*n

def _build_workload(
    n: int,
    input_tokens: int,
    output_tokens: int,
    workload_type: str,
    arrival_rate: float,
    arrival_distribution: str,
    prefix_reuse: float,  # 0..1 fraction of requests sharing prefix
    seed: int,
) -> List[Dict[str, Any]]:
    rnd = random.Random(seed)
    offsets = _arrival_offsets(n, arrival_rate, arrival_distribution, seed)
    prompts: List[Dict[str, Any]] = []
    # shared prefix pool
    shared = SHARED_PREFIX + _make_filler(400, seed=seed)  # ~512 tokens total
    shared_toks = _estimate_tokens(shared)
    for i in range(n):
        # decide prefix reuse
        use_prefix = rnd.random() < prefix_reuse
        # remaining tokens for unique part
        if use_prefix:
            unique_needed = max(16, input_tokens - shared_toks)
            unique = _make_filler(unique_needed, seed=seed+i*1000)
            prompt = shared + "\n\nUser question " + str(i) + ": " + unique[: min(len(unique), unique_needed*4)]
            # ensure token target
            ids = ENC.encode(prompt)
            if len(ids) > input_tokens:
                prompt = ENC.decode(ids[:input_tokens])
            elif len(ids) < input_tokens:
                prompt = prompt + " " + _make_filler(input_tokens - len(ids), seed=seed+i*9999)
        else:
            # no shared prefix, fully synthetic controlled or random chat
            if workload_type == "synthetic":
                prompt = _make_filler(input_tokens, seed=seed+i)
            elif workload_type == "chat":
                base = rnd.choice(CHAT_PROMPTS)
                filler_needed = max(0, input_tokens - _estimate_tokens(base))
                prompt = base + " " + _make_filler(filler_needed, seed=seed+i) if filler_needed>0 else base
            elif workload_type == "long_context":
                prompt = _make_filler(input_tokens, seed=seed+i)
            elif workload_type == "rag":
                ctx_needed = max(32, input_tokens - 80)
                ctx = _make_filler(ctx_needed, seed=seed+i)
                q = rnd.choice(CHAT_PROMPTS)
                prompt = RAG_CONTEXT_TEMPLATE.format(context=ctx, question=q)
                # trim/pad
                ids = ENC.encode(prompt)
                if len(ids) > input_tokens:
                    prompt = ENC.decode(ids[:input_tokens])
            elif workload_type == "agent":
                prompt = AGENT_TOOL_TRACE.format(
                    user_query=rnd.choice(CHAT_PROMPTS),
                    thought="I need to search for relevant information.",
                    action=f"search(query='topic {i}')",
                    observation=_make_filler(min(200, max(16, input_tokens//3)), seed=seed+i)
                )
                ids = ENC.encode(prompt)
                if len(ids) < input_tokens:
                    prompt += " " + _make_filler(input_tokens - len(ids), seed=seed+i*2)
                elif len(ids) > input_tokens:
                    prompt = ENC.decode(ids[:input_tokens])
            else:
                prompt = _make_filler(input_tokens, seed=seed+i)
        # final trim to exact input_tokens (best effort)
        ids = ENC.encode(prompt)
        if len(ids) > input_tokens:
            prompt = ENC.decode(ids[:input_tokens])
        elif len(ids) < input_tokens and input_tokens - len(ids) > 4:
            prompt = prompt + " " + _make_filler(input_tokens - len(ids), seed=seed+i*777)

        prompts.append({
            "prompt": prompt,
            "max_tokens": output_tokens,
            "temperature": 0.0,
            "arrival_offset": offsets[i],
            "workload_type": workload_type,
            "input_tokens_target": input_tokens,
            "input_tokens_actual": _estimate_tokens(prompt),
        })
    return prompts

# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------
def synthetic_controlled(n=20, input_tokens=512, output_tokens=64, arrival_rate=0, arrival_distribution="closed", seed=0) -> List[Dict[str, Any]]:
    return _build_workload(n, input_tokens, output_tokens, "synthetic", arrival_rate, arrival_distribution, 0.0, seed)

def chat_like(n=20, input_tokens=256, output_tokens=128, arrival_rate=0, arrival_distribution="closed", seed=1) -> List[Dict[str, Any]]:
    return _build_workload(n, input_tokens, output_tokens, "chat", arrival_rate, arrival_distribution, 0.0, seed)

def prefix_reuse(n=20, input_tokens=1024, output_tokens=64, reuse_fraction=0.8, arrival_rate=0, arrival_distribution="closed", seed=2) -> List[Dict[str, Any]]:
    return _build_workload(n, input_tokens, output_tokens, "prefix_reuse", arrival_rate, arrival_distribution, reuse_fraction, seed)

def long_context(n=20, input_tokens=4096, output_tokens=64, arrival_rate=0, arrival_distribution="closed", seed=3) -> List[Dict[str, Any]]:
    return _build_workload(n, input_tokens, output_tokens, "long_context", arrival_rate, arrival_distribution, 0.0, seed)

def rag_like(n=20, input_tokens=2048, output_tokens=128, arrival_rate=0, arrival_distribution="closed", seed=4) -> List[Dict[str, Any]]:
    return _build_workload(n, input_tokens, output_tokens, "rag", arrival_rate, arrival_distribution, 0.0, seed)

def agent_like(n=20, input_tokens=1024, output_tokens=128, arrival_rate=0, arrival_distribution="closed", seed=5) -> List[Dict[str, Any]]:
    return _build_workload(n, input_tokens, output_tokens, "agent", arrival_rate, arrival_distribution, 0.0, seed)

WORKLOAD_FACTORIES = {
    "synthetic": synthetic_controlled,
    "chat": chat_like,
    "prefix_reuse": prefix_reuse,
    "long_context": long_context,
    "rag": rag_like,
    "agent": agent_like,
}

def generate(
    workload_type: str = "synthetic",
    n: int = 20,
    input_tokens: int = 512,
    output_tokens: int = 64,
    concurrency: Optional[int] = None,  # kept for config completeness, not used in generation
    arrival_rate: float = 0,
    arrival_distribution: str = "closed",
    prefix_reuse_fraction: float = 0.0,
    seed: int = 0,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Unified entry point — mirrors harness config schema.
    prefix_reuse_fraction only meaningful for prefix_reuse type; for others it is forced to 0
    unless explicitly requested (allows controlled reuse experiments on any type).
    """
    if workload_type not in WORKLOAD_FACTORIES:
        raise ValueError(f"unknown workload_type {workload_type}, choose from {list(WORKLOAD_FACTORIES)}")
    # allow prefix reuse override for any type for controlled experiments
    if workload_type == "prefix_reuse":
        return prefix_reuse(n=n, input_tokens=input_tokens, output_tokens=output_tokens, reuse_fraction=prefix_reuse_fraction or 0.8, arrival_rate=arrival_rate, arrival_distribution=arrival_distribution, seed=seed)
    # generic with optional reuse
    base = WORKLOAD_FACTORIES[workload_type](n=n, input_tokens=input_tokens, output_tokens=output_tokens, arrival_rate=arrival_rate, arrival_distribution=arrival_distribution, seed=seed)
    if prefix_reuse_fraction and prefix_reuse_fraction > 0:
        # retroactively inject shared prefix into fraction of prompts
        rnd = random.Random(seed)
        shared = SHARED_PREFIX + _make_filler(400, seed=seed)
        for item in base:
            if rnd.random() < prefix_reuse_fraction:
                # prepend shared prefix, then trim to target
                combined = shared + "\n\n" + item["prompt"]
                ids = ENC.encode(combined)
                if len(ids) > input_tokens:
                    combined = ENC.decode(ids[:input_tokens])
                item["prompt"] = combined
                item["input_tokens_actual"] = _estimate_tokens(combined)
                item["workload_type"] = item["workload_type"] + "+prefix"
    return base

def describe_workload(wl: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not wl:
        return {}
    toks = [x["input_tokens_actual"] for x in wl]
    return {
        "count": len(wl),
        "input_tokens": {"mean": sum(toks)/len(toks), "min": min(toks), "max": max(toks)},
        "output_tokens_target": wl[0]["max_tokens"],
        "arrival_distribution": "closed" if all(x["arrival_offset"]==0 for x in wl) else "open",
        "workload_types": sorted(set(x["workload_type"] for x in wl)),
    }
