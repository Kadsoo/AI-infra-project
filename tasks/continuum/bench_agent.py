"""Agent-workload bench: shared system prefix + multi-turn tool gaps.
Timing via streaming SSE; hit rate via /metrics delta. Usage:
  python bench_agent.py --mode single|multi --n 60 --conc 4 --port 8000 --out /tmp/b.json
"""
import argparse, json, statistics, threading, time, urllib.request

SYS = ("You are a function-calling agent. Tools: get_weather(city), search_issues(repo,query), "
       "run_tests(path), fetch_url(url), calc(expr). Always respond with one tool call per turn "
       "in JSON like {\"tool\":...,\"args\":...}. Be concise. Trace format: thought then action. "
       "Rules: verify args, handle errors, stop when done. ") * 6  # ~350 tok shared prefix

TASKS = [f"Task {i}: investigate issue #{1000+i} in repo proj-{i%10}, run relevant tests, summarize."
         for i in range(200)]


MODEL = "/root/models/Qwen2.5-3B-Instruct"


def post_stream(port, messages, max_tokens):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "stream": True}).encode()
    t0 = time.time()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    first = None
    n_tok = 0
    t_last = t0
    itls = []
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r:
                if not line.startswith(b"data: "):
                    continue
                if line.strip() == b"data: [DONE]":
                    break
                if first is None:
                    first = time.time()
                n_tok += 1
                now = time.time()
                itls.append(now - t_last)
                t_last = now
    except Exception as e:
        return {"err": str(e)[:100]}
    return {"ttft": (first - t0) if first else -1, "e2e": t_last - t0,
            "n": n_tok, "itl": statistics.mean(itls[1:]) if len(itls) > 1 else 0}


def metrics(port):
    out = {}
    try:
        txt = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=30).read().decode()
        for line in txt.splitlines():
            if line.startswith("#") or "_total" not in line or "{" not in line:
                continue
            if "prefix_cache_hits_total" in line or "prefix_cache_queries_total" in line:
                head = line.split("{")[0]
                if head.startswith("vllm:gpu_"):
                    continue  # deprecated twin of vllm:prefix_cache_*
                name = "hits" if "hits_total" in head else "queries"
                try:
                    out[name] = out.get(name, 0) + float(line.split()[-1])
                except ValueError:
                    pass
    except Exception as e:
        out["err"] = str(e)[:100]
    return out


def worker(port, jobs, results, mode, tool_gap, doc):
    for j in jobs:
        udoc = (f"\n[unique ref {j}: {doc}]" if doc else "")
        user0 = TASKS[j % len(TASKS)] + udoc
        if mode == "single":
            r = post_stream(port, [{"role": "system", "content": SYS},
                                   {"role": "user", "content": user0}], 64)
            r["prog"] = j
            results.append(r)
        else:
            hist = [{"role": "system", "content": SYS},
                    {"role": "user", "content": user0}]
            t_prog0 = time.time()
            for turn in range(4):
                r = post_stream(port, hist, 48)
                r["prog"] = j
                r["turn"] = turn
                results.append(r)
                hist.append({"role": "assistant", "content": '{"tool":"fetch_url","args":{}}'})
                if turn < 3:
                    time.sleep(tool_gap)  # simulated tool execution
                    hist.append({"role": "user", "content": f"[tool result {turn}: ok, 42 rows]"})
            results.append({"prog": j, "makespan": time.time() - t_prog0, "turn": -1})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="single", choices=["single", "multi"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tool-gap", type=float, default=0.5)
    ap.add_argument("--doc-tokens", type=int, default=0,
                    help="append a unique ~N-token doc per request (pressure)")
    ap.add_argument("--out", default="/tmp/b.json")
    a = ap.parse_args()
    doc = ""
    if a.doc_tokens > 0:
        # ~4 chars/token surrogate; unique per request id via suffix
        unit = "Reference document passage about distributed systems caching and scheduling. "
        doc = (unit * (a.doc_tokens // 12 + 1))[:a.doc_tokens * 4]
    m0 = metrics(a.port)
    t0 = time.time()
    results = []
    jobs = list(range(a.n))
    chunks = [jobs[i::a.conc] for i in range(a.conc)]
    ths = [threading.Thread(target=worker, args=(a.port, ch, results, a.mode, a.tool_gap, doc)) for ch in chunks]
    [t.start() for t in ths]
    [t.join() for t in ths]
    wall = time.time() - t0
    m1 = metrics(a.port)
    reqs = [r for r in results if r.get("ttft", -1) >= 0]
    tt = sorted(r["ttft"] for r in reqs)
    it = sorted(r["itl"] for r in reqs if r.get("itl", 0) > 0)
    mk = [r.get("makespan") for r in results if "makespan" in r]
    def pct(s, p): return s[min(len(s)-1, int(p*len(s)))] if s else -1
    hits = m1.get("hits", 0) - m0.get("hits", 0)
    ques = m1.get("queries", 0) - m0.get("queries", 0)
    by_turn = {}
    for t in sorted(set(r.get("turn", 0) for r in reqs if "turn" in r)):
        s = sorted(r["ttft"] for r in reqs if r.get("turn") == t)
        by_turn[f"turn{t}"] = round(pct(s, .5), 4) if s else -1
    out = {"mode": a.mode, "n": a.n, "conc": a.conc, "wall": wall,
           "ok": len(reqs), "err": len(results) - len(reqs) - len(mk),
           "ttft_p50": pct(tt, .5), "ttft_p95": pct(tt, .95),
           "itl_p50": pct(it, .5),
           "out_tps": sum(r["n"] for r in reqs)/wall if wall else 0,
           "hit_rate": hits/ques if ques else 0,
           "makespan_p50": pct(sorted(mk), .5) if mk else None,
           "ttft_by_turn": by_turn if a.mode == "multi" else None}
    print(json.dumps(out, indent=1))
    open(a.out, "w").write(json.dumps({"summary": out, "m0": m0, "m1": m1}))


if __name__ == "__main__":
    main()
