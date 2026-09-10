import pathlib, json, statistics
from collections import defaultdict

raw_dir = pathlib.Path('F:/AIinfraResearch/research/measurement/raw')
procs = list(raw_dir.glob('*_processed.json'))
baseline_names = ['baseline_low_concurrency','baseline_medium_concurrency','baseline_high_concurrency','baseline_short_context','baseline_medium_context','baseline_long_context','baseline_prefix_reuse','baseline_rag','baseline_high_memory','baseline_agent']
selected = [p for p in procs if any(name in p.name for name in baseline_names) and 'test_fix' not in p.name and 'overhead' not in p.name and 'smoke' not in p.name and 'real_smoke' not in p.name]
# Only keep the latest 3 reps per config (the real baseline matrix from last run)
# Filter to those with timestamps > 1787899386 (the real matrix)
# Actually all with name baseline_* and processed count, filter to those with success>0 and raw contains tiny-gpt2? Let's just group
groups = defaultdict(list)
for p in selected:
    j=json.loads(p.read_text(encoding='utf-8'))
    try:
        rid = j.get('run_id','')
        ts = int(rid.split('-')[-1]) if '-' in rid else 0
    except:
        ts = 0
    if ts < 1787899386:
        continue
    name=j['config']['name']
    groups[name].append(j)

print('Found groups', list(groups.keys()))
for name, lst in sorted(groups.items()):
    print(name, len(lst), 'reps')
    thr = [x['throughput_rps'] for x in lst if x['throughput_rps'] is not None]
    lat_p50 = [x['latency']['p50'] for x in lst if x['latency']['p50'] is not None]
    lat_mean = [x['latency']['mean'] for x in lst if x['latency']['mean'] is not None]
    ttft_p50 = [x['ttft']['p50'] for x in lst if x['ttft']['p50'] is not None]
    tpot_med = [x['tpot']['median'] for x in lst if x['tpot']['median'] is not None]
    print(f"  thr {statistics.mean(thr):.3f} +/- {statistics.stdev(thr) if len(thr)>1 else 0:.3f} range {min(thr):.3f}-{max(thr):.3f}")
    print(f"  lat p50 {statistics.mean(lat_p50):.3f} mean {statistics.mean(lat_mean):.3f} std {statistics.stdev(lat_p50) if len(lat_p50)>1 else 0:.3f}")
    if ttft_p50:
        print(f"  ttft p50 {statistics.mean(ttft_p50):.4f}")
    if tpot_med:
        print(f"  tpot median {statistics.mean(tpot_med):.4f}")
    succ = [x['success'] for x in lst]
    print(f"  success {succ}")

print("\n=== Sanity: concurrency vs throughput (512/64) ===")
for name in ['baseline_low_concurrency','baseline_medium_concurrency','baseline_high_concurrency']:
    lst=groups[name]
    thr=statistics.mean([x['throughput_rps'] for x in lst])
    thr_tok=statistics.mean([x['token_throughput'] for x in lst])
    lat=statistics.mean([x['latency']['p50'] for x in lst])
    print(f"{name} thr {thr:.3f} tok_thr {thr_tok:.1f} lat p50 {lat:.3f}")

print("\n=== Sanity: context length vs TTFT/latency ===")
for name in ['baseline_short_context','baseline_low_concurrency','baseline_medium_context','baseline_long_context','baseline_rag','baseline_high_memory']:
    lst=groups[name]
    cfg=lst[0]['config']
    inp=cfg['input_tokens']
    ttft=statistics.mean([x['ttft']['p50'] for x in lst if x['ttft']['p50']])
    lat=statistics.mean([x['latency']['p50'] for x in lst])
    print(f"{name} input {inp} ttft p50 {ttft:.4f} lat p50 {lat:.4f}")

print("\n=== Sanity: output length vs decode ===")
for name in ['baseline_short_context','baseline_low_concurrency','baseline_medium_context','baseline_high_memory']:
    lst=groups[name]
    cfg=lst[0]['config']
    out=cfg['output_tokens']
    thr_tok=statistics.mean([x['token_throughput'] for x in lst])
    tpot=statistics.mean([x['tpot']['median'] for x in lst if x['tpot']['median']])
    lat=statistics.mean([x['latency']['p50'] for x in lst])
    print(f"{name} out {out} token_thr {thr_tok:.1f} tpot {tpot:.4f} lat {lat:.3f}")

print("\n=== Sanity: memory vs workload ===")
for name in ['baseline_short_context','baseline_low_concurrency','baseline_medium_context','baseline_long_context','baseline_high_memory']:
    lst=groups[name]
    ram=max([x['system']['ram_max_gb'] for x in lst])
    gpu_vals = []
    for x in lst:
        g = x['system']['gpu_mem_percent']
        if 'mean' in g and g['mean'] is not None:
            gpu_vals.append(g['mean'])
    gpu_mem = statistics.mean(gpu_vals) if gpu_vals else 0
    print(f"{name} ram_max {ram:.2f} gpu_mem% {gpu_mem:.2f}")
