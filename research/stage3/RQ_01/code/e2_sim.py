"""RQ-1 E2: calibrated event-driven queue simulator (CPU-only).
Spec: research/stage3/RQ_01/experiment_plan.md # E2.
All H2-verdict cells use NVLink-class transfer (mean 8 ms); 25 Gbps transfer never enters H2.
Events are (time, kind, payload...) so heapq orders strictly by time.
"""
import heapq
import numpy as np

MS_PER_TOKEN = 2.0 * 8e9 / (312e12 * 0.5)   # 1.0256e-4 s/token (E1-corrected prefill model)
CHUNK_OVERHEAD = 1.25
HOT_PREFIX = 6144
NV_TRANSFER_MEAN = 8e-3
TRANSFER_SIGMA = 0.5


class Request:
    __slots__ = ("id", "cls", "prompt_len", "hot", "arrival", "pair",
                 "t_queue", "t_prefill", "t_transfer_wait", "t_transfer", "ttft")


class SimPD:
    """P/D system: 2 pairs (1P+1D); prefill FIFO + per-pair transfer engine (decode side)."""

    def __init__(self, rng, lam, cv, policy, n_prefill, service_scale, transfer_scale, trace):
        self.rng = rng
        self.trace = trace
        self.lam = lam
        self.cv = cv
        self.policy = policy          # "affine" (longest-prefix) or "loadbal" (least backlog)
        self.n_prefill = n_prefill
        self.service_scale = service_scale
        self.transfer_scale = transfer_scale
        self.hot_pair = None
        self.events = []
        self.now = 0.0
        self.pair_queue = [[], []]
        self.prefill_free = [n_prefill, n_prefill]
        self.transfer_queue = [[], []]
        self.transfer_free = [1, 1]
        self.pending_tokens = [0.0, 0.0]
        self.busy_tokens = [0.0, 0.0]
        self.arrivals_per_pair = [0, 0]
        self.results = []
        self._seq = 0

    def _push(self, t, kind, *payload):
        self._seq += 1
        heapq.heappush(self.events, (t, kind, self._seq, *payload))

    def uncached_tokens(self, req, pair):
        if req.hot and pair == self.hot_pair:
            return max(0, req.prompt_len - HOT_PREFIX)
        return req.prompt_len

    def t_prefill(self, req, pair):
        return self.uncached_tokens(req, pair) * MS_PER_TOKEN * self.service_scale

    def t_transfer(self):
        mu = np.log(NV_TRANSFER_MEAN * self.transfer_scale) - 0.5 * TRANSFER_SIGMA ** 2
        return float(self.rng.lognormal(mu, TRANSFER_SIGMA))

    def predicted_ttft(self, pair, req):
        backlog = (self.pending_tokens[pair] + self.busy_tokens[pair]) / self.n_prefill
        return backlog * MS_PER_TOKEN + self.t_prefill(req, pair) + NV_TRANSFER_MEAN

    def route(self, req):
        if self.policy == "affine":
            if req.hot and self.hot_pair is not None:
                return self.hot_pair
            return min((0, 1), key=lambda p: self.predicted_ttft(p, req))
        return int(np.argmin([self.pending_tokens[p] + self.busy_tokens[p] for p in (0, 1)]))

    def push_arrivals(self):
        a = 1.0 / (self.cv * self.cv)
        theta = 1.0 / (a * self.lam)
        t = 0.0
        for row in self.trace.itertuples(index=False):
            t += self.rng.gamma(a, theta)
            req = Request()
            req.id = row.request_id
            req.cls = row.cls
            req.prompt_len = row.prompt_len
            req.hot = bool(row.hot)
            req.arrival = t
            self._push(t, "arr", req)

    def start_prefill(self, pair):
        if not self.pair_queue[pair] or self.prefill_free[pair] == 0:
            return
        req = self.pair_queue[pair].pop(0)
        self.pending_tokens[pair] -= self.uncached_tokens(req, pair)
        self.prefill_free[pair] -= 1
        self.busy_tokens[pair] += self.uncached_tokens(req, pair)
        svc = self.t_prefill(req, pair)
        req.t_queue = self.now - req.arrival
        req.t_prefill = svc
        req.pair = pair
        self._push(self.now + svc, "prefill_done", req, pair)

    def start_transfer(self, pair):
        if not self.transfer_queue[pair] or self.transfer_free[pair] == 0:
            return
        req = self.transfer_queue[pair].pop(0)
        self.transfer_free[pair] -= 1
        req.t_transfer_wait = self.now - (req.arrival + req.t_queue + req.t_prefill)
        req.t_transfer = self.t_transfer()
        self._push(self.now + req.t_transfer, "transfer_done", req, pair)

    def run(self):
        self.push_arrivals()
        while self.events:
            ev = heapq.heappop(self.events)
            self.now = ev[0]
            kind = ev[1]
            if kind == "arr":
                req = ev[3]
                pair = self.route(req)
                self.arrivals_per_pair[pair] += 1
                self.pair_queue[pair].append(req)
                self.pending_tokens[pair] += self.uncached_tokens(req, pair)
                self.start_prefill(pair)
            elif kind == "prefill_done":
                _, _, _, req, pair = ev
                self.prefill_free[pair] += 1
                self.busy_tokens[pair] -= self.uncached_tokens(req, pair)
                if req.hot and self.hot_pair is None:
                    self.hot_pair = pair
                self.transfer_queue[pair].append(req)
                self.start_transfer(pair)
                self.start_prefill(pair)
            elif kind == "transfer_done":
                _, _, _, req, pair = ev
                self.transfer_free[pair] += 1
                req.ttft = self.now - req.arrival
                self.results.append(req)
                self.start_transfer(pair)
        return self.results


class SimColocated:
    """Two independent 1-GPU Sarathi-style chunked-prefill queues (tau=1024, +25%)."""

    def __init__(self, rng, lam, cv, service_scale, trace):
        self.rng = rng
        self.trace = trace
        self.lam = lam
        self.cv = cv
        self.service_scale = service_scale
        self.events = []
        self.now = 0.0
        self.free = [1, 1]
        self.queue = [[], []]
        self.results = []
        self._seq = 0

    def _push(self, t, kind, *payload):
        self._seq += 1
        heapq.heappush(self.events, (t, kind, self._seq, *payload))

    def push_arrivals(self):
        a = 1.0 / (self.cv * self.cv)
        theta = 1.0 / (a * self.lam)
        t = 0.0
        for row in self.trace.itertuples(index=False):
            t += self.rng.gamma(a, theta)
            self._push(t, "arr", row)

    def service(self, row):
        uncached = max(0, row.prompt_len - HOT_PREFIX) if bool(row.hot) else row.prompt_len
        return uncached * MS_PER_TOKEN * self.service_scale * CHUNK_OVERHEAD

    def start(self, inst):
        if not self.queue[inst] or self.free[inst] == 0:
            return
        row, arr = self.queue[inst].pop(0)
        self.free[inst] -= 1
        svc = self.service(row)
        self._push(self.now + svc, "done", row, arr, inst)

    def run(self):
        self.push_arrivals()
        while self.events:
            ev = heapq.heappop(self.events)
            self.now = ev[0]
            kind = ev[1]
            if kind == "arr":
                row = ev[3]
                inst = min((0, 1), key=lambda i: len(self.queue[i]) + (1 - self.free[i]))
                self.queue[inst].append((row, self.now))
                self.start(inst)
            else:
                _, _, _, row, arr, inst = ev
                svc = self.service(row)
                self.free[inst] += 1
                r = Request()
                r.id = row.request_id
                r.cls = row.cls
                r.prompt_len = row.prompt_len
                r.hot = bool(row.hot)
                r.arrival = arr
                r.pair = inst
                r.t_queue = (self.now - svc) - arr
                r.t_prefill = svc
                r.t_transfer_wait = 0.0
                r.t_transfer = 0.0
                r.ttft = self.now - arr
                self.results.append(r)
                self.start(inst)
        return self.results


def summarize(results, n_arrivals):
    """Per-run metrics from request list."""
    ttft = np.array([r.ttft for r in results])
    qfrac = np.array([(r.t_queue + r.t_transfer_wait) / max(r.ttft, 1e-12) for r in results])
    q = lambda a: float(np.quantile(ttft, a))
    attainment = float((ttft <= 4.0).mean())
    return {
        "n_completed": len(results),
        "n_arrivals": n_arrivals,
        "ttft_p50": q(0.5), "ttft_p95": q(0.95), "ttft_p99": q(0.99),
        "qfrac_p99": float(np.quantile(qfrac, 0.99)),
        "qfrac_p95": float(np.quantile(qfrac, 0.95)),
        "attainment_p95_slo": attainment,
        "throughput": len(results) / max(float(results[-1].arrival) if results else 1.0, 1e-9),
        "max_pair_share": None,  # filled by caller
    }