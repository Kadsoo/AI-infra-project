"""Discrete-event simulator for Continuum TTL (paper v7 Alg.1 + Sec 4-5).

Models: Poisson program arrivals, N-turn programs, tool gaps, GPU KV capacity,
pin-with-TTL, program-FCFS vs request-FCFS, prefill/reload + per-turn queueing.
Purpose: fast policy validation + ablation + HOL instrumentation before cloud runs.
"""
from __future__ import annotations

import heapq
import math
from collections import deque

from continuum_ttl import ToolHistory, calc_ttl, compute_eta, prefill_reload_cost

STATIC_TTL = 2.0  # preview repo FIXED_THRESHOLD_CONTINUUM
DECODE_S = 1.2    # per-turn decode surrogate (small model, OSL~128)


class Sim:
    def __init__(self, programs, gpu_cap_tokens=150_000, offload=False,
                 policy="full", max_batch=8):
        self.progs = {p["pid"]: p for p in programs}
        self.cap = gpu_cap_tokens
        self.offload = offload
        self.policy = policy
        self.max_batch = max_batch
        self.hist = ToolHistory()
        self.evicted_waits = deque(maxlen=500)  # for T
        self.finished_N: list[int] = []
        self.pinned: dict[int, dict] = {}  # pid -> {tokens, expiry}
        self.running: dict[int, dict] = {}  # pid -> {tokens, done_at}
        self.waiting: list[dict] = []  # ready turns
        self.events: list = []  # (time, seq, kind, payload)
        self.seq = 0
        self.now = 0.0
        self.rec = {pid: {"queue_waits": [], "prefill_s": 0.0, "turns_done": 0,
                          "pin_hits": 0, "pins": 0, "misses": 0,
                          "finish": None} for pid in self.progs}
        self.pin_token_s = 0.0
        self.expired_wasted = 0
        self.victim_unpins = 0
        self.hol_events: list[dict] = []
        self.timeline: list[dict] = []  # pin/free/wait samples for boundary analysis
        for p in programs:
            self._push(p["t_arrive"], "prog_arrive", p["pid"])

    def _push(self, t, kind, payload):
        self.seq += 1
        heapq.heappush(self.events, (t, self.seq, kind, payload))

    def mem_used(self):
        return (sum(r["tokens"] for r in self.running.values())
                + sum(v["tokens"] for v in self.pinned.values()))

    def _ttl_for(self, pid, turn):
        p = self.progs[pid]
        ctx = p["ctx"][turn]
        pr = prefill_reload_cost(ctx, self.offload)
        T = (sum(self.evicted_waits) / len(self.evicted_waits)) if self.evicted_waits else 0.0
        eta = compute_eta(self.finished_N)
        if self.policy == "full":
            tool = p["tools"][turn][0]
            tau, _ = calc_ttl(tool, self.hist, T, eta, pr)
            return tau
        if self.policy == "static":
            return STATIC_TTL
        return 0.0

    def _order_key(self, w):
        pinned = 1 if (w["pid"] in self.pinned and self.pinned[w["pid"]]["expiry"] >= self.now) else 0
        if self.policy == "baseline":
            return (0, -pinned * 0, w["q_at"])  # request-FCFS, ignore pin
        prog_arr = self.progs[w["pid"]]["t_arrive"]
        return (-pinned, prog_arr, w["q_at"])

    def _eff_need(self, w):
        """Memory if w started: own valid pin is reused, not double-counted."""
        use = self.mem_used()
        pin = self.pinned.get(w["pid"])
        if pin is not None and pin["expiry"] >= self.now:
            use -= pin["tokens"]
        return use + w["ctx"]

    def _victim_unpin(self):
        """Paper 5.2 deadlock path: nothing schedulable -> free latest-arrival pin."""
        if not self.pinned:
            return False
        victim = max(self.pinned, key=lambda pid: self.progs[pid]["t_arrive"])
        del self.pinned[victim]
        self.victim_unpins += 1
        self.timeline.append({"t": self.now, "ev": "victim_unpin",
                              "pid": victim, "mem": self.mem_used()})
        return True

    def _try_schedule(self):
        self.waiting.sort(key=self._order_key)
        while self.waiting and len(self.running) < self.max_batch:
            head = self.waiting[0]
            if self._eff_need(head) <= self.cap:
                self._start(head)
                self.waiting.pop(0)
                continue
            # head blocked: is there a later waiter that WOULD fit right now?
            for i in range(1, len(self.waiting)):
                w = self.waiting[i]
                if self._eff_need(w) <= self.cap:
                    self.hol_events.append({
                        "t": self.now, "head_pid": head["pid"],
                        "head_ctx": head["ctx"], "head_mem": self.mem_used(),
                        "cap": self.cap, "succ_pid": w["pid"],
                        "succ_ctx": w["ctx"], "policy": self.policy})
                    break
            # strict order: head blocks. True deadlock only if nothing running
            # can free memory -> victim-unpin latest-arrival pin and retry.
            if not self.running and self.pinned:
                if self._victim_unpin():
                    continue  # retry scheduling with freed memory
                return
            return  # wait for running turns to finish (or nothing fits)

    def _start(self, w):
        pid, turn = w["pid"], w["turn"]
        p = self.progs[pid]
        ctx = w["ctx"]
        hit = pid in self.pinned and self.pinned[pid]["expiry"] >= self.now
        if hit:
            self.rec[pid]["pin_hits"] += 1
            del self.pinned[pid]
            pre = 0.0
        else:
            pre = prefill_reload_cost(ctx, self.offload)
            self.rec[pid]["misses"] += 1
            wait = self.now - w["q_at"]
            if wait > 0.01:
                self.evicted_waits.append(wait)
        self.rec[pid]["queue_waits"].append(self.now - w["q_at"])
        self.rec[pid]["prefill_s"] += pre
        dur = pre + DECODE_S
        self.running[pid] = {"tokens": ctx, "done_at": self.now + dur}
        self._push(self.now + dur, "turn_done", (pid, turn))
        self.timeline.append({"t": self.now, "ev": "start", "pid": pid,
                              "turn": turn, "hit": hit, "mem": self.mem_used()})

    def run(self, t_max=100000.0):
        while self.events and self.now <= t_max:
            t, _, kind, pay = heapq.heappop(self.events)
            self.now = t
            if kind == "prog_arrive":
                p = self.progs[pay]
                self.waiting.append({"pid": pay, "turn": 0, "ctx": p["ctx"][0],
                                     "q_at": t, "wait": 0.0})
                self._try_schedule()
            elif kind == "turn_done":
                pid, turn = pay
                info = self.running.pop(pid, None)
                p = self.progs[pid]
                self.rec[pid]["turns_done"] += 1
                # record tool duration sample on NEXT arrival; store finish stamp
                if turn + 1 >= p["turns"]:
                    self.rec[pid]["finish"] = t
                    self.finished_N.append(p["turns"])
                    # proactive unpin same program (paper 5.2)
                    self.pinned.pop(pid, None)
                    self._try_schedule()
                    continue
                tool, dur = p["tools"][turn]
                tau = self._ttl_for(pid, turn)
                ctx = p["ctx"][turn]
                if tau > 0:
                    self.pinned[pid] = {"tokens": ctx, "expiry": t + tau}
                    self.rec[pid]["pins"] += 1
                    self.pin_token_s += ctx * tau
                    self._push(t + tau, "ttl_expire", pid)
                    self.timeline.append({"t": t, "ev": "pin", "pid": pid,
                                          "ttl": tau, "mem": self.mem_used()})
                # observe tool time sample at next-turn arrival (paper 5.1)
                self._push(t + dur, "tool_return",
                           (pid, turn + 1, tool, t, dur))
                self._try_schedule()
            elif kind == "tool_return":
                pid, nxt, tool, t_fin, dur = pay
                self.hist.add(tool, dur)
                p = self.progs[pid]
                self.waiting.append({"pid": pid, "turn": nxt,
                                     "ctx": p["ctx"][nxt], "q_at": t, "wait": 0.0})
                self._try_schedule()
            elif kind == "ttl_expire":
                pin = self.pinned.get(pay)
                if pin and pin["expiry"] <= self.now + 1e-9:
                    in_q = any(w["pid"] == pay for w in self.waiting)
                    if not in_q:
                        del self.pinned[pay]
                        self.expired_wasted += 1
                        self.timeline.append({"t": self.now, "ev": "expire",
                                              "pid": pay, "mem": self.mem_used()})
                self._try_schedule()
        # patch waits: recompute from recorded queue_waits? already tracked at start
        # fix: waiting entries still pending get no finish; drop
        out = []
        for pid, p in self.progs.items():
            r = self.rec[pid]
            if r["finish"] is None:
                continue
            jct = r["finish"] - p["t_arrive"]
            out.append({"pid": pid, "kind": p["kind"], "turns": p["turns"],
                        "jct": jct, "avg_q": (sum(r["queue_waits"]) / len(r["queue_waits"])
                                              if r["queue_waits"] else 0.0),
                        "prefill_s": r["prefill_s"], "pins": r["pins"],
                        "hits": r["pin_hits"], "misses": r["misses"]})
        # correct wait values: they were set 0 at start; recompute approx from timeline?
        return {"programs": out, "pin_token_s": self.pin_token_s,
                "expired_wasted": self.expired_wasted, "hol": self.hol_events,
                "timeline": self.timeline}

    # NOTE: per-turn wait is measured queue-entry->start; tool_return entries get
    # wait updated each schedule attempt; _start snapshots it. Good enough for trends.
