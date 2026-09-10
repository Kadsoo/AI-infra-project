# A_01 — Causal hypotheses for the concurrency knee

## Scope and evidence boundary

This experiment concerns one implementation only: the local
hf_transformers_naive FastAPI/Uvicorn serving shim, sshleifer/tiny-gpt2,
FP32 CPU inference, Windows, and streaming generation.  It does not test
vLLM, SGLang, continuous batching, GPU execution, KV-cache management, or a
general LLM-serving claim.

The prior Stage 3R-C observation is used only to select the three fixed load
points: c=1 (below the knee), c=4 (at/just beyond it), and c=8 (above it).
It does not establish any internal cause.  The prior harness had only
client-side TTFT and a client semaphore wait; neither is a server queue
measurement.

## Causal graph under test

Offered client concurrency
  -> HTTP/client connection and handler admission
  -> runtime before executor submission
  -> default asyncio executor waiting
  -> worker-local preparation and model-step scheduling
  -> CPU model forward execution
  -> first SSE content and completion

The experiment changes only offered concurrency.  Model, prompt length,
maximum output length, arrival semantics, warmup, process, source snapshot,
device, precision, and tracing implementation remain fixed within a session.

## Primary hypotheses

### H1 — Runtime / pre-executor queueing

At c=4 or c=8, the pre-executor server interval grows while executor waiting
and model-forward intervals remain comparatively stable.

Operational support requires all of the following:

1. The end-to-end TTFT knee reproduces under the locked protocol.
2. The p95 of the pre-executor component has a c4/c1 ratio of at least 1.5
   and a c8/c4 ratio of at least 1.25.
3. Its mean increase from c1 to c8 accounts for at least 50% of the mean
   increase in server time to first content.
4. The direction holds in at least two of the three formal repetition blocks,
   and its bootstrap 95% interval for the pooled c8-minus-c1 mean increase
   excludes zero.
5. Executor-wait and model-forward components each account for less than 25%
   of the same mean increase.

Falsification: if the executor-wait or model execution component is dominant,
or if no stable pre-executor change meets the above criteria, H1 is rejected.

### H2 — Executor queueing

At c=4 or c=8, the interval from an asyncio.to_thread submission to actual
worker entry grows.  This is measured directly for preparation and every
model step; it is not inferred from a private queue depth.

Operational support requires the same knee, alignment, stability, and
bootstrap requirements as H1, with preparation executor wait plus the
first-step executor wait accounting for at least 50% of the c1-to-c8
server-first-content increase.  The p95 first-step executor wait must satisfy
c4/c1 >= 1.5 and c8/c4 >= 1.25, and the aggregated later-step executor wait
must increase in the same direction in at least two blocks as completion-side
corroboration.

Falsification: stable forward execution growth without executor-wait dominance
rejects H2; so does no stable executor-wait change.

### H3 — Model execution serialization / scheduling boundary

At c=4 or c=8, work after a worker has begun, especially first and later
Torch forward calls, grows materially even after executor wait is separated.
This would indicate execution-stage serialization, contention, or
step-scheduling behavior in this runtime, not merely a queue in front of it.

Operational support requires the same knee, alignment, stability, and
bootstrap requirements as H1, with first worker-to-forward plus first
Torch-forward execution accounting for at least 50% of the c1-to-c8
server-first-content increase.  At least one of first-forward p95 and
later-step forward p95 must meet c4/c1 >= 1.5 and c8/c4 >= 1.25.
Executor wait must account for less than 25% of the increase.

Falsification: executor wait dominates, or forward/worker-local execution has
no stable aligned increase.

### H4 — No distinct internal knee

If the end-to-end knee reproduces but no internal component satisfies the
precommitted dominance, alignment, and stability rules, the result is
ORDINARY QUEUEING.  It is not promoted into a new runtime bottleneck.

### Multi-stage alternative

MULTI-STAGE is used only when two or more disjoint components each account for
at least 25% of the c1-to-c8 server-first-content mean increase, jointly
account for at least 70%, and each passes the alignment, stability, and
bootstrap rules.  A collection of weak changes is not enough.

## Non-hypotheses and hard limitations

- There is no GPU model-execution hypothesis in this session: health must
  report device=cpu and the trace records CPU/Torch forward wall time only.
- The server has no independent scheduler object, batch state, or exposed
  continuous-batching queue.  Those fields remain NOT AVAILABLE.
- A client/HTTP result is not re-labelled as a server-runtime cause.  If the
  pooled-client control materially changes the knee while server components
  do not, the server localization is INCONCLUSIVE.
- This experiment diagnoses a boundary; it does not modify or optimize that
  boundary.
