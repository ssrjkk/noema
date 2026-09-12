# Performance Tuning

## Where the time goes

A single `/think` spends time in order of typical cost:

1. **LLM calls** — dominated by the Reflexion loop (up to `3` attempts) and any
   module/kernel calls. This is **the** lever for latency and token spend.
2. **Verification** — neurosymbolic verify (Z3 solver, `verification_timeout` 5 s) and
   sandbox execution.
3. **In-process work** — task graph planning, knowledge retrieval, ontology checks.

## LLM layer (biggest wins)

- `NOEMA_LLM__REQUEST_TIMEOUT` — cap per-call latency (default 120 s).
- `NOEMA_LLM__TEMPERATURE` (0.4 default) and `NOEMA_LLM__MAX_TOKENS` — lower for
  cheap/quick refinements.
- Circuit breaker (`NOEMA_LLM__CIRCUIT_BREAKER_THRESHOLD`=5, `RECOVERY`=30 s) prevents
  hammering a degraded provider.
- `RetryPolicy`: `retry_max`=3, `retry_base_delay`=1 s, `retry_max_delay`=30 s
  (exponential + jitter).
- Set `NOEMA_LLM__PROVIDER` to your real backend; never use `fallback` in prod
  (it returns stub answers with no LLM latency, which makes load tests lie).

## Concurrency & queues

- **Worker pool** (`NOEMA_WORKER__POOL_SIZE`, default 10) is the number of concurrently
  executing tasks. Each runs async, so raise it with available IO concurrency.
- **Max queue**: `NOEMA_WORKER__MAX_QUEUE` (default 100) bounds backlog. Unbounded growth
  is prevented by rejection instead of memory pressure.
- **Worker hierarchy** (`HIERARCHY_MAX_CONCURRENT`, default 50): too low → parent waits
  for children; too high → thundering herd at the LLM. Align with provider RPM.
- **`think_timeout_seconds`** (900 s) bounds a whole run; `api.request_timeout_seconds`
  (960 s) is the HTTP backstop above it.
- P99 latency is sensitive to rate limiting: with `RATE_LIMIT_RPM`=60 the burst window
  shapes request arrival. For bulk workloads use `POST /tasks/enqueue` + `arq` workers
  instead of synchronous `/think`.

## Memory & caching

- Multi-layer cache (`SEMANTIC_CACHE`) keeps hot tasks out of the LLM. `GET/POST
  /knowledge` are cheap; `CacheControlMiddleware` emits `ETag`/`Cache-Control`.
- Persisted stores (memory, checkpoints, reasoning traces) write under `.noema/`;
  put that on fast storage, and keep `.noema/traces` pruning/archiving scheduled.

## Observability while tuning

- `GET /metrics` (or sidecar on `NOEMA_METRICS_PORT`, default 9090) exposes
  `noema_llm_latency_seconds`, `noema_llm_tokens_used_total`, `noema_llm_circuit_state`,
  `noema_http_request_duration_seconds`, `noema_worker_*`, `noema_memory_*`.
- `GET /workers/stats` and `GET /health/infra` give live queue depth.
- The Locust harness (`tests/locustfile.py`) drives `/think` load with the tags
  `health think admin knowledge ops stream`. Use it to find your P99 ceiling, then
  tune `POOL_SIZE`/`MAX_QUEUE`/`RATE_LIMIT_RPM`/`RANGE` against measured latency.

## Bottleneck decision rules

| Symptom | First thing to change |
|---|---|
| P50 high, LLM busy | lower `MAX_TOKENS`, raise `circuit_breaker_threshold`, add provider capacity |
| Queue builds, workers idle-ish | raise `POOL_SIZE` |
| 429s under forecasted load | raise `RATE_LIMIT_RPM`/`BURST`, offload to `arq` |
| Verification dominates | reduce `max_refinement_attempts`, raise `verification_timeout` only if false-negatives |
| Disk IO spikes | move `.noema/` and audit writes to NVMe; batch trace commits |
| Memory high | lower `HIERARCHY_MAX_CONCURRENT`, ensure cache eviction (see chaos tests) |

## Scaling out

- Stateless API: run `NOEMA_API__WORKERS`>1 uvicorn processes behind a load balancer.
- Distributed queue: more `noema arq worker` processes; the fleet is visible in
  `GET /grid` (heartbeats + draining).
- PostgreSQL/Redis scale independently; watch `circuit_breaker` recovery windows so
  failover storms don't reopen circuits.