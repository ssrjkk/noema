# Troubleshooting

Frequent operational issues and how to check them. Every failure is designed to be
observable via structlog (correlation `X-Request-ID`), Prometheus metrics, and the
RFC 7807 problem responses (`error` field).

## The API returns 503 "Service starting"

The engine is initializing (or never got initialized). Wait for lifespane startup to
finish; in deployment, only route traffic once `GET /ready` returns `ready: true`.

## Requests get 429 Too Many Requests

The per-key sliding-window rate limiter fired. Settings: `NOEMA_API__RATE_LIMIT_RPM`
(default 60/min) and `NOEMA_API__RATE_LIMIT_BURST` (default 10). Headers
`X-RateLimit-Remaining` tell you how close you are. If you run behind a proxy, set
`NOEMA_API__TRUSTED_PROXIES`; otherwise forwarded headers are ignored and every client
may be attributed to the proxy IP.

## Requests get 504 request_timeout

The API-level safety net in `RequestTimeoutMiddleware` cancelled a handler that
exceeded `NOEMA_API__REQUEST_TIMEOUT_SECONDS` (default 960 s). This is a *backstop*
— check first whether the engine's own timeouts are too tight:
`NOEMA_WORKER__TASK_TIMEOUT` (300 s), `think_timeout_seconds` (900 s),
`NOEMA_LLM__REQUEST_TIMEOUT` (120 s). Exempt paths from the middleware with
`NOEMA_API__REQUEST_TIMEOUT_EXEMPT`, or raise the value.

## LLM errors are not surfacing as 5xx

LLM providers are fail-closed: `create_llm_provider()` returns a `FallbackProvider`
only when the provider is unknown/unset. If `NOEMA_LLM__PROVIDER=fallback` you are
running the stub and answers are non-functional — this is intentional for demo mode
and set by default in the Dockerfile.

## Redis is down

Graceful degradation should engage automatically:
- Cache falls back to in-memory (`SEMANTIC_CACHE` → memory dict).
- `arq` workers stop consuming jobs — enqueue calls will fail; check worker logs and
  `GET /grid`.
Verify with the chaos suite: `pytest tests/chaos/test_redis_failover.py`.

## PostgreSQL is down

- Memory and audit fall back to file-based storage (`resilience.graceful_degradation`,
  `AuditLogger(pg_pool=None)`).
- Quotas (`QuotaManager`) degrade similarly. Confirm no `page_not_found`-style 500s
  and inspect `/admin/diagnostics`.
Verify with `pytest tests/chaos/test_db_failover.py`.

## A task runs forever / worker pool is saturated

- `GET /workers/stats` shows pool usage; `GET /health/infra` reports queue depth.
- `GET /tasks/active` lists in-flight tasks; `DELETE /think/{task_id}` cancels one.
- Bounded queue is `NOEMA_WORKER__MAX_QUEUE` (default 100); exceeding it rejects
  work instead of unbounded growth.

## Audit verification fails

Run `noema audit import --import <export_file>` (verifies chain integrity on import)
or `POST /admin/audit/verify`. A mismatch means the artifact or chain was tampered
with or the LLM's result drifted from the recorded proof. Reasoning traces
(`.noema/traces`) can be replayed deterministically without an LLM — use
`reverify_trace_file()`; a replay mismatch is a signal of data corruption, not model
regress. For a single-task proof, use
`noema audit verify --proof-file FILE` or `GET /admin/audit/proof/<task_id>`.

## gRPC / streaming hangs

Streaming (`/think/stream`, gRPC unary/stream `think`) does not enforce the HTTP
handler timeout on the body; a silent client that stops reading will not be killed by
the server. Use client-side timeouts and cancel tasks server-side via
`DELETE /think/{task_id}` / `Cancel`.

## `noema arq worker` crashes

Since arq is a declared dependency (`pyproject.toml`), a fresh `pip install -e .`
provides it. If it still fails, confirm Redis is reachable at
`redis://localhost:6379/0` (override with `noema arq worker --redis <url>`).

## Trace files grow unboundedly

Reasoning traces accumulate in `NOEMA_NS__TRACE_DIR`. Archiving/reverification is an
ops task: use the eval harness (`tests/eval/run_eval.py`) as a reference, and back up
or prune according to your audit-retention policy.

## Checklist when nothing obvious works

1. `noema health` — is the API up?
2. `GET /ready` — what check failed (db / llm / workers)?
3. `GET /metrics` or the metrics sidecar — error counters, request latency histograms.
4. Read the structlog entry with your `X-Request-ID`.
5. Run the relevant slice of `tests/chaos/` and `tests/adversarial/` to reproduce.