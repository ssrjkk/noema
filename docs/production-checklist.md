# Production Checklist

Run through this list before every production deploy of Noema.

## Pre-deployment

- [ ] `pytest` is green on `main` (CI gate).
- [ ] `ruff check` and `ruff format --check` pass.
- [ ] `mypy noema/` reports 0 issues.
- [ ] `bandit -r noema/`, `safety check`, and `pip-audit` report no HIGH/CRITICAL findings.
- [ ] `pip install -e ".[dev,db,full,sentry]"` resolves cleanly (Dockerfile installs the same extras).
- [ ] `NOEMA_LLM__PROVIDER` is set to a real provider (`ollama`, `openai`, `anthropic`); `fallback` returns stub answers and must never be used outside staging.

## Configuration

- [ ] `NOEMA_API__API_KEY` is set and your clients send it via `X-API-Key`. Without it auth is OFF (see logs).
- [ ] `NOEMA_API__RATE_LIMIT_ENABLED=true` (default) with sane `RATE_LIMIT_RPM`/`RATE_LIMIT_BURST` per tenant.
- [ ] `NOEMA_API__TRUSTED_PROXIES` lists your reverse-proxy CIDRs so rate limiting cannot be spoofed via `X-Forwarded-For`.
- [ ] `NOEMA_API__WEBHOOK_SECRET` is set; verify inbound webhook HMACs before enabling incident automation.
- [ ] `NOEMA_API__REQUEST_TIMEOUT_SECONDS` (default 960 s) is above the engine's `think_timeout_seconds` (900 s).
- [ ] `NOEMA_AUDIT__MERKLE_CHAIN_ENABLED=true` (default) for tamper-evident audit chains.
- [ ] `NOEMA_NS__ENABLED=true` enables neurosymbolic verification; reasoning traces are committed to `NOEMA_NS__TRACE_DIR` (default `.noema/traces`) for later re-audit.
- [ ] Secrets are injected via env/`SecretStr`, never committed to the image or repo.

## Infrastructure

- [ ] PostgreSQL 16 reachable; `make migrate` (Alembic `upgrade head`) applied.
- [ ] Redis 7 reachable; `arq` workers `noema arq worker` running (worker fleet visible in `GET /grid`).
- [ ] Persistence fallbacks are in place (`resilience.graceful_degradation`): Redis down → in-memory cache, PostgreSQL down → file storage. Confirm logs show no silent data loss.
- [ ] Object/volume mounts exist for `/app/data` and `/app/.noema` (trace/memory/checkpoint persistence).

## Observability

- [ ] Metrics endpoint exposed (`/metrics`, or sidecar on `NOEMA_METRICS_PORT`, default 9090) and scraped by Prometheus.
- [ ] Prometheus alert rules loaded from `deploy/monitoring/prometheus-rules.yml`; AlertManager receivers configured.
- [ ] Logs shipped to Loki/ELK; correlation `X-Request-ID` flows from request into structlog context.
- [ ] Sentry DSN set (`NOEMA_OBS__SENTRY_DSN`) if error tracking is desired.
- [ ] Reasoning traces committed to `NOEMA_NS__TRACE_DIR` and periodically archived/reverified (see `tests/eval` for the re-audit harness).

## Load & correctness

- [ ] Load test with the included Locust harness (`tests/locustfile.py`) at expected peak traffic; watch 429/504/5xx rates.
- [ ] `GET /ready` returns `ready: true` and all checks `ok` before routing traffic.
- [ ] Circuit-breaker thresholds (`NOEMA_LLM__CIRCUIT_BREAKER_THRESHOLD`/`RECOVERY`) tuned for your LLM provider's SLO.
- [ ] Graceful shutdown verified (drain workers first, then stop API).

## Safety

- [ ] CORS `allow_origins` restricted (default `*` is dev-only).
- [ ] Reverse proxy enforces TLS; `X-Forwarded-Proto` handled.
- [ ] No secrets in images: build with the repo's `.dockerignore`.
- [ ] `ingest_sanitizer.scan()` applied to untrusted knowledge before ingestion (blocks prompt injection / dangerous imports).