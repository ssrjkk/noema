# Monitoring Setup

Noema ships observability primitives in `noema/observability/` and ready-made configs
under `deploy/monitoring/`.

## Metrics

Metrics are exposed over Prometheus text format at:

- `GET /metrics` on the API, **and**
- a dedicated sidecar (`build_metrics_app`/`spawn_metrics_server`) on
  `NOEMA_OBS__METRICS_PORT` (default **9090**) if the metric server is spawned.

If `prometheus_client` is absent, all metrics degrade to no-ops (`_NoopMetric`), so a
minimal install never crashes on import.

Useful series:

```
noema_http_requests_total{method,endpoint,status}
noema_http_request_duration_seconds
noema_llm_requests_total / noema_llm_latency_seconds / noema_llm_tokens_used_total
noema_llm_circuit_state
noema_worker_tasks_total{status} / noema_worker_queue_depth
noema_memory_*            (episodic/semantic store size, hit rate)
noema_evolution_patches_total
noema_module_*            (per-module invocations, failures)
noema_pr_cost_usd_total   (cost attribution)
```

## Prometheus

Reference config: `deploy/monitoring/prometheus-config.yml`, with alert rules in
`deploy/monitoring/prometheus-rules.yml`. Run the whole stack with:

```bash
docker compose -f deploy/monitoring/docker-compose-monitoring.yml up
```

It starts Prometheus, Grafana (dashboard `grafana-dashboard.json`), Loki and
Promtail.

## AlertManager

AlertManager is expected on the standard compose network. `prometheus-rules.yml`
covers the critical scenarios:

- **LLM circuit open** (`noema_llm_circuit_state == 1`) — provider degraded, traffic
  falling back.
- **High error rate** on `noema_http_requests_total{status=~"5.."}`.
- **Queue depth sustained high** — workers cannot keep up.
- **Worker node disappearing** — heartbeat TTL expiry (see `GET /grid`).

Wire receivers in `alertmanager-config.yml` before going live.

## Sentry

Optional error tracking — set `NOEMA_OBS__SENTRY_DSN` (plus
`NOEMA_OBS__SENTRY_ENVIRONMENT`, `NOEMA_OBS__SENTRY_TRACES_SAMPLE_RATE`,
`NOEMA_OBS__SENTRY_PROFILES_SAMPLE_RATE`).
Initialized in the API lifespan with structlog, asyncio and logging integrations;
`send_default_pii=False`.

## Logging & correlation

- Structured JSON logs via structlog; `RequestIDMiddleware` stamps every response with
  `X-Request-ID` (your value or a generated one) and it flows into the log context as
  `correlation_id`.
- Ship logs with Promtail (reference `deploy/monitoring/promtail-config.yml`) or your
  preferred shipper.

## Reasoning traces (verifiable auditing)

- When neurosymbolic is enabled (`NOEMA_NS__ENABLED=true`), each `think` commits a
  **self-contained, replayable reasoning trace** to `NOEMA_NS__TRACE_DIR`
  (default `.noema/traces`): task input, per-round hypothesis, AST + Z3 verdicts,
  terminal outcome.
- Re-audit without any LLM: load the artifact and call `reverify_trace_file()` — it
  re-runs only the deterministic AST + symbolic checks and returns a
  `ReplayVerdict {matches, static_matches, symbolic_matches}`.
- The in-process `Tracer` keeps live OpenTelemetry-style spans in memory
  (`get_trace()`/`get_stats()`); `NOEMA_OBS__TRACING_ENDPOINT`
  (default `http://localhost:4318`) and `TraceConfig.export_endpoint` are the reserved
  OTLP export hook and are currently unused — spans are bounded in memory, traces on
  disk.

## Health endpoints for uptime

- `GET /health` — liveness (DB connectivity, provider, worker counts).
- `GET /ready` — readiness gate (`db`, `llm`, `workers` sub-checks; `ready:false`
  until everything is up).
- `GET /health/infra` — Redis/PostgreSQL/queue/worker-pool depth for dashboards.
- Kubernetes: the Helm chart configures `/health` as the probe and the runtime image
  ships a Docker `HEALTHCHECK`.

## Dashboards

- Import `deploy/monitoring/grafana-dashboard.json` into Grafana (datasource =
  Prometheus).
- `NOEMA_METRICS_PORT` should match the scrape port in `prometheus-config.yml`.