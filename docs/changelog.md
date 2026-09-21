# Changelog

## v1.3.0 (2026-09-19)

### Features

- **Fallback provider rewritten as a template engine** — `noema/llm/providers.py`
  `FallbackProvider` now classifies the task domain (api, docker, terraform,
  database, auth, pipeline, frontend, security, test) and returns a structured
  JSON solution (architecture, stack, code files with real FastAPI/RBAC/SQL/Docker/
  Terraform/Airflow/React templates) instead of a stub string. The demo pipeline
  (`python demo.py`) now produces well-shaped solutions with zero API keys.
  Domain classification is token-aware: `"ui"` no longer matches inside `"build"`,
  `"api"` inside `"capital"`, and `test_` matches identifiers (`test_api`).
- **Ollama health check with self-healing** — a failed 1s health-check is cached
  for `NOEMA_LLM_CIRCUIT_BREAKER_RECOVERY` seconds and then re-attempted, so a
  long-lived worker recovers when Ollama comes back instead of staying poisoned
  by its first failure. Failures are wrapped in `LLMProviderError` (fail-closed,
  retry policy treats them as non-retryable).
- **`python -m noema` entry point** — `noema/__main__.py` makes the CLI runnable
  as a module, which is the reliable option on Windows where the console script
  sometimes misses `PATH`.
- **Real OTLP/HTTP trace exporter** — `noema/observability/otlp.py` pushes
  traces to any OpenTelemetry Collector (`POST {endpoint}/v1/traces`,
  JSON/protobuf mapping) from a daemon worker thread with its own asyncio loop,
  so it works from sync workers, the CLI and the API server alike. Enable with
  `NOEMA_OBS_TRACING_ENABLED=true`; `TraceSpan` now records epoch-nanosecond
  start/end for correct `startTimeUnixNano`/`endTimeUnixNano`. Transport failure
  logs at debug level and never breaks instrumentation.
- **Webhook management hardening** — new settings `NOEMA_API__WEBHOOK_ADMIN_TOKEN`
  (locks `/webhooks/register|list|unregister` when the master API key is empty)
  and `NOEMA_API__WEBHOOK_ALLOW_UNSIGNED_INCIDENTS` (default `false`, fail-closed).

### Fixes

- **Fallback becomes the default provider** — `NOEMA_LLM_PROVIDER` defaults to
  `fallback` so a fresh install works out of the box; set it to `ollama`/`openai`/
  `anthropic` for real reasoning.

### Quality

- New coverage: `tests/test_main_module.py` (module entry point, UTF-8-safe on
  Windows cp1251 consoles), full `FallbackProvider` suite (classification matrix,
  template contract, determinism, round-trip through the pipeline parser),
  Ollama health-check self-healing (`tests/test_refactor_llm_providers.py`), and
  the OTLP exporter (`tests/test_otlp.py`: payload shape, id/time mapping,
  batch delivery, transport failure survival, tracer→exporter push).
- `noema/py.typed` added so type consumers see annotations.
- Repo hygiene: `.trae/`, `.zed/`, `docs/.opencode/` are gitignored.
- 0 mypy errors, 0 ruff issues; full suite: 1242 passed, 1 skipped.

## v1.2.1 (2026-09-12)

### Features

- **Request timeout guard** — `RequestTimeoutMiddleware` in `noema/api/middleware.py` fails closed with `504 {error: request_timeout}` when a handler exceeds `NOEMA_API_REQUEST_TIMEOUT_SECONDS` (default `960.0`; `0` disables it). Paths in `NOEMA_API_REQUEST_TIMEOUT_EXEMPT` bypass the guard; the response still carries the `X-Request-ID` header.
- **Trace persistence** — reasoning traces are now persisted to `NOEMA_NS_TRACE_DIR` (default `.noema/traces` under the project root, `""` disables). OpenTelemetry/OTLP export stays a reserved hook (`NOEMA_OBS_TRACING_ENDPOINT`, default `http://localhost:4318`).
- **`arq` declared as a dependency** (`arq>=0.28.0`) — the async job worker is now formally part of the project; `noema/workers/arq_worker.py` updated for the arq 0.28 `Worker`/`Monitor` signatures (`max_tries` instead of the removed `max_retries`/`retry_delay`).
- **Docker hygiene** — new `.dockerignore` keeps builds lean.

### Fixes

- **Dependency conflict** — `redis` pin relaxed to `>=5.0.3,<6.0` so the declared `arq` requirement (`redis<6`) resolves in a clean install (pip previously failed with `ResolutionImpossible` in CI).
- **Grid API test flakiness** — `tests/test_grid_api.py` no longer crosses asyncio event-loop boundaries with `fakeredis`; the fixture seeds on the same loop the ASGI app runs on.

### Docs

- New: `docs/api-examples.md`, `docs/monitoring-setup.md`, `docs/performance-tuning.md`, `docs/production-checklist.md`, `docs/troubleshooting.md`.
- Updated: `README.md`, `docs/configuration.md`, `docs/index.md`, `docs/getting-started.md`, `docs/contributing.md`, `mkdocs.yml` (new pages + repo URL).

### Quality

- 0 mypy errors, 0 ruff issues; full suite: 1201 passed, 1 skipped.

## v1.2.0 (2026-08-31)

### Features

- **Global Noema Grid (Phase 3 complete)** — the roadmap's final phase is done:
  - **T3.1 Multi-node worker pool** — worker heartbeats now advertise `metrics_port`; `noema arq workers` lists the live fleet from Redis; stress-tested 100 tasks over 3 nodes with zero double-execution (`tests/test_grid_workers.py`).
  - **T3.2 Federation protocol** — new `noema/federation/` package: `FederationRouter` splits sub-tasks, delegates them round-robin to the next healthy peer over gRPC Think with per-peer circuit breaker + exponential-backoff retries (open circuits fail fast), falls back to the local executor when no peer is healthy, and re-joins results positionally. Config: `NOEMA_FEDERATION__*`; CLI: `noema grid federate` (`tests/test_federation.py`).
  - **T3.3 Token/ledger economy** — `noema/billing/ledger.py` `ContributionLedger`: bounded append-only JSONL entries (node, task, kind, model, tokens, cost, peer, artifact) with `per_node()`/`entries_for()`/`audit()`; the arq worker records every completed think job (`NOEMA_WORKER_LEDGER_PATH`), federated delegations are recorded by the router, `noema arq ledger` audits a node's file (`tests/test_ledger.py`).
  - **T3.4 Grid dashboard** — `noema/observability/grid.py` `GridDashboard` joins Redis heartbeats with each node's Prometheus `/metrics` into a per-node latency/token/error view + cluster totals (graceful degradation without `prometheus_client`; unreachable nodes are reported, never raised). Exposed as `GET /grid` and `noema grid status` (`tests/test_grid_dashboard.py`, `tests/test_grid_api.py`).
- **Per-file cost attribution (T2.5)** — `noema/experiments/runner.py` `attribute_file_costs` distributes each run's measured tokens across generated files weighted by line count; per-file rows sum back exactly to run totals. `results.json` now includes `file_costs`; `runs.csv` stays flat (`tests/test_file_costs.py`).

### Quality

- 0 mypy errors, 0 ruff issues; full suite: 1197 passed, 1 skipped.
- Roadmap Phases 1–3 fully closed (`docs/ROADMAP.md`).

## v1.1.0 (2026-08-21)

### Features

- **Ontological Reinforcement Learning (ORL)** with epistemic validator (`noema/ontology/`); ontological axioms are injected into LLM context for deterministic architecture generation.
- **Pluggable sandbox environments**, ontology graph, chaos test harness.
- **Foundation hardening**: the merge gate never raises; verifier failures are fail-closed; LLM errors, judge gate, tenant hygiene, ORL mutation guards; tenant precedence in `think()`; bound static gate input; kernel error isolation with fail-degraded peripheral stores.

### Performance

- Incremental HNSW adds, shared embedder, vectorized cache scan, lean checkpoint IO.

### Fixes

- Billing: default Anthropic model bumped to `claude-sonnet-4-5` after the 2026-06 EOL; zero suite warnings.
- Sandbox: Docker lint/test repairs, rlimit crash fix, walrus static gate.

## v1.0.1 (2026-08-07)

### Fixes

- **Encoding corruption** — restored Cyrillic text in 21 source files (`kernels/*`, `agents/*`, `knowledge/*`, `core/engine.py`, `pipelines/engine.py`, `scaffolder/*`, `plugins/*`, `feedback/*`, `workers/*`, `noema_knowledge.json`) that had been corrupted by a cp1251 round-trip during the rename, producing mojibake in CLI output, LLM prompts, and generated code.
- **CLI rendering** — fixed `UnicodeEncodeError` on Windows cp1251 consoles for status glyphs (`✓ ✗ ⚠ ●`); glyphs now fall back to ASCII when the output stream cannot encode them, while output is emitted deterministically as UTF-8.
- **`noema health check`** — no longer crashes on encoding; returns correct exit code (1 on issues) and supports `--output json`.
- **`noema discover`** — fixed `asyncio.run()` from a running event loop.

### Quality

- Added `scripts/check_encoding.py` encoding guard (Makefile `encoding-check`, wired into `make ci` and CI).

### Dependencies

- Raised minimum versions: pydantic ≥2.13.4, pydantic-settings ≥2.14.2, aiohttp ≥3.14.3, asyncio-mqtt ≥0.16.2, uvicorn ≥0.52.1, asyncpg ≥0.31.0, mypy ≥2.3.0, rich ≥15.0.0, chromadb ≥1.5.9, hvac ≥2.4.0.
- Bumped GitHub Actions: upload-artifact v4→v7, codecov-action v4→v7, docker/login-action v3→v4, docker/setup-buildx-action v3→v4, azure/setup-helm v3→v5.

### CI

- Fixed `Tests` job failing at startup (pytest exit code 4/2): added `pytest-cov`, `pytest-timeout`, `pytest-benchmark`, `hypothesis`, `httpx` to the `dev` extras.
- `test_default_values` no longer depends on ambient `NOEMA_LLM_PROVIDER` (uses `monkeypatch.delenv`).
- Removed redundant `renovate.json`; Dependabot is the single update bot.

## v1.0.0 (2024-07-30)

Initial release of Noema.

### Features

- **NoemaEngine** — LLM-first orchestration with DAG-based Chain-of-Thought
- **Dynamic Step Planner** — Automatically selects reasoning steps based on task tags and complexity
- **Parallel Execution** — Independent CoT steps execute concurrently
- **Reflexion** — Up to 3 retry attempts with LLM-as-a-Judge feedback
- **Checkpointing** — DAG progress saved for resumable task execution
- **NeuroSymbolic Engine** — Z3 formal verification + LLM hypothesis generation with refinement loop
- **22 Domain Modules** — monitoring, testing, docs, database, queues, caching, auth, graphql, websocket, mobile, i18n, CLI, security, performance, config, events, quality, containers, terraform, data pipeline, ML ops, gateway
- **9 Reasoning Kernels** — analysis, architecture, codegen, optimization, security, frontend, devops, data, ai_ml
- **Sub-Agent System** — specialized agents (Architect, Developer, Security, DevOps, DBA, AI Engineer)
- **Three-Tier Memory** — episodic, semantic, procedural with HNSW vector search
- **Knowledge Store** — persistent knowledge base with embedding-based search
- **Knowledge Graph** — relationship-aware architecture suggestions
- **Resilience** — Circuit breaker, retry with exponential backoff + jitter, graceful degradation (Redis→memory, PostgreSQL→file), task cancellation
- **Multi-Tenant** — contextvar isolation, per-tenant quotas, feature flags, audit logging
- **Billing** — Cost tracking per tenant/task/step, quota enforcement (monthly budget, hourly rate, concurrency)
- **API Server** — FastAPI with SSE streaming, rate limiting, API key auth, CORS, request ID tracking
- **Observability** — Prometheus metrics (HTTP, LLM, workers, memory, modules), Sentry error tracking with structlog, OpenTelemetry tracing
- **Security** — PII redaction, RAG injection sanitizer, sandboxed code execution (Docker)
- **Self-Evolution** — Automated prompt optimization via OPRO pattern, trace analysis, and judge feedback
- **Self-Healing** — Executor with multi-strategy recovery (retry, fallback, skip, escalate)
- **Plugin System** — Pluggable kernels and agents via external packages
- **CLI** — Typer-based command-line interface
- **SSE Streaming** — Real-time step progress via Server-Sent Events
- **Webhook Support** — Event-driven integration with external services
- **Project Scaffolding** — Auto-generate project files from solutions
- **Container Ready** — Multi-stage Dockerfile, Docker Compose with PostgreSQL + Redis
- **Kubernetes** — Kustomize manifests
- **Helm Chart** — Production-grade Helm deployment
- **Terraform** — AWS infrastructure (EKS, RDS, ElastiCache)
- **Monitoring Stack** — Prometheus + Grafana + Loki + Promtail
- **Postman Collection** — API exploration and testing
