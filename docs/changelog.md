# Changelog

## v1.0.1 (2026-08-07)

### Fixes

- **Encoding corruption** — restored Cyrillic text in 21 source files (`kernels/*`, `agents/*`, `knowledge/*`, `core/engine.py`, `pipelines/engine.py`, `scaffolder/*`, `plugins/*`, `feedback/*`, `workers/*`, `noema_knowledge.json`) that had been corrupted by a cp1251 round-trip during the rename, producing mojibake in CLI output, LLM prompts, and generated code.
- **CLI rendering** — fixed `UnicodeEncodeError` on Windows cp1251 consoles for status glyphs (`✓ ✗ ⚠ ●`); glyphs now fall back to ASCII when the output stream cannot encode them, while output is emitted deterministically as UTF-8.
- **`noema health check`** — no longer crashes on encoding; returns correct exit code (1 on issues) and supports `--output json`.
- **`noema discover`** — fixed `asyncio.run()` from a running event loop.

### Quality

- Added `scripts/check_encoding.py` encoding guard (Makefile `encoding-check`, wired into `make ci` and CI).

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
