"""Domain-seeded knowledge for the 22 built-in Noema modules.

Each domain module gets a dedicated :class:`KnowledgeEntry` whose content uses
the module's own vocabulary (name, keywords from its ``DESCRIPTION``, concrete
patterns) so TF-IDF retrieval returns a relevant hit for a plain-language
query in that domain — no per-query tuning required.
"""

from __future__ import annotations

from noema.core.types import KnowledgeEntry

#: Domain name → (title, content, tags). ``tags`` always starts with the
#: module name so ``noema knowledge search <module>`` matches immediately.
DOMAIN_KNOWLEDGE_SPECS: list[tuple[str, str, str, list[str]]] = [
    (
        "caching",
        "Multi-Layer Caching",
        "Use multi-layer caching with L1 in-memory (LRU/dict), L2 Redis/Memcached and L3 CDN for "
        "static content. Patterns: cache-aside for reads, write-through for consistency, "
        "write-behind for write performance. Manage TTL expiration and event-driven cache "
        "invalidation; preload (cache warming) for hot keys. Key metrics: hit rate and stale "
        "read latency.",
        ["caching", "cache", "ttl", "redis", "invalidation", "memcached", "cache-warming"],
    ),
    (
        "auth",
        "Authentication & Authorization",
        "Secure authentication stack: JWT access tokens with short TTL plus refresh tokens, "
        "OAuth2 / OIDC for third-party login, password hashing with Argon2/bcrypt, RBAC/ABAC "
        "for authorization, rate limiting on login endpoints, and audit logging of auth events. "
        "Apply MFA for privileged actions and secure secret storage for signing keys.",
        ["auth", "authentication", "authorization", "jwt", "oauth2", "rbac", "mfa", "security"],
    ),
    (
        "graphql",
        "GraphQL API Design",
        "GraphQL schema generation with clear types and resolvers. Use DataLoader to batch and "
        "deduplicate N+1 queries, subscriptions for real-time pushes, persisted queries for "
        "performance, and schema federation or stitching across services. Guard resolvers with "
        "authorization and enforce query depth/complexity limits to prevent abuse.",
        ["graphql", "api", "schema", "resolvers", "subscriptions", "dataloader", "federation"],
    ),
    (
        "database",
        "Database Schema & Query Design",
        "Relational schema design with normalisation, primary/foreign keys and appropriate "
        "indexes (B-tree, GIN, GiST). Plan and optimise queries with EXPLAIN ANALYZE, use "
        "connection pooling (PgBouncer), read replicas and partitioning for large tables. "
        "Manage schema evolution through versioned migrations and generate typed ORM models "
        "(SQLAlchemy, Prisma).",
        ["database", "schema", "migrations", "orm", "sql", "query", "indexes", "postgresql"],
    ),
    (
        "documentation",
        "Automated Documentation",
        "Generate README, API reference, OpenAPI/Swagger specs, changelogs and docstrings "
        "automatically. Keep API docs in sync with the implementation (OpenAPI from code, "
        "docstring-driven markdown, MkDocs/ Sphinx sites), and include runnable examples plus "
        "architecture overviews so a solution is self-explanatory.",
        ["documentation", "docs", "openapi", "readme", "api-docs", "docstrings", "mkdocs"],
    ),
    (
        "cli_generator",
        "CLI Tool Generation",
        "Generate command-line tools with argparse, click or typer. Structure: subcommands, "
        "positional/optional args, help text, exit codes and error handling. Add progress "
        "bars, rich output, config-file and env overrides, and shell completion. Test CLI "
        "entry points via pytest and the typer/click test runners.",
        ["cli", "command-line", "argparse", "click", "typer", "cli-tools", "terminal"],
    ),
    (
        "containers",
        "Containerization & Images",
        "Build lean Docker images with multi-stage builds, minimal base images and .dockerignore. "
        "Run containers non-root, pin image digests, add health checks and resource limits, "
        "prefer compose or Kubernetes manifests with probes and network policies. Keep layers "
        "cached by ordering; scan images for vulnerabilities before shipping.",
        ["containers", "docker", "kubernetes", "image", "containerization", "compose"],
    ),
    (
        "config",
        "Configuration Management",
        "Follow 12-factor configuration: settings via environment variables with typed parsing "
        "(pydantic-settings), separate per environment, feature flags for gradual rollout, and "
        "secret management via Vault or injected secrets — never commit secrets. Provide sane "
        "defaults and validate config at startup with clear failure messages.",
        ["config", "configuration", "env", "feature-flags", "secrets", "settings"],
    ),
    (
        "data_pipeline",
        "Data Pipeline / ETL",
        "Build ETL and streaming pipelines with Airflow or Prefect for batch scheduling and dbt "
        "for transformations. Handle idempotent loads, retries, schema drift and backfills; "
        "store raw and processed layers separately (data lakehouse). Monitor pipeline health, "
        "data quality checks, and lineage so failures are caught early.",
        ["data-pipeline", "etl", "airflow", "dbt", "streaming", "data-processing", "batch"],
    ),
    (
        "websocket",
        "Real-Time WebSockets",
        "Implement real-time WebSocket servers with pub/sub, channels/rooms and connection "
        "management (heartbeat ping/pong, reconnects with backoff, graceful shutdown). "
        "Use FastAPI/websockets, Socket.IO or Phoenix Channels; scale horizontally via a "
        "shared pub/sub broker (Redis) and sticky sessions or connection registry.",
        ["websocket", "realtime", "pubsub", "rooms", "socket", "connection"],
    ),
    (
        "gateway",
        "API Gateway & Middleware",
        "Design an API gateway (nginx, envoy, Kong) with path/header-based routing, middleware "
        "chaining, authentication, rate limiting, circuit breaking and request logging. Centralise "
        "cross-cutting concerns, aggregate microservice APIs, and enforce timeouts/retries and "
        "canary routing for safe deployments.",
        ["gateway", "api-gateway", "routing", "middleware", "nginx", "envoy", "circuit-breaker"],
    ),
    (
        "events",
        "Event-Driven Architecture",
        "Event-driven systems: event sourcing for an append-only event store, CQRS with separate "
        "read models, an event bus (Kafka, RabbitMQ) for decoupling, and the saga pattern for "
        "distributed transactions. Design event schemas with versioning, idempotent consumers, "
        "and outbox pattern for reliable publishing.",
        ["events", "event-driven", "event-sourcing", "cqrs", "saga", "kafka", "rabbitmq"],
    ),
    (
        "performance",
        "Performance & Profiling",
        "Profile with cProfile/py-spy and benchmark with pytest-benchmark; load-test with Locust "
        "or k6 to find bottlenecks (CPU, memory, I/O, lock contention). Set SLOs, use caching and "
        "async I/O, batch database access, and avoid N+1 queries. Track regressions in CI with "
        "baseline comparisons.",
        ["performance", "profiling", "benchmark", "load-testing", "bottleneck", "optimization"],
    ),
    (
        "mobile",
        "Mobile App Development",
        "Generate mobile apps with React Native or Flutter, or native iOS/Android. Structure: "
        "clean architecture with state management (Provider/Redux/Bloc), typed API client, "
        "local persistence (SQLite), push notifications, and offline support. Ship via CI "
        "builds (Fastlane) with code signing and over-the-air updates.",
        ["mobile", "flutter", "react-native", "android", "ios", "app"],
    ),
    (
        "monitoring",
        "Observability & Monitoring",
        "Full observability: Prometheus metrics, Grafana dashboards, structured logging, and "
        "distributed tracing (OpenTelemetry, Jaeger/Tempo). Health/readiness endpoints, "
        "SLO-based alerting, and error budgets. Instrument RED (rate/errors/duration) and USE "
        "(utilisation/saturation/errors) metrics for every service.",
        ["monitoring", "observability", "metrics", "alerting", "tracing", "prometheus", "grafana"],
    ),
    (
        "ml_ops",
        "MLOps Pipeline",
        "Production ML: experiment tracking (MLflow), data versioning (DVC), feature store, "
        "training pipelines, model registry with versioned artifacts, and model serving "
        "(Triton/TorchServe/FastAPI). Monitor prediction drift and model quality in production; "
        "automate retraining and canary deploys with automated evaluation gates.",
        ["mlops", "ml", "machine-learning", "model-serving", "mlflow", "drift", "training"],
    ),
    (
        "security_scanner",
        "SAST & Security Scanning",
        "Static application security testing (SAST) plus dependency scanning, secret detection "
        "and OWASP Top 10 checks. Run scanners in CI, enforce thresholds that block merges, and "
        "remediate with parameterised queries, output encoding and validated auth. Produce an "
        "auditable SBOM and signed commits for supply-chain integrity.",
        ["security", "sast", "vulnerability", "owasp", "dependency-scanning", "secrets"],
    ),
    (
        "testing",
        "Automated Testing",
        "Generate unit, integration and end-to-end tests with pytest/unittest, plus coverage "
        "analysis and mutation testing to gauge test strength. Follow the testing pyramid, "
        "use fixtures and parametrisation, mock external I/O, and run tests in CI with "
        "reliable, fast isolation (testcontainers, factories).",
        ["testing", "tests", "pytest", "coverage", "mutation", "unit-tests"],
    ),
    (
        "terraform",
        "Infrastructure as Code",
        "Manage infrastructure with Terraform or Pulumi: versioned IaC, remote state with "
        "locking, reusable modules, and environments (dev/staging/prod) via workspaces. "
        "Plan in CI, apply with approval gates, tag resources, and enable drift detection. "
        "Use providers for cloud/kubernetes and destroy/validate in sandboxes.",
        ["terraform", "iac", "pulumi", "infrastructure", "infrastructure-as-code", "providers"],
    ),
    (
        "i18n",
        "Internationalization & Localization",
        "Internationalise applications with message catalogs, locale-aware formatting "
        "(dates, numbers, currencies, plurals), right-to-left support and translation "
        "management. Load localised resources lazily, fall back to default locale, and "
        "automate translation sync in CI.",
        ["i18n", "internationalization", "localization", "translation", "l10n", "locale"],
    ),
    (
        "queues",
        "Async Job Queues & Scheduling",
        "Background processing with Celery, RQ or native async queues (arq); brokers Redis or "
        "RabbitMQ. Design idempotent tasks with retries, dead-letter queues, and backoff; "
        "schedule periodic jobs and set worker concurrency and rate limits. Track job status, "
        "latency and failure rates in observability dashboards.",
        ["queues", "jobs", "celery", "message-broker", "scheduling", "async", "worker"],
    ),
    (
        "quality",
        "Code Quality & Metrics",
        "Enforce code quality with linting (ruff), formatting, complexity limits, and code-smell "
        "detection. Measure cyclomatic complexity, duplication and maintainability; compute a "
        "grade per module and fail CI on regressions. Keep public APIs documented and typed, "
        "and gate merges on quality thresholds.",
        ["quality", "code-quality", "complexity", "code-smells", "linting", "metrics"],
    ),
]

DOMAIN_KNOWLEDGE: list[KnowledgeEntry] = [
    KnowledgeEntry(
        category="domain",
        title=title,
        content=content,
        tags=[module_name, *tags],
        weight=0.9,
        source=f"domain-seed:{module_name}",
    )
    for module_name, title, content, tags in DOMAIN_KNOWLEDGE_SPECS
]

#: Domain name → seed title (for tests and the CLI).
DOMAIN_INDEX: dict[str, str] = {name: title for name, title, _, _ in DOMAIN_KNOWLEDGE_SPECS}
