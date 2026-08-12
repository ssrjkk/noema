"""Хранилище знаний — база паттернов, решений, технологий."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from noema.core.types import KnowledgeEntry, Pattern, Task, TechStack
from noema.knowledge.domains import DOMAIN_KNOWLEDGE
from noema.logging import get_logger

logger = get_logger(__name__)

# ── Встроенные базы знаний ──────────────────────────────────────────────────

BUILTIN_PATTERNS: list[Pattern] = [
    Pattern(
        name="api-gateway-microservices",
        category="architecture",
        description="API Gateway паттерн для микросервисов",
        template={
            "gateway": {"type": "nginx/envoy", "routing": "path-based"},
            "services": {"discovery": "consul", "communication": "gRPC"},
        },
        applicable_stacks=["python", "go", "java"],
        success_rate=0.85,
    ),
    Pattern(
        name="event-sourcing-cqrs",
        category="architecture",
        description="CQRS + Event Sourcing для event-driven систем",
        template={
            "write_model": "event_store",
            "read_model": "projections",
            "event_bus": "kafka/rabbitmq",
        },
        applicable_stacks=["python", "java", "go", "rust"],
        success_rate=0.80,
    ),
    Pattern(
        name="ml-pipeline-mlops",
        category="ml",
        description="ML Pipeline с MLOps практиками",
        template={
            "data": "feature_store",
            "training": "mlflow",
            "serving": "triton/torchserve",
            "monitoring": "evidently/whylabs",
        },
        applicable_stacks=["python"],
        success_rate=0.75,
    ),
    Pattern(
        name="react-fastapi-fullstack",
        category="web",
        description="Fullstack: React + FastAPI + PostgreSQL",
        template={
            "frontend": "React/Next.js",
            "backend": "FastAPI",
            "database": "PostgreSQL",
            "cache": "Redis",
            "auth": "JWT + OAuth2",
        },
        applicable_stacks=["python", "typescript"],
        success_rate=0.88,
    ),
    Pattern(
        name="go-microservice",
        category="backend",
        description="Высокопроизводительный микросервис на Go",
        template={
            "framework": "gin/chi",
            "database": "postgresql",
            "cache": "redis",
            "messaging": "kafka",
            "observability": "opentelemetry",
        },
        applicable_stacks=["go"],
        success_rate=0.87,
    ),
    Pattern(
        name="rust-web-service",
        category="backend",
        description="Высокопроизводительный веб-сервис на Rust",
        template={
            "framework": "axum/actix",
            "database": "sqlx+postgres",
            "serialization": "serde",
            "async": "tokio",
        },
        applicable_stacks=["rust"],
        success_rate=0.82,
    ),
    Pattern(
        name="flutter-mobile-backend",
        category="mobile",
        description="Mobile приложение Flutter + бэкенд",
        template={
            "mobile": "Flutter/Dart",
            "backend": "Firebase/Supabase",
            "auth": "Firebase Auth",
            "storage": "Cloud Firestore",
        },
        applicable_stacks=["dart", "typescript"],
        success_rate=0.83,
    ),
    Pattern(
        name="data-lakehouse",
        category="data",
        description="Data Lakehouse архитектура",
        template={
            "ingestion": "kafka/flink",
            "storage": "S3/delta-lake",
            "processing": "spark/dbt",
            "serving": "presto/trino",
        },
        applicable_stacks=["python", "scala"],
        success_rate=0.78,
    ),
    Pattern(
        name="spring-boot-enterprise",
        category="enterprise",
        description="Enterprise приложение на Spring Boot",
        template={
            "framework": "Spring Boot 3",
            "security": "Spring Security",
            "database": "JPA + PostgreSQL",
            "messaging": "Spring Kafka",
            "api": "OpenAPI/Swagger",
        },
        applicable_stacks=["java", "kotlin"],
        success_rate=0.85,
    ),
]

BUILTIN_KNOWLEDGE: list[KnowledgeEntry] = [
    KnowledgeEntry(
        category="best-practice",
        title="12-Factor App Principles",
        content="Follow the 12-factor methodology: codebase in VCS, explicit dependencies, "
        "config via environment, backing services as attached resources, build/release/run "
        "separated, stateless processes, port binding, concurrency via processes, fast startup "
        "and graceful shutdown, dev/prod parity, logs as event streams, admin tasks as one-off.",
        tags=["architecture", "best-practice", "devops"],
        weight=0.9,
    ),
    KnowledgeEntry(
        category="best-practice",
        title="Database Optimization Strategies",
        content="Use: 1) Proper indexing (B-tree, GIN, GiST), 2) Query plan analysis "
        "(EXPLAIN ANALYZE), 3) Connection pooling (PgBouncer), 4) Read replicas, 5) Partitioning "
        "for large tables, 6) Materialized views for complex queries, 7) Vacuum tuning, "
        "8) pg_stat_statements monitoring.",
        tags=["database", "postgresql", "performance"],
        weight=0.85,
    ),
    KnowledgeEntry(
        category="best-practice",
        title="Microservices Communication Patterns",
        content="Synchronous: REST (simple), gRPC (fast, typed). Asynchronous: event-driven "
        "(loose coupling), saga (distributed transactions), CQRS (read/write split). Use circuit "
        "breakers for fault tolerance and a service mesh (Istio/Linkerd) for traffic control.",
        tags=["microservices", "distributed", "architecture"],
        weight=0.88,
    ),
    KnowledgeEntry(
        category="best-practice",
        title="Kubernetes Production Checklist",
        content="1) Resource limits & requests, 2) Health probes (liveness/readiness/startup), "
        "3) Pod disruption budgets, 4) HorizontalPodAutoscaler, 5) Network policies, 6) RBAC, "
        "7) Secret management (Vault), 8) Log aggregation (EFK/Loki), 9) Metrics (Prometheus), "
        "10) Distributed tracing (Jaeger/Tempo).",
        tags=["kubernetes", "devops", "production"],
        weight=0.9,
    ),
    KnowledgeEntry(
        category="stack",
        title="Python High-Performance Stack",
        content="FastAPI (async), SQLAlchemy 2.0 (async ORM), Pydantic v2 (validation), uvicorn "
        "(ASGI), Redis (cache), PostgreSQL (DB), Celery (task queue), Docker + Kubernetes, "
        "pytest + httpx (testing), structlog (logging).",
        tags=["python", "web", "api", "high-performance"],
        weight=0.87,
    ),
    KnowledgeEntry(
        category="stack",
        title="Go Microservices Stack",
        content="Chi/Gin (router), sqlx (database), go-redis, Kafka/NATS (messaging), gRPC, "
        "OpenTelemetry, Zap (logging), Viper (config), GoReleaser, Docker, Kubernetes.",
        tags=["go", "microservices", "high-performance"],
        weight=0.86,
    ),
    KnowledgeEntry(
        category="stack",
        title="TypeScript Full-Stack",
        content="Next.js (frontend), Fastify/Express (backend), Prisma (ORM), PostgreSQL, Redis, "
        "tRPC (type-safe API), Tailwind CSS, Vitest (testing), Turborepo (monorepo), Vercel "
        "(deployment).",
        tags=["typescript", "fullstack", "react", "nextjs"],
        weight=0.85,
    ),
    KnowledgeEntry(
        category="stack",
        title="ML/AI Production Stack",
        content="Python, PyTorch/TensorFlow, FastAPI (inference API), MLflow (experiment "
        "tracking), DVC (data versioning), Airflow/Prefect (orchestration), Redis (feature "
        "cache), PostgreSQL (metadata), Docker + GPU support, Prometheus + Grafana (monitoring).",
        tags=["ml", "ai", "python", "production"],
        weight=0.84,
    ),
    KnowledgeEntry(
        category="security",
        title="OWASP Top 10 Mitigations",
        content="A01 Broken Access Control -> RBAC + resource-level perms. A02 Cryptographic "
        "Failures -> TLS 1.3 + AES-256. A03 Injection -> parameterized queries. A04 Insecure "
        "Design -> threat modeling. A05 Security Misconfiguration -> hardened defaults. A06 "
        "Vulnerable Components -> SCA scanning. A07 Auth Failures -> MFA + rate limiting. A08 "
        "Data Integrity -> signed commits + SBOM. A09 Logging -> audit trail. A10 SSRF -> "
        "allowlist outbound.",
        tags=["security", "owasp", "web", "api"],
        weight=0.92,
    ),
    KnowledgeEntry(
        category="performance",
        title="Caching Strategy L1/L2/L3",
        content="L1: In-memory (dict/LRU) — nanoseconds, per-process. L2: Redis/Memcached — "
        "milliseconds, shared. L3: CDN — edge, static + API caching. Cache-aside for reads, "
        "write-through for consistency, write-behind for write performance. TTL-based expiration "
        "+ event-driven invalidation.",
        tags=["caching", "performance", "redis", "architecture"],
        weight=0.86,
    ),
    *DOMAIN_KNOWLEDGE,
]


class KnowledgeStore:
    """
    Хранилище знаний с TF-IDF поиском.

    Содержит базу паттернов, best practices и стеков технологий
    для генерации обоснованных решений.
    """

    def __init__(self, persist_path: str | None = None) -> None:
        self.persist_path = Path(persist_path) if persist_path else Path("noema_knowledge.json")
        self.entries: list[KnowledgeEntry] = list(BUILTIN_KNOWLEDGE)
        self.patterns: list[Pattern] = list(BUILTIN_PATTERNS)
        self._vectorizer: TfidfVectorizer | None = None
        self._vectors = None
        self._corpus: list[str] = []

    async def load(self) -> None:
        """Загрузка знаний из файла (если есть)."""
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text(encoding="utf-8"))
                for entry_data in data.get("entries", []):
                    self.entries.append(KnowledgeEntry(**entry_data))
                for pattern_data in data.get("patterns", []):
                    self.patterns.append(Pattern(**pattern_data))
                logger.info(
                    f"Загружено {len(data.get('entries', []))} entries, {len(data.get('patterns', []))} patterns"
                )
            except Exception as e:
                logger.warning(f"Ошибка загрузки знаний: {e}")

        self._build_index()

    async def persist(self) -> None:
        """Сохранение знаний в файл."""
        data = {
            "entries": [e.model_dump(mode="json") for e in self.entries if not e.embeddings],
            "patterns": [p.model_dump(mode="json") for p in self.patterns],
        }
        self.persist_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info(f"Сохранено {len(self.entries)} entries, {len(self.patterns)} patterns")

    def _build_index(self) -> None:
        """Построение TF-IDF индекса."""
        self._corpus = []
        for entry in self.entries:
            text = f"{entry.title} {entry.content} {' '.join(entry.tags)}"
            self._corpus.append(text)
        for pattern in self.patterns:
            text = f"{pattern.name} {pattern.description} {' '.join(pattern.applicable_stacks)}"
            self._corpus.append(text)

        if self._corpus:
            self._vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words=None,
                ngram_range=(1, 2),
            )
            self._vectors = self._vectorizer.fit_transform(self._corpus)

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Поиск по базе знаний."""
        if not self._vectorizer or self._vectors is None:
            return []

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._vectors).flatten()
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            if idx < len(self.entries):
                entry = self.entries[idx]
                results.append(
                    {
                        "type": "knowledge",
                        "title": entry.title,
                        "content": entry.content,
                        "category": entry.category,
                        "tags": entry.tags,
                        "score": float(scores[idx]),
                    }
                )
            else:
                pattern_idx = idx - len(self.entries)
                if pattern_idx < len(self.patterns):
                    pattern = self.patterns[pattern_idx]
                    results.append(
                        {
                            "type": "pattern",
                            "name": pattern.name,
                            "description": pattern.description,
                            "category": pattern.category,
                            "applicable_stacks": pattern.applicable_stacks,
                            "score": float(scores[idx]),
                        }
                    )

        return results

    async def find_relevant_stacks(self, task: Task) -> list[TechStack]:
        """Поиск релевантных стеков для задачи."""
        query = f"{task.title} {task.description} {' '.join(task.tags)}"
        results = await self.search(query, top_k=10)

        stacks = []
        for result in results:
            if result["type"] == "pattern" and result.get("score", 0) > 0.3:
                languages = result.get("applicable_stacks", [])
                if languages:
                    stacks.append(TechStack(languages=languages))

        if not stacks:
            stacks.append(
                TechStack(
                    languages=["Python", "TypeScript"],
                    frameworks=["FastAPI", "React"],
                    databases=["PostgreSQL", "Redis"],
                )
            )

        return stacks

    async def add_entry(self, entry: KnowledgeEntry) -> None:
        """Добавить новую запись."""
        self.entries.append(entry)
        self._build_index()

    async def add_pattern(self, pattern: Pattern) -> None:
        """Добавить новый паттерн."""
        self.patterns.append(pattern)
        self._build_index()

    def get_stats(self) -> dict[str, Any]:
        """Статистика хранилища."""
        return {
            "total_entries": len(self.entries),
            "total_patterns": len(self.patterns),
            "categories": list({e.category for e in self.entries}),
            "pattern_categories": list({p.category for p in self.patterns}),
        }
