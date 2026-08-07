"""Ядро архитектуры — проектирование системных решений."""

from __future__ import annotations

from typing import Any

from noema.core.types import (
    ArchitecturePattern,
    Task,
    TaskComplexity,
)
from noema.kernels.base import BaseKernel
from noema.logging import get_logger

logger = get_logger(__name__)

# ── Архитектурные паттерны ─────────────────────────────────────────────────

PATTERNS_DB: dict[str, ArchitecturePattern] = {
    "microservices": ArchitecturePattern(
        name="Microservices",
        description="Микросервисная архитектура с независимыми сервисами",
        pros=[
            "Независимое деплоивание",
            "Масштабирование по сервисам",
            "Технологическая свобода",
            "Fault isolation",
        ],
        cons=[
            "Сложность распределённых вызовов",
            "Дистрибутивные транзакции",
            "Операционная сложность",
        ],
        use_cases=[
            "Высоконагруженные системы",
            "Большие команды",
            "Сложный домен",
        ],
        complexity=TaskComplexity.COMPLEX,
    ),
    "event_driven": ArchitecturePattern(
        name="Event-Driven",
        description="Событийно-управляемая архитектура с асинхронной обработкой",
        pros=[
            "Слабая связанность",
            "Высокая吞吐ость",
            "Естественная шина данных",
            "Replay событий",
        ],
        cons=[
            "Сложность отладки",
            "Eventual consistency",
            "Сложность order guarantee",
        ],
        use_cases=["Real-time системы", "IoT", "Финансовые системы"],
        complexity=TaskComplexity.COMPLEX,
    ),
    "clean_architecture": ArchitecturePattern(
        name="Clean Architecture",
        description="Чистая архитектура с чёткими слоями и dependency rule",
        pros=[
            "Тестируемость",
            "Независимость от фреймворков",
            "Ясная структура",
            "Легкая замена слоёв",
        ],
        cons=[
            "Больше boilerplate",
            "Кривая обучения",
            "Over-engineering для простых задач",
        ],
        use_cases=[
            "Долгосрочные проекты",
            "Сложная бизнес-логика",
            "Enterprise",
        ],
        complexity=TaskComplexity.MODERATE,
    ),
    "serverless": ArchitecturePattern(
        name="Serverless",
        description="Бессерверная архитектура на managed functions",
        pros=[
            "Нулевая серверная нагрузка",
            "Pay-per-use",
            "Автоматическое масштабирование",
            "Быстрый старт",
        ],
        cons=[
            "Vendor lock-in",
            "Cold starts",
            "Ограничения на execution time",
            "Сложность локальной отладки",
        ],
        use_cases=["Event processing", "APIs с переменной нагрузкой", "MVP"],
        complexity=TaskComplexity.SIMPLE,
    ),
    "modular_monolith": ArchitecturePattern(
        name="Modular Monolith",
        description="Модульный монолит с чёткими границами между модулями",
        pros=[
            "Простота деплоя",
            "Единая транзакция",
            "Лёгкая отладка",
            "Путь к микросервисам",
        ],
        cons=[
            "Сложность масштабирования",
            "Тесная связанность модулей",
            "Единая точка отказа",
        ],
        use_cases=[
            "Средние проекты",
            "Малые команды",
            "Стартапы",
        ],
        complexity=TaskComplexity.MODERATE,
    ),
    "cqrs_event_sourcing": ArchitecturePattern(
        name="CQRS + Event Sourcing",
        description="Разделение чтения/записи с хранением событий",
        pros=[
            "Полная история изменений",
            "Оптимизация read/write путей",
            "Time-travel debugging",
            "Высокая吞吐ость записи",
        ],
        cons=[
            "Сложность implementation",
            "Event versioning",
            "Объём данных",
        ],
        use_cases=["Финансовые системы", "Audit-heavy", "Collaborative editing"],
        complexity=TaskComplexity.EXTREME,
    ),
    "pipeline": ArchitecturePattern(
        name="Pipeline / ETL",
        description="Конвейерная обработка данных с этапами трансформации",
        pros=[
            "Параллелизм",
            "Модульность этапов",
            "Легко масштабировать",
            "Простая отладка",
        ],
        cons=[
            "Латентность",
            "Потеря данных при сбоях",
            "Сложность управления состоянием",
        ],
        use_cases=["Data processing", "ML pipelines", "ETL"],
        complexity=TaskComplexity.MODERATE,
    ),
    "layered": ArchitecturePattern(
        name="Layered (N-Tier)",
        description="Классическая многоуровневая архитектура",
        pros=[
            "Простота понимания",
            "Стандартный паттерн",
            "Лёгкое разделение обязанностей",
        ],
        cons=[
            "Рigid layers",
            "Сложность модификации",
            "Overhead между слоями",
        ],
        use_cases=[
            "CRUD-приложения",
            "Внутренние инструменты",
            "Legacy modernization",
        ],
        complexity=TaskComplexity.SIMPLE,
    ),
}


class ArchitectureKernel(BaseKernel):
    """Ядро проектирования архитектуры."""

    @property
    def name(self) -> str:
        return "architecture"

    @property
    def description(self) -> str:
        return "Проектирование архитектуры системы и выбор паттернов"

    async def execute(self, task: Task, **kwargs) -> dict[str, Any]:
        phase = kwargs.get("phase", "design")

        if phase == "analyze":
            return await self._analyze(task)
        return await self._design(task)

    async def _analyze(self, task: Task) -> dict[str, Any]:
        """Анализ задачи для определения архитектурных требований."""
        constraints = []
        for req in task.requirements:
            if req.priority >= 7:
                constraints.append(req.description)

        tags = {t.lower() for t in task.tags}

        return {
            "type": "analysis",
            "constraints": constraints,
            "scale_requirements": self._infer_scale(tags, task),
            "complexity": task.complexity.value,
            "suggested_patterns": self._suggest_patterns(tags, task),
            "_confidence": 0.75,
        }

    async def _design(self, task: Task) -> dict[str, Any]:
        """Проектирование архитектуры."""
        tags = {t.lower() for t in task.tags}
        pattern = self._select_pattern(tags, task)

        knowledge_context = await self._query_knowledge(
            f"architecture {task.title} {task.description}"
        )

        components = self._design_components(pattern, task)

        return {
            "type": "architecture",
            "pattern": pattern.model_dump(),
            "components": components,
            "communication": self._design_communication(pattern),
            "deployment": self._design_deployment(task),
            "knowledge_references": [k.get("title", "") for k in knowledge_context[:3]],
            "_confidence": 0.72,
        }

    def _select_pattern(self, tags: set[str], task: Task) -> ArchitecturePattern:
        """Выбор паттерна на основе тегов и сложности."""
        if "microservice" in tags or "distributed" in tags:
            return PATTERNS_DB["microservices"]
        if "event" in tags or "stream" in tags or "real-time" in tags:
            return PATTERNS_DB["event_driven"]
        if "clean" in tags or "ddd" in tags:
            return PATTERNS_DB["clean_architecture"]
        if "serverless" in tags or "lambda" in tags:
            return PATTERNS_DB["serverless"]
        if "etl" in tags or "pipeline" in tags or "data" in tags:
            return PATTERNS_DB["pipeline"]
        if "cqrs" in tags or "event-sourcing" in tags:
            return PATTERNS_DB["cqrs_event_sourcing"]

        if task.complexity in (TaskComplexity.COMPLEX, TaskComplexity.EXTREME):
            return PATTERNS_DB["microservices"]
        if task.complexity == TaskComplexity.SIMPLE:
            return PATTERNS_DB["layered"]

        return PATTERNS_DB["modular_monolith"]

    def _suggest_patterns(self, tags: set[str], task: Task) -> list[str]:
        """Предложение нескольких подходящих паттернов."""
        suggested = []
        for _name, pattern in PATTERNS_DB.items():
            score = 0
            if task.complexity == pattern.complexity:
                score += 2
            for use_case in pattern.use_cases:
                if any(t in use_case.lower() for t in tags):
                    score += 1
            if score >= 1:
                suggested.append(pattern.name)
        return suggested or ["Modular Monolith"]

    def _design_components(self, pattern: ArchitecturePattern, task: Task) -> list[dict]:
        """Проектирование компонентов архитектуры."""
        base_components = [
            {
                "name": "API Gateway",
                "type": "gateway",
                "responsibility": "Маршрутизация и аутентификация",
            },
            {
                "name": "Auth Service",
                "type": "service",
                "responsibility": "Управление пользователями и токенами",
            },
        ]

        tags = {t.lower() for t in task.tags}
        if "web" in tags or "api" in tags:
            base_components.append(
                {"name": "Web API", "type": "service", "responsibility": "HTTP/REST API"}
            )
        if "ml" in tags or "ai" in tags:
            base_components.append(
                {
                    "name": "ML Service",
                    "type": "service",
                    "responsibility": "Обучение и инференс моделей",
                }
            )
        if "data" in tags or "analytics" in tags:
            base_components.append(
                {
                    "name": "Data Pipeline",
                    "type": "pipeline",
                    "responsibility": "Сбор и обработка данных",
                }
            )

        base_components.extend(
            [
                {
                    "name": "Database",
                    "type": "database",
                    "responsibility": "Персистентное хранение данных",
                },
                {
                    "name": "Cache",
                    "type": "cache",
                    "responsibility": "Кэширование горячих данных",
                },
                {
                    "name": "Message Queue",
                    "type": "messaging",
                    "responsibility": "Асинхронный обмен сообщениями",
                },
            ]
        )

        return base_components

    def _design_communication(self, pattern: ArchitecturePattern) -> dict[str, Any]:
        """Проектирование коммуникации между компонентами."""
        if pattern.name in ("Microservices", "Event-Driven"):
            return {
                "sync": "gRPC / REST",
                "async": "Kafka / RabbitMQ",
                "discovery": "Consul / K8s Service",
            }
        return {
            "sync": "Function calls",
            "async": "In-process event bus",
            "discovery": "Direct import",
        }

    def _design_deployment(self, task: Task) -> dict[str, Any]:
        """Проектирование деплоя."""
        return {
            "containerization": "Docker",
            "orchestration": "Kubernetes",
            "ci_cd": "GitHub Actions / GitLab CI",
            "monitoring": "Prometheus + Grafana",
            "logging": "ELK Stack / Loki",
        }

    def _infer_scale(self, tags: set[str], task: Task) -> str:
        """Определение требований к масштабируемости."""
        if "high-load" in tags or "scale" in tags:
            return "high"
        if task.complexity in (TaskComplexity.COMPLEX, TaskComplexity.EXTREME):
            return "medium-high"
        return "standard"
