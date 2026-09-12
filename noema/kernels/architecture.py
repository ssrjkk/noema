"""Architecture kernel — designing system solutions."""

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

# ── Architecture patterns ─────────────────────────────────────────────────

PATTERNS_DB: dict[str, ArchitecturePattern] = {
    "microservices": ArchitecturePattern(
        name="Microservices",
        description="Microservice architecture with independent services",
        pros=[
            "Independent deployment",
            "Scaling per service",
            "Technology freedom",
            "Fault isolation",
        ],
        cons=[
            "Complexity of distributed calls",
            "Distributed transactions",
            "Operational complexity",
        ],
        use_cases=[
            "High-load systems",
            "Large teams",
            "Complex domain",
        ],
        complexity=TaskComplexity.COMPLEX,
    ),
    "event_driven": ArchitecturePattern(
        name="Event-Driven",
        description="Event-driven architecture with asynchronous processing",
        pros=[
            "Loose coupling",
            "High throughput",
            "Natural data bus",
            "Event replay",
        ],
        cons=[
            "Debugging complexity",
            "Eventual consistency",
            "Order guarantee complexity",
        ],
        use_cases=["Real-time systems", "IoT", "Financial systems"],
        complexity=TaskComplexity.COMPLEX,
    ),
    "clean_architecture": ArchitecturePattern(
        name="Clean Architecture",
        description="Clean architecture with clear layers and dependency rule",
        pros=[
            "Testability",
            "Framework independence",
            "Clear structure",
            "Easy layer replacement",
        ],
        cons=[
            "More boilerplate",
            "Learning curve",
            "Over-engineering for simple tasks",
        ],
        use_cases=[
            "Long-term projects",
            "Complex business logic",
            "Enterprise",
        ],
        complexity=TaskComplexity.MODERATE,
    ),
    "serverless": ArchitecturePattern(
        name="Serverless",
        description="Serverless architecture on managed functions",
        pros=[
            "Zero server load",
            "Pay-per-use",
            "Automatic scaling",
            "Fast start",
        ],
        cons=[
            "Vendor lock-in",
            "Cold starts",
            "Execution time limits",
            "Local debugging complexity",
        ],
        use_cases=["Event processing", "APIs with variable load", "MVP"],
        complexity=TaskComplexity.SIMPLE,
    ),
    "modular_monolith": ArchitecturePattern(
        name="Modular Monolith",
        description="Modular monolith with clear boundaries between modules",
        pros=[
            "Deployment simplicity",
            "Single transaction",
            "Easy debugging",
            "Path to microservices",
        ],
        cons=[
            "Scaling complexity",
            "Tight coupling between modules",
            "Single point of failure",
        ],
        use_cases=[
            "Medium projects",
            "Small teams",
            "Startups",
        ],
        complexity=TaskComplexity.MODERATE,
    ),
    "cqrs_event_sourcing": ArchitecturePattern(
        name="CQRS + Event Sourcing",
        description="Read/write separation with event storage",
        pros=[
            "Full change history",
            "Read/write path optimization",
            "Time-travel debugging",
            "High write throughput",
        ],
        cons=[
            "Implementation complexity",
            "Event versioning",
            "Data volume",
        ],
        use_cases=["Financial systems", "Audit-heavy", "Collaborative editing"],
        complexity=TaskComplexity.EXTREME,
    ),
    "pipeline": ArchitecturePattern(
        name="Pipeline / ETL",
        description="Pipeline data processing with transformation stages",
        pros=[
            "Parallelism",
            "Stage modularity",
            "Easy to scale",
            "Simple debugging",
        ],
        cons=[
            "Latency",
            "Data loss on failures",
            "State management complexity",
        ],
        use_cases=["Data processing", "ML pipelines", "ETL"],
        complexity=TaskComplexity.MODERATE,
    ),
    "layered": ArchitecturePattern(
        name="Layered (N-Tier)",
        description="Classic multi-layered architecture",
        pros=[
            "Ease of understanding",
            "Standard pattern",
            "Easy separation of concerns",
        ],
        cons=[
            "Rigid layers",
            "Modification complexity",
            "Overhead between layers",
        ],
        use_cases=[
            "CRUD applications",
            "Internal tools",
            "Legacy modernization",
        ],
        complexity=TaskComplexity.SIMPLE,
    ),
}


class ArchitectureKernel(BaseKernel):
    """Architecture design kernel."""

    @property
    def name(self) -> str:
        return "architecture"

    @property
    def description(self) -> str:
        return "System architecture design and pattern selection"

    async def execute(self, task: Task, **kwargs) -> dict[str, Any]:
        phase = kwargs.get("phase", "design")

        if phase == "analyze":
            return await self._analyze(task)
        return await self._design(task)

    async def _analyze(self, task: Task) -> dict[str, Any]:
        """Analyze the task to determine architectural requirements."""
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
        """Architecture design."""
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
        """Select a pattern based on tags and complexity."""
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
        """Suggest several suitable patterns."""
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
        """Design architecture components."""
        base_components = [
            {
                "name": "API Gateway",
                "type": "gateway",
                "responsibility": "Routing and authentication",
            },
            {
                "name": "Auth Service",
                "type": "service",
                "responsibility": "User and token management",
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
                    "responsibility": "Model training and inference",
                }
            )
        if "data" in tags or "analytics" in tags:
            base_components.append(
                {
                    "name": "Data Pipeline",
                    "type": "pipeline",
                    "responsibility": "Data collection and processing",
                }
            )

        base_components.extend(
            [
                {
                    "name": "Database",
                    "type": "database",
                    "responsibility": "Persistent data storage",
                },
                {
                    "name": "Cache",
                    "type": "cache",
                    "responsibility": "Caching hot data",
                },
                {
                    "name": "Message Queue",
                    "type": "messaging",
                    "responsibility": "Asynchronous message exchange",
                },
            ]
        )

        return base_components

    def _design_communication(self, pattern: ArchitecturePattern) -> dict[str, Any]:
        """Design communication between components."""
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
        """Design deployment."""
        return {
            "containerization": "Docker",
            "orchestration": "Kubernetes",
            "ci_cd": "GitHub Actions / GitLab CI",
            "monitoring": "Prometheus + Grafana",
            "logging": "ELK Stack / Loki",
        }

    def _infer_scale(self, tags: set[str], task: Task) -> str:
        """Determine scalability requirements."""
        if "high-load" in tags or "scale" in tags:
            return "high"
        if task.complexity in (TaskComplexity.COMPLEX, TaskComplexity.EXTREME):
            return "medium-high"
        return "standard"
