"""Центральные типы фреймворка."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel, Field

# ── Enums ──────────────────────────────────────────────────────────────────


class TaskComplexity(StrEnum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXTREME = "extreme"


class KernelType(StrEnum):
    ARCHITECTURE = "architecture"
    CODEGEN = "codegen"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    SECURITY = "security"
    SCALING = "scaling"
    DATA = "data"
    DEVOPS = "devops"
    FRONTEND = "frontend"
    AI_ML = "ai_ml"


class AgentRole(StrEnum):
    ARCHITECT = "architect"
    DEVELOPER = "developer"
    REVIEWER = "reviewer"
    ANALYST = "analyst"
    DEVOPS = "devops"
    SECURITY = "security"
    DBA = "dba"
    FRONTEND = "frontend"
    AI_ENGINEER = "ai_engineer"
    LEAD = "lead"


class SolutionQuality(StrEnum):
    DRAFT = "draft"
    ACCEPTABLE = "acceptable"
    GOOD = "good"
    EXCELLENT = "excellent"
    MASTERPIECE = "masterpiece"


# ── Core Models ────────────────────────────────────────────────────────────

T = TypeVar("T")


class TechStack(BaseModel):
    """Описание технологического стека."""

    languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    databases: list[str] = Field(default_factory=list)
    infrastructure: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    cloud: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.languages:
            parts.append(f"Языки: {', '.join(self.languages)}")
        if self.frameworks:
            parts.append(f"Фреймворки: {', '.join(self.frameworks)}")
        if self.databases:
            parts.append(f"БД: {', '.join(self.databases)}")
        if self.infrastructure:
            parts.append(f"Инфра: {', '.join(self.infrastructure)}")
        if self.cloud:
            parts.append(f"Облако: {', '.join(self.cloud)}")
        return " | ".join(parts) if parts else "Не определён"


class ArchitecturePattern(BaseModel):
    """Паттерн архитектуры."""

    name: str
    description: str
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    complexity: TaskComplexity = TaskComplexity.MODERATE


class Requirement(BaseModel):
    """Требование к решению."""

    category: str
    description: str
    priority: int = Field(ge=1, le=10, default=5)
    constraints: list[str] = Field(default_factory=list)


class Task(BaseModel):
    """Входная задача для генерации решения."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str
    description: str = ""
    requirements: list[Requirement] = Field(default_factory=list)
    preferred_stack: TechStack | None = None
    complexity: TaskComplexity = TaskComplexity.MODERATE
    context: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CodeBlock(BaseModel):
    """Блок сгенерированного кода."""

    filename: str
    language: str
    content: str
    description: str = ""
    dependencies: list[str] = Field(default_factory=list)


class Solution(BaseModel):
    """Генерируемое решение."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str
    title: str
    summary: str
    architecture: ArchitecturePattern | None = None
    stack: TechStack = Field(default_factory=TechStack)
    code_blocks: list[CodeBlock] = Field(default_factory=list)
    diagrams: list[str] = Field(default_factory=list)
    deployment: dict[str, Any] = Field(default_factory=dict)
    performance_notes: list[str] = Field(default_factory=list)
    security_notes: list[str] = Field(default_factory=list)
    quality: SolutionQuality = SolutionQuality.DRAFT
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ThoughtProcess(BaseModel):
    """Траектория мышления мозга при генерации решения."""

    task_id: str
    steps: list[ThoughtStep] = Field(default_factory=list)
    alternatives_considered: list[str] = Field(default_factory=list)
    final_reasoning: str = ""
    duration_ms: float = 0.0

    def add_step(
        self, kernel: str, input_summary: str, output_summary: str, confidence: float
    ) -> None:
        self.steps.append(
            ThoughtStep(
                step_number=len(self.steps) + 1,
                kernel=kernel,
                input_summary=input_summary,
                output_summary=output_summary,
                confidence=confidence,
            )
        )

    @property
    def avg_confidence(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.confidence for s in self.steps) / len(self.steps)


class ThoughtStep(BaseModel):
    """Один шаг процесса мышления."""

    step_number: int
    kernel: str
    input_summary: str
    output_summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── Knowledge Models ───────────────────────────────────────────────────────


class KnowledgeEntry(BaseModel):
    """Запись в базе знаний."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    category: str
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    embeddings: list[float] = Field(default_factory=list)
    weight: float = Field(ge=0.0, le=1.0, default=0.5)
    source: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Pattern(BaseModel):
    """Паттерн решения из базы знаний."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    category: str
    description: str
    template: dict[str, Any] = Field(default_factory=dict)
    applicable_stacks: list[str] = Field(default_factory=list)
    success_rate: float = Field(ge=0.0, le=1.0, default=0.5)
    usage_count: int = 0
