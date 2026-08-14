"""Ядро анализа — анализ требований, сложности, рисков."""

from __future__ import annotations

from typing import Any

from noema.core.types import Task, TaskComplexity
from noema.kernels.base import BaseKernel
from noema.logging import get_logger

logger = get_logger(__name__)


class AnalysisKernel(BaseKernel):
    """Ядро глубокого анализа задач."""

    @property
    def name(self) -> str:
        return "analysis"

    @property
    def description(self) -> str:
        return "Анализ требований, оценка сложности, выявление рисков"

    async def execute(self, task: Task, **kwargs) -> dict[str, Any]:
        phase = kwargs.get("phase", "full")

        requirements_analysis = self._analyze_requirements(task)
        complexity = self._estimate_complexity(task)
        risks = self._identify_risks(task)

        result: dict[str, Any] = {
            "type": "analysis",
            "requirements": requirements_analysis,
            "complexity": complexity,
            "risks": risks,
            "_confidence": 0.78,
        }

        if phase in ("full", "design"):
            result["timeline"] = self._estimate_timeline(task, complexity)
            result["team"] = self._suggest_team(task, complexity)
            result["tech_debt_estimate"] = self._estimate_tech_debt(task)

        return result

    def _analyze_requirements(self, task: Task) -> dict[str, Any]:
        functional = []
        non_functional = []

        for req in task.requirements:
            entry = {
                "description": req.description,
                "priority": req.priority,
                "constraints": req.constraints,
            }
            nf_keywords = {
                "performance",
                "security",
                "scale",
                "availability",
                "monitoring",
                "logging",
            }
            if any(k in req.category.lower() for k in nf_keywords):
                non_functional.append(entry)
            else:
                functional.append(entry)

        return {
            "functional": functional,
            "non_functional": non_functional,
            "total": len(task.requirements),
            "high_priority": sum(1 for r in task.requirements if r.priority >= 7),
        }

    def _estimate_complexity(self, task: Task) -> dict[str, Any]:
        score = 0
        factors = []

        req_count = len(task.requirements)
        score += min(req_count * 2, 20)
        if req_count > 10:
            factors.append(f"Много требований ({req_count})")

        tags = {t.lower() for t in task.tags}
        complex_tags = {"distributed", "microservice", "ml", "real-time", "high-load", "security"}
        intersection = tags & complex_tags
        score += len(intersection) * 10
        if intersection:
            factors.append(f"Сложные теги: {', '.join(intersection)}")

        if task.complexity in (TaskComplexity.COMPLEX, TaskComplexity.EXTREME):
            score += 20
            factors.append(f"Указанная сложность: {task.complexity.value}")

        if task.preferred_stack:
            stack_complexity = len(task.preferred_stack.languages) + len(
                task.preferred_stack.frameworks
            )
            score += stack_complexity * 3

        if not factors:
            factors.append("Стандартная задача")

        level = (
            "extreme"
            if score >= 80
            else "complex"
            if score >= 50
            else "moderate"
            if score >= 25
            else "simple"
        )

        return {"score": score, "level": level, "factors": factors}

    def _identify_risks(self, task: Task) -> list[dict[str, Any]]:
        risks = []
        tags = {t.lower() for t in task.tags}

        if "distributed" in tags or "microservice" in tags:
            risks.append(
                {
                    "type": "technical",
                    "name": "Distributed System Complexity",
                    "probability": "high",
                    "impact": "high",
                    "mitigation": "Start with modular monolith, extract services later",
                }
            )

        if "ml" in tags or "ai" in tags:
            risks.append(
                {
                    "type": "technical",
                    "name": "Model Drift",
                    "probability": "medium",
                    "impact": "high",
                    "mitigation": "Implement ML monitoring, retraining pipeline, A/B testing",
                }
            )

        if "high-load" in tags:
            risks.append(
                {
                    "type": "performance",
                    "name": "Performance Bottlenecks",
                    "probability": "medium",
                    "impact": "critical",
                    "mitigation": "Load testing from day 1, profiling, horizontal scaling",
                }
            )

        if len(task.requirements) > 15:
            risks.append(
                {
                    "type": "project",
                    "name": "Scope Creep",
                    "probability": "high",
                    "impact": "medium",
                    "mitigation": "MVP first, iterative delivery, strict requirement management",
                }
            )

        risks.append(
            {
                "type": "dependency",
                "name": "Third-party Service Dependency",
                "probability": "low",
                "impact": "medium",
                "mitigation": "Abstract integrations, circuit breakers, fallbacks",
            }
        )

        return risks

    def _estimate_timeline(self, task: Task, complexity: dict) -> dict[str, Any]:
        base_days = {
            "simple": 5,
            "moderate": 15,
            "complex": 40,
            "extreme": 90,
        }
        days = base_days.get(complexity["level"], 20)

        return {
            "estimated_days": days,
            "phases": [
                {"name": "Discovery & Planning", "days": max(1, days // 10)},
                {"name": "Core Development", "days": int(days * 0.5)},
                {"name": "Integration & Testing", "days": int(days * 0.25)},
                {"name": "Deployment & Optimization", "days": int(days * 0.15)},
            ],
        }

    def _suggest_team(self, task: Task, complexity: dict) -> dict[str, Any]:
        tags = {t.lower() for t in task.tags}
        team = {"backend": 1, "frontend": 0, "devops": 0, "qa": 0}

        if complexity["level"] in ("complex", "extreme"):
            team["backend"] = 3
            team["qa"] = 1
            team["devops"] = 1
        elif complexity["level"] == "moderate":
            team["backend"] = 2

        if "web" in tags or "frontend" in tags:
            team["frontend"] = 2 if complexity["level"] in ("complex", "extreme") else 1
        if "ml" in tags:
            team["ml_engineer"] = 1
        if "mobile" in tags:
            team["mobile"] = 1

        team["total"] = sum(v for k, v in team.items() if isinstance(v, int))
        return team

    def _estimate_tech_debt(self, task: Task) -> dict[str, Any]:
        tags = {t.lower() for t in task.tags}
        debt_items = []

        if "legacy" in tags:
            debt_items.append({"item": "Legacy code migration", "effort": "high"})
        if "rapid-prototyping" in tags:
            debt_items.append({"item": "Quick prototype to production", "effort": "medium"})
        if not task.preferred_stack:
            debt_items.append({"item": "Stack selection needs validation", "effort": "low"})

        return {
            "items": debt_items,
            "estimated_effort": sum(
                1 if d["effort"] == "high" else 0.5 if d["effort"] == "medium" else 0.2
                for d in debt_items
            ),
        }
