"""Subagent — a specialized agent for a specific domain."""

from __future__ import annotations

import abc
from typing import Any

from noema.core.types import AgentRole, Solution, Task
from noema.logging import get_logger

logger = get_logger(__name__)


class BaseAgent(abc.ABC):
    """
    Base subagent.

    Each agent specializes in its own domain
    and contributes to the final solution.
    """

    def __init__(self, role: AgentRole, name: str | None = None) -> None:
        self.role = role
        self.name = name or f"{role.value}_agent"
        self._history: list[dict[str, Any]] = []

    @property
    @abc.abstractmethod
    def expertise(self) -> list[str]:
        """Domains of agent expertise."""
        ...

    @abc.abstractmethod
    async def analyze(self, task: Task) -> dict[str, Any]:
        """Analyze the task from the agent's expertise perspective."""
        ...

    @abc.abstractmethod
    async def contribute(
        self,
        task: Task,
        solution: Solution,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Agent's contribution to the solution."""
        ...

    async def review(self, solution: Solution) -> dict[str, Any]:
        """Review the solution as an agent."""
        return {
            "agent": self.name,
            "role": self.role.value,
            "approved": True,
            "comments": [],
            "suggestions": [],
        }

    def _log(self, message: str) -> None:
        logger.info(f"[{self.name}] {message}")


class ArchitectAgent(BaseAgent):
    """Architect agent."""

    def __init__(self) -> None:
        super().__init__(AgentRole.ARCHITECT, "lead_architect")

    @property
    def expertise(self) -> list[str]:
        return ["system-design", "patterns", "scalability", "trade-offs"]

    async def analyze(self, task: Task) -> dict[str, Any]:
        self._log(f"Analyzing architectural requirements: {task.title}")
        return {
            "domain": self._classify_domain(task),
            "scale": self._estimate_scale(task),
            "key_decisions": [],
        }

    async def contribute(
        self,
        task: Task,
        solution: Solution,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        self._log("Designing the solution architecture")
        return {
            "layer": "architecture",
            "patterns_recommended": ["modular", "clean-separation"],
            "component_count": len(context.get("components", [])),
        }

    def _classify_domain(self, task: Task) -> str:
        tags = {t.lower() for t in task.tags}
        if "web" in tags:
            return "web-application"
        if "ml" in tags or "ai" in tags:
            return "ml-system"
        if "data" in tags:
            return "data-platform"
        return "general"

    def _estimate_scale(self, task: Task) -> str:
        tags = {t.lower() for t in task.tags}
        if "high-load" in tags:
            return "high"
        if "enterprise" in tags:
            return "enterprise"
        return "standard"


class DeveloperAgent(BaseAgent):
    """Developer agent."""

    def __init__(self) -> None:
        super().__init__(AgentRole.DEVELOPER, "lead_developer")

    @property
    def expertise(self) -> list[str]:
        return ["implementation", "patterns", "testing", "refactoring"]

    async def analyze(self, task: Task) -> dict[str, Any]:
        self._log(f"Analyzing implementation requirements: {task.title}")
        return {
            "modules_needed": len(task.requirements),
            "complexity_assessment": task.complexity.value,
            "recommended_patterns": ["repository", "service-layer", "dependency-injection"],
        }

    async def contribute(
        self,
        task: Task,
        solution: Solution,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        self._log("Generating the codebase")
        return {
            "layer": "implementation",
            "files_generated": len(solution.code_blocks),
            "patterns_applied": ["repository", "service", "handler"],
        }


class SecurityAgent(BaseAgent):
    """Security agent."""

    def __init__(self) -> None:
        super().__init__(AgentRole.SECURITY, "security_specialist")

    @property
    def expertise(self) -> list[str]:
        return ["security-audit", "hardening", "compliance", "penetration-testing"]

    async def analyze(self, task: Task) -> dict[str, Any]:
        self._log("Running security assessment")
        tags = {t.lower() for t in task.tags}
        attack_surface = []
        if "web" in tags or "api" in tags:
            attack_surface.extend(["xss", "sqli", "csrf", "rate-limiting"])
        if "auth" in tags:
            attack_surface.extend(["brute-force", "token-theft", "session-hijack"])
        return {
            "attack_surface": attack_surface,
            "compliance_requirements": ["OWASP Top 10"],
        }

    async def contribute(
        self,
        task: Task,
        solution: Solution,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        self._log("Adding security measures")
        return {
            "layer": "security",
            "checks_added": ["input-validation", "rate-limiting", "cors", "csp"],
        }

    async def review(self, solution: Solution) -> dict[str, Any]:
        issues = []
        for note in solution.security_notes:
            if "vulnerability" in note.lower():
                issues.append(note)
        return {
            "agent": self.name,
            "role": self.role.value,
            "approved": len(issues) == 0,
            "comments": solution.security_notes,
            "issues": issues,
        }


class DevOpsAgent(BaseAgent):
    """DevOps agent."""

    def __init__(self) -> None:
        super().__init__(AgentRole.DEVOPS, "devops_engineer")

    @property
    def expertise(self) -> list[str]:
        return ["ci-cd", "containerization", "orchestration", "monitoring", "iac"]

    async def analyze(self, task: Task) -> dict[str, Any]:
        self._log("Analyzing infrastructure requirements")
        return {
            "deployment_target": "kubernetes",
            "ci_cd": "github-actions",
            "monitoring_stack": ["prometheus", "grafana", "alertmanager"],
        }

    async def contribute(
        self,
        task: Task,
        solution: Solution,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        self._log("Setting up infrastructure and CI/CD")
        return {
            "layer": "infrastructure",
            "configs_generated": [
                "Dockerfile",
                "docker-compose.yml",
                "k8s-manifests",
                "github-actions",
            ],
        }


class DBAAgent(BaseAgent):
    """Database agent."""

    def __init__(self) -> None:
        super().__init__(AgentRole.DBA, "database_architect")

    @property
    def expertise(self) -> list[str]:
        return ["database-design", "optimization", "migration", "replication"]

    async def analyze(self, task: Task) -> dict[str, Any]:
        self._log("Analyzing data requirements")
        return {
            "data_volume": "unknown",
            "consistency_requirements": "eventual",
            "recommended_db": "postgresql",
        }

    async def contribute(
        self,
        task: Task,
        solution: Solution,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        self._log("Designing the database schema")
        return {
            "layer": "database",
            "tables": [],
            "indexes": [],
            "migrations": [],
        }


class AIEngineerAgent(BaseAgent):
    """ML/AI agent."""

    def __init__(self) -> None:
        super().__init__(AgentRole.AI_ENGINEER, "ml_engineer")

    @property
    def expertise(self) -> list[str]:
        return ["model-training", "inference-optimization", "data-pipeline", "mlops"]

    async def analyze(self, task: Task) -> dict[str, Any]:
        self._log("Analyzing ML/AI requirements")
        return {
            "model_type": "unknown",
            "inference_requirements": "real-time",
            "training_data": "required",
        }

    async def contribute(
        self,
        task: Task,
        solution: Solution,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        self._log("Designing the ML pipeline")
        return {
            "layer": "ml",
            "pipeline_steps": ["ingestion", "preprocessing", "training", "evaluation", "serving"],
        }
