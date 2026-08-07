"""AgentOrchestrator — parallel subagent dispatch and solution assembly.

Architecture:
- Registers a fixed roster of specialized :class:`BaseAgent` sub-agents
  (architect, developer, security, devops, DBA, ML engineer).
- Fan-out methods (``get_analyses`` / ``get_contributions`` /
  ``review_solution``) run every agent concurrently, bounded by a semaphore,
  so an orchestrator request completes in the slowest agent, not the sum.

Concurrency contract:
- All fan-out is via :func:`asyncio.gather` under :class:`asyncio.Semaphore`;
  per-agent failures are captured into the result dict, never raised.
- The event loop is never blocked: agents are pure coroutine work.

Complexity:
- Fan-out: ``O(A)`` for A agents, wall time ``O(max_agent)`` (parallel), with
  at most ``max_concurrent`` agents in flight.
- ``assemble_solution``: ``O(B + O + S)`` for B code blocks, O optimizations,
  and S security checks, plus one parallel contribution pass.
"""

from __future__ import annotations

import asyncio
from typing import Any

from noema.agents.base import (
    AIEngineerAgent,
    ArchitectAgent,
    BaseAgent,
    DBAAgent,
    DeveloperAgent,
    DevOpsAgent,
    SecurityAgent,
)
from noema.core.types import (
    AgentRole,
    ArchitecturePattern,
    CodeBlock,
    Solution,
    Task,
    TechStack,
)
from noema.logging import get_logger

logger = get_logger(__name__)


class AgentOrchestrator:
    """Orchestrator that fans tasks out to sub-agents and merges their work.

    Args:
        max_concurrent: Hard cap on agents running simultaneously during any
            fan-out call; extra agents queue behind the semaphore.
    """

    def __init__(self, max_concurrent: int = 10) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.agents: dict[str, BaseAgent] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._initialized = False

    async def initialize(self) -> None:
        """РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ Рё СЂРµРіРёСЃС‚СЂР°С†РёСЏ Р°РіРµРЅС‚РѕРІ РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ."""
        if self._initialized:
            return

        default_agents = [
            ArchitectAgent(),
            DeveloperAgent(),
            SecurityAgent(),
            DevOpsAgent(),
            DBAAgent(),
            AIEngineerAgent(),
        ]

        for agent in default_agents:
            self.agents[agent.name] = agent

        self._initialized = True
        logger.info(f"AgentOrchestrator инициализирован с {len(self.agents)} агентами")

    async def shutdown(self) -> None:
        """Завершение работы."""
        self.agents.clear()
        self._initialized = False

    def register_agent(self, agent: BaseAgent) -> None:
        """Регистрация нового агента."""
        self.agents[agent.name] = agent
        logger.info(f"Зарегистрирован агент: {agent.name} (role={agent.role.value})")

    async def get_analyses(self, task: Task) -> dict[str, dict]:
        """Collect analyses from all agents, concurrently.

        Complexity: ``O(A)`` coroutines, wall time bounded by the slowest
        agent (max ``max_concurrent`` in flight).
        """
        return await self._gather_agent_results(
            {name: agent.analyze(task) for name, agent in self.agents.items()}
        )

    async def get_contributions(
        self,
        task: Task,
        solution: Solution,
        context: dict[str, Any],
    ) -> dict[str, dict]:
        """Collect per-agent contributions, concurrently.

        Complexity: ``O(A)`` coroutines, wall time bounded by the slowest
        agent (max ``max_concurrent`` in flight).
        """
        return await self._gather_agent_results(
            {name: agent.contribute(task, solution, context) for name, agent in self.agents.items()}
        )

    async def review_solution(self, solution: Solution) -> dict[str, Any]:
        """Review the solution with every agent, concurrently.

        Complexity: ``O(A)`` coroutines, wall time bounded by the slowest
        agent (max ``max_concurrent`` in flight).
        """
        return await self._gather_agent_results(
            {name: agent.review(solution) for name, agent in self.agents.items()}
        )

    async def _gather_agent_results(
        self,
        coro_by_name: dict[str, Any],
    ) -> dict[str, dict]:
        """Run every agent coroutine concurrently, capturing failures.

        Each agent's exception is converted into a ``{"error": ...}`` result so
        one failing agent cannot abort the whole fan-out.

        Complexity: ``O(A)`` coroutines scheduled; wall time ``O(max_agent)``
        under the ``max_concurrent`` semaphore.
        """

        async def _run(name: str, coro: Any) -> tuple[str, dict]:
            async with self._semaphore:
                try:
                    return name, await coro
                except Exception as e:
                    logger.error("agent_failed", agent=name, error=str(e))
                    return name, {"error": str(e)}

        results = await asyncio.gather(*(_run(name, coro) for name, coro in coro_by_name.items()))
        return dict(results)

    async def assemble_solution(
        self,
        task: Task,
        stack: TechStack,
        architecture: dict[str, Any],
        code_blocks: list[dict[str, Any]],
        optimizations: dict[str, Any],
        security_notes: dict[str, Any],
    ) -> Solution:
        """Assemble the final solution from kernels and agent contributions.

        Complexity: ``O(B + O + S)`` for B code blocks, O optimizations, and
        S security checks, plus one parallel contribution pass.
        """
        # Создаём базовое решение
        solution = Solution(
            task_id=task.id,
            title=f"Решение: {task.title}",
            summary=self._build_summary(task, architecture, stack),
            stack=stack,
        )

        # Архитектура
        if architecture and "pattern" in architecture:
            pattern_data = architecture["pattern"]
            solution.architecture = ArchitecturePattern(**pattern_data)

        # Кодовые блоки
        for block_data in code_blocks:
            if isinstance(block_data, dict) and "filename" in block_data:
                solution.code_blocks.append(
                    CodeBlock(
                        filename=block_data["filename"],
                        language=block_data.get("language", "python"),
                        content=block_data.get("content", ""),
                        description=block_data.get("description", ""),
                    )
                )

        # Оптимизации
        if optimizations:
            for strategy in optimizations.get("strategies", []):
                solution.performance_notes.append(
                    f"[{strategy.get('category', 'general')}] {strategy.get('strategy', 'N/A')}: "
                    f"{strategy.get('description', '')}"
                )

        # Безопасность
        if security_notes:
            for check in security_notes.get("checks", []):
                solution.security_notes.append(
                    f"[{check.get('severity', 'info')}] {check.get('check', 'N/A')}: "
                    f"{check.get('description', '')}"
                )

        # Deployment
        solution.deployment = architecture.get("deployment", {}) if architecture else {}

        # Получаем вклад от агентов
        context = {
            "architecture": architecture,
            "optimizations": optimizations,
            "security": security_notes,
        }
        contributions = await self.get_contributions(task, solution, context)

        solution.metadata["agent_contributions"] = {
            name: contrib.get("layer", "unknown")
            for name, contrib in contributions.items()
            if "error" not in contrib
        }

        return solution

    def _build_summary(self, task: Task, architecture: dict, stack: TechStack) -> str:
        """Построение краткого описания решения."""
        parts = [f"Решение для: {task.title}"]
        parts.append(f"Стек: {stack.summary()}")

        if architecture and "pattern" in architecture:
            pattern = architecture["pattern"]
            parts.append(f"Архитектура: {pattern.get('name', 'N/A')}")

        if architecture and "components" in architecture:
            parts.append(f"Компоненты: {len(architecture['components'])}")

        parts.append(f"Сложность: {task.complexity.value}")
        return " | ".join(parts)

    def get_agent_for_role(self, role: AgentRole) -> BaseAgent | None:
        """Получить агента по роли."""
        for agent in self.agents.values():
            if agent.role == role:
                return agent
        return None
