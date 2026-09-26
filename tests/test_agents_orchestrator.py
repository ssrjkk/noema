"""Tests for agents/orchestrator.py — fan-out, assemble, role lookup."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from noema.agents.orchestrator import AgentOrchestrator
from noema.core.types import (
    AgentRole,
    Solution,
    Task,
    TaskComplexity,
    TechStack,
)


@pytest.fixture
def task():
    return Task(title="Test task", description="desc", complexity=TaskComplexity.SIMPLE)


@pytest.fixture
def solution(task):
    return Solution(task_id=task.id, title="Test solution", summary="")


def _make_agent(name: str, role: AgentRole, analyze_result: dict | None = None):
    agent = MagicMock()
    agent.name = name
    agent.role = role
    agent.expertise = ["python"]
    agent.analyze = AsyncMock(return_value=analyze_result or {"analysis": f"{name} ok"})
    agent.contribute = AsyncMock(return_value={"layer": f"{name}_layer"})
    agent.review = AsyncMock(return_value={"verdict": "approve"})
    return agent


@pytest.mark.asyncio
async def test_initialize_registers_default_agents():
    orch = AgentOrchestrator()
    await orch.initialize()
    assert len(orch.agents) == 6
    assert orch._initialized is True
    await orch.shutdown()


@pytest.mark.asyncio
async def test_initialize_idempotent():
    orch = AgentOrchestrator()
    await orch.initialize()
    count = len(orch.agents)
    await orch.initialize()
    assert len(orch.agents) == count
    await orch.shutdown()


@pytest.mark.asyncio
async def test_shutdown_clears_agents():
    orch = AgentOrchestrator()
    await orch.initialize()
    await orch.shutdown()
    assert len(orch.agents) == 0
    assert orch._initialized is False


def test_invalid_max_concurrent():
    with pytest.raises(ValueError, match="max_concurrent must be >= 1"):
        AgentOrchestrator(max_concurrent=0)


@pytest.mark.asyncio
async def test_register_agent():
    orch = AgentOrchestrator()
    agent = _make_agent("custom", AgentRole.DEVELOPER)
    orch.register_agent(agent)
    assert "custom" in orch.agents
    assert orch.agents["custom"] is agent


@pytest.mark.asyncio
async def test_get_analyses(task):
    orch = AgentOrchestrator()
    orch.agents["arch"] = _make_agent("arch", AgentRole.ARCHITECT, {"pattern": "microservices"})
    orch.agents["dev"] = _make_agent("dev", AgentRole.DEVELOPER, {"stack": "python"})

    results = await orch.get_analyses(task)
    assert "arch" in results
    assert "dev" in results
    assert results["arch"]["pattern"] == "microservices"
    assert results["dev"]["stack"] == "python"


@pytest.mark.asyncio
async def test_get_analyses_captures_failures(task):
    orch = AgentOrchestrator()
    good = _make_agent("good", AgentRole.DEVELOPER)
    bad = _make_agent("bad", AgentRole.SECURITY)
    bad.analyze = AsyncMock(side_effect=RuntimeError("analysis boom"))
    orch.agents = {"good": good, "bad": bad}

    results = await orch.get_analyses(task)
    assert "error" not in results["good"]
    assert "error" in results["bad"]
    assert "analysis boom" in results["bad"]["error"]


@pytest.mark.asyncio
async def test_get_contributions(task, solution):
    orch = AgentOrchestrator()
    orch.agents["arch"] = _make_agent("arch", AgentRole.ARCHITECT)
    context = {"architecture": {}}

    results = await orch.get_contributions(task, solution, context)
    assert "arch" in results
    assert results["arch"]["layer"] == "arch_layer"


@pytest.mark.asyncio
async def test_review_solution(solution):
    orch = AgentOrchestrator()
    orch.agents["sec"] = _make_agent("sec", AgentRole.SECURITY)

    results = await orch.review_solution(solution)
    assert "sec" in results
    assert results["sec"]["verdict"] == "approve"


@pytest.mark.asyncio
async def test_assemble_solution(task):
    orch = AgentOrchestrator()
    agent = _make_agent("dev", AgentRole.DEVELOPER, {"layer": "backend"})
    orch.agents = {"dev": agent}

    stack = TechStack(languages=["Python"], frameworks=["FastAPI"])
    architecture = {
        "pattern": {"name": "Layered", "description": "3-tier"},
        "components": ["api", "db", "cache"],
        "deployment": {"target": "docker"},
    }
    code_blocks = [
        {
            "filename": "main.py",
            "language": "python",
            "content": "print('hi')",
            "description": "entry",
        },
    ]
    optimizations = {
        "strategies": [{"category": "cache", "strategy": "redis", "description": "cache layer"}],
    }
    security_notes = {
        "checks": [{"severity": "high", "check": "auth", "description": "add RBAC"}],
    }

    result = await orch.assemble_solution(
        task, stack, architecture, code_blocks, optimizations, security_notes
    )
    assert isinstance(result, Solution)
    assert result.task_id == task.id
    assert result.architecture is not None
    assert result.architecture.name == "Layered"
    assert len(result.code_blocks) == 1
    assert result.code_blocks[0].filename == "main.py"
    assert len(result.performance_notes) == 1
    assert "cache" in result.performance_notes[0]
    assert len(result.security_notes) == 1
    assert "RBAC" in result.security_notes[0]
    assert result.deployment == {"target": "docker"}


@pytest.mark.asyncio
async def test_assemble_solution_empty_architecture(task):
    orch = AgentOrchestrator()
    orch.agents = {}
    stack = TechStack(languages=["Go"])

    result = await orch.assemble_solution(task, stack, {}, [], {}, {})
    assert result.architecture is None
    assert len(result.code_blocks) == 0
    assert result.deployment == {}


@pytest.mark.asyncio
async def test_assemble_solution_skips_code_without_filename(task):
    orch = AgentOrchestrator()
    orch.agents = {}
    stack = TechStack(languages=["Python"])

    result = await orch.assemble_solution(
        task,
        stack,
        {},
        [{"content": "no filename"}, {"filename": "ok.py", "content": "x = 1"}],
        {},
        {},
    )
    assert len(result.code_blocks) == 1
    assert result.code_blocks[0].filename == "ok.py"


def test_get_agent_for_role():
    orch = AgentOrchestrator()
    dev = _make_agent("dev", AgentRole.DEVELOPER)
    sec = _make_agent("sec", AgentRole.SECURITY)
    orch.agents = {"dev": dev, "sec": sec}

    found = orch.get_agent_for_role(AgentRole.SECURITY)
    assert found is sec

    not_found = orch.get_agent_for_role(AgentRole.DBA)
    assert not_found is None


def test_build_summary():
    orch = AgentOrchestrator()
    task = Task(title="Build API", complexity=TaskComplexity.MODERATE)
    stack = TechStack(languages=["Python"], frameworks=["FastAPI"])
    arch = {
        "pattern": {"name": "Layered"},
        "components": ["a", "b", "c"],
    }
    summary = orch._build_summary(task, arch, stack)
    assert "Build API" in summary
    assert "Layered" in summary
    assert "3" in summary
    assert "moderate" in summary


def test_build_summary_no_architecture():
    orch = AgentOrchestrator()
    task = Task(title="Simple", complexity=TaskComplexity.TRIVIAL)
    stack = TechStack(languages=["Rust"])
    summary = orch._build_summary(task, {}, stack)
    assert "Simple" in summary
    assert "trivial" in summary
