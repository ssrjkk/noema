"""Coverage for agent subclasses in noema.agents.base."""

import pytest

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
    CodeBlock,
    Solution,
    Task,
    TaskComplexity,
)


class _ConcreteAgent(BaseAgent):
    @property
    def expertise(self) -> list[str]:
        return ["generic"]

    async def analyze(self, task: Task) -> dict[str, object]:
        return {"ok": True}

    async def contribute(
        self,
        task: Task,
        solution: Solution,
        context: dict[str, object],
    ) -> dict[str, object]:
        return {"ok": True}


def _task(
    *,
    title: str = "Some task",
    tags: list[str] | None = None,
    complexity: TaskComplexity = TaskComplexity.MODERATE,
) -> Task:
    return Task(title=title, tags=tags or [], complexity=complexity)


def _solution(*, security_notes: list[str] | None = None) -> Solution:
    return Solution(
        task_id="t1",
        title="Solution",
        summary="Summary",
        security_notes=security_notes or [],
    )


def test_base_agent_default_name_from_role():
    agent = _ConcreteAgent(AgentRole.REVIEWER)
    assert agent.name == "reviewer_agent"
    assert agent.role == AgentRole.REVIEWER
    assert agent._history == []


def test_base_agent_explicit_name():
    agent = _ConcreteAgent(AgentRole.REVIEWER, name="custom")
    assert agent.name == "custom"


@pytest.mark.asyncio
async def test_base_agent_default_review():
    agent = _ConcreteAgent(AgentRole.ANALYST)
    review = await agent.review(_solution())
    assert review["approved"] is True
    assert review["agent"] == agent.name
    assert review["role"] == "analyst"
    assert review["comments"] == []
    assert review["suggestions"] == []


@pytest.mark.asyncio
async def test_architect_classifies_domain_and_scale():
    architect = ArchitectAgent()
    web = await architect.analyze(_task(tags=["web", "high-load"]))
    assert web["domain"] == "web-application"
    assert web["scale"] == "high"

    ml = await architect.analyze(_task(tags=["ai"]))
    assert ml["domain"] == "ml-system"
    assert ml["scale"] == "standard"

    data = await architect.analyze(_task(tags=["data", "enterprise"]))
    assert data["domain"] == "data-platform"
    assert data["scale"] == "enterprise"

    general = await architect.analyze(_task())
    assert general["domain"] == "general"
    assert general["scale"] == "standard"


@pytest.mark.asyncio
async def test_architect_expertise_and_contribute():
    architect = ArchitectAgent()
    assert architect.expertise == ["system-design", "patterns", "scalability", "trade-offs"]
    contribution = await architect.contribute(
        _task(), _solution(), {"components": ["a", "b"]}
    )
    assert contribution["layer"] == "architecture"
    assert contribution["component_count"] == 2


@pytest.mark.asyncio
async def test_developer_analyze_and_contribute():
    developer = DeveloperAgent()
    task = _task(tags=["api"], complexity=TaskComplexity.COMPLEX)
    analysis = await developer.analyze(task)
    assert analysis["modules_needed"] == 0
    assert analysis["complexity_assessment"] == "complex"
    assert "repository" in analysis["recommended_patterns"]

    solution = _solution()
    solution.code_blocks.append(
        CodeBlock(filename="a.py", language="python", content="x = 1")
    )
    contribution = await developer.contribute(task, solution, {})
    assert contribution["layer"] == "implementation"
    assert contribution["files_generated"] == 1


@pytest.mark.asyncio
async def test_security_analyze_attack_surface():
    agent = SecurityAgent()
    assert agent.name == "security_specialist"
    web = await agent.analyze(_task(tags=["web", "api"]))
    assert "xss" in web["attack_surface"]
    assert "sqli" in web["attack_surface"]
    assert "csrf" in web["attack_surface"]
    assert "rate-limiting" in web["attack_surface"]
    assert "brute-force" not in web["attack_surface"]

    auth = await agent.analyze(_task(tags=["auth"]))
    assert "brute-force" in auth["attack_surface"]
    assert "token-theft" in auth["attack_surface"]
    assert "session-hijack" in auth["attack_surface"]

    bare = await agent.analyze(_task())
    assert bare["attack_surface"] == []


@pytest.mark.asyncio
async def test_security_contribute():
    agent = SecurityAgent()
    contribution = await agent.contribute(_task(), _solution(), {})
    assert contribution["layer"] == "security"
    assert "input-validation" in contribution["checks_added"]


@pytest.mark.asyncio
async def test_security_review_rejects_vulnerability_notes():
    agent = SecurityAgent()
    clean = await agent.review(_solution(security_notes=["checked OWASP", "no issues"]))
    assert clean["approved"] is True
    assert clean["issues"] == []

    dirty = await agent.review(_solution(security_notes=["Found a vulnerability in auth"]))
    assert dirty["approved"] is False
    assert dirty["issues"] == ["Found a vulnerability in auth"]


@pytest.mark.asyncio
async def test_devops_analyze_and_contribute():
    agent = DevOpsAgent()
    assert agent.name == "devops_engineer"
    assert "ci-cd" in agent.expertise
    analysis = await agent.analyze(_task())
    assert analysis["deployment_target"] == "kubernetes"
    assert analysis["ci_cd"] == "github-actions"
    assert "prometheus" in analysis["monitoring_stack"]

    contribution = await agent.contribute(_task(), _solution(), {})
    assert contribution["layer"] == "infrastructure"
    assert "Dockerfile" in contribution["configs_generated"]


@pytest.mark.asyncio
async def test_dba_analyze_and_contribute():
    agent = DBAAgent()
    assert agent.name == "database_architect"
    analysis = await agent.analyze(_task())
    assert analysis["recommended_db"] == "postgresql"
    assert analysis["consistency_requirements"] == "eventual"

    contribution = await agent.contribute(_task(), _solution(), {})
    assert contribution["layer"] == "database"
    assert contribution["tables"] == []
    assert contribution["migrations"] == []


@pytest.mark.asyncio
async def test_ai_engineer_analyze_and_contribute():
    agent = AIEngineerAgent()
    assert agent.name == "ml_engineer"
    analysis = await agent.analyze(_task())
    assert analysis["model_type"] == "unknown"
    assert analysis["inference_requirements"] == "real-time"

    contribution = await agent.contribute(_task(), _solution(), {})
    assert contribution["layer"] == "ml"
    assert "training" in contribution["pipeline_steps"]
