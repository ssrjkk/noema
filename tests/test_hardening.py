"""Trust-hardening tests: fail-closed LLM errors, judge enforcement, tenant hygiene."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from noema.config.settings import JudgeSettings, SandboxSettings
from noema.context import get_tenant_id, reset_tenant_id, set_tenant_id
from noema.core.engine import NoemaEngine
from noema.core.types import JudgeError, Solution, Task, ThoughtProcess
from noema.llm.providers import BaseLLMProvider, LLMMessage, LLMProviderError, LLMResponse


class _ErrorLLM(BaseLLMProvider):
    """Provider whose responses always carry an error marker."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def name(self) -> str:
        return "erroring"

    @property
    def model_name(self) -> str:
        return "erroring"

    async def _complete(self, messages, temperature=0.7, max_tokens=4096):
        self.calls += 1
        return LLMResponse(
            content="[Fallback mode] looks like content",
            model="erroring",
            error="simulated provider failure",
        )


# ── LLM fail-closed ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_raises_on_error_response():
    llm = _ErrorLLM()
    with pytest.raises(LLMProviderError):
        await llm.complete([LLMMessage(role="user", content="hi")])
    assert llm.calls == 1  # permanent failure: no retry storm


@pytest.mark.asyncio
async def test_complete_error_response_is_not_cached():
    llm = _ErrorLLM()
    with pytest.raises(LLMProviderError):
        await llm.complete([LLMMessage(role="user", content="hi")])
    assert llm.calls == 1


# ── Judge enforcement ──────────────────────────────────────────────────


def _solution(task_id: str = "t-1") -> Solution:
    return Solution(task_id=task_id, title="s", summary="sum")


def _span() -> SimpleNamespace:
    return SimpleNamespace(attributes={})


@pytest.mark.asyncio
async def test_judge_enforce_raises_on_failed_verdict(tmp_path):
    engine = NoemaEngine(project_root=str(tmp_path), tenant_id="ten-j")
    engine._settings = engine._settings.model_copy(update={"judge": JudgeSettings(enforce=True)})
    task = Task(title="t", description="d")
    solution = _solution()
    solution.metadata["judge_passed"] = False

    with pytest.raises(JudgeError):
        await engine._finalize_solution(task, solution, ThoughtProcess(task_id="t-1"), 0.0, _span())


@pytest.mark.asyncio
async def test_judge_not_enforced_by_default(tmp_path):
    engine = NoemaEngine(project_root=str(tmp_path), tenant_id="ten-j")
    task = Task(title="t", description="d")
    solution = _solution()
    solution.metadata["judge_passed"] = False

    await engine._finalize_solution(task, solution, ThoughtProcess(task_id="t-1"), 0.0, _span())


@pytest.mark.asyncio
async def test_judge_enforce_allows_passed_verdict(tmp_path):
    engine = NoemaEngine(project_root=str(tmp_path), tenant_id="ten-j")
    engine._settings = engine._settings.model_copy(update={"judge": JudgeSettings(enforce=True)})
    task = Task(title="t", description="d")
    solution = _solution()
    solution.metadata["judge_passed"] = True

    await engine._finalize_solution(task, solution, ThoughtProcess(task_id="t-1"), 0.0, _span())


# ── Tenant hygiene ─────────────────────────────────────────────────────


def test_engine_init_does_not_mutate_global_tenant():
    before = get_tenant_id()
    NoemaEngine(tenant_id="engine-tenant")
    assert get_tenant_id() == before


@pytest.mark.asyncio
async def test_think_restores_prior_tenant(tmp_path):
    engine = NoemaEngine(project_root=str(tmp_path), tenant_id="engine-tenant")
    token = set_tenant_id("caller-tenant")
    try:
        with pytest.raises(ValueError):
            await engine.think(Task(title="x", description="y" * 100_001))
        assert get_tenant_id() == "caller-tenant"
    finally:
        reset_tenant_id(token)


# ── Defaults ───────────────────────────────────────────────────────────


def test_verify_think_defaults_to_enabled():
    assert SandboxSettings().verify_think is True
    assert SandboxSettings().verify_think_enforce is False
    assert JudgeSettings().enforce is False
