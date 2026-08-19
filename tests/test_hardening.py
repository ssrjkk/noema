"""Trust-hardening tests: fail-closed LLM errors, judge enforcement, tenant hygiene."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from noema.billing.quotas import QuotaExceededError, QuotaManager, TenantQuota
from noema.config.settings import JudgeSettings, SandboxSettings
from noema.context import get_tenant_id, reset_tenant_id, set_tenant_id
from noema.core.engine import NoemaEngine
from noema.core.types import (
    CodeBlock,
    JudgeError,
    Solution,
    Task,
    ThinkTimeoutError,
    ThoughtProcess,
)
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


class _TenantRecordingLLM(BaseLLMProvider):
    """Mock LLM that records the ambient tenant on every call."""

    def __init__(self) -> None:
        super().__init__()
        self.tenants: list[str] = []

    @property
    def name(self) -> str:
        return "recorder"

    @property
    def model_name(self) -> str:
        return "recorder"

    async def _complete(self, messages, temperature=0.7, max_tokens=4096):
        self.tenants.append(get_tenant_id())
        return LLMResponse(content="{}", model="recorder", tokens_used=10)


class _SlowLLM(BaseLLMProvider):
    """Mock LLM that sleeps far beyond any test timeout."""

    @property
    def name(self) -> str:
        return "slow"

    @property
    def model_name(self) -> str:
        return "slow"

    async def _complete(self, messages, temperature=0.7, max_tokens=4096):
        await asyncio.sleep(5.0)
        return LLMResponse(content="{}", model="slow", tokens_used=10)


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


# ── Tenant precedence in think() ───────────────────────────────────────


@pytest.mark.asyncio
async def test_think_prefers_caller_tenant_over_engine_default(tmp_path):
    engine = NoemaEngine(project_root=str(tmp_path))
    recorder = _TenantRecordingLLM()
    engine.llm = recorder
    token = set_tenant_id("request-tenant")
    try:
        await engine.think(Task(title="t", description="build a payment service"))
        assert recorder.tenants, "CoT path must call the LLM"
        assert all(t == "request-tenant" for t in recorder.tenants)
        assert get_tenant_id() == "request-tenant"
    finally:
        reset_tenant_id(token)


# ── think() end-to-end timeout ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_think_timeout_raises_fail_closed(tmp_path):
    engine = NoemaEngine(project_root=str(tmp_path))
    engine.llm = _SlowLLM()
    await engine.initialize()
    engine._settings = engine._settings.model_copy(update={"think_timeout_seconds": 0.2})

    with pytest.raises(ThinkTimeoutError):
        await engine.think(Task(title="t", description="will hang forever"))


# ── Quota: max input tokens per task ───────────────────────────────────


@pytest.mark.asyncio
async def test_quota_enforces_max_input_tokens():
    manager = QuotaManager()
    await manager.set_quota(
        "limited", TenantQuota(tenant_id="limited", max_input_tokens_per_task=1000)
    )

    with pytest.raises(QuotaExceededError):
        await manager.check_quota("limited", estimated_input_tokens=2000)
    assert await manager.check_quota("limited", estimated_input_tokens=500) is True
    assert await manager.check_quota("limited") is True  # no estimate → no check


# ── Sandbox gate language normalization ────────────────────────────────


@pytest.mark.asyncio
async def test_verify_think_normalizes_python_language(tmp_path):
    engine = NoemaEngine(project_root=str(tmp_path))
    solution = _solution()
    solution.code_blocks.append(
        CodeBlock(filename="a.py", language="Python", content="def f():\n    return 1\n")
    )

    verdict = await engine._verify_think_solution(solution)

    assert verdict["enabled"] is True
    assert len(verdict["files"]) == 1
    assert verdict["passed"] is True
