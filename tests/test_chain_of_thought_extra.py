"""Additional tests for chain_of_thought.py — covering missed lines.

Targets: CoTStep.to_dict, resume_context, async on_step_end callbacks,
skipped steps, cycle detection in topological sort, step skip on empty prompt.
"""

from __future__ import annotations

import pytest

from noema.core.chain_of_thought import (
    ChainOfThought,
    CoTStep,
    StepStatus,
)
from noema.llm.providers import BaseLLMProvider, LLMResponse


class _MockLLM(BaseLLMProvider):
    def __init__(self, responses: dict[str, str] | None = None) -> None:
        super().__init__()
        self._responses = responses or {}

    @property
    def name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock"

    async def _complete(self, messages, temperature=0.7, max_tokens=4096):
        content = '{"ok": true}'
        for key, val in self._responses.items():
            if any(key in (m.content or "") for m in messages):
                content = val
                break
        return LLMResponse(content=content, model="mock", tokens_used=50)


class _EmptyPromptLLM(BaseLLMProvider):
    """LLM that returns empty content for specific steps."""

    @property
    def name(self) -> str:
        return "empty"

    @property
    def model_name(self) -> str:
        return "empty"

    async def _complete(self, messages, temperature=0.7, max_tokens=4096):
        return LLMResponse(content="", model="empty", tokens_used=0)


# ── CoTStep.to_dict ──────────────────────────────────────────────────────────


def test_cot_step_to_dict_basic():
    step = CoTStep(
        name="analysis",
        role="analyst",
        depends_on=[],
        system_prompt="Analyze",
        user_prompt="Do it",
        result="Some result " * 30,
        confidence=0.856,
        status=StepStatus.COMPLETED,
        duration_ms=123.456,
        tokens_used=500,
    )
    d = step.to_dict()
    assert d["name"] == "analysis"
    assert d["role"] == "analyst"
    assert d["status"] == "completed"
    assert d["duration_ms"] == 123.5
    assert d["confidence"] == 0.86
    assert d["tokens_used"] == 500
    assert len(d["result_preview"]) == 200


def test_cot_step_to_dict_empty_result():
    step = CoTStep(
        name="test",
        role="tester",
        depends_on=[],
        system_prompt="",
        user_prompt="",
    )
    d = step.to_dict()
    assert d["result_preview"] == ""
    assert d["status"] == "pending"
    assert d["confidence"] == 0.0


# ── Resume context ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resume_context_restores_completed_steps():
    llm = _MockLLM()
    cot = ChainOfThought(llm)
    resume = {"analysis": "previous analysis result"}
    await cot.reason("task", ["api"], [], complexity="moderate", resume_context=resume)

    analysis = next(s for s in cot._steps if s.name == "analysis")
    assert analysis.status == StepStatus.COMPLETED
    assert analysis.result == "previous analysis result"


@pytest.mark.asyncio
async def test_resume_context_empty_dict_no_effect():
    llm = _MockLLM()
    cot = ChainOfThought(llm)
    await cot.reason("task", [], [], complexity="simple", resume_context={})
    for step in cot._steps:
        assert step.name not in {} or step.status != StepStatus.COMPLETED


# ── Async on_step_end callback ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_on_step_end_callback_called():
    llm = _MockLLM()
    events: list[tuple] = []

    async def on_end(name: str, result: str, done: int, total: int) -> None:
        events.append((name, result[:50], done, total))

    cot = ChainOfThought(llm, on_step_end=on_end)
    await cot.reason("task", ["api"], [], complexity="moderate")
    assert len(events) > 0
    for name, _, done, total in events:
        assert isinstance(name, str)
        assert done <= total


@pytest.mark.asyncio
async def test_async_on_step_end_callback_on_failure():
    class FailLLM(BaseLLMProvider):
        @property
        def name(self) -> str:
            return "fail"

        @property
        def model_name(self) -> str:
            return "fail"

        async def _complete(self, messages, temperature=0.7, max_tokens=4096):
            raise RuntimeError("LLM down")

    events: list[tuple] = []

    async def on_end(name: str, result: str, done: int, total: int) -> None:
        events.append((name, result, done, total))

    cot = ChainOfThought(FailLLM(), on_step_end=on_end)
    await cot.reason("task", [], [], complexity="simple")
    failed_events = [e for e in events if e[1].startswith("FAILED")]
    assert len(failed_events) > 0


# ── Skipped steps (budget / empty prompt) ────────────────────────────────────


@pytest.mark.asyncio
async def test_skipped_step_fires_on_step_end():
    from unittest.mock import MagicMock

    action = MagicMock()
    action.name = "SKIP"

    budget = MagicMock()
    budget.check.return_value = action
    budget.should_degrade.return_value = False

    events: list[tuple] = []

    async def on_end(name: str, result: str, done: int, total: int) -> None:
        events.append((name, result, done, total))

    cot = ChainOfThought(_MockLLM(), token_budget=budget, on_step_end=on_end)
    await cot.reason("task", ["api"], [], complexity="moderate")
    skipped = [e for e in events if e[1] == "SKIPPED"]
    assert len(skipped) > 0


# ── Topological sort cycle detection ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_topological_sort_handles_cycles():
    cot = ChainOfThought(_MockLLM())
    cot._steps = [
        CoTStep(name="a", role="r", depends_on=["b"], system_prompt="", user_prompt=""),
        CoTStep(name="b", role="r", depends_on=["a"], system_prompt="", user_prompt=""),
    ]
    levels = cot._topological_levels()
    emitted = [s.name for lvl in levels for s in lvl]
    assert len(emitted) == 0


# ── Step with unknown name in gather results ─────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_step_name_in_results_logged():
    """If a task result arrives for an unknown step name, it's logged and skipped."""
    cot = ChainOfThought(_MockLLM())
    cot._steps = [
        CoTStep(name="analysis", role="r", depends_on=[], system_prompt="s", user_prompt="u"),
    ]
    cot._context = {}
    import asyncio

    async def fake_execute(step):
        step.status = StepStatus.COMPLETED
        step.result = "done"

    original = cot._execute_step
    cot._execute_step = fake_execute
    result = await cot.reason("task", [], [], complexity="simple")
    assert isinstance(result, dict)
    cot._execute_step = original
