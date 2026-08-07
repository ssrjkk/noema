"""Hypothesis property tests for the refactored DAG Chain-of-Thought engine.

Covers ``noema/core/chain_of_thought.py``:
- planner invariants (valid step universe, tag-driven inclusion, reflexion repair),
- topological-level correctness (dependencies precede dependents, no loss/dup),
- context compression bounds,
- DAG execution ordering with a mocked LLM provider.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from noema.core.chain_of_thought import (
    ChainOfThought,
    StepPlanner,
    StepStatus,
)
from noema.llm.providers import BaseLLMProvider, LLMResponse

ALL_STEPS = StepPlanner.ALL_STEPS

TAG_TEXT = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=16
)


class _MockLLM(BaseLLMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock"

    async def _complete(self, messages, temperature=0.7, max_tokens=4096):
        self.calls.append(messages[-1].content[:80] if messages else "")
        return LLMResponse(content='{"ok": true}', model="mock", tokens_used=100)


# ── StepPlanner invariants ───────────────────────────────────────────


@given(
    tags=st.lists(TAG_TEXT, max_size=8),
    complexity=st.sampled_from(["trivial", "simple", "moderate", "complex", "extreme"]),
    error=st.text(max_size=60),
    has_past=st.booleans(),
)
@settings(max_examples=100)
def test_plan_steps_valid_universe(tags, complexity, error, has_past):
    planner = StepPlanner()
    steps = planner.plan(
        tags, complexity, error_context=error, past_attempts=["x"] if has_past else None
    )
    assert isinstance(steps, list)
    assert all(s in ALL_STEPS for s in steps)
    assert len(steps) == len(set(steps))  # no duplicates
    assert steps[0] == "analysis" or (has_past and error and steps[0] == "repair")


@given(
    tags=st.lists(TAG_TEXT, max_size=8),
)
@settings(max_examples=50)
def test_plan_trivial_is_minimal(tags):
    planner = StepPlanner()
    assert planner.plan(tags, "trivial") == StepPlanner.MINIMAL_STEPS
    assert planner.plan(tags, "SIMPLE") == StepPlanner.MINIMAL_STEPS


@given(
    tags=st.lists(
        st.sampled_from(
            [
                "api",
                "web",
                "rest",
                "data",
                "database",
                "test",
                "qa",
                "security",
                "auth",
                "performance",
                "high-load",
            ]
        ),
        max_size=10,
    )
)
@settings(max_examples=100)
def test_plan_tag_driven_inclusion(tags):
    planner = StepPlanner()
    steps = planner.plan(tags, "moderate")
    lowered = {t.lower() for t in tags}
    if lowered & {"api", "web", "rest"}:
        assert "api_design" in steps
    if lowered & {"data", "database"}:
        assert "data_model" in steps
    if lowered & {"test", "qa"}:
        assert "testing" in steps
    if lowered & {"security", "auth"}:
        assert "security" in steps
    if lowered & {"performance", "high-load"}:
        assert "optimization" in steps
    assert steps[-1] == "review"


@given(error=st.text(min_size=1))
@settings(max_examples=30)
def test_plan_reflexion_injects_repair(error):
    planner = StepPlanner()
    steps = planner.plan(["api"], "moderate", error_context=error, past_attempts=["a"])
    assert "repair" in steps
    assert steps.index("repair") == 1  # right after analysis


# ── Topological levels ───────────────────────────────────────────────


@given(
    subset=st.lists(st.sampled_from(ALL_STEPS), max_size=20),
    tags=st.lists(
        st.sampled_from(
            [
                "api",
                "web",
                "data",
                "database",
                "test",
                "security",
                "performance",
                "qa",
                "auth",
                "high-load",
            ]
        ),
        max_size=8,
    ),
)
@settings(max_examples=60)
def test_topological_levels_partition_and_order(subset, tags):
    cot = ChainOfThought(_MockLLM())
    planned = list(dict.fromkeys(subset))  # dedupe: DAG keeps one step per name
    planned = planned or ["analysis"]
    cot._build_dag(planned, "t", tags, [], "", "")
    levels = cot._topological_levels()

    emitted = [step.name for level in levels for step in level]
    assert len(emitted) == len(set(emitted)) == len(planned)
    assert set(emitted) == set(planned)

    # dependency ordering: each step's deps appear strictly before it
    index = {name: i for i, name in enumerate(emitted)}
    for step in cot._steps:
        for dep in step.depends_on:
            assert index[dep] < index[step.name]


@given(
    subset=st.lists(st.sampled_from(ALL_STEPS), max_size=20),
)
@settings(max_examples=30)
def test_topological_levels_single_steps_run_first(subset):
    cot = ChainOfThought(_MockLLM())
    planned = subset or ["analysis"]
    cot._build_dag(planned, "t", [], [], "", "")
    levels = cot._topological_levels()
    assert levels[0] == [step for step in cot._steps if not step.depends_on]


# ── Context compression ──────────────────────────────────────────────


@given(text=st.text(max_size=2000), max_chars=st.integers(min_value=1, max_value=2000))
@settings(max_examples=100)
def test_compress_bound(text, max_chars):
    cot = ChainOfThought(_MockLLM())
    out = cot._compress(text, max_chars)
    assert isinstance(out, str)
    # truncation appends a "..." suffix, so the hard bound is max_chars + 3
    assert len(out) <= max_chars + 3


@given(
    payload=st.dictionaries(
        st.text(min_size=1, max_size=12),
        st.one_of(st.text(max_size=300), st.integers(), st.lists(st.integers(), max_size=10)),
        max_size=8,
    ),
    max_chars=st.integers(min_value=50, max_value=2000),
)
@settings(max_examples=100)
def test_compress_json_stays_bounded(payload, max_chars):
    cot = ChainOfThought(_MockLLM())
    import json

    text = json.dumps(payload)
    out = cot._compress(text, max_chars)
    if len(text) <= max_chars:
        assert out == text
    else:
        assert len(out) <= max_chars + 3  # truncation suffix


# ── DAG execution ────────────────────────────────────────────────────


@pytest.mark.asyncio
@settings(max_examples=20, deadline=None)
@given(
    tags=st.lists(
        st.sampled_from(["api", "web", "data", "test", "security", "performance"]), max_size=6
    ),
)
async def test_reason_runs_all_steps_in_order(tags):
    llm = _MockLLM()
    cot = ChainOfThought(llm)
    context = await cot.reason("build an api", tags, [], complexity="moderate")

    planned = {step.name for step in cot._steps}
    assert planned <= set(StepPlanner.ALL_STEPS)
    # every planned step executed (not skipped) and its result stored
    for step in cot._steps:
        assert step.status in (StepStatus.COMPLETED, StepStatus.FAILED)
        if step.status == StepStatus.COMPLETED:
            assert context.get(step.name)


@pytest.mark.asyncio
async def test_reason_respects_dependency_order():
    llm = _MockLLM()
    cot = ChainOfThought(llm)
    order: list[str] = []

    async def track(step_name: str, label: str, done: int, total: int) -> None:
        if step_name != "planner":
            order.append(step_name)

    cot.on_step_start = track
    await cot.reason("task", ["api"], [], complexity="moderate")

    index = {name: i for i, name in enumerate(order)}
    for step in cot._steps:
        for dep in step.depends_on:
            assert index[dep] < index[step.name]


@pytest.mark.asyncio
async def test_reason_step_failure_is_captured():
    class _FailingLLM(_MockLLM):
        async def _complete(self, messages, temperature=0.7, max_tokens=4096):
            if any("analysis" in (m.content or "") for m in messages):
                raise RuntimeError("provider down")
            return LLMResponse(content="{}", model="mock", tokens_used=10)

    cot = ChainOfThought(_FailingLLM())
    await cot.reason("t", [], [], complexity="moderate")
    analysis = next(s for s in cot._steps if s.name == "analysis")
    assert analysis.status == StepStatus.FAILED
    assert "Error" in analysis.result
