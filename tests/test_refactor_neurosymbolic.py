"""Hypothesis property tests for the refactored neurosymbolic engine and symbolic layer.

Covers the hardening directives applied to
``noema/neurosymbolic/engine.py`` and ``symbolic.py``:
strict input validation, streamed-event invariants, bounded refinement,
deterministic source hashing, priority coercion and solver-pool hygiene.
"""

from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from noema.neurosymbolic.engine import (
    NeuroSymbolicEngine,
    NeuroSymbolicTaskInput,
    _coerce_priority,
)
from noema.neurosymbolic.symbolic import SymbolicEngine, TaskGraph

try:
    import z3  # noqa: F401

    has_z3 = True
except ImportError:
    has_z3 = False

_TERMINAL_STAGES = frozenset({"completed", "failed", "error"})
_KNOWN_STAGES = frozenset(
    {
        "parsing",
        "hypothesis_generation",
        "verification",
        "refinement",
        "causal_analysis",
        "completed",
        "failed",
        "error",
    }
)


@st.composite
def structured_task(draw):
    """Build a well-formed task dict accepted by the strict engine input model."""
    requirement = st.fixed_dictionaries(
        {
            "name": st.text(min_size=1, max_size=16),
            "type": st.sampled_from(["numeric", "boolean", "string"]),
            "min": st.integers(),
            "max": st.integers(),
            "priority": st.integers(min_value=0, max_value=10),
        }
    )
    constraint = st.fixed_dictionaries(
        {
            "name": st.text(min_size=1, max_size=16),
            "condition": st.text(min_size=1, max_size=32),
        }
    )
    return draw(
        st.fixed_dictionaries(
            {
                "requirements": st.lists(requirement, max_size=6),
                "constraints": st.lists(constraint, max_size=6),
                "goals": st.lists(st.text(max_size=24), max_size=3),
                "variables": st.dictionaries(
                    st.text(min_size=1, max_size=16), st.integers(), max_size=6
                ),
            }
        )
    )


def _mock_collaborators(engine, valid: bool) -> None:
    engine.symbolic.parse_task = AsyncMock(return_value=TaskGraph())
    engine.neural.generate_hypothesis = AsyncMock(return_value={"result": "ok"})
    engine.neural.refine_hypothesis = AsyncMock(return_value={"result": "refined"})
    if valid:
        engine.symbolic.verify_solution = AsyncMock(return_value=(True, []))
    else:
        engine.symbolic.verify_solution = AsyncMock(return_value=(False, ["violated"]))


# ── Engine: streamed-event invariants ────────────────────────────────


@pytest.mark.asyncio
@settings(max_examples=50, deadline=None)
@given(task=structured_task())
async def test_think_stream_invariants(task):
    engine = NeuroSymbolicEngine(max_refinement_attempts=3, enable_causal=False)
    await engine.start()
    _mock_collaborators(engine, valid=True)

    events = []
    async for event in engine.think(task):
        events.append(event)

    # every event carries a stable correlation id for the whole stream
    correlation_ids = {event["correlation_id"] for event in events}
    assert len(correlation_ids) == 1

    # all stages belong to the closed set and exactly one terminal event fires
    assert all(event["stage"] in _KNOWN_STAGES for event in events)
    terminals = [event for event in events if event["stage"] in _TERMINAL_STAGES]
    assert len(terminals) == 1
    assert terminals[0]["stage"] == "completed"

    # deterministic pipeline ordering
    order = [
        event["stage"]
        for event in events
        if event["stage"] in ("parsing", "hypothesis_generation", "verification", "completed")
    ]
    assert order == sorted(
        order,
        key={
            "parsing": 0,
            "hypothesis_generation": 1,
            "verification": 2,
            "completed": 3,
        }.__getitem__,
    )

    await engine.stop()


@pytest.mark.asyncio
@settings(max_examples=30, deadline=None)
@given(task=structured_task())
async def test_think_completed_event_carries_solution(task):
    engine = NeuroSymbolicEngine(max_refinement_attempts=3, enable_causal=False)
    await engine.start()
    _mock_collaborators(engine, valid=True)

    completed = None
    async for event in engine.think(task):
        if event["stage"] == "completed":
            completed = event
    assert completed is not None
    assert completed["solution"] == {"result": "ok"}
    assert completed["attempts"] == 1

    metrics = engine.get_metrics()
    assert metrics["tasks_processed"] == 1
    assert metrics["tasks_successful"] == 1
    assert metrics["tasks_failed"] == 0
    assert 0.0 <= metrics["success_rate"] <= 1.0
    await engine.stop()


@pytest.mark.asyncio
@settings(max_examples=30, deadline=None)
@given(attempts=st.integers(min_value=1, max_value=5), task=structured_task())
async def test_refinement_bounded_by_max_attempts(attempts, task):
    engine = NeuroSymbolicEngine(max_refinement_attempts=attempts, enable_causal=False)
    await engine.start()
    _mock_collaborators(engine, valid=False)

    events = []
    async for event in engine.think(task):
        events.append(event)

    verifications = [
        event
        for event in events
        if event["stage"] == "verification" and event["status"] == "started"
    ]
    assert len(verifications) == attempts

    refinements = [
        event for event in events if event["stage"] == "refinement" and event["status"] == "started"
    ]
    assert len(refinements) == attempts - 1

    failed = [event for event in events if event["stage"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["attempts"] == attempts

    # the refine step is never invoked on the final (exhausted) round
    assert engine.neural.refine_hypothesis.await_count == attempts - 1
    assert engine.get_metrics()["tasks_failed"] == 1
    await engine.stop()


@pytest.mark.asyncio
@settings(max_examples=20, deadline=None)
@given(task=structured_task())
async def test_think_requires_started_engine(task):
    engine = NeuroSymbolicEngine()
    with pytest.raises(RuntimeError, match="not started"):
        async for _ in engine.think(task):
            pass


@pytest.mark.asyncio
@settings(max_examples=20, deadline=None)
@given(task=structured_task())
async def test_think_streams_error_and_reraised(task):
    engine = NeuroSymbolicEngine(enable_causal=False)
    await engine.start()
    engine.symbolic.parse_task = AsyncMock(side_effect=ValueError("boom"))

    events = []
    with pytest.raises(ValueError, match="boom"):
        async for event in engine.think(task):
            events.append(event)

    errors = [event for event in events if event["stage"] == "error"]
    assert len(errors) == 1
    assert errors[0]["error"] == "boom"
    assert errors[0]["error_type"] == "ValueError"
    assert engine.get_metrics()["tasks_failed"] == 1
    await engine.stop()


# ── Engine: fail-fast configuration ──────────────────────────────────


@given(bad_attempts=st.integers(max_value=0))
def test_init_rejects_non_positive_refinement_attempts(bad_attempts):
    with pytest.raises(ValueError, match="max_refinement_attempts"):
        NeuroSymbolicEngine(max_refinement_attempts=bad_attempts)


@given(bad_timeout=st.floats(max_value=0.0, allow_nan=False))
def test_init_rejects_non_positive_timeout(bad_timeout):
    with pytest.raises(ValueError, match="verification_timeout"):
        NeuroSymbolicEngine(verification_timeout=bad_timeout)


@given(bad_count=st.integers(max_value=-1))
def test_init_rejects_negative_counterfactuals(bad_count):
    with pytest.raises(ValueError, match="max_counterfactuals"):
        NeuroSymbolicEngine(max_counterfactuals=bad_count)


# ── Engine: strict zero-trust input validation ───────────────────────


@given(task=structured_task())
def test_strict_input_round_trips(task):
    model = NeuroSymbolicTaskInput.model_validate(task)
    assert model.model_dump() == task


@given(
    key=st.text(min_size=1).filter(
        lambda k: k not in {"requirements", "constraints", "goals", "variables"}
    ),
    value=st.integers(),
)
def test_strict_input_rejects_unknown_keys(key, value):
    with pytest.raises(ValidationError):
        NeuroSymbolicTaskInput.model_validate({"requirements": [], key: value})


@given(items=st.lists(st.integers(), min_size=1))
def test_strict_input_rejects_non_dict_requirements(items):
    with pytest.raises(ValidationError):
        NeuroSymbolicTaskInput.model_validate({"requirements": items})


@given(value=st.integers())
def test_strict_input_rejects_non_string_goal(value):
    with pytest.raises(ValidationError):
        NeuroSymbolicTaskInput.model_validate({"goals": [value]})


# ── Engine: priority coercion ────────────────────────────────────────


@given(value=st.integers())
def test_coerce_priority_bounded(value):
    priority = _coerce_priority(value)
    assert isinstance(priority, int)
    assert 0 <= priority <= 10


@given(value=st.integers(min_value=0, max_value=10))
def test_coerce_priority_identity(value):
    assert _coerce_priority(value) == value


@given(
    value=st.one_of(
        st.none(),
        st.just(float("nan")),
        st.just(float("inf")),
        st.just(float("-inf")),
        st.lists(st.integers()),
        st.dictionaries(st.text(), st.integers()),
        st.text(min_size=1).filter(lambda s: not (s.isdigit() or s.lstrip("+-").isdigit())),
    )
)
def test_coerce_priority_junk_defaults(value):
    assert _coerce_priority(value) == 5


# ── Symbolic: deterministic source hashing ───────────────────────────


@pytest.mark.asyncio
@settings(max_examples=30, deadline=None)
@given(task=structured_task())
async def test_parse_task_source_hash_stable(task):
    engine = SymbolicEngine()
    graph_a = await engine.parse_task(task)
    graph_b = await engine.parse_task(task)

    source_hash = graph_a.metadata["source_hash"]
    assert isinstance(source_hash, str)
    assert len(source_hash) == 64
    assert source_hash == graph_b.metadata["source_hash"]

    if task["requirements"]:
        requirements = list(task["requirements"])
        first = dict(requirements[0])
        first["name"] = first["name"] + "_mutated"
        mutated_task = dict(task)
        mutated_task["requirements"] = [first] + requirements[1:]
        mutated_graph = await engine.parse_task(mutated_task)
        assert mutated_graph.metadata["source_hash"] != source_hash


@pytest.mark.asyncio
@settings(max_examples=30, deadline=None)
@given(task=structured_task())
async def test_parse_task_hash_cross_engine_deterministic(task):
    engine_a = SymbolicEngine()
    engine_b = SymbolicEngine()

    hash_a = (await engine_a.parse_task(task)).metadata["source_hash"]
    hash_b = (await engine_b.parse_task(task)).metadata["source_hash"]
    assert hash_a == hash_b


# ── Symbolic: solver pool hygiene (z3 required) ──────────────────────


@pytest.mark.skipif(not has_z3, reason="z3 not installed")
@pytest.mark.asyncio
@settings(max_examples=15, deadline=None)
@given(solution=st.dictionaries(st.text(min_size=1, max_size=8), st.integers(), max_size=4))
async def test_verify_solution_returns_solver_to_pool(solution):
    engine = SymbolicEngine(verification_timeout=5.0)
    await engine.initialize()
    pool_size = engine._solver_pool.qsize()
    assert pool_size == 10

    graph = TaskGraph(metadata={"source_hash": "x" * 64})
    is_valid, violations = await engine.verify_solution(solution, graph)

    assert isinstance(is_valid, bool)
    assert isinstance(violations, list)
    assert all(isinstance(v, str) for v in violations)
    assert engine._solver_pool.qsize() == pool_size


# ── Integration: metric bookkeeping matches terminal outcome ─────────


@pytest.mark.asyncio
@settings(max_examples=20, deadline=None)
@given(task=structured_task())
async def test_metrics_balance_after_failed_stream(task):
    engine = NeuroSymbolicEngine(max_refinement_attempts=2, enable_causal=False)
    await engine.start()
    _mock_collaborators(engine, valid=False)

    async for _ in engine.think(task):
        pass

    metrics = engine.get_metrics()
    assert metrics["tasks_processed"] == metrics["tasks_successful"] + metrics["tasks_failed"]
    assert metrics["tasks_failed"] == 1
    assert metrics["total_refinements"] == 1
    await engine.stop()
