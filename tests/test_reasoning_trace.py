"""T4.2 tests: verifiable reasoning traces.

Covers ``noema/tracing/reasoning_trace.py`` and its wiring into
``NeuroSymbolicEngine.think``:
- committed traces are atomic JSON artifacts that round-trip,
- ``reverify`` reproduces the recorded verdict deterministically (no LLM),
- the engine commits a trace on the terminal event (completed / failed / error),
- a re-audit of a previous run's artifact matches the original verdict.
"""

from unittest.mock import AsyncMock

import pytest

from noema.neurosymbolic.engine import NeuroSymbolicEngine
from noema.neurosymbolic.symbolic import SymbolicEngine, TaskGraph
from noema.tracing.reasoning_trace import (
    ReasoningTrace,
    VerificationRound,
    build_reasoning_trace,
    commit_reasoning_trace,
    load_reasoning_trace,
    reverify_reasoning_trace,
)

TASK = {
    "requirements": [{"name": "x", "type": "numeric", "min": 0, "max": 10}],
    "constraints": [],
    "goals": [],
    "variables": {},
}
HYPOTHESIS = {"x": 5}
STATIC_OK = {"analyzed": False, "code_snippets": 0, "passed": True, "issues": []}


def _trace(outcome: str = "completed") -> ReasoningTrace:
    return build_reasoning_trace(
        correlation_id="corr-1",
        task=dict(TASK),
        rounds=[
            VerificationRound(
                attempt=1,
                hypothesis=dict(HYPOTHESIS),
                static_verdict=dict(STATIC_OK),
                symbolic_valid=True,
                violations=[],
            )
        ],
        outcome=outcome,
        attempts=1,
        final_hypothesis=dict(HYPOTHESIS),
        final_static_verdict=dict(STATIC_OK),
        final_symbolic_valid=True,
        final_violations=[],
    )


# ── Commit / load round-trip ───────────────────────────────────────────


def test_commit_and_load_round_trip(tmp_path):
    trace = _trace()
    path = commit_reasoning_trace(trace, tmp_path)
    assert path.is_file()
    assert path.suffix == ".json"
    assert not path.with_suffix(path.suffix + ".tmp").exists()

    loaded = load_reasoning_trace(path)
    assert loaded is not None
    assert loaded.run_id == trace.run_id
    assert loaded.correlation_id == "corr-1"
    assert loaded.outcome == "completed"
    assert loaded.final_hypothesis == HYPOTHESIS
    assert loaded.final_symbolic_valid is True
    assert loaded.rounds[0].attempt == 1
    assert loaded.rounds[0].hypothesis == HYPOTHESIS
    assert loaded.rounds[0].symbolic_valid is True


def test_load_missing_returns_none(tmp_path):
    assert load_reasoning_trace(tmp_path / "nope.json") is None


def test_load_corrupt_returns_none(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_reasoning_trace(bad) is None


# ── Reverify reproduces the verdict without the LLM ────────────────────


@pytest.mark.asyncio
async def test_reverify_matches_when_verdict_reproduced():
    trace = _trace()
    engine = AsyncMock(spec=SymbolicEngine)
    engine.parse_task = AsyncMock(return_value=TaskGraph())
    engine.verify_solution = AsyncMock(return_value=(True, []))

    verdict = await reverify_reasoning_trace(trace, engine)
    assert verdict.matches is True
    assert verdict.static_matches is True
    assert verdict.symbolic_matches is True
    assert verdict.replayed_final_symbolic_valid is True
    assert verdict.replayed_final_violations == []
    # The re-audit never touches the LLM side.
    engine.parse_task.assert_awaited_once_with(trace.task)


@pytest.mark.asyncio
async def test_reverify_detects_verdict_drift():
    trace = _trace()
    engine = AsyncMock(spec=SymbolicEngine)
    engine.parse_task = AsyncMock(return_value=TaskGraph())
    # A later re-audit no longer reproduces the recorded (True) verdict.
    engine.verify_solution = AsyncMock(return_value=(False, ["requirement_x_violated"]))

    verdict = await reverify_reasoning_trace(trace, engine)
    assert verdict.matches is False
    assert verdict.symbolic_matches is False
    assert verdict.recorded_final_symbolic_valid is True
    assert verdict.replayed_final_symbolic_valid is False


@pytest.mark.asyncio
async def test_reverify_detects_static_verdict_drift():
    trace = _trace()
    trace.final_hypothesis = {"solution_code": "def solve():\n    return missing(1)\n"}
    trace.final_static_verdict = {
        "analyzed": True,
        "code_snippets": 1,
        "passed": False,
        "issues": ["snippet 1: line 2: [undefined-name] 'missing' is never defined"],
    }
    engine = AsyncMock(spec=SymbolicEngine)
    engine.parse_task = AsyncMock(return_value=TaskGraph())
    engine.verify_solution = AsyncMock(return_value=(True, []))

    verdict = await reverify_reasoning_trace(trace, engine)
    assert verdict.matches is True  # both static verdicts agree on the failure
    assert verdict.static_matches is True


@pytest.mark.asyncio
async def test_reverify_error_trace_without_hypothesis():
    trace = _trace(outcome="error")
    trace.final_hypothesis = None
    trace.final_static_verdict = None
    trace.final_symbolic_valid = None
    trace.error = "SomeError: boom"
    engine = AsyncMock(spec=SymbolicEngine)

    verdict = await reverify_reasoning_trace(trace, engine)
    assert verdict.matches is True
    assert verdict.static_matches is None
    assert verdict.symbolic_matches is None
    engine.parse_task.assert_not_awaited()


# ── Engine integration ─────────────────────────────────────────────────


async def _collect(engine, task):
    return [event async for event in engine.think(task)]


@pytest.mark.asyncio
async def test_engine_commits_trace_on_completion(tmp_path):
    engine = NeuroSymbolicEngine(
        max_refinement_attempts=2, enable_causal=False, trace_dir=str(tmp_path)
    )
    await engine.start()
    try:
        engine.symbolic.parse_task = AsyncMock(return_value=TaskGraph())
        engine.symbolic.verify_solution = AsyncMock(return_value=(True, []))
        engine.neural.generate_hypothesis = AsyncMock(return_value={"x": 5})

        events = await _collect(engine, TASK)
        completed = [e for e in events if e["stage"] == "completed"][0]
        assert completed["static_verdict"]["passed"] is True

        artifacts = list(tmp_path.glob("*.json"))
        assert len(artifacts) == 1
        loaded = load_reasoning_trace(artifacts[0])
        assert loaded is not None
        assert loaded.outcome == "completed"
        assert loaded.correlation_id == completed["correlation_id"]
        assert loaded.final_symbolic_valid is True
        assert loaded.rounds[0].hypothesis == {"x": 5}
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_engine_commits_trace_on_failure(tmp_path):
    engine = NeuroSymbolicEngine(
        max_refinement_attempts=2, enable_causal=False, trace_dir=str(tmp_path)
    )
    await engine.start()
    try:
        engine.symbolic.parse_task = AsyncMock(return_value=TaskGraph())
        engine.symbolic.verify_solution = AsyncMock(return_value=(False, ["violated"]))
        engine.neural.generate_hypothesis = AsyncMock(return_value={"x": 99})
        engine.neural.refine_hypothesis = AsyncMock(return_value={"x": 100})

        events = await _collect(engine, TASK)
        failed = [e for e in events if e["stage"] == "failed"][0]
        assert failed["reason"] == "max_refinements_exceeded"

        artifacts = list(tmp_path.glob("*.json"))
        assert len(artifacts) == 1
        loaded = load_reasoning_trace(artifacts[0])
        assert loaded is not None
        assert loaded.outcome == "failed"
        assert loaded.final_symbolic_valid is False
        assert len(loaded.rounds) == 2
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_previous_run_artifacts_reproduce_verdict(tmp_path):
    """Done-when: a previous run's artifacts fully reproduce its verdict
    without re-running the LLM — replayed with a fresh symbolic engine."""
    engine = NeuroSymbolicEngine(
        max_refinement_attempts=2, enable_causal=False, trace_dir=str(tmp_path)
    )
    await engine.start()
    try:
        engine.symbolic.parse_task = AsyncMock(return_value=TaskGraph())
        engine.symbolic.verify_solution = AsyncMock(return_value=(True, []))
        engine.neural.generate_hypothesis = AsyncMock(return_value={"x": 5})
        await _collect(engine, TASK)
    finally:
        await engine.stop()

    artifact = list(tmp_path.glob("*.json"))[0]
    trace = load_reasoning_trace(artifact)
    assert trace is not None

    # A fresh engine (no LLM wiring used at all) replays the artifact.
    reaudit = NeuroSymbolicEngine(max_refinement_attempts=1, enable_causal=False)
    await reaudit.start()
    try:
        reaudit.symbolic.parse_task = AsyncMock(return_value=TaskGraph())
        reaudit.symbolic.verify_solution = AsyncMock(return_value=(True, []))
        verdict = await reverify_reasoning_trace(trace, reaudit.symbolic)
        assert verdict.matches is True
        assert verdict.replayed_final_symbolic_valid is True
        # No LLM calls happened during the re-audit (reverify only touches
        # the deterministic symbolic side).
        assert reaudit._metrics["total_llm_calls"] == 0
    finally:
        await reaudit.stop()


@pytest.mark.asyncio
async def test_engine_without_trace_dir_writes_nothing(tmp_path):
    engine = NeuroSymbolicEngine(max_refinement_attempts=2, enable_causal=False)
    await engine.start()
    try:
        engine.symbolic.parse_task = AsyncMock(return_value=TaskGraph())
        engine.symbolic.verify_solution = AsyncMock(return_value=(True, []))
        engine.neural.generate_hypothesis = AsyncMock(return_value={"x": 5})
        await _collect(engine, TASK)
    finally:
        await engine.stop()
    assert list(tmp_path.glob("*.json")) == []
