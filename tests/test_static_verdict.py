"""T4.1 tests: AST static analysis verdict inside the neurosymbolic pipeline.

Covers ``noema/neurosymbolic/static.py`` and its wiring into
``NeuroSymbolicEngine.think``:
- code snippets embedded in a hypothesis are analyzed (structure before
  symbolic verification),
- the pipeline reports the static-analysis verdict alongside the Z3 verdict,
- hypotheses without code are skipped (analyzed=False, passed=True).
"""

from unittest.mock import AsyncMock

from noema.neurosymbolic.engine import NeuroSymbolicEngine
from noema.neurosymbolic.static import StaticAnalysisVerdict, analyze_solution_static

BROKEN_CODE = "def solve():\n    return missing_function(1)\n"
GOOD_CODE = "def solve():\n    return 42\n"


# ── analyze_solution_static unit behaviour ─────────────────────────────


def test_no_code_yields_skipped_verdict() -> None:
    verdict = analyze_solution_static({"result": "ok"})
    assert isinstance(verdict, StaticAnalysisVerdict)
    assert verdict.analyzed is False
    assert verdict.passed is True
    assert verdict.code_snippets == 0


def test_code_hint_key_is_detected() -> None:
    verdict = analyze_solution_static({"solution_code": BROKEN_CODE})
    assert verdict.analyzed is True
    assert verdict.code_snippets == 1
    assert verdict.passed is False
    assert any("undefined-name" in issue for issue in verdict.issues)


def test_nested_code_detected_recursively() -> None:
    hypothesis = {"config": {"implementation": {"source": GOOD_CODE}}}
    verdict = analyze_solution_static(hypothesis)
    assert verdict.analyzed is True
    assert verdict.passed is True
    assert verdict.issues == []


def test_valid_code_passes() -> None:
    verdict = analyze_solution_static({"code": GOOD_CODE})
    assert verdict.passed is True
    assert verdict.issues == []


def test_issue_messages_are_indexed() -> None:
    verdict = analyze_solution_static({"code": BROKEN_CODE})
    assert verdict.issues[0].startswith("snippet 1: ")


def test_verdict_as_dict() -> None:
    verdict = StaticAnalysisVerdict(analyzed=True, code_snippets=2, passed=False, issues=["a"])
    assert verdict.as_dict() == {
        "analyzed": True,
        "code_snippets": 2,
        "passed": False,
        "issues": ["a"],
    }


# ── Pipeline wiring ─────────────────────────────────────────────────────


async def _run_think(engine, task, hypothesis):
    engine.neural.generate_hypothesis = AsyncMock(return_value=hypothesis)
    engine.symbolic.verify_solution = AsyncMock(return_value=(True, []))
    return [event async for event in engine.think(task)]


async def test_pipeline_reports_static_verdict_alongside_z3() -> None:
    """T4.1 done-when: static verdict is reported next to the Z3 verdict."""
    engine = NeuroSymbolicEngine(max_refinement_attempts=2, enable_causal=False)
    await engine.start()
    try:
        task = {"requirements": [{"name": "x", "type": "numeric", "min": 0, "max": 10}]}
        events = await _run_think(engine, task, {"solution_code": BROKEN_CODE})

        verification = [e for e in events if e["stage"] == "verification"][-1]
        assert verification["is_valid"] is True  # Z3 verdict
        assert verification["static_passed"] is False  # AST verdict
        assert verification["static_analyzed"] is True
        assert any("undefined-name" in i for i in verification["static_issues"])

        completed = [e for e in events if e["stage"] == "completed"][0]
        assert completed["static_verdict"]["passed"] is False
        assert completed["static_verdict"]["code_snippets"] == 1
    finally:
        await engine.stop()


async def test_pipeline_static_pass_for_clean_hypothesis() -> None:
    engine = NeuroSymbolicEngine(max_refinement_attempts=2, enable_causal=False)
    await engine.start()
    try:
        task = {"requirements": [{"name": "x", "type": "numeric", "min": 0, "max": 10}]}
        events = await _run_think(engine, task, {"result": "ok"})

        verification = [e for e in events if e["stage"] == "verification"][-1]
        assert verification["static_passed"] is True
        assert verification["static_analyzed"] is False

        completed = [e for e in events if e["stage"] == "completed"][0]
        assert completed["static_verdict"]["analyzed"] is False
        assert completed["static_verdict"]["passed"] is True
    finally:
        await engine.stop()


async def test_pipeline_refinement_updates_static_verdict() -> None:
    """Refining a broken snippet to valid code flips the static verdict."""
    engine = NeuroSymbolicEngine(max_refinement_attempts=3, enable_causal=False)
    await engine.start()
    try:
        engine.symbolic.verify_solution = AsyncMock(side_effect=[(False, ["violated"]), (True, [])])
        engine.neural.generate_hypothesis = AsyncMock(return_value={"code": BROKEN_CODE})
        engine.neural.refine_hypothesis = AsyncMock(return_value={"code": GOOD_CODE})

        task = {"requirements": [{"name": "x", "type": "numeric", "min": 0, "max": 10}]}
        events = [event async for event in engine.think(task)]

        verifications = [
            e for e in events if e["stage"] == "verification" and e["status"] == "completed"
        ]
        assert verifications[0]["static_passed"] is False
        assert verifications[1]["static_passed"] is True
        completed = [e for e in events if e["stage"] == "completed"][0]
        assert completed["static_verdict"]["passed"] is True
    finally:
        await engine.stop()
