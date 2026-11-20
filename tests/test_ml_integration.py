"""Integration tests for the ML/AI upgrade wiring (not the base components).

test_ml_upgrades.py covers the standalone bandit / CEGIS / RLVR / memory units.
This file covers the points where those units were *integrated* into the running
system:

- #47 LinUCB inside the unified ``ModelRouter`` (config gating, fail-closed
  budget, online outcome feedback, cross-task misattribution guard) and the
  CoT executor's budget abort.
- #48 CEGIS feedback (real unsat_core / satisfying counterexample) surfaced by
  ``SymbolicEngine.verify_solution`` and consumed by the refine loop.
- #51 the ``noema train`` CLI commands and their malformed-input gates.

Every gate here is written so it can actually fail: removing the guard it
targets turns the assertion red.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from noema.cli.train import train_app
from noema.config.settings import NoemaSettings
from noema.llm.router_bandit import BudgetExceededError, LinUCBModelRouter, ModelArm
from noema.routing.model_router import ModelRouter

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()

_ARMS = (
    ModelArm("local", "ollama", "qwen", 0.0, 0.7, is_local=True),
    ModelArm("paid", "openai", "gpt-4o", 0.05, 0.7, is_local=False),
)


def _settings(router: str = "static", checkpoint: str = "") -> NoemaSettings:
    s = NoemaSettings()
    s.llm.router = router
    s.llm.router_checkpoint = checkpoint
    return s


def _patch_settings(monkeypatch: pytest.MonkeyPatch, s: NoemaSettings) -> None:
    monkeypatch.setattr("noema.routing.model_router.get_settings", lambda: s)


# ---------------------------------------------------------------------------
# #47 — LinUCB integrated into the unified ModelRouter
# ---------------------------------------------------------------------------


def test_model_router_static_policy_has_no_bandit(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, _settings("static"))
    router = ModelRouter()
    assert router.bandit_active is False
    assert router.record_outcome(verified_pass=True) == 0
    assert router.bandit_summary() is None
    router.select(complexity="moderate")  # static path still works
    assert router.record_outcome(verified_pass=True) == 0


def test_model_router_linucb_enabled_via_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, _settings("linucb"))
    router = ModelRouter()
    assert router.bandit_active is True
    router.select(complexity="moderate", tags=["api"], prompt_tokens=800)
    assert router.record_outcome(verified_pass=True) >= 1
    # Pending decisions are consumed: a second verdict must not re-credit them.
    assert router.record_outcome(verified_pass=True) == 0
    assert router.bandit_summary() is not None


def test_model_router_unknown_policy_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, _settings("bogus"))
    with pytest.raises(ValueError, match="Unknown NOEMA_LLM__ROUTER"):
        ModelRouter()


def test_model_router_budget_exhaustion_is_fail_closed() -> None:
    router = ModelRouter(bandit=LinUCBModelRouter(_ARMS))
    # A free local arm exists, but the default policy must NOT silently fall
    # back to it when the USD budget is exhausted.
    with pytest.raises(BudgetExceededError):
        router.select(complexity="moderate", budget_remaining_usd=0.0)


def test_model_router_discard_prevents_cross_task_misattribution() -> None:
    router = ModelRouter(bandit=LinUCBModelRouter(_ARMS))
    router.select(complexity="moderate")
    router.discard_pending_outcomes()
    assert router.record_outcome(verified_pass=True) == 0


def test_model_router_record_outcome_dedups_repeated_arm() -> None:
    router = ModelRouter(bandit=LinUCBModelRouter((_ARMS[0],)))
    router.select(complexity="simple")
    router.select(complexity="simple")
    assert router.record_outcome(verified_pass=True) == 1


def test_model_router_corrupt_checkpoint_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bandit = LinUCBModelRouter(_ARMS)
    good = tmp_path / "good.json"
    bandit.save(good)
    corrupted = json.loads(good.read_text(encoding="utf-8"))
    corrupted["states"]["local"]["b"] = [1.0, 2.0]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(corrupted), encoding="utf-8")

    _patch_settings(monkeypatch, _settings("linucb", str(bad)))
    with pytest.raises(ValueError, match="Invalid b vector"):
        ModelRouter()


def test_model_router_missing_checkpoint_starts_fresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_settings(monkeypatch, _settings("linucb", str(tmp_path / "absent.json")))
    router = ModelRouter()
    assert router.bandit_active is True


def test_model_router_save_bandit_roundtrip(tmp_path: Path) -> None:
    router = ModelRouter(bandit=LinUCBModelRouter(_ARMS))
    dest = tmp_path / "state.json"
    router.save_bandit(dest)
    assert dest.is_file()
    restored = LinUCBModelRouter.load(dest)
    assert set(restored.arms) == {"local", "paid"}


async def test_cot_budget_exceeded_aborts_run_not_failed_step() -> None:
    from noema.core.chain_of_thought import ChainOfThought, CoTStep, StepStatus

    cot = ChainOfThought(llm=MagicMock())
    step = CoTStep(name="analysis", role="analyst", depends_on=[], system_prompt="", user_prompt="")
    cot._planner.plan = MagicMock(return_value=["analysis"])

    def _fake_build_dag(*_a: object, **_k: object) -> None:
        cot._steps = [step]

    cot._build_dag = _fake_build_dag  # type: ignore[method-assign]

    async def _boom(_step: CoTStep) -> None:
        raise BudgetExceededError("usd budget exhausted")

    cot._execute_step = _boom  # type: ignore[method-assign]

    with pytest.raises(BudgetExceededError):
        await cot.reason("task", [], [])
    # The run aborted; the step was NOT degraded into a "failed" step that
    # would still let the executor return a partial solution.
    assert step.status != StepStatus.FAILED


# ---------------------------------------------------------------------------
# #48 — CEGIS feedback surfaced by SymbolicEngine (real Z3 required)
# ---------------------------------------------------------------------------


async def _symbolic_engine():
    from noema.neurosymbolic.symbolic import SymbolicEngine

    eng = SymbolicEngine(verification_timeout=5.0)
    await eng.initialize()
    return eng


async def test_symbolic_unsat_core_names_conflicting_requirement() -> None:
    pytest.importorskip("z3")
    eng = await _symbolic_engine()
    task = {"requirements": [{"name": "a", "type": "numeric", "min": 10, "max": 3}]}
    graph = await eng.parse_task(task)
    valid, violations = await eng.verify_solution({"a": 5}, graph)
    assert not valid
    assert "constraints_unsatisfiable" in violations
    fb = eng.last_feedback
    assert fb.verdict == "UNSAT"
    assert fb.unsat_core == ("a",)


async def test_symbolic_counterexample_is_real_variable_value() -> None:
    pytest.importorskip("z3")
    eng = await _symbolic_engine()
    task = {"requirements": [{"name": "x", "type": "numeric", "min": 0, "max": 5}]}
    graph = await eng.parse_task(task)
    valid, violations = await eng.verify_solution({"x": 10}, graph)
    assert not valid
    assert any("requirement_x_violated" in v for v in violations)
    fb = eng.last_feedback
    assert fb.verdict == "UNSAT"
    assert fb.counterexample
    # Regression: an assertion-tracking Bool literal named 'x' must never
    # shadow the Int variable 'x' (the old bug produced {'x': 'True'}).
    assert fb.counterexample["x"] != "True"
    assert 0 <= int(fb.counterexample["x"]) <= 5


async def test_symbolic_sat_feedback_on_valid_candidate() -> None:
    pytest.importorskip("z3")
    eng = await _symbolic_engine()
    task = {"requirements": [{"name": "x", "type": "numeric", "min": 0, "max": 5}]}
    graph = await eng.parse_task(task)
    valid, violations = await eng.verify_solution({"x": 3}, graph)
    assert valid
    assert violations == []
    assert eng.last_feedback.verdict == "SAT"


async def test_verify_solution_with_feedback_returns_consistent_tuple() -> None:
    pytest.importorskip("z3")
    eng = await _symbolic_engine()
    task = {"requirements": [{"name": "x", "type": "numeric", "min": 0, "max": 5}]}
    graph = await eng.parse_task(task)
    valid, violations, fb = await eng.verify_solution_with_feedback({"x": 3}, graph)
    assert valid
    assert violations == []
    assert fb is eng.last_feedback
    assert fb.verdict == "SAT"


async def test_symbolic_degraded_engine_fails_closed_with_feedback() -> None:
    from noema.neurosymbolic.symbolic import SymbolicEngine, TaskGraph

    eng = SymbolicEngine()
    eng._degraded = True
    eng._initialized = True
    valid, violations = await eng.verify_solution({"x": 1}, TaskGraph())
    assert not valid
    assert "symbolic_engine_unavailable_z3_not_installed" in violations
    assert eng.last_feedback.verdict == "FAIL_CLOSED"


async def test_symbolic_uninitialized_graph_fails_closed() -> None:
    from noema.neurosymbolic.symbolic import SymbolicEngine

    eng = SymbolicEngine()
    valid, violations = await eng.verify_solution({}, None)
    assert not valid
    assert "task_graph_not_initialized" in violations
    assert eng.last_feedback.verdict == "FAIL_CLOSED"


# ---------------------------------------------------------------------------
# #51 — noema train CLI
# ---------------------------------------------------------------------------


def _rollouts_jsonl(tmp_path: Path) -> Path:
    rows = [
        {
            "task_id": "t1",
            "prompt": "p",
            "response": "good",
            "ast_ok": True,
            "z3_sat": True,
            "mutation_kill_score": 0.9,
            "docker_pass_rate": 1.0,
            "normalized_cost": 0.1,
            "raw_judge_score": 0.9,
        },
        {
            "task_id": "t1",
            "prompt": "p",
            "response": "bad",
            "ast_ok": True,
            "z3_sat": False,
            "mutation_kill_score": 0.0,
            "docker_pass_rate": 0.0,
            "normalized_cost": 0.1,
            "raw_judge_score": 0.3,
        },
    ]
    src = tmp_path / "rollouts.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return src


def test_train_export_dpo_writes_pairs(tmp_path: Path) -> None:
    src = _rollouts_jsonl(tmp_path)
    out = tmp_path / "dpo.jsonl"
    result = runner.invoke(train_app, ["export-dpo", "-i", str(src), "-o", str(out)])
    assert result.exit_code == 0, result.stdout
    pairs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "good"
    assert pairs[0]["rejected"] == "bad"
    assert pairs[0]["margin"] > 0.25


def test_train_export_dpo_missing_input_is_bad_parameter(tmp_path: Path) -> None:
    result = runner.invoke(
        train_app,
        ["export-dpo", "-i", str(tmp_path / "nope.jsonl"), "-o", str(tmp_path / "o.jsonl")],
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_train_export_dpo_invalid_json_line_reports_lineno(tmp_path: Path) -> None:
    src = tmp_path / "bad.jsonl"
    src.write_text('{"task_id": "t1"}\nnot-json\n', encoding="utf-8")
    result = runner.invoke(
        train_app, ["export-dpo", "-i", str(src), "-o", str(tmp_path / "o.jsonl")]
    )
    assert result.exit_code != 0
    assert ":2:" in result.output


def test_train_export_dpo_non_object_line_rejected(tmp_path: Path) -> None:
    src = tmp_path / "arr.jsonl"
    src.write_text("[1, 2, 3]\n", encoding="utf-8")
    result = runner.invoke(
        train_app, ["export-dpo", "-i", str(src), "-o", str(tmp_path / "o.jsonl")]
    )
    assert result.exit_code != 0
    assert "expected a JSON object" in result.output


def test_train_export_dpo_malformed_rollout_rejected(tmp_path: Path) -> None:
    src = tmp_path / "missing.jsonl"
    src.write_text('{"task_id": "t1", "prompt": "p"}\n', encoding="utf-8")
    result = runner.invoke(
        train_app, ["export-dpo", "-i", str(src), "-o", str(tmp_path / "o.jsonl")]
    )
    assert result.exit_code != 0
    assert "malformed rollout" in result.output


def test_train_export_grpo_normalizes_advantage(tmp_path: Path) -> None:
    src = _rollouts_jsonl(tmp_path)
    out = tmp_path / "grpo.jsonl"
    result = runner.invoke(train_app, ["export-grpo", "-i", str(src), "-o", str(out)])
    assert result.exit_code == 0, result.stdout
    groups = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(groups) == 1
    advantages = [c["advantage"] for c in groups[0]["completions"]]
    assert abs(sum(advantages)) < 1e-6  # within-group normalization centers to ~0


def test_train_calibrate_judge_reports_out_of_sample_and_saves(tmp_path: Path) -> None:
    src = tmp_path / "judge.jsonl"
    rows = [
        {"raw_judge_score": 0.9, "docker_pass": 1},
        {"raw_judge_score": 0.8, "docker_pass": 1},
        {"raw_judge_score": 0.3, "docker_pass": 0},
        {"raw_judge_score": 0.2, "docker_pass": 0},
        {"raw_judge_score": 0.6, "docker_pass": 1},
        {"raw_judge_score": 0.4, "docker_pass": 0},
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    dest = tmp_path / "cal.json"
    result = runner.invoke(
        train_app, ["calibrate-judge", "-i", str(src), "-k", "3", "--out", str(dest)]
    )
    assert result.exit_code == 0, result.stdout
    assert "out-of-sample" in result.stdout.lower()
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert "slope" in payload and "intercept" in payload
    assert payload["cross_validate"]["samples"] == 6


def test_train_calibrate_judge_needs_two_samples(tmp_path: Path) -> None:
    src = tmp_path / "one.jsonl"
    src.write_text('{"raw_judge_score": 0.9, "docker_pass": 1}\n', encoding="utf-8")
    result = runner.invoke(train_app, ["calibrate-judge", "-i", str(src)])
    assert result.exit_code == 1


def test_train_calibrate_judge_malformed_sample_rejected(tmp_path: Path) -> None:
    src = tmp_path / "bad.jsonl"
    src.write_text('{"raw_judge_score": 0.9}\n', encoding="utf-8")
    result = runner.invoke(train_app, ["calibrate-judge", "-i", str(src)])
    assert result.exit_code != 0
    assert "malformed sample" in result.output
