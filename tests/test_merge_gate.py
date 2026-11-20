"""T2.2 tests: merge gate (judge score + sandbox) blocks merges.

Covers ``noema/experiments/gate``:
- changed-file collection from a git diff,
- sandbox failures block the gate,
- judge score below the threshold blocks the gate,
- judge score at/above the threshold passes,
- CLI returns a non-zero exit code when blocked and writes the JSON report.
"""

import json
from types import SimpleNamespace

import pytest

from noema.core.types import Solution
from noema.experiments.gate import (
    CalibrationReport,
    GateConfig,
    GateReport,
    _changed_file_paths,
    _environment_import_roots,
    _repo_lint_config,
    run_gate_cli,
    run_merge_gate,
)
from noema.training.rlvr_pipeline import JudgeCalibrator


def _fake_git(paths: list[str]):
    def _run(args: list[str]) -> str:
        return "\n".join(paths) + ("\n" if paths else "")

    return _run


async def _judge_factory(score: float, passed: bool = True):
    async def _judge(solution: Solution):
        return SimpleNamespace(
            passed=passed,
            summary="fake judge",
            scores=SimpleNamespace(overall=score),
        )

    return _judge


class _FakeSandbox:
    def __init__(self, all_valid: bool):
        self._all_valid = all_valid

    async def validate_files(self, files, run_tests: bool = False):
        return SimpleNamespace(
            all_valid=self._all_valid,
            summary=f"{len(files)} files valid" if self._all_valid else "broken",
        )


def test_collect_changed_files_lists_paths(tmp_path):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# hi\n", encoding="utf-8")

    paths = _changed_file_paths(_fake_git(["app.py", "readme.md", "old.txt"]), "origin/main")
    assert paths == ["app.py", "readme.md", "old.txt"]


@pytest.mark.asyncio
async def test_gate_blocks_on_sandbox_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "broken.py").write_text("def f():\n    return missing(1)\n", encoding="utf-8")
    cfg = GateConfig(
        diff_target="origin/main",
        git=_fake_git(["broken.py"]),
        judge=await _judge_factory(0.9),
        sandbox=_FakeSandbox(all_valid=False),
    )
    report = await run_merge_gate(cfg)
    assert report.passed is False
    assert "sandbox" in report.blocked_by
    assert report.sandbox_all_valid is False


@pytest.mark.asyncio
async def test_gate_blocks_on_low_judge_score(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    cfg = GateConfig(
        diff_target="origin/main",
        judge_threshold=0.7,
        git=_fake_git(["app.py"]),
        judge=await _judge_factory(0.4),
        sandbox=_FakeSandbox(all_valid=True),
    )
    report = await run_merge_gate(cfg)
    assert report.passed is False
    assert any("judge_score" in b for b in report.blocked_by)
    assert report.judge_score == 0.4


@pytest.mark.asyncio
async def test_gate_passes_when_score_meets_threshold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    cfg = GateConfig(
        diff_target="origin/main",
        judge_threshold=0.7,
        git=_fake_git(["app.py"]),
        judge=await _judge_factory(0.8),
        sandbox=_FakeSandbox(all_valid=True),
    )
    report = await run_merge_gate(cfg)
    assert report.passed is True
    assert report.judge_score == 0.8
    assert report.changed_files == 1


@pytest.mark.asyncio
async def test_gate_no_changes_passes(tmp_path):
    cfg = GateConfig(
        diff_target="origin/main",
        judge_threshold=0.7,
        git=_fake_git([]),
        judge=await _judge_factory(0.0),
        sandbox=_FakeSandbox(all_valid=True),
    )
    report = await run_merge_gate(cfg)
    assert report.passed is True
    assert report.note == "no_changed_files"


async def _blocked_report(cfg: GateConfig) -> GateReport:
    return GateReport(passed=False, changed_files=1, judge_score=0.3, blocked_by=["sandbox"])


@pytest.mark.asyncio
async def test_cli_writes_report_and_blocks(tmp_path, monkeypatch):
    app = tmp_path / "app.py"
    app.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("noema.experiments.gate.run_merge_gate", _blocked_report)

    out = tmp_path / "gate.json"
    code = await run_gate_cli(
        ["--diff-target", "origin/main", "--judge-threshold", "0.7", "--out", str(out)]
    )
    assert code == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["blocked_by"] == ["sandbox"]


@pytest.mark.asyncio
async def test_cli_verifier_flags_plumb_into_config(tmp_path, monkeypatch):
    app = tmp_path / "app.py"
    app.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    captured: dict[str, object] = {}

    async def _capture(cfg):
        captured["cfg"] = cfg
        return GateReport(passed=True, changed_files=1)

    monkeypatch.setattr("noema.experiments.gate.run_merge_gate", _capture)

    out = tmp_path / "gate.json"
    code = await run_gate_cli(["--verifier", "--require-spec", "crypto/", "--out", str(out)])
    assert code == 0
    cfg = captured["cfg"]
    assert cfg.verifier is not None
    assert cfg.require_spec_patterns == ("crypto/",)


@pytest.mark.asyncio
async def test_gate_blocks_when_verifier_crashes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")

    class _CrashyVerifier:
        async def verify_files(self, files):
            raise RuntimeError("verifier exploded")

    (tmp_path / "specs").mkdir()
    (tmp_path / "specs" / "app.lean").write_text(
        "theorem t : True := by trivial\n", encoding="utf-8"
    )
    cfg = GateConfig(
        diff_target="origin/main",
        git=_fake_git(["app.py", "specs/app.lean"]),
        judge=await _judge_factory(0.9),
        sandbox=_FakeSandbox(all_valid=True),
        verifier=_CrashyVerifier(),
    )
    report = await run_merge_gate(cfg)
    assert report.passed is False
    assert "formal_verification" in report.blocked_by
    assert report.formal_verified is False
    assert "verifier_crashed" in report.formal_error


@pytest.mark.asyncio
async def test_gate_blocks_when_sandbox_crashes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")

    class _CrashySandbox:
        async def validate_files(self, files, run_tests: bool = False):
            raise OSError("sandbox tool missing")

    cfg = GateConfig(
        diff_target="origin/main",
        git=_fake_git(["app.py"]),
        judge=await _judge_factory(0.9),
        sandbox=_CrashySandbox(),
    )
    report = await run_merge_gate(cfg)
    assert report.passed is False
    assert "sandbox" in report.blocked_by
    assert report.sandbox_all_valid is False
    assert "sandbox_error" in report.sandbox_summary


@pytest.mark.asyncio
async def test_gate_blocks_when_judge_crashes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")

    async def _crashing_judge(solution):
        raise RuntimeError("llm unreachable")

    cfg = GateConfig(
        diff_target="origin/main",
        git=_fake_git(["app.py"]),
        judge=_crashing_judge,
        sandbox=_FakeSandbox(all_valid=True),
    )
    report = await run_merge_gate(cfg)
    assert report.passed is False
    assert any("judge_error" in b for b in report.blocked_by)
    assert report.judge_score == 0.0
    assert report.judge_passed is False


@pytest.mark.asyncio
async def test_cli_writes_blocked_artifact_on_gate_crash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def _crash(cfg):
        raise RuntimeError("git diff exploded")

    monkeypatch.setattr("noema.experiments.gate.run_merge_gate", _crash)

    out = tmp_path / "gate.json"
    code = await run_gate_cli(["--out", str(out)])
    assert code == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert any("gate_error" in b for b in report["blocked_by"])


# -- Ontological semantics (requires / forbids axioms) -----------------------


@pytest.mark.asyncio
async def test_gate_blocks_on_ontology_requires_violation(tmp_path, monkeypatch):
    """A file referencing User without AuthMethod violates the requires axiom."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("class User:\n    pass\n", encoding="utf-8")

    from noema.ontology import Entity, OntologyGraph, Relation

    graph = OntologyGraph()
    graph.add_entity(Entity(name="User", type="domain"))
    graph.add_entity(Entity(name="AuthMethod", type="domain"))
    graph.add_relation(Relation("User", "requires", "AuthMethod"))

    cfg = GateConfig(
        diff_target="origin/main",
        git=_fake_git(["app.py"]),
        judge=await _judge_factory(0.9),
        sandbox=_FakeSandbox(all_valid=True),
        ontology=graph,
    )
    report = await run_merge_gate(cfg)
    assert report.passed is False
    assert "ontology:app.py:User requires AuthMethod" in report.blocked_by
    assert report.blocked_by == report.ontology_violations


@pytest.mark.asyncio
async def test_gate_passes_when_requires_satisfied(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text(
        "class User:\n    pass\n\nclass AuthMethod:\n    pass\n", encoding="utf-8"
    )

    from noema.ontology import Entity, OntologyGraph, Relation

    graph = OntologyGraph()
    graph.add_entity(Entity(name="User", type="domain"))
    graph.add_entity(Entity(name="AuthMethod", type="domain"))
    graph.add_relation(Relation("User", "requires", "AuthMethod"))

    cfg = GateConfig(
        diff_target="origin/main",
        git=_fake_git(["app.py"]),
        judge=await _judge_factory(0.9),
        sandbox=_FakeSandbox(all_valid=True),
        ontology=graph,
    )
    report = await run_merge_gate(cfg)
    assert report.passed is True
    assert report.ontology_violations == []


@pytest.mark.asyncio
async def test_gate_blocks_on_ontology_forbids_violation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text(
        "class PlaintextStorage:\n    pass\n\nclass Passwords:\n    pass\n", encoding="utf-8"
    )

    from noema.ontology import Entity, OntologyGraph, Relation

    graph = OntologyGraph()
    graph.add_entity(Entity(name="PlaintextStorage", type="anti"))
    graph.add_entity(Entity(name="Passwords", type="domain"))
    graph.add_relation(Relation("PlaintextStorage", "forbids", "Passwords"))

    cfg = GateConfig(
        diff_target="origin/main",
        git=_fake_git(["app.py"]),
        judge=await _judge_factory(0.9),
        sandbox=_FakeSandbox(all_valid=True),
        ontology=graph,
    )
    report = await run_merge_gate(cfg)
    assert report.passed is False
    assert "ontology:app.py:PlaintextStorage forbids Passwords" in report.blocked_by


@pytest.mark.asyncio
async def test_gate_without_ontology_never_blocks_on_semantics(tmp_path, monkeypatch):
    """No ontology configured > no semantic stage, behavior unchanged."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("class User:\n    pass\n", encoding="utf-8")
    cfg = GateConfig(
        diff_target="origin/main",
        git=_fake_git(["app.py"]),
        judge=await _judge_factory(0.9),
        sandbox=_FakeSandbox(all_valid=True),
    )
    report = await run_merge_gate(cfg)
    assert report.passed is True
    assert report.ontology_violations == []


# ---------------------------------------------------------------------------
# The real sandbox, not a fake one. Every test above injects _FakeSandbox, so
# none of them could notice that the gate's own SandboxEngine disagreed with
# the repository's lint rules and import policy. These drive the engine the
# gate actually builds.
# ---------------------------------------------------------------------------

_RUFF_CONFIG = """\
[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F"]
ignore = ["E501"]
"""


@pytest.mark.asyncio
async def test_gate_lints_repo_files_with_the_repository_ruff_config(tmp_path, monkeypatch):
    """E501 is ignored by this repo, so the gate must not block on it."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(_RUFF_CONFIG, encoding="utf-8")
    content = "x = " + repr("a" * 130) + "\n"
    assert len(content.splitlines()[0]) > 100

    cfg = GateConfig(
        diff_target="origin/main",
        judge=await _judge_factory(0.9),
        explicit_files=[{"path": "wide.py", "content": content}],
    )
    report = await run_merge_gate(cfg)

    assert report.sandbox_all_valid is True, report.sandbox_issues
    assert report.passed is True


@pytest.mark.asyncio
async def test_gate_still_blocks_rules_the_repository_does_not_ignore(tmp_path, monkeypatch):
    """Counterweight: the pass above must not be the sandbox going blind."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(_RUFF_CONFIG, encoding="utf-8")

    cfg = GateConfig(
        diff_target="origin/main",
        judge=await _judge_factory(0.9),
        explicit_files=[
            {
                "path": "unused.py",
                "content": """import os

x = 1
""",
            }
        ],
    )
    report = await run_merge_gate(cfg)

    assert report.passed is False
    assert "sandbox" in report.blocked_by
    assert report.sandbox_all_valid is False
    assert any("F401" in issue for issue in report.sandbox_issues), report.sandbox_issues


@pytest.mark.asyncio
async def test_gate_allows_imports_the_environment_provides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(_RUFF_CONFIG, encoding="utf-8")
    content = """import pytest


def marker():
    return pytest.mark.skip
"""

    cfg = GateConfig(
        diff_target="origin/main",
        judge=await _judge_factory(0.9),
        explicit_files=[{"path": "uses_pytest.py", "content": content}],
    )
    report = await run_merge_gate(cfg)

    assert report.sandbox_all_valid is True, report.sandbox_issues


@pytest.mark.asyncio
async def test_gate_blocks_import_the_environment_does_not_provide(tmp_path, monkeypatch):
    """A dependency the CI job forgot to install must still block, naming itself."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(_RUFF_CONFIG, encoding="utf-8")
    content = """import definitely_not_installed_pkg_xyz

x = 1
"""

    cfg = GateConfig(
        diff_target="origin/main",
        judge=await _judge_factory(0.9),
        explicit_files=[{"path": "missing_dep.py", "content": content}],
    )
    report = await run_merge_gate(cfg)

    assert report.passed is False
    assert "sandbox" in report.blocked_by
    assert report.sandbox_all_valid is False
    assert any("definitely_not_installed_pkg_xyz" in issue for issue in report.sandbox_issues)


@pytest.mark.asyncio
async def test_gate_allows_first_party_relative_imports(tmp_path, monkeypatch):
    """Package internals legitimately import their own siblings. The gate
    validates first-party files, so relative imports must not block it (the
    untrusted-code default still rejects them — see test_sandbox_static_check)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(_RUFF_CONFIG, encoding="utf-8")
    content = """from .store import MemoryRecord
from ..config import get_settings

__all__ = ["MemoryRecord", "get_settings"]
"""

    cfg = GateConfig(
        diff_target="origin/main",
        judge=await _judge_factory(0.9),
        explicit_files=[{"path": "pkg/__init__.py", "content": content}],
    )
    report = await run_merge_gate(cfg)

    assert report.sandbox_all_valid is True, report.sandbox_issues
    assert not any("relative-import" in issue for issue in report.sandbox_issues)


def test_environment_import_roots_skips_stdlib_and_unresolvable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    files = [
        {
            "path": "a.py",
            "content": """import os
import pytest
import absent_pkg_qq
""",
        },
        {
            "path": "b.py",
            "content": """from __future__ import annotations
""",
        },
        {
            "path": "notes.md",
            "content": """import pytest
""",
        },
        {
            "path": "broken.py",
            "content": """def f(:
""",
        },
    ]
    roots = _environment_import_roots(files)
    assert "pytest" in roots
    assert "absent_pkg_qq" not in roots
    assert "os" not in roots and "__future__" not in roots


def test_repo_lint_config_follows_ruff_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(_RUFF_CONFIG, encoding="utf-8")
    assert _repo_lint_config().endswith("pyproject.toml")
    (tmp_path / "ruff.toml").write_text("line-length = 100\n", encoding="utf-8")
    assert _repo_lint_config().endswith("ruff.toml")
    (tmp_path / "nowhere").mkdir()
    monkeypatch.chdir(tmp_path / "nowhere")
    assert _repo_lint_config() == ""


def test_environment_import_roots_is_fail_closed_when_probe_raises(tmp_path, monkeypatch):
    """A probe that errors must leave the root blocked, not allowed."""
    import importlib.util as iu

    monkeypatch.chdir(tmp_path)
    real_find_spec = iu.find_spec

    def flaky_find_spec(name, *args, **kwargs):
        if name == "cursed_pkg":
            raise ValueError("probe exploded")
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(iu, "find_spec", flaky_find_spec)
    files = [
        {
            "path": "a.py",
            "content": """import cursed_pkg
import pytest
""",
        }
    ]
    roots = _environment_import_roots(files)
    assert "cursed_pkg" not in roots
    assert "pytest" in roots


# ---------------------------------------------------------------------------
# RLVR judge calibration inside the merge gate. The history below is a clean
# score/outcome split so the Platt fit is reproducible: cross-validated Brier is
# 0.089172, the same fit scored on its own training data reports 0.069334, and
# calibrate(0.9) = 0.9334 while calibrate(0.2) = 0.2315.
# ---------------------------------------------------------------------------

_CAL_SCORES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
_CAL_OUTCOMES = (0, 0, 0, 0, 1, 1, 1, 1, 1, 1)
_CAL_FILE = {"scores": list(_CAL_SCORES), "outcomes": list(_CAL_OUTCOMES)}


def _cal_cfg(judge, **overrides):
    kwargs = {
        "explicit_files": [{"path": "a.py", "language": "python", "content": "x = 1\n"}],
        "judge": judge,
        "sandbox": _FakeSandbox(all_valid=True),
        "calibration_scores": _CAL_SCORES,
        "calibration_outcomes": _CAL_OUTCOMES,
    }
    kwargs.update(overrides)
    return GateConfig(**kwargs)


@pytest.mark.asyncio
async def test_gate_reports_out_of_sample_calibration_metrics(tmp_path, monkeypatch):
    """The calibration numbers in the report are held-out, never the fit's own training set."""
    monkeypatch.chdir(tmp_path)
    cfg = _cal_cfg(await _judge_factory(0.9))
    report = await run_merge_gate(cfg)

    assert report.passed is True
    assert report.calibration is not None
    expected = JudgeCalibrator.cross_validate(_CAL_SCORES, _CAL_OUTCOMES, k_folds=5)
    assert report.calibration.method == "5-fold-out-of-sample"
    assert report.calibration.brier_score == expected.brier_score
    assert report.calibration.ece == expected.ece
    assert report.calibration.samples == 10
    # The training-set score is the optimistic one; reporting it as the headline
    # number would overstate how much the judge can be trusted.
    assert report.calibration.in_sample_brier < report.calibration.brier_score
    assert report.judge_calibrated == pytest.approx(0.9334, abs=1e-3)


@pytest.mark.asyncio
async def test_gate_blocks_when_calibrated_probability_below_threshold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _cal_cfg(await _judge_factory(0.9), calibrate_threshold=0.95)
    report = await run_merge_gate(cfg)

    assert report.passed is False
    assert any("judge_calibrated=0.933 < threshold=0.950" in b for b in report.blocked_by)


@pytest.mark.asyncio
async def test_gate_passes_when_calibrated_probability_meets_threshold(tmp_path, monkeypatch):
    """Counterweight: the calibrated gate must be able to pass, not only to block."""
    monkeypatch.chdir(tmp_path)
    cfg = _cal_cfg(await _judge_factory(0.9), calibrate_threshold=0.9)
    report = await run_merge_gate(cfg)

    assert report.passed is True
    assert report.blocked_by == []


@pytest.mark.asyncio
async def test_calibration_never_rescues_a_raw_score_below_the_judge_threshold(
    tmp_path, monkeypatch
):
    """Adding calibration may only tighten the gate, never replace the existing compare."""
    monkeypatch.chdir(tmp_path)
    cfg = _cal_cfg(await _judge_factory(0.4), judge_threshold=0.7, calibrate_threshold=0.0)
    report = await run_merge_gate(cfg)

    assert report.passed is False
    assert any("judge_score=0.400 < threshold=0.700" in b for b in report.blocked_by)
    assert report.calibration is not None


@pytest.mark.asyncio
async def test_gate_reads_calibration_history_from_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    history = tmp_path / "history.json"
    history.write_text(json.dumps(_CAL_FILE), encoding="utf-8")

    cfg = _cal_cfg(
        await _judge_factory(0.9),
        calibration_scores=(),
        calibration_outcomes=(),
        calibration_file=str(history),
    )
    report = await run_merge_gate(cfg)

    assert report.passed is True
    assert report.calibration is not None
    assert report.calibration.samples == 10
    assert report.judge_calibrated == pytest.approx(0.9334, abs=1e-3)


@pytest.mark.asyncio
async def test_gate_blocks_on_malformed_calibration_file(tmp_path, monkeypatch):
    """A history file that is not a scores/outcomes object blocks instead of skipping."""
    monkeypatch.chdir(tmp_path)
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"judge": [0.1, 0.2]}), encoding="utf-8")

    cfg = _cal_cfg(
        await _judge_factory(0.9),
        calibration_scores=(),
        calibration_outcomes=(),
        calibration_file=str(history),
    )
    report = await run_merge_gate(cfg)

    assert report.passed is False
    assert "scores" in report.calibration_error
    assert any(b.startswith("calibration_error:") for b in report.blocked_by)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scores", "outcomes", "reason"),
    [
        ((0.1, 0.2), (1,), "lengths differ"),
        ((0.1,), (1,), "at least 2 labeled samples"),
        ((0.1, 0.2), (1, 5), "must be 0"),
        ((0.1, 17.0), (1, 0), "must lie in [0, 1]"),
    ],
)
async def test_gate_fail_closed_on_unusable_calibration_history(
    tmp_path, monkeypatch, scores, outcomes, reason
):
    """Every unusable history blocks by name; a requested stage cannot switch itself off."""
    monkeypatch.chdir(tmp_path)
    cfg = _cal_cfg(
        await _judge_factory(0.9), calibration_scores=scores, calibration_outcomes=outcomes
    )
    report = await run_merge_gate(cfg)

    assert report.passed is False
    assert report.calibration is None
    assert reason in report.calibration_error
    assert any(b.startswith("calibration_error:") for b in report.blocked_by)


@pytest.mark.asyncio
async def test_gate_blocks_when_threshold_requested_without_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _cal_cfg(
        await _judge_factory(0.9),
        calibration_scores=(),
        calibration_outcomes=(),
        calibrate_threshold=0.5,
    )
    report = await run_merge_gate(cfg)

    assert report.passed is False
    assert "no history supplied" in report.calibration_error


@pytest.mark.asyncio
async def test_gate_without_calibration_keeps_previous_verdict(tmp_path, monkeypatch):
    """Opt-in stage: unset, the report fields stay empty and the verdict is unchanged."""
    monkeypatch.chdir(tmp_path)
    cfg = GateConfig(
        judge_threshold=0.7,
        explicit_files=[{"path": "a.py", "language": "python", "content": "x = 1\n"}],
        judge=await _judge_factory(0.9),
        sandbox=_FakeSandbox(all_valid=True),
    )
    report = await run_merge_gate(cfg)

    assert report.passed is True
    assert report.calibration is None
    assert report.judge_calibrated is None
    assert report.calibration_error == ""


@pytest.mark.asyncio
async def test_cli_calibration_flags_plumb_into_config_and_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    history = tmp_path / "history.json"
    history.write_text(json.dumps(_CAL_FILE), encoding="utf-8")

    captured: dict[str, object] = {}

    async def _capture(cfg):
        captured["cfg"] = cfg
        return GateReport(
            passed=True,
            changed_files=1,
            calibration=CalibrationReport(
                method="5-fold-out-of-sample",
                brier_score=0.089172,
                ece=0.152678,
                samples=10,
                in_sample_brier=0.069334,
                slope=5.486982,
                intercept=-2.297466,
            ),
        )

    monkeypatch.setattr("noema.experiments.gate.run_merge_gate", _capture)

    out = tmp_path / "gate.json"
    code = await run_gate_cli(
        [
            "--calibration-file",
            str(history),
            "--calibration-folds",
            "4",
            "--calibrate-threshold",
            "0.9",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    cfg = captured["cfg"]
    assert cfg.calibration_file == str(history)
    assert cfg.calibration_folds == 4
    assert cfg.calibrate_threshold == 0.9

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["calibration"]["brier_score"] == 0.089172
    assert report["calibration"]["method"] == "5-fold-out-of-sample"
