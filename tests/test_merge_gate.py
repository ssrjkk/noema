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
    GateConfig,
    GateReport,
    _changed_file_paths,
    run_gate_cli,
    run_merge_gate,
)


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
