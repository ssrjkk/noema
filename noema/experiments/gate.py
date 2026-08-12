"""Merge gate — block merges when the judge score is below a threshold or the sandbox fails.

Consumes a PR's changed files (git diff against a target branch), runs them through
the existing sandbox pipeline (AST + static + lint, optionally execution/tests) and
the LLM judge, then reports a pass/block verdict. The CLI exits non-zero when the
gate blocks, so it can be wired into CI as a required status check (T2.2).

Usage::

    python -m noema.experiments.gate --diff-target origin/main \
        --judge-threshold 0.7 --out gate.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from noema.core.types import CodeBlock, Solution
from noema.judge import JudgeVerdict, evaluate_solution
from noema.logging import get_logger
from noema.sandbox.engine import SandboxConfig, SandboxEngine, SandboxResult

log = get_logger(__name__)

GitRunner = Callable[[list[str]], str]
JudgeRunner = Callable[[Solution], Awaitable[JudgeVerdict]]


def run_git(args: list[str]) -> str:
    """Run a git command and return stdout (used as the default git runner)."""
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False, timeout=120
    )
    return result.stdout


def _changed_file_paths(git: GitRunner, diff_target: str) -> list[str]:
    stdout = git(["diff", "--name-only", "--diff-filter=ACMR", diff_target, "--", "."]).strip()
    if not stdout:
        return []
    return [line for line in stdout.splitlines() if line.strip()]


@dataclass
class GateConfig:
    """Tunables for one merge-gate run."""

    judge_threshold: float = 0.0
    sandbox_enabled: bool = True
    sandbox_run: bool = False
    run_tests: bool = False
    diff_target: str = "origin/main"
    include_suffixes: tuple[str, ...] = (".py",)
    git: GitRunner = run_git
    judge: JudgeRunner | None = None
    sandbox: SandboxEngine | None = None


@dataclass
class GateReport:
    """The verdict of one merge-gate run."""

    passed: bool
    changed_files: int
    judge_score: float = 0.0
    judge_passed: bool = False
    judge_summary: str = ""
    sandbox_all_valid: bool | None = None
    sandbox_summary: str = ""
    blocked_by: list[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _collect_files(cfg: GateConfig) -> list[dict[str, str]]:
    """Read the changed files (filtered by suffix) from the worktree."""
    files: list[dict[str, str]] = []
    for path in _changed_file_paths(cfg.git, cfg.diff_target):
        if not path.endswith(cfg.include_suffixes):
            continue
        file_path = Path(path)
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            continue  # deleted or unreadable files are not gated
        language = "python" if path.endswith(".py") else "text"
        files.append({"path": path, "language": language, "content": content})
    return files


async def run_merge_gate(cfg: GateConfig) -> GateReport:
    """Evaluate the PR's changed files and decide pass/block.

    Blocks when:
    - the sandbox verdict is not ``all_valid`` (structural/lint failures), or
    - the judge's overall score is below ``judge_threshold``.
    """
    files = _collect_files(cfg)
    if not files:
        return GateReport(passed=True, changed_files=0, note="no_changed_files")

    blocked_by: list[str] = []

    sandbox_all_valid: bool | None = None
    sandbox_summary = ""
    if cfg.sandbox_enabled:
        sandbox = cfg.sandbox or SandboxEngine(
            config=SandboxConfig(
                enabled=True,
                lint_enabled=True,
                type_check_enabled=True,
                run_enabled=cfg.sandbox_run,
                test_enabled=False,
            )
        )
        result: SandboxResult = await sandbox.validate_files(files, run_tests=cfg.run_tests)
        sandbox_all_valid = bool(result.all_valid)
        sandbox_summary = str(result.summary)
        if not sandbox_all_valid:
            blocked_by.append("sandbox")

    solution = Solution(
        task_id="merge-gate",
        title="Pull request changes",
        summary=f"Changed files: {len(files)}",
        code_blocks=[
            CodeBlock(filename=f["path"], language=f["language"], content=f["content"])
            for f in files
        ],
    )
    judge = cfg.judge or _default_judge()
    verdict: JudgeVerdict = await judge(solution)
    judge_score = float(verdict.scores.overall or 0.0)
    if judge_score < cfg.judge_threshold:
        blocked_by.append(f"judge_score={judge_score:.3f} < threshold={cfg.judge_threshold:.3f}")

    return GateReport(
        passed=not blocked_by,
        changed_files=len(files),
        judge_score=judge_score,
        judge_passed=bool(verdict.passed),
        judge_summary=str(verdict.summary),
        sandbox_all_valid=sandbox_all_valid,
        sandbox_summary=sandbox_summary,
        blocked_by=blocked_by,
    )


def _default_judge() -> JudgeRunner:
    from noema.llm.providers import create_llm_provider

    llm = create_llm_provider()

    async def _judge(solution: Solution) -> JudgeVerdict:
        return await evaluate_solution(
            llm,
            solution,
            task_description="Merge gate: review the pull request changes for quality and correctness.",
            task_tags=["merge-gate", "ci"],
        )

    return _judge


async def run_gate_cli(argv: list[str] | None = None) -> int:
    """Parse CLI args, run the merge gate, write the report, return the exit code."""
    parser = argparse.ArgumentParser(
        prog="noema.experiments.gate",
        description="Block merges when judge_score < threshold or the sandbox fails.",
    )
    parser.add_argument(
        "--diff-target", default="origin/main", help="Git diff base (default: origin/main)"
    )
    parser.add_argument(
        "--judge-threshold", type=float, default=0.0, help="Minimum judge score to pass"
    )
    parser.add_argument(
        "--sandbox", choices=["on", "off"], default="on", help="Run the sandbox stage"
    )
    parser.add_argument(
        "--sandbox-run",
        action="store_true",
        help="Execute the changed code (default: static/lint only)",
    )
    parser.add_argument("--out", default="gate.json", help="Path for the JSON report")
    args = parser.parse_args(argv)

    cfg = GateConfig(
        judge_threshold=args.judge_threshold,
        sandbox_enabled=args.sandbox == "on",
        sandbox_run=args.sandbox_run,
        diff_target=args.diff_target,
    )
    report = await run_merge_gate(cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.as_dict(), f, ensure_ascii=False, indent=2)

    status = "BLOCKED" if not report.passed else "PASSED"
    log.info(
        "merge_gate",
        status=status,
        judge_score=report.judge_score,
        sandbox_all_valid=report.sandbox_all_valid,
        blocked_by=report.blocked_by,
    )
    print(
        f"merge gate: {status} | judge_score={report.judge_score:.3f} | blocked_by={report.blocked_by}"
    )
    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run_gate_cli(argv))


if __name__ == "__main__":
    raise SystemExit(main())
