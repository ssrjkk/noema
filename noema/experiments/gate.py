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
    """Tunables for one merge-gate run.

    ``verifier``: optional formal verifier (e.g. :class:`LeanVerifier`).
    When set, every ``.lean`` file in the change set is theorem-checked and a
    failed proof blocks the gate. ``require_spec_patterns`` (path prefixes)
    additionally blocks changed ``.py`` files under those prefixes that have
    no matching ``.lean`` spec — the anti-vacuum guarantee for critical code.
    """

    judge_threshold: float = 0.0
    sandbox_enabled: bool = True
    sandbox_run: bool = False
    run_tests: bool = False
    diff_target: str = "origin/main"
    include_suffixes: tuple[str, ...] = (".py",)
    explicit_files: list[dict[str, str]] | None = None
    git: GitRunner = run_git
    judge: JudgeRunner | None = None
    sandbox: SandboxEngine | None = None
    verifier: Any | None = None
    require_spec_patterns: tuple[str, ...] = ()


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
    formal_verified: bool | None = None
    formal_error: str = ""
    blocked_by: list[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _raw_changed_files(cfg: GateConfig) -> list[dict[str, str]]:
    """Read every changed file (no suffix filter) from explicit files or git.

    ``cfg.explicit_files`` short-circuits the git diff so the autonomy loop
    can gate in-memory PR contents without a repository checkout.
    """
    if cfg.explicit_files is not None:
        return [
            {
                "path": str(f["path"]),
                "language": "python" if str(f["path"]).endswith(".py") else "text",
                "content": str(f["content"]),
            }
            for f in cfg.explicit_files
        ]
    files: list[dict[str, str]] = []
    for path in _changed_file_paths(cfg.git, cfg.diff_target):
        file_path = Path(path)
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            continue  # deleted or unreadable files are not gated
        language = "python" if path.endswith(".py") else "text"
        files.append({"path": path, "language": language, "content": content})
    return files


def _collect_files(cfg: GateConfig) -> list[dict[str, str]]:
    """Read the changed files (filtered by suffix) from the worktree.

    When ``cfg.explicit_files`` is set (e.g. the autonomy loop has the PR's
    file contents in memory), those are used verbatim instead of diffing the
    local checkout — no git repository is required.
    """
    return [f for f in _raw_changed_files(cfg) if f["path"].endswith(cfg.include_suffixes)]


def _missing_formal_specs(
    files: list[dict[str, str]],
    lean_paths: list[str],
    require_patterns: tuple[str, ...],
) -> list[str]:
    """Changed ``.py`` files under a required prefix without a ``.lean`` spec.

    A spec is considered matching when it sits next to the file
    (``app/billing.lean``) or under ``specs/`` (``specs/app/billing.lean`` or
    ``specs/billing.lean``). Any other layout must be covered by a follow-up
    change to this policy.
    """
    spec_set = set(lean_paths)
    missing: list[str] = []
    for f in files:
        path = f["path"]
        if not path.endswith(".py") or not any(path.startswith(p) for p in require_patterns):
            continue
        stem = path[: -len(".py")]
        candidates = (f"{stem}.lean", f"specs/{stem}.lean", f"specs/{Path(stem).name}.lean")
        if not any(c in spec_set for c in candidates):
            missing.append(path)
    return missing


async def run_merge_gate(cfg: GateConfig) -> GateReport:
    """Evaluate the PR's changed files and decide pass/block.

    Blocks when:
    - the sandbox verdict is not ``all_valid`` (structural/lint failures), or
    - a configured formal verifier rejects a ``.lean`` proof obligation, or
    - a changed file under ``require_spec_patterns`` has no matching ``.lean``
      spec (``missing_formal_spec``), or
    - the judge's overall score is below ``judge_threshold``.

    The formal stage runs over *every* changed file (including PRs that touch
    only ``.lean`` specs), so spec-only changes are still theorem-checked.

    The gate never raises: a crashing verifier, sandbox, or judge maps to a
    blocked verdict (``verifier_crashed`` / ``sandbox_error`` / ``judge_error``)
    so every run produces a report artifact.
    """
    raw_files = _raw_changed_files(cfg)
    files = [f for f in raw_files if f["path"].endswith(cfg.include_suffixes)]

    blocked_by: list[str] = []
    formal_verified: bool | None = None
    formal_error = ""
    if cfg.verifier is not None:
        lean_files = [(f["path"], f["content"]) for f in raw_files if f["path"].endswith(".lean")]
        if lean_files:
            try:
                vres = await cfg.verifier.verify_files(lean_files)
                formal_verified = bool(vres.verified)
                formal_error = str(getattr(vres, "error", ""))
            except Exception as e:  # noqa: BLE001 - a crashing verifier blocks
                formal_verified = False
                formal_error = f"verifier_crashed:{e}"
            if not formal_verified:
                blocked_by.append("formal_verification")
        if cfg.require_spec_patterns:
            for path in _missing_formal_specs(
                files, [p for p, _ in lean_files], cfg.require_spec_patterns
            ):
                blocked_by.append(f"missing_formal_spec:{path}")

    if not files:
        return GateReport(
            passed=not blocked_by,
            changed_files=0,
            formal_verified=formal_verified,
            formal_error=formal_error,
            blocked_by=blocked_by,
            note="no_changed_files",
        )

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
        try:
            result: SandboxResult = await sandbox.validate_files(files, run_tests=cfg.run_tests)
            sandbox_all_valid = bool(result.all_valid)
            sandbox_summary = str(result.summary)
        except Exception as e:  # noqa: BLE001 - a crashing sandbox blocks
            sandbox_all_valid = False
            sandbox_summary = f"sandbox_error:{e}"
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
    judge_score = 0.0
    judge_passed = False
    judge_summary = ""
    try:
        verdict: JudgeVerdict = await judge(solution)
        judge_score = float(verdict.scores.overall or 0.0)
        judge_passed = bool(verdict.passed)
        judge_summary = str(verdict.summary)
    except Exception as e:  # noqa: BLE001 - a failing judge blocks, never crashes
        blocked_by.append(f"judge_error:{e}")
    if judge_score < cfg.judge_threshold:
        blocked_by.append(f"judge_score={judge_score:.3f} < threshold={cfg.judge_threshold:.3f}")

    return GateReport(
        passed=not blocked_by,
        changed_files=len(files),
        judge_score=judge_score,
        judge_passed=judge_passed,
        judge_summary=judge_summary,
        sandbox_all_valid=sandbox_all_valid,
        sandbox_summary=sandbox_summary,
        formal_verified=formal_verified,
        formal_error=formal_error,
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
    parser.add_argument(
        "--verifier",
        action="store_true",
        help="Enable the Lean 4 formal verification stage (compiles .lean "
        "proof obligations; fail-closed when the binary is missing)",
    )
    parser.add_argument(
        "--require-spec",
        action="append",
        default=[],
        metavar="PREFIX",
        help="Path prefix whose changed .py files must ship a matching .lean "
        "spec or the gate blocks (repeatable)",
    )
    parser.add_argument("--out", default="gate.json", help="Path for the JSON report")
    args = parser.parse_args(argv)

    cfg = GateConfig(
        judge_threshold=args.judge_threshold,
        sandbox_enabled=args.sandbox == "on",
        sandbox_run=args.sandbox_run,
        diff_target=args.diff_target,
        require_spec_patterns=tuple(args.require_spec),
    )
    if args.verifier:
        from noema.verifiers.lean import LeanVerifier

        cfg.verifier = LeanVerifier()
    try:
        report = await run_merge_gate(cfg)
    except Exception as e:  # noqa: BLE001 - every gate run must yield an artifact
        log.error("merge_gate_crash", error=str(e))
        report = GateReport(
            passed=False,
            changed_files=0,
            blocked_by=[f"gate_error:{e}"],
            note="gate_error",
        )

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
