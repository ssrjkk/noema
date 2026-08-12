"""Regression tests for confirmed bug fixes (no mocks, real behavior only).

Covers:
- BUG1: WorkerPool no longer leaks task bookkeeping.
- BUG2: judge score parsing coerces string/boolean/NaN scores instead of crashing.
- BUG3: sandbox ``max_parallel`` actually bounds concurrent execution.
- BUG4: sandbox test counting parses the pytest summary (not uppercase markers).
"""

import json
import time

import pytest

from noema.core.types import Solution
from noema.judge import _parse_pairwise, _parse_verdict
from noema.sandbox.engine import SandboxConfig, SandboxEngine, _parse_pytest_counts
from noema.workers.pool import WorkerPool

# ── BUG1: WorkerPool bookkeeping leak ────────────────────────────────────


@pytest.mark.asyncio
async def test_workerpool_releases_task_bookkeeping():
    pool = WorkerPool(max_workers=2)
    await pool.start()
    try:

        async def _identity(x):
            return x

        for i in range(10):
            assert await pool.submit(_identity, i) == i
        assert pool._tasks == {}
        assert pool._done_events == {}
        assert pool.stats["total_submitted"] == 10
        assert pool.stats["total_completed"] == 10
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_workerpool_releases_bookkeeping_on_failure():
    pool = WorkerPool(max_workers=1)
    await pool.start()
    try:

        async def _boom():
            raise ValueError("kaboom")

        with pytest.raises(ValueError, match="kaboom"):
            await pool.submit(_boom)
        assert pool._tasks == {}
        assert pool._done_events == {}
        assert pool.stats["total_failed"] == 1
    finally:
        await pool.shutdown()


# ── BUG2: judge score coercion ───────────────────────────────────────────


def _solution() -> Solution:
    return Solution(task_id="t1", title="task", summary="summary")


def test_judge_parses_string_scores_without_crashing():
    raw = json.dumps(
        {
            "scores": {
                "architecture": "0.8",
                "code_quality": 0.6,
                "security": "0.9",
                "performance": 1.7,  # out of range
                "maintainability": -0.2,  # out of range
                "completeness": "NaN",
                "overall": "0.75",
            },
            "summary": "ok",
        }
    )
    verdict = _parse_verdict(raw, _solution())
    assert verdict.passed is True
    assert verdict.scores.architecture == 0.8
    assert verdict.scores.code_quality == 0.6
    assert verdict.scores.security == 0.9
    assert verdict.scores.performance == 1.0
    assert verdict.scores.maintainability == 0.0
    assert verdict.scores.completeness == 0.0
    assert verdict.scores.overall == 0.75


def test_judge_passed_false_when_score_below_threshold():
    verdict = _parse_verdict(json.dumps({"scores": {"overall": "0.3"}}), _solution())
    assert verdict.passed is False
    assert verdict.scores.overall == 0.3


def test_judge_handles_boolean_and_null_scores():
    verdict = _parse_verdict(
        json.dumps({"scores": {"overall": True, "architecture": None, "security": 0}}),
        _solution(),
    )
    assert verdict.scores.overall == 1.0
    assert verdict.scores.architecture == 0.0
    assert verdict.scores.security == 0.0


def test_judge_pairwise_coerces_string_scores():
    raw = json.dumps(
        {
            "winner": "A",
            "scores_a": {"overall": "0.9", "architecture": "0.7"},
            "scores_b": {"overall": 0.4},
        }
    )
    result = _parse_pairwise(raw, _solution(), _solution())
    assert result.winner == "A"
    assert result.winner_index == 0
    assert result.scores_a.overall == 0.9
    assert result.scores_b.overall == 0.4


# ── BUG4: pytest count parsing ───────────────────────────────────────────


def test_pytest_counts_parse_summary_line():
    green = "=============== 3 passed, 1 skipped in 0.42s ===============\n"
    assert _parse_pytest_counts(green, 0) == (3, 0)

    red = "================ 1 failed, 2 passed in 0.30s ================\n"
    assert _parse_pytest_counts(red, 1) == (2, 1)

    errored = "=============== 1 error, 2 passed in 0.30s ================\n"
    assert _parse_pytest_counts(errored, 1) == (2, 1)


def test_pytest_counts_fallback_to_returncode():
    assert _parse_pytest_counts("no summary here", 0) == (1, 0)
    assert _parse_pytest_counts("no summary here", 1) == (0, 1)


@pytest.mark.asyncio
async def test_sandbox_counts_green_tests():
    engine = SandboxEngine(
        config=SandboxConfig(
            enabled=True,
            lint_enabled=False,
            type_check_enabled=False,
            run_enabled=False,
            test_enabled=True,
            max_parallel=1,
        )
    )
    engine._has_docker = False
    engine._has_bwrap = False
    result = await engine.validate_files(
        [
            {
                "path": "test_ok.py",
                "language": "python",
                "content": "def test_ok():\n    assert 1 + 1 == 2\n",
            }
        ],
        run_tests=True,
    )
    assert result.tests_passed == 1
    assert result.tests_failed == 0


@pytest.mark.asyncio
async def test_sandbox_counts_failing_tests():
    engine = SandboxEngine(
        config=SandboxConfig(
            enabled=True,
            lint_enabled=False,
            type_check_enabled=False,
            run_enabled=False,
            test_enabled=True,
            max_parallel=1,
        )
    )
    engine._has_docker = False
    engine._has_bwrap = False
    result = await engine.validate_files(
        [
            {
                "path": "test_bad.py",
                "language": "python",
                "content": "def test_bad():\n    assert 1 == 2\n",
            }
        ],
        run_tests=True,
    )
    assert result.tests_passed == 0
    assert result.tests_failed == 1


# ── BUG3: sandbox max_parallel bounds concurrency ────────────────────────


@pytest.mark.asyncio
async def test_sandbox_max_parallel_serializes_execution():
    engine = SandboxEngine(
        config=SandboxConfig(
            enabled=True,
            lint_enabled=False,
            type_check_enabled=False,
            run_enabled=True,
            test_enabled=False,
            max_parallel=1,
            max_cpu_seconds=20,
        )
    )
    engine._has_docker = False
    engine._has_bwrap = False
    files = [
        {
            "path": f"f{i}.py",
            "language": "python",
            "content": "import time\ntime.sleep(0.8)\n",
        }
        for i in range(3)
    ]
    t0 = time.monotonic()
    result = await engine.validate_files(files)
    elapsed = time.monotonic() - t0
    assert result.all_valid is True
    # 3 files x 0.8s serialized must take clearly longer than a parallel run (~0.9s).
    assert elapsed >= 2.0
