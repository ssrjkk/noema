"""Tests for core/engine.py — helper functions and utility methods."""

from __future__ import annotations

import pytest

from noema.core.engine import NoemaEngine, _parse_memory_mb, _sandbox_config_from_settings
from noema.core.types import (
    Solution,
    SolutionQuality,
    Task,
    TaskComplexity,
    ThoughtProcess,
)

# ── _parse_memory_mb ──────────────────────────────────────────────────────────


def test_parse_memory_mb_plain_number():
    assert _parse_memory_mb("256") == 256


def test_parse_memory_mb_megabytes():
    assert _parse_memory_mb("512m") == 512


def test_parse_memory_mb_gigabytes():
    assert _parse_memory_mb("1g") == 1024


def test_parse_memory_mb_kilobytes():
    assert _parse_memory_mb("2048k") == 2


def test_parse_memory_mb_invalid():
    assert _parse_memory_mb("invalid") == 256


def test_parse_memory_mb_empty():
    assert _parse_memory_mb("") == 256


def test_parse_memory_mb_with_spaces():
    assert _parse_memory_mb("  512 m  ") == 512


def test_parse_memory_mb_zero_g():
    assert _parse_memory_mb("0g") == 1


def test_parse_memory_mb_zero_k():
    assert _parse_memory_mb("500k") == 1  # 500 // 1024 = 0, max(1, 0) = 1


# ── _sandbox_config_from_settings ──────────────────────────────────────────────


def test_sandbox_config_from_settings():
    sb = pytest.importorskip("noema.config.settings").SandboxSettings(
        enabled=True,
        timeout=30,
        max_memory="512m",
        max_cpus=2,
        network_disabled=True,
        docker_image="python:3.12-slim",
    )
    config = _sandbox_config_from_settings(sb)
    assert config.enabled is True
    assert config.timeout == 30
    assert config.max_memory_mb == 512
    assert config.max_cpus == 2
    assert config.network_isolation is True
    assert config.docker_image == "python:3.12-slim"


# ── _evaluate_quality ─────────────────────────────────────────────────────────


def _make_engine_minimal():
    """Create a NoemaEngine without full initialization for testing helpers."""
    with pytest.importorskip("unittest.mock").patch("noema.core.engine.get_settings") as ms:
        settings = ms.return_value
        settings.sandbox.enabled = False
        settings.sandbox.timeout = 30
        settings.sandbox.max_memory = "256m"
        settings.sandbox.max_cpus = 1
        settings.sandbox.network_disabled = True
        settings.sandbox.docker_image = "python:3.12"
        settings.sandbox.verify_think = False
        settings.sandbox.verify_think_enforce = False
        settings.neurosymbolic.enabled = False
        settings.neurosymbolic.trace_dir = ""
        settings.think_timeout_seconds = 120
        settings.judge.enforce = False
        settings.ontology_persist_path = ".noema/ontology.json"

        with pytest.importorskip("unittest.mock").patch("noema.core.engine.get_tracer") as mt:
            tracer = mt.return_value
            tracer.current_span_id = "test-span"

            engine = NoemaEngine()
    return engine


def test_evaluate_quality_masterpiece():
    engine = _make_engine_minimal()
    task = Task(title="test", complexity=TaskComplexity.SIMPLE)
    solution = Solution(task_id=task.id, title="sol", summary="")
    thought = ThoughtProcess(task_id=task.id)
    for _ in range(3):
        thought.add_step("kernel", "input", "output", 0.95)

    result = engine._evaluate_quality(solution, thought)
    assert result.quality == SolutionQuality.MASTERPIECE
    assert result.confidence >= 0.9


def test_evaluate_quality_excellent():
    engine = _make_engine_minimal()
    task = Task(title="test", complexity=TaskComplexity.SIMPLE)
    solution = Solution(task_id=task.id, title="sol", summary="")
    thought = ThoughtProcess(task_id=task.id)
    for _ in range(3):
        thought.add_step("kernel", "input", "output", 0.8)

    result = engine._evaluate_quality(solution, thought)
    assert result.quality == SolutionQuality.EXCELLENT


def test_evaluate_quality_good():
    engine = _make_engine_minimal()
    task = Task(title="test", complexity=TaskComplexity.SIMPLE)
    solution = Solution(task_id=task.id, title="sol", summary="")
    thought = ThoughtProcess(task_id=task.id)
    for _ in range(3):
        thought.add_step("kernel", "input", "output", 0.65)

    result = engine._evaluate_quality(solution, thought)
    assert result.quality == SolutionQuality.GOOD


def test_evaluate_quality_acceptable():
    engine = _make_engine_minimal()
    task = Task(title="test", complexity=TaskComplexity.SIMPLE)
    solution = Solution(task_id=task.id, title="sol", summary="")
    thought = ThoughtProcess(task_id=task.id)
    for _ in range(3):
        thought.add_step("kernel", "input", "output", 0.45)

    result = engine._evaluate_quality(solution, thought)
    assert result.quality == SolutionQuality.ACCEPTABLE


def test_evaluate_quality_draft():
    engine = _make_engine_minimal()
    task = Task(title="test", complexity=TaskComplexity.SIMPLE)
    solution = Solution(task_id=task.id, title="sol", summary="")
    thought = ThoughtProcess(task_id=task.id)
    for _ in range(3):
        thought.add_step("kernel", "input", "output", 0.2)

    result = engine._evaluate_quality(solution, thought)
    assert result.quality == SolutionQuality.DRAFT


# ── _safe_parse ───────────────────────────────────────────────────────────────


def test_safe_parse_dict():
    engine = _make_engine_minimal()
    d = {"key": "value"}
    assert engine._safe_parse(d) is d


def test_safe_parse_list():
    engine = _make_engine_minimal()
    lst = [1, 2, 3]
    assert engine._safe_parse(lst) is lst


def test_safe_parse_json_string():
    engine = _make_engine_minimal()
    result = engine._safe_parse('{"key": "value"}')
    assert isinstance(result, dict)
    assert result["key"] == "value"


def test_safe_parse_fenced_json():
    engine = _make_engine_minimal()
    result = engine._safe_parse('```json\n{"key": "value"}\n```')
    assert isinstance(result, dict)
    assert result["key"] == "value"


def test_safe_parse_raw_string():
    engine = _make_engine_minimal()
    result = engine._safe_parse("just plain text")
    assert isinstance(result, dict)
    assert "raw" in result


def test_safe_parse_non_string():
    engine = _make_engine_minimal()
    assert engine._safe_parse(42) == {}
    assert engine._safe_parse(None) == {}


# ── _extract_summary ─────────────────────────────────────────────────────────


def test_extract_summary_with_architecture():
    engine = _make_engine_minimal()
    reasoning = {
        "architecture": '{"high_level_design": "Layered architecture with 3 tiers"}',
        "review": '{"final_summary": "Good solution"}',
    }
    summary = engine._extract_summary(reasoning)
    assert "Layered" in summary
    assert "Good" in summary


def test_extract_summary_empty():
    engine = _make_engine_minimal()
    summary = engine._extract_summary({})
    assert "Chain-of-Thought" in summary


def test_extract_summary_only_review():
    engine = _make_engine_minimal()
    reasoning = {"review": '{"final_summary": "Review only"}'}
    summary = engine._extract_summary(reasoning)
    assert "Review only" in summary


# ── _build_reasoning ─────────────────────────────────────────────────────────


def test_build_reasoning():
    engine = _make_engine_minimal()
    task = Task(title="test", complexity=TaskComplexity.SIMPLE)
    solution = Solution(task_id=task.id, title="Test Solution", summary="")
    solution.quality = SolutionQuality.GOOD
    thought = ThoughtProcess(task_id=task.id)
    thought.add_step("analysis", "input", "output summary", 0.8)

    reasoning = engine._build_reasoning(thought, solution)
    assert "Test Solution" in reasoning
    assert "Steps: 1" in reasoning
    assert "analysis" in reasoning
    assert "GOOD" in reasoning or "good" in reasoning


# ── _prepare_neurosymbolic_task ───────────────────────────────────────────────


def test_prepare_neurosymbolic_task():
    from noema.core.types import Requirement

    engine = _make_engine_minimal()
    task = Task(
        title="Build API",
        description="REST API",
        requirements=[
            Requirement(category="perf", description="fast", priority=8, constraints=["<100ms"]),
            Requirement(category="security", description="auth", priority=9, constraints=[]),
        ],
    )
    result = engine._prepare_neurosymbolic_task(task)
    assert "requirements" in result
    assert len(result["requirements"]) == 2
    assert result["requirements"][0]["name"] == "perf"
    assert result["requirements"][0]["constraints"] == ["<100ms"]
    assert result["goals"] == ["Build API", "REST API"]


def test_prepare_neurosymbolic_task_no_requirements():
    engine = _make_engine_minimal()
    task = Task(title="Simple", description="task")
    result = engine._prepare_neurosymbolic_task(task)
    assert result["requirements"] == []


# ── search_memory / memory_stats ──────────────────────────────────────────────


def test_search_memory():
    engine = _make_engine_minimal()
    result = engine.search_memory("test query")
    assert "episodes" in result
    assert "knowledge" in result
    assert "procedures" in result


def test_search_memory_episodes_only():
    engine = _make_engine_minimal()
    result = engine.search_memory("test", kind="episodes")
    assert "episodes" in result
    assert "knowledge" not in result


def test_memory_stats():
    engine = _make_engine_minimal()
    stats = engine.memory_stats()
    assert "episodic_count" in stats


# ── list_modules / modules_stats ──────────────────────────────────────────────


def test_list_modules():
    engine = _make_engine_minimal()
    mods = engine.list_modules()
    assert isinstance(mods, list)


def test_modules_stats():
    engine = _make_engine_minimal()
    stats = engine.modules_stats()
    assert "total_modules" in stats


# ── discover_resources ────────────────────────────────────────────────────────


def test_discover_resources():
    engine = _make_engine_minimal()
    result = engine.discover_resources()
    assert "providers_available" in result


# ── register_kernel ───────────────────────────────────────────────────────────


def test_register_kernel():
    engine = _make_engine_minimal()
    mock_kernel = pytest.importorskip("unittest.mock").MagicMock()
    mock_kernel.name = "test_kernel"
    engine.register_kernel(mock_kernel)
    assert "test_kernel" in engine.kernels
