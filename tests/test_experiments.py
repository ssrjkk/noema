"""Tests for the reproducible experiment runner."""

from __future__ import annotations

import csv
import json
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from noema.config.settings import reset_settings
from noema.core.types import Solution, ThoughtProcess
from noema.experiments.runner import RECORD_FIELDS, load_experiment, run_experiment
from noema.llm.providers import FallbackProvider
from noema.tracing.tracer import get_tracer

if TYPE_CHECKING:
    from pathlib import Path


class FakeEngine:
    """Deterministic engine stand-in that never touches a real LLM."""

    def __init__(self, provider: str, model: str | None, project_root: str) -> None:
        self.provider = provider
        self.model = model
        self.project_root = project_root
        self.llm = FallbackProvider()
        self.tracer = get_tracer()
        self.fail_on_think = False
        self.think_calls = 0

    async def think(self, task: Any) -> tuple[Solution, ThoughtProcess]:
        self.think_calls += 1
        if self.fail_on_think:
            raise RuntimeError("boom")
        solution = Solution(
            task_id=task.id,
            title=f"Solution for {task.title}",
            summary="A synthetic deterministic solution.",
        )
        return solution, ThoughtProcess(task_id=task.id)

    async def validate_solution(self, solution: Any, run_tests: bool = False) -> SimpleNamespace:
        return SimpleNamespace(all_valid=True, summary="1/1 files valid")


def make_factory(fail_on_think: bool = False) -> Any:
    def factory(provider: str, model: str | None, project_root: str) -> FakeEngine:
        engine = FakeEngine(provider, model, project_root)
        engine.fail_on_think = fail_on_think
        return engine

    return factory


@pytest.fixture()
def restore_settings() -> Any:
    saved = os.environ.get("NOEMA_NEUROSYMBOLIC__ENABLED")
    yield
    if saved is None:
        os.environ.pop("NOEMA_NEUROSYMBOLIC__ENABLED", None)
    else:
        os.environ["NOEMA_NEUROSYMBOLIC__ENABLED"] = saved
    reset_settings()


def write_config(tmp_path: Path, extra: dict[str, Any] | None = None) -> Path:
    cfg: dict[str, Any] = {
        "meta": {"name": "test-exp", "seed": 42},
        "providers": [{"provider": "fallback", "models": ["fallback"]}],
        "tasks": [
            {"title": "Chat App", "description": "Design a chat app.", "tags": ["python"]},
            {"title": "REST API", "description": "Design a REST API.", "tags": ["fastapi"]},
        ],
        "repetitions": 2,
    }
    if extra:
        cfg.update(extra)
    path = tmp_path / "experiment.yaml"
    path.write_text(_to_yaml(cfg), encoding="utf-8")
    return path


def _to_yaml(cfg: dict[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(cfg, allow_unicode=True)


def test_load_experiment_valid(tmp_path: Path) -> None:
    path = write_config(tmp_path)
    cfg = load_experiment(path)
    assert cfg["meta"]["name"] == "test-exp"
    assert cfg["providers"][0]["provider"] == "fallback"
    assert len(cfg["tasks"]) == 2


@pytest.mark.parametrize(
    "mutator",
    [
        lambda c: c.update({"providers": []}),
        lambda c: c.update({"providers": [{"provider": "x"}]}),
        lambda c: c.update({"tasks": []}),
        lambda c: c.update({"repetitions": 0}),
        lambda c: c.pop("meta"),
    ],
)
def test_load_experiment_rejects_bad_config(tmp_path: Path, mutator: Any) -> None:
    cfg: dict[str, Any] = {
        "meta": {"name": "bad"},
        "providers": [{"provider": "fallback", "models": ["fallback"]}],
        "tasks": [{"title": "t", "description": "d"}],
        "repetitions": 1,
    }
    mutator(cfg)
    path = tmp_path / "bad.yaml"
    path.write_text(_to_yaml(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment(path)


def test_run_experiment_writes_schema(tmp_path: Path, restore_settings: Any) -> None:
    cfg_path = write_config(tmp_path)
    cfg = load_experiment(cfg_path)
    out = tmp_path / "results"
    results = run_experiment(cfg, out_dir=out, engine_factory=make_factory(), run_id="run-1")

    assert results["experiment_id"] == "test-exp"
    assert results["run_id"] == "run-1"
    assert len(results["records"]) == 2 * 2 * 1  # tasks x reps x (provider, model)

    records_path = out / "test-exp" / "run-1" / "results.json"
    assert records_path.is_file()
    records = json.loads(records_path.read_text(encoding="utf-8"))
    assert len(records) == 4
    for record in records:
        for field in RECORD_FIELDS:
            assert field in record, f"missing field {field} in {record}"
    assert all(r["provider"] == "fallback" for r in records)
    assert all(r["error"] is None for r in records)
    assert all(r["repeat_index"] in (1, 2) for r in records)


def test_run_experiment_captures_errors(tmp_path: Path, restore_settings: Any) -> None:
    cfg = load_experiment(write_config(tmp_path))
    out = tmp_path / "results"
    results = run_experiment(cfg, out_dir=out, engine_factory=make_factory(fail_on_think=True))

    assert len(results["records"]) == 4
    assert all(r["error"] is not None for r in results["records"])
    assert all("boom" in r["error"] for r in results["records"])


def test_summary_csv_has_expected_columns(tmp_path: Path, restore_settings: Any) -> None:
    cfg = load_experiment(write_config(tmp_path))
    out = tmp_path / "results"
    run_experiment(cfg, out_dir=out, engine_factory=make_factory(), run_id="run-1")

    summary_path = out / "test-exp" / "run-1" / "summary.csv"
    assert summary_path.is_file()
    with open(summary_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["provider"] == "fallback"
    for col in ("mean_judge_score", "mean_latency_ms", "mean_total_tokens", "mean_cost"):
        assert col in rows[0]
    assert rows[0]["n"] == "4"


def test_sandbox_recorded_when_enabled(tmp_path: Path, restore_settings: Any) -> None:
    cfg = load_experiment(
        write_config(tmp_path, {"settings": {"sandbox": True, "retrieval": False}})
    )
    out = tmp_path / "results"
    results = run_experiment(cfg, out_dir=out, engine_factory=make_factory(), run_id="run-1")

    assert all(r["sandbox_all_valid"] is True for r in results["records"])
    assert all(r["sandbox_summary"] == "1/1 files valid" for r in results["records"])
