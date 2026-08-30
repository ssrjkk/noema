"""Tests for per-file token/cost attribution in experiment artifacts (T2.5)."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace
from typing import Any

import pytest

from noema.config.settings import reset_settings
from noema.core.types import CodeBlock, Solution, ThoughtProcess
from noema.experiments.runner import attribute_file_costs, load_experiment, run_experiment
from noema.tracing.tracer import get_tracer


class BlockEngine:
    """Engine whose solutions carry code blocks; tokens come from the tracer."""

    def __init__(self, provider: str, model: str | None, project_root: str) -> None:
        self.provider = provider
        self.model = model
        self.project_root = project_root
        self.tracer = get_tracer()

    async def think(self, task: Any) -> tuple[Solution, ThoughtProcess]:
        solution = Solution(
            task_id=task.id,
            title=f"Solution for {task.title}",
            summary="synthetic",
            code_blocks=[
                CodeBlock(
                    filename="src/main.py",
                    language="python",
                    content="\n".join(f"line {i}" for i in range(30)),
                ),
                CodeBlock(
                    filename="src/db.py",
                    language="python",
                    content="\n".join(f"row {i}" for i in range(10)),
                ),
            ],
        )
        return solution, ThoughtProcess(task_id=task.id)


def _config(tmp_path: Any) -> dict[str, Any]:
    cfg = {
        "meta": {"name": "cost-exp"},
        "providers": [{"provider": "fallback", "models": ["fallback"], "cost_per_token": 1e-5}],
        "tasks": [{"title": "Chat", "description": "d"}],
        "repetitions": 1,
    }
    import yaml

    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return load_experiment(path)


@pytest.fixture(autouse=True)
def _clean_settings():
    import os

    saved = os.environ.get("NOEMA_NEUROSYMBOLIC__ENABLED")
    yield
    if saved is None:
        os.environ.pop("NOEMA_NEUROSYMBOLIC__ENABLED", None)
    else:
        os.environ["NOEMA_NEUROSYMBOLIC__ENABLED"] = saved
    reset_settings()


class TestAttributeFileCosts:
    def test_splits_tokens_by_lines_and_sums_back(self):
        blocks = [
            SimpleNamespace(filename="a.py", content="x\n" * 30),
            SimpleNamespace(filename="b.py", content="y\n" * 10),
        ]
        rows = attribute_file_costs(blocks, tokens_input=80, tokens_output=40, cost_per_token=0.001)
        assert len(rows) == 2
        assert rows[0]["file_path"] == "a.py"
        assert rows[0]["lines"] == 30
        # 75%/25% line split, sums exact.
        assert rows[0]["tokens_input"] + rows[1]["tokens_input"] == 80
        assert rows[0]["tokens_output"] + rows[1]["tokens_output"] == 40
        assert rows[0]["tokens_input"] == 60 and rows[1]["tokens_input"] == 20
        # a.py: 60+30 tokens, b.py: 20+10; total = run total of 120.
        assert rows[0]["cost_estimate"] == pytest.approx(0.001 * 90)
        assert rows[1]["cost_estimate"] == pytest.approx(0.001 * 30)
        assert rows[0]["cost_estimate"] + rows[1]["cost_estimate"] == pytest.approx(0.001 * 120)

    def test_empty_when_no_blocks(self):
        assert attribute_file_costs([], 100, 100, 1.0) == []

    def test_empty_when_no_tokens(self):
        blocks = [SimpleNamespace(filename="a.py", content="x\n" * 10)]
        assert attribute_file_costs(blocks, 0, 0, 1.0) == []

    def test_empty_when_no_lines(self):
        blocks = [SimpleNamespace(filename="a.py", content="")]
        assert attribute_file_costs(blocks, 10, 10, 1.0) == []

    def test_rounding_drift_absorbed_by_last_file(self):
        blocks = [SimpleNamespace(filename=f"f{i}.py", content="x\n" * 3) for i in range(3)]
        rows = attribute_file_costs(blocks, tokens_input=100, tokens_output=0, cost_per_token=0.0)
        assert sum(r["tokens_input"] for r in rows) == 100


class TestArtifacts:
    def test_results_json_contains_file_costs(self, tmp_path):
        cfg = _config(tmp_path)
        out = tmp_path / "out"
        results = run_experiment(cfg, out_dir=out, engine_factory=BlockEngine, run_id="r1")
        record = results["records"][0]
        assert "file_costs" in record
        # fallback provider produced no real tokens → nothing to attribute
        # but the key is always present (schema stability).
        assert record["file_costs"] == [] or record["file_costs"][0]["file_path"] == "src/main.py"

    def test_runs_csv_stays_flat_without_file_costs(self, tmp_path):
        cfg = _config(tmp_path)
        out = tmp_path / "out"
        run_experiment(cfg, out_dir=out, engine_factory=BlockEngine, run_id="r1")
        with open(out / "cost-exp" / "r1" / "runs.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows
        assert "file_costs" not in rows[0]
        assert "cost_estimate" in rows[0]

    def test_results_json_on_disk_has_file_costs_key(self, tmp_path):
        cfg = _config(tmp_path)
        out = tmp_path / "out"
        run_experiment(cfg, out_dir=out, engine_factory=BlockEngine, run_id="r1")
        records = json.loads((out / "cost-exp" / "r1" / "results.json").read_text(encoding="utf-8"))
        assert all("file_costs" in r for r in records)
