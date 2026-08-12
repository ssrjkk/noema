"""End-to-end tests for the Benchmark-as-a-service API.

Real engine run (fallback provider, no network) through ``POST /experiments``:
the matrix executes, artifacts land in ``results/``, and the read endpoints
serve the existing ``results.json``/``summary.json`` schema.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from noema.api.server import app

if TYPE_CHECKING:
    from pathlib import Path

EXPERIMENT_YAML = """\
meta:
  name: "api-smoke"
  description: "API smoke benchmark."
  seed: 42
providers:
  - provider: fallback
    models:
      - fallback
tasks:
  - title: "Hello CLI"
    description: "Write a Python CLI that prints hello."
    tags: ["python"]
repetitions: 1
settings:
  neurosymbolic: false
  timeout_seconds: 60
"""

BROKEN_YAML = """\
meta:
  name: "broken"
providers: []
tasks: []
"""


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("experiments")


def _post(client: TestClient, out_dir: Path, body: str) -> Any:
    return client.post(
        f"/experiments?out_dir={out_dir}",
        content=body,
        headers={"Content-Type": "text/yaml"},
    )


@pytest.fixture(scope="module")
def run_result(client: TestClient, out_dir: Path) -> dict:
    resp = _post(client, out_dir, EXPERIMENT_YAML)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ── POST /experiments ─────────────────────────────────────────────────────


def test_post_returns_run_id_and_status(run_result: dict) -> None:
    assert run_result["experiment_id"] == "api-smoke"
    assert run_result["run_id"]
    assert run_result["status"] == "completed"
    assert run_result["n_records"] == 1
    assert run_result["generated_at"]
    assert isinstance(run_result["summary"], list)


def test_post_lands_artifacts_in_results(run_result: dict, out_dir: Path) -> None:
    run_dir = out_dir / "api-smoke" / run_result["run_id"]
    assert (run_dir / "results.json").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "runs.csv").is_file()

    records = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert len(records) == 1
    record = records[0]
    assert record["experiment_id"] == "api-smoke"
    assert record["provider"] == "fallback"
    assert record["model"] == "fallback"
    assert record["repeat_index"] == 1
    assert record["task_id"]
    assert record["duration_ms"] > 0
    assert record["error"] is None

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == run_result["run_id"]
    assert summary["n_records"] == 1
    assert summary["summary"][0]["provider"] == "fallback"
    assert summary["summary"][0]["n"] == 1


def test_post_rejects_invalid_config(client: TestClient, out_dir: Path) -> None:
    resp = _post(client, out_dir, BROKEN_YAML)
    assert resp.status_code == 400
    assert "providers" in resp.json()["detail"]


def test_post_rejects_non_utf8(client: TestClient, out_dir: Path) -> None:
    resp = client.post(
        f"/experiments?out_dir={out_dir}",
        content=b"\xff\xfe\x00",
        headers={
            "Content-Type": "text/plain",
        },
    )
    assert resp.status_code == 400


# ── GET endpoints ──────────────────────────────────────────────────────────


def test_get_run_returns_records_and_summary(
    client: TestClient, out_dir: Path, run_result: dict
) -> None:
    resp = client.get(f"/experiments/api-smoke/runs/{run_result['run_id']}?out_dir={out_dir}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["experiment_id"] == "api-smoke"
    assert body["run_id"] == run_result["run_id"]
    assert len(body["records"]) == 1
    assert body["records"][0]["provider"] == "fallback"
    assert isinstance(body["summary"], dict)
    assert body["summary"]["run_id"] == run_result["run_id"]


def test_list_runs_contains_posted_run(client: TestClient, out_dir: Path, run_result: dict) -> None:
    resp = client.get(f"/experiments/api-smoke/runs?out_dir={out_dir}")
    assert resp.status_code == 200, resp.text
    runs = resp.json()
    assert any(r["run_id"] == run_result["run_id"] for r in runs)
    assert any(r["n_records"] == 1 for r in runs)


def test_list_experiments_contains_posted_experiment(
    client: TestClient, out_dir: Path, run_result: dict
) -> None:
    resp = client.get(f"/experiments?out_dir={out_dir}")
    assert resp.status_code == 200, resp.text
    assert any(e["experiment_id"] == "api-smoke" and e["n_runs"] >= 1 for e in resp.json())


def test_unknown_run_is_404(client: TestClient, out_dir: Path) -> None:
    resp = client.get(f"/experiments/api-smoke/runs/does-not-exist?out_dir={out_dir}")
    assert resp.status_code == 404


def test_unknown_experiment_is_404(client: TestClient, out_dir: Path) -> None:
    resp = client.get(f"/experiments/nope/runs?out_dir={out_dir}")
    assert resp.status_code == 404


def test_path_traversal_rejected(client: TestClient, out_dir: Path) -> None:
    resp = client.get(f"/experiments/..%2F..%2F/runs?out_dir={out_dir}")
    assert resp.status_code == 404
