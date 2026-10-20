"""Benchmark as a service — run the experiment matrix over HTTP.

Exposes the reproducible runner (:mod:`noema.experiments.runner`) as an API.
The request body is the experiment config as YAML or JSON (YAML is a superset
of JSON, so one parser serves both); the runner executes the matrix with the
real engine and writes the existing ``results.json``/``summary.json``/CSV
artifacts under ``results/<experiment_id>/<run_id>/``.

Endpoints:
- ``POST /experiments``  — run a matrix, return the run id + artifacts info.
- ``GET  /experiments``  — list experiments persisted on disk.
- ``GET  /experiments/{experiment_id}/runs`` — list runs of an experiment.
- ``GET  /experiments/{experiment_id}/runs/{run_id}`` — records + summary.

Concurrency contract: the runner serializes env-mutating matrix runs with an
internal thread lock; the blocking run is offloaded with ``asyncio.to_thread``
(it manages its own event loop), so the server loop is never blocked.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from noema.experiments.runner import parse_experiment, run_experiment
from noema.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["experiments"])

DEFAULT_OUT_DIR = "results"

# Имена экспериментов и id прогонов генерирует runner: буква/цифра/`.`/`-`,
# но не с точки — иначе `..` проходил бы как «безопасный сегмент» и выводил
# запрос в родительский каталог (это и была находка CodeQL).
_PATH_SEGMENT = re.compile(r"^(?![.])[\w.\-]+$")


async def _read_body_async(request: Request) -> str:
    raw = await request.body()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Request body must be UTF-8") from None


def _require_segment(value: str, kind: str) -> None:
    if not _PATH_SEGMENT.fullmatch(value):
        raise HTTPException(status_code=404, detail=f"{kind} not found")


def _artifacts_root() -> Path:
    """Каталог артефактов задаёт сервер, а не запрос.

    Раньше корень приходил в query-параметре `out_dir`, и запрос с абсолютным
    путём заставлял сервис читать и писать в любую директорию машины. Параметра
    больше нет: базовый каталог берётся из окружения (``NOEMA_RESULTS_DIR``, по
    умолчанию ``results``), а внутри него имена экспериментов и прогонов
    фильтруются `_require_segment`.
    """
    return Path(os.environ.get("NOEMA_RESULTS_DIR", DEFAULT_OUT_DIR)).resolve()


def _existing_dir(parent: Path, name: str, kind: str) -> Path:
    """Каталог берётся из уже лежащих в `parent`, а не склеивается с именем из запроса.

    Так путь вообще не зависит от пользовательского значения: `name` только
    сравнивается с уже существующими записями. Плюс это честный 404 вместо
    «сходили в несуществующий путь».
    """
    _require_segment(name, kind)
    if parent.is_dir():
        for candidate in parent.iterdir():
            if candidate.name == name and candidate.is_dir():
                return candidate
    raise HTTPException(status_code=404, detail=f"{kind} not found")


def _experiment_dir(experiment_id: str) -> Path:
    return _existing_dir(_artifacts_root(), experiment_id, "Experiment")


def _run_dir(experiment_id: str, run_id: str) -> Path:
    return _existing_dir(_experiment_dir(experiment_id), run_id, "Run")


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Corrupt artifact: {path}") from None


@router.post("/experiments")
async def create_experiment(
    request: Request,
    run_id: str | None = Query(None, description="Optional explicit run id"),
) -> dict[str, Any]:
    """Run an experiment matrix and land its artifacts under the server artifacts root."""
    text = await _read_body_async(request)
    try:
        cfg = parse_experiment(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    # Off-loop so the request never blocks the server event loop; the runner
    # manages its own loop and serializes env mutation with an internal lock.
    results = await asyncio.to_thread(
        run_experiment, cfg, out_dir=str(_artifacts_root()), run_id=run_id
    )

    experiment_id = str(results["experiment_id"])
    run_id = str(results["run_id"])
    out_path = _experiment_dir(experiment_id)
    log.info(
        "experiment_api_completed",
        experiment=experiment_id,
        run=run_id,
        cells=len(results["records"]),
    )
    return {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "status": "completed",
        "out_dir": str(out_path),
        "results_dir": str(out_path / run_id),
        "n_records": len(results["records"]),
        "generated_at": results["generated_at"],
        "summary": results["summary"],
    }


@router.get("/experiments")
async def list_experiments() -> list[dict[str, Any]]:
    """List experiments persisted under the server artifacts root."""
    root = _artifacts_root()
    if not root.is_dir():
        return []
    experiments: list[dict[str, Any]] = []
    for exp_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        runs = sorted(d.name for d in exp_dir.iterdir() if d.is_dir())
        experiments.append({"experiment_id": exp_dir.name, "n_runs": len(runs), "runs": runs})
    return experiments


@router.get("/experiments/{experiment_id}/runs")
async def list_runs(
    experiment_id: str,
) -> list[dict[str, Any]]:
    """List runs of an experiment, each merged with its ``summary.json``."""
    exp_dir = _experiment_dir(experiment_id)

    runs: list[dict[str, Any]] = []
    for run_dir in sorted((d for d in exp_dir.iterdir() if d.is_dir()), key=lambda p: p.name):
        entry: dict[str, Any] = {"run_id": run_dir.name}
        summary_path = run_dir / "summary.json"
        if summary_path.is_file():
            data = _load_json(summary_path)
            if isinstance(data, dict):
                entry.update(data)
        runs.append(entry)
    return runs


@router.get("/experiments/{experiment_id}/runs/{run_id}")
async def get_run(
    experiment_id: str,
    run_id: str,
) -> dict[str, Any]:
    """Return the run's records + summary (the existing runner schema)."""
    run_dir = _run_dir(experiment_id, run_id)
    return {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "records": _load_json(run_dir / "results.json"),
        "summary": _load_json(run_dir / "summary.json"),
    }
