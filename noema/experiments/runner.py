"""Reproducible experiment runner for LLM benchmarks.

Runs a matrix of ``(task, provider, model, repetition)`` through
:class:`~noema.core.engine.NoemaEngine` and collects wall time, tokens, judge
score, cost estimate and optional sandbox validation into JSON + CSV
artifacts under ``results/``.

The runner is CI-safe: with the built-in ``fallback`` provider it exercises
the full pipeline without any API keys. Real providers (openai/anthropic)
are enabled by listing them in the experiment YAML and setting the usual
``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` environment variables.

Usage::

    python -m noema.experiments.runner experiments/experiments.yaml --out results
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

from noema.config.settings import reset_settings
from noema.core.engine import NoemaEngine
from noema.core.types import Solution, Task, ThoughtProcess
from noema.judge import evaluate_solution
from noema.logging import get_logger

log = get_logger(__name__)

RECORD_FIELDS = [
    "experiment_id",
    "run_id",
    "task_id",
    "provider",
    "model",
    "repeat_index",
    "started_at",
    "duration_ms",
    "tokens_input",
    "tokens_output",
    "total_tokens",
    "llm_calls",
    "judge_score",
    "judge_passed",
    "judge_summary",
    "solution_quality",
    "code_blocks",
    "sandbox_all_valid",
    "sandbox_summary",
    "cost_estimate",
    "error",
]


@dataclass
class RunRecord:
    """One measurement cell of the experiment matrix."""

    experiment_id: str
    run_id: str
    task_id: str
    provider: str
    model: str
    repeat_index: int
    started_at: str
    duration_ms: float
    tokens_input: int | None
    tokens_output: int | None
    total_tokens: int
    llm_calls: int
    judge_score: float
    judge_passed: bool
    judge_summary: str
    solution_quality: str
    code_blocks: int
    sandbox_all_valid: bool | None
    sandbox_summary: str
    cost_estimate: float
    error: str | None = None


EngineFactory = Callable[[str, str | None, str], Any]


def _default_engine_factory(provider: str, model: str | None, project_root: str) -> Any:
    return NoemaEngine(llm_provider=provider, llm_model=model, project_root=project_root)


# в”Ђв”Ђ Config в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ


def load_experiment(path: str | Path) -> dict[str, Any]:
    """Parse and validate an experiment YAML file.

    Raises ``ValueError`` on structural problems so CI fails loudly instead of
    silently skipping a misconfigured experiment.
    """
    cfg_path = Path(path)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"experiment {path}: top-level must be a mapping")

    meta = cfg.get("meta")
    if not isinstance(meta, dict) or not meta.get("name"):
        raise ValueError(f"experiment {path}: missing 'meta.name'")

    providers = cfg.get("providers")
    if not isinstance(providers, list) or not providers:
        raise ValueError(f"experiment {path}: 'providers' must be a non-empty list")

    seen: set[str] = set()
    for p in providers:
        if not isinstance(p, dict) or not p.get("provider"):
            raise ValueError(f"experiment {path}: each provider needs a 'provider' name")
        models = p.get("models")
        if not isinstance(models, list) or not models:
            raise ValueError(
                f"experiment {path}: provider {p['provider']!r} needs non-empty 'models'"
            )
        seen.add(p["provider"])
        for m in models:
            if not isinstance(m, str) or not m:
                raise ValueError(f"experiment {path}: models must be non-empty strings")

    tasks = cfg.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"experiment {path}: 'tasks' must be a non-empty list")
    for t in tasks:
        if not isinstance(t, dict) or not t.get("title"):
            raise ValueError(f"experiment {path}: each task needs a 'title'")

    repetitions = cfg.get("repetitions", 1)
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError(f"experiment {path}: 'repetitions' must be a positive integer")

    return cfg


# в”Ђв”Ђ Measurement в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ


def _delta_stats(after: dict[str, Any], before: dict[str, Any]) -> tuple[int, int]:
    return (
        int(after.get("total_tokens", 0) - before.get("total_tokens", 0)),
        int(after.get("llm_calls", 0) - before.get("llm_calls", 0)),
    )


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _stddev(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _apply_settings(settings_cfg: dict[str, Any]) -> None:
    """Apply experiment toggles via env + settings reload.

    ``neurosymbolic`` is honored end-to-end. ``retrieval`` and ``sandbox`` are
    reserved knobs that Phase 2/3 wire into the engine; they are accepted here
    so experiment files remain forward-compatible.
    """
    os_env: dict[str, str] = {
        "NOEMA_NEUROSYMBOLIC__ENABLED": "true" if settings_cfg.get("neurosymbolic") else "false",
    }
    for key, value in os_env.items():
        os.environ[key] = value
    reset_settings()
    log.info("experiment_settings_applied", neurosymbolic=os_env["NOEMA_NEUROSYMBOLIC__ENABLED"])


async def _run_once(
    engine: Any,
    cfg: dict[str, Any],
    task: Task,
    provider: str,
    model: str,
    repeat_index: int,
    experiment_id: str,
    run_id: str,
    cost_per_token: float,
) -> RunRecord:
    started_at = datetime.now(UTC).isoformat()
    before = engine.tracer.get_stats()
    t0 = time.monotonic()
    error: str | None = None
    solution: Solution | None = None
    thought: ThoughtProcess | None = None
    try:
        solution, thought = await engine.think(task)
    except Exception as exc:  # noqa: BLE001 - benchmark must never die on one cell
        error = f"{type(exc).__name__}: {exc}"

    duration_ms = (time.monotonic() - t0) * 1000
    after = engine.tracer.get_stats()
    total_tokens, llm_calls = _delta_stats(after, before)

    judge_score = 0.0
    judge_passed = False
    judge_summary = ""
    if solution is not None and error is None:
        try:
            verdict = await evaluate_solution(engine.llm, solution, task.description, task.tags)
            judge_score = float(verdict.scores.overall or 0.0)
            judge_passed = bool(verdict.passed)
            judge_summary = verdict.summary
        except Exception as exc:  # noqa: BLE001
            error = f"judge:{type(exc).__name__}: {exc}"

    sandbox_all_valid: bool | None = None
    sandbox_summary = ""
    if solution is not None and error is None and cfg.get("settings", {}).get("sandbox"):
        try:
            result = await engine.validate_solution(solution, run_tests=False)
            sandbox_all_valid = bool(result.all_valid)
            sandbox_summary = str(result.summary)
        except Exception as exc:  # noqa: BLE001
            error = f"sandbox:{type(exc).__name__}: {exc}"

    return RunRecord(
        experiment_id=experiment_id,
        run_id=run_id,
        task_id=task.id,
        provider=provider,
        model=model,
        repeat_index=repeat_index,
        started_at=started_at,
        duration_ms=round(duration_ms, 2),
        tokens_input=None,
        tokens_output=None,
        total_tokens=total_tokens,
        llm_calls=llm_calls,
        judge_score=judge_score,
        judge_passed=judge_passed,
        judge_summary=judge_summary,
        solution_quality=solution.quality.value if solution else "",
        code_blocks=len(solution.code_blocks) if solution else 0,
        sandbox_all_valid=sandbox_all_valid,
        sandbox_summary=sandbox_summary,
        cost_estimate=round(cost_per_token * total_tokens, 8),
        error=error,
    )


# в”Ђв”Ђ Orchestration в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ


def run_experiment(
    cfg: dict[str, Any],
    out_dir: str | Path = "results",
    engine_factory: EngineFactory = _default_engine_factory,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Execute the experiment matrix and persist JSON + CSV artifacts."""
    experiment_id = str(cfg["meta"]["name"])
    run_id = run_id or (
        f"{experiment_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    )
    seed = cfg.get("meta", {}).get("seed")
    if seed is not None:
        random.seed(seed)

    out_path = Path(out_dir) / experiment_id
    out_path.mkdir(parents=True, exist_ok=True)
    _apply_settings(cfg.get("settings", {}))

    records: list[RunRecord] = asyncio.run(
        _run_matrix(cfg, out_path, engine_factory, experiment_id, run_id)
    )

    results: dict[str, Any] = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "meta": cfg.get("meta", {}),
        "generated_at": datetime.now(UTC).isoformat(),
        "records": [asdict(r) for r in records],
    }
    _write_artifacts(out_path, run_id, records)
    log.info(
        "experiment_complete",
        experiment=experiment_id,
        run=run_id,
        cells=len(records),
        out=str(out_path),
    )
    return results


async def _run_matrix(
    cfg: dict[str, Any],
    out_path: Path,
    engine_factory: EngineFactory,
    experiment_id: str,
    run_id: str,
) -> list[RunRecord]:
    tasks = [_build_task(t) for t in cfg["tasks"]]
    repetitions = int(cfg.get("repetitions", 1))
    records: list[RunRecord] = []
    engines: dict[tuple[str, str | None], Any] = {}

    for provider_cfg in cfg["providers"]:
        provider = provider_cfg["provider"]
        cost_per_token = float(provider_cfg.get("cost_per_token", 0.0))
        for model in provider_cfg["models"]:
            key = (provider, model)
            if key not in engines:
                engines[key] = engine_factory(provider, model, str(out_path))
            engine = engines[key]
            for task in tasks:
                for rep in range(1, repetitions + 1):
                    record = await _run_once(
                        engine,
                        cfg,
                        task,
                        provider,
                        model,
                        rep,
                        experiment_id,
                        run_id,
                        cost_per_token,
                    )
                    records.append(record)
                    log.info(
                        "run_cell",
                        task=record.task_id,
                        provider=record.provider,
                        model=record.model,
                        repeat=record.repeat_index,
                        ms=record.duration_ms,
                        error=record.error,
                    )
    return records


def _build_task(raw: dict[str, Any]) -> Task:
    return Task(
        title=str(raw["title"]),
        description=str(raw.get("description", "")),
        tags=[str(t) for t in raw.get("tags", [])],
    )


# в”Ђв”Ђ Artifacts в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ


def _write_artifacts(out_path: Path, run_id: str, records: list[RunRecord]) -> None:
    (out_path / run_id).mkdir(parents=True, exist_ok=True)
    _write_json(out_path / run_id / "results.json", [asdict(r) for r in records])
    _write_csv(out_path / run_id / "runs.csv", [asdict(r) for r in records])
    _write_csv(out_path / run_id / "summary.csv", _summarize(records))


def _write_json(path: Path, data: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(records: list[RunRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for r in records:
        groups[(r.provider, r.model)].append(r)

    rows: list[dict[str, Any]] = []
    for (provider, model), recs in sorted(groups.items()):
        scores = [r.judge_score for r in recs]
        durations = [r.duration_ms for r in recs]
        tokens = [float(r.total_tokens) for r in recs]
        costs = [r.cost_estimate for r in recs]
        sandbox = [r.sandbox_all_valid for r in recs if r.sandbox_all_valid is not None]
        rows.append(
            {
                "provider": provider,
                "model": model,
                "n": len(recs),
                "mean_judge_score": round(_mean(scores), 4),
                "stddev_judge_score": round(_stddev(scores), 4),
                "mean_latency_ms": round(_mean(durations), 2),
                "stddev_latency_ms": round(_stddev(durations), 2),
                "mean_total_tokens": round(_mean(tokens), 1),
                "mean_cost": round(_mean(costs), 8),
                "sandbox_pass_rate": round(sum(sandbox) / len(sandbox), 4) if sandbox else "",
                "error_rate": round(_mean([1.0 if r.error else 0.0 for r in recs]), 4),
            }
        )
    return rows


# в”Ђв”Ђ CLI в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="noema.experiments.runner",
        description="Run a reproducible LLM benchmark experiment.",
    )
    parser.add_argument("config", nargs="?", default="experiments/experiments.yaml")
    parser.add_argument("--out", default="results", help="Output directory (default: results)")
    args = parser.parse_args(argv)

    cfg = load_experiment(args.config)
    run_experiment(cfg, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
