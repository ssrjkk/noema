"""CLI commands for RLVR dataset export and honest judge calibration.

Thin wrappers over :mod:`noema.training.rlvr_pipeline`:

- ``export-dpo`` / ``export-grpo`` turn a JSONL of audited rollouts into DPO
  preference pairs / GRPO advantage-normalized groups.
- ``calibrate-judge`` reports out-of-sample k-fold Brier/ECE for the Platt judge
  calibrator (never in-sample-only) and can persist the fitted parameters.

Input rollout JSONL fields: ``task_id``, ``prompt``, ``response``, ``ast_ok``,
``z3_sat``, ``mutation_kill_score``, ``docker_pass_rate`` and optionally
``normalized_cost``, ``raw_judge_score``. Calibration JSONL fields:
``raw_judge_score`` and ``docker_pass`` (0/1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from noema.cli.ui import info, kv_panel, ok, warn

if TYPE_CHECKING:
    from noema.training.rlvr_pipeline import TraceRollout

train_app = typer.Typer(
    help="RLVR training-data & judge-calibration commands", rich_markup_mode="rich"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise typer.BadParameter(f"Input file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise typer.BadParameter(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            if not isinstance(obj, dict):
                raise typer.BadParameter(f"{path}:{lineno}: expected a JSON object")
            records.append(obj)
    return records


def _load_rollouts(path: Path) -> list[TraceRollout]:
    from noema.training.rlvr_pipeline import TraceRollout

    rollouts: list[TraceRollout] = []
    for lineno, rec in enumerate(_read_jsonl(path), start=1):
        try:
            rollouts.append(
                TraceRollout(
                    task_id=str(rec["task_id"]),
                    prompt=str(rec["prompt"]),
                    response=str(rec["response"]),
                    ast_ok=bool(rec["ast_ok"]),
                    z3_sat=bool(rec["z3_sat"]),
                    mutation_kill_score=float(rec["mutation_kill_score"]),
                    docker_pass_rate=float(rec["docker_pass_rate"]),
                    normalized_cost=float(rec.get("normalized_cost", 0.0)),
                    raw_judge_score=float(rec.get("raw_judge_score", 0.5)),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise typer.BadParameter(f"{path}: record {lineno}: malformed rollout ({exc})") from exc
    return rollouts


@train_app.command("export-dpo")
def export_dpo(
    input_path: str = typer.Option(..., "--input", "-i", help="JSONL of audited rollouts"),
    output: str = typer.Option(..., "--output", "-o", help="Destination JSONL for DPO pairs"),
    min_margin: float = typer.Option(0.25, "--min-margin", help="Minimum reward margin for a pair"),
    max_pairs_per_task: int = typer.Option(
        0, "--max-pairs-per-task", help="Cap pairs per task (0 = unlimited)"
    ),
) -> None:
    """Export DPO preference pairs from verified rollouts."""
    from noema.training.rlvr_pipeline import RLVRDatasetBuilder

    rollouts = _load_rollouts(Path(input_path))
    builder = RLVRDatasetBuilder(min_reward_margin=min_margin)
    pairs = builder.build_dpo_pairs(
        rollouts, max_pairs_per_task=max_pairs_per_task if max_pairs_per_task > 0 else None
    )
    out = builder.export_jsonl(pairs, output, create_parents=True)
    ok(f"{len(pairs)} DPO pairs from {len(rollouts)} rollouts -> {out}")


@train_app.command("export-grpo")
def export_grpo(
    input_path: str = typer.Option(..., "--input", "-i", help="JSONL of audited rollouts"),
    output: str = typer.Option(..., "--output", "-o", help="Destination JSONL for GRPO groups"),
) -> None:
    """Export GRPO groups with within-group advantage normalization."""
    from noema.training.rlvr_pipeline import RLVRDatasetBuilder

    rollouts = _load_rollouts(Path(input_path))
    builder = RLVRDatasetBuilder()
    groups = builder.build_grpo_groups(rollouts)
    out = builder.export_jsonl(groups, output, create_parents=True)
    ok(f"{len(groups)} GRPO groups from {len(rollouts)} rollouts -> {out}")


@train_app.command("calibrate-judge")
def calibrate_judge(
    input_path: str = typer.Option(
        ..., "--input", "-i", help="JSONL of {raw_judge_score, docker_pass}"
    ),
    folds: int = typer.Option(5, "--folds", "-k", help="k-fold cross-validation folds (>=2)"),
    out: str = typer.Option("", "--out", help="Optional JSON file for fitted slope/intercept"),
) -> None:
    """Report honest out-of-sample judge calibration (Brier / ECE)."""
    from noema.training.rlvr_pipeline import JudgeCalibrator

    records = _read_jsonl(Path(input_path))
    scores: list[float] = []
    outcomes: list[int] = []
    for lineno, rec in enumerate(records, start=1):
        try:
            scores.append(float(rec["raw_judge_score"]))
            outcomes.append(int(rec["docker_pass"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise typer.BadParameter(
                f"{input_path}: record {lineno}: malformed sample ({exc})"
            ) from exc
    if len(scores) < 2:
        warn("Need at least 2 paired samples for cross-validation.")
        raise typer.Exit(code=1)

    metrics = JudgeCalibrator.cross_validate(scores, outcomes, k_folds=folds)
    kv_panel(
        "Judge calibration (out-of-sample)",
        [
            ("Brier score", f"{metrics.brier_score:.6f}"),
            ("ECE", f"{metrics.ece:.6f}"),
            ("Samples", str(metrics.samples)),
            ("Folds", str(min(max(2, folds), len(scores)))),
        ],
    )

    if out:
        cal = JudgeCalibrator()
        cal.fit(scores, outcomes)
        payload = {
            "slope": cal.slope,
            "intercept": cal.intercept,
            "cross_validate": {
                "brier_score": metrics.brier_score,
                "ece": metrics.ece,
                "samples": metrics.samples,
            },
        }
        dest = Path(out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        info(f"Fitted calibrator saved to {dest}")
    ok("Calibration reported on out-of-sample folds (not in-sample).")
