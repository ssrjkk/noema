"""Eval leaderboard — aggregate experiment summaries into a ranked table.

Scans a results root (``results/<experiment_id>/<run_id>/summary.json`` or the
bare ``results/<run_id>/`` layout) and merges every ``(provider, model)`` row
into one leaderboard entry: weighted mean judge score, latency, tokens, cost
and error rate across runs, ranked by score then latency then cost.

Purpose: "who wins today" must be reproducible from committed artifacts, so
the leaderboard reads only the persisted ``summary.json`` files the runner
writes — no re-execution.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from noema.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

log = get_logger(__name__)

# Summary keys the runner emits per (provider, model) row (see
# ``noema.experiments.runner._summarize``).
_SCORE_KEY = "mean_judge_score"
_LATENCY_KEY = "mean_latency_ms"
_TOKENS_KEY = "mean_total_tokens"
_COST_KEY = "mean_cost"
_ERROR_KEY = "error_rate"
_SANDBOX_KEY = "sandbox_pass_rate"


@dataclass
class LeaderboardRow:
    provider: str
    model: str
    runs: int = 0
    cells: int = 0
    mean_judge_score: float = 0.0
    best_score: float = 0.0
    worst_score: float = 0.0
    mean_latency_ms: float = 0.0
    mean_total_tokens: float = 0.0
    mean_cost: float = 0.0
    error_rate: float = 0.0
    sandbox_pass_rate: float | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.provider, self.model)

    def as_table(self) -> list[str]:
        sandbox = (
            "—" if self.sandbox_pass_rate is None else f"{self.sandbox_pass_rate * 100:.1f}%"
        )
        return [
            self.provider,
            self.model,
            str(self.runs),
            str(self.cells),
            f"{self.mean_judge_score:.3f}",
            f"{self.mean_latency_ms:.0f}",
            f"{self.mean_total_tokens:.0f}",
            f"{self.mean_cost:.6f}",
            f"{self.error_rate * 100:.2f}%",
            sandbox,
        ]


def _read_summary_rows(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("leaderboard_skip_corrupt", path=str(path), error=str(exc))
        return []
    summary = data.get("summary") if isinstance(data, dict) else None
    if not isinstance(summary, list):
        return []
    return [row for row in summary if isinstance(row, dict)]


def _run_dirs(root: Path) -> Iterable[Path]:
    """Yield all run directories that carry a ``summary.json``."""
    seen: set[Path] = set()
    for pattern in ("*/*/summary.json", "*/summary.json"):
        for path in sorted(root.glob(pattern)):
            run_dir = path.parent
            if run_dir in seen:
                continue
            seen.add(run_dir)
            yield run_dir


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_leaderboard(root: Path | str, top_n: int = 0) -> list[LeaderboardRow]:
    """Merge all summaries under ``root`` into a ranked leaderboard.

    ``top_n<=0`` returns all rows. Ranking: mean judge score desc, then mean
    latency asc, then mean cost asc (provider/model as stable tie-breakers).
    """
    root = Path(root)
    acc: dict[tuple[str, str], LeaderboardRow] = {}

    for run_dir in _run_dirs(root):
        for row in _read_summary_rows(run_dir / "summary.json"):
            provider = str(row.get("provider", ""))
            model = str(row.get("model", ""))
            if not provider or not model:
                continue
            n = max(1, int(_to_float(row.get("n", 0))))
            score = _to_float(row.get(_SCORE_KEY))
            out = acc.setdefault((provider, model), LeaderboardRow(provider, model))
            prev_cells = out.cells
            out.runs += 1
            out.cells += n
            out.mean_judge_score = _weighted(out.mean_judge_score, prev_cells, score, n)
            out.mean_latency_ms = _weighted(
                out.mean_latency_ms, prev_cells, _to_float(row.get(_LATENCY_KEY)), n
            )
            out.mean_total_tokens = _weighted(
                out.mean_total_tokens, prev_cells, _to_float(row.get(_TOKENS_KEY)), n
            )
            out.mean_cost = _weighted(
                out.mean_cost, prev_cells, _to_float(row.get(_COST_KEY)), n
            )
            out.error_rate = _weighted(
                out.error_rate, prev_cells, _to_float(row.get(_ERROR_KEY)), n
            )
            out.best_score = max(out.best_score, score)
            out.worst_score = score if out.worst_score == 0.0 else min(out.worst_score, score)
            sandbox = row.get(_SANDBOX_KEY)
            if isinstance(sandbox, (int, float)):
                val = float(sandbox)
                current = out.sandbox_pass_rate
                out.sandbox_pass_rate = val if current is None else (current + val) / 2.0

    rows = list(acc.values())
    rows.sort(
        key=lambda r: (
            -r.mean_judge_score,
            r.mean_latency_ms,
            r.mean_cost,
            r.provider.lower(),
            r.model.lower(),
        )
    )
    if top_n > 0:
        rows = rows[:top_n]
    return rows


def _weighted(current: float, current_n: int, value: float, new_n: int) -> float:
    total = current_n + new_n
    if total <= 0:
        return 0.0
    # current is the previous weighted mean over `current_n` cells already.
    return round((current * current_n + value * new_n) / total, 4)


def render_markdown(rows: list[LeaderboardRow]) -> str:
    """Render the leaderboard as a GitHub-style Markdown table."""
    if not rows:
        return "_No experiment runs found. Run `noema eval run` first._\n"
    lines = [
        "| Provider | Model | Runs | Cells | Score | Latency ms | Tokens | Cost | Errors | Sandbox |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        sandbox = "—" if r.sandbox_pass_rate is None else f"{r.sandbox_pass_rate * 100:.1f}%"
        lines.append(
            f"| {r.provider} | {r.model} | {r.runs} | {r.cells} | "
            f"{r.mean_judge_score:.3f} | {r.mean_latency_ms:.0f} | "
            f"{r.mean_total_tokens:.0f} | {r.mean_cost:.6f} | "
            f"{r.error_rate * 100:.2f}% | {sandbox} |"
        )
    return "\n".join(lines) + "\n"


def to_dict(rows: list[LeaderboardRow]) -> list[dict[str, Any]]:
    return [asdict(r) for r in rows]
