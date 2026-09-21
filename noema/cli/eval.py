"""CLI commands for evaluation: the artifact-based leaderboard."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from noema.cli.ui import data_table, ok, warn

eval_app = typer.Typer(help="Evaluation commands", rich_markup_mode="rich")

_HEADERS = [
    "Provider",
    "Model",
    "Runs",
    "Cells",
    "Score",
    "Latency ms",
    "Tokens",
    "Cost",
    "Errors",
    "Sandbox",
]


@eval_app.command("leaderboard")
def leaderboard(
    out: str = typer.Option("results", help="Root dir with experiment results"),
    top: int = typer.Option(20, "--top", help="Show at most N rows (0 = all)"),
    markdown: bool = typer.Option(
        False, "--markdown", "-m", help="Emit a Markdown table instead of a Rich table"
    ),
) -> None:
    """Rank provider/model combinations from persisted experiment summaries.

    Reads every ``summary.json`` under ``out`` (from ``noema.eval.run`` / the
    experiment runner) and merges rows by provider+model — no re-execution.
    """
    from noema.eval.leaderboard import compute_leaderboard, render_markdown, to_dict

    root = Path(out)
    if not root.is_dir():
        warn(f"Results dir not found: {root}")
        return
    rows = compute_leaderboard(root, top_n=top)
    if not rows:
        warn(f"No experiment summaries found under {root} (looked for */summary.json).")
        return

    if markdown:
        print(render_markdown(rows))
        return

    data_table(
        "Eval leaderboard",
        _HEADERS,
        [r.as_table() for r in rows],
    )
    ok(f"{len(rows)} entries from {root}")

    compact = json.loads(json.dumps(to_dict(rows), default=str))
    print(json.dumps(compact, indent=2))
