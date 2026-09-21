"""Eval API — the artifact-based leaderboard over HTTP.

Reads the persisted ``summary.json`` artifacts written by the reproducible
runner (``results/<experiment_id>/<run_id>/summary.json``) and returns the
ranked leaderboard, so a dashboard can render "which provider/model wins
today" without re-execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from noema.eval.leaderboard import compute_leaderboard, to_dict
from noema.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["eval"])

DEFAULT_OUT_DIR = "results"


@router.get("/eval/leaderboard")
def leaderboard(
    top: int = Query(20, ge=1, le=500, description="Max entries (ranked list)"),
) -> dict[str, Any]:
    """Rank provider/model combinations from all persisted experiment summaries."""
    root = Path(DEFAULT_OUT_DIR)
    rows = compute_leaderboard(root, top_n=top)
    return {"count": len(rows), "leaderboard": to_dict(rows)}
