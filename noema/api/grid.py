"""Grid dashboard API — live per-node health for the Noema fleet (T3.4).

``GET /grid`` folds the Redis worker heartbeats and each node's Prometheus
``/metrics`` endpoint into a per-node latency/token/error view plus cluster
totals. A node that is down shows up as ``reachable: false`` with the
transport error — the dashboard must always answer, never fail.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from noema.config.settings import get_settings
from noema.logging import get_logger
from noema.observability.grid import GridDashboard

log = get_logger(__name__)

router = APIRouter(prefix="/grid", tags=["grid"])

_dashboard: GridDashboard | None = None


def get_dashboard() -> GridDashboard:
    """Process-wide dashboard bound to the configured Redis."""
    global _dashboard
    if _dashboard is None:
        settings = get_settings()
        _dashboard = GridDashboard(redis_url=settings.redis.url)
    return _dashboard


def set_dashboard(dashboard: GridDashboard | None) -> None:
    """Swap the process-wide dashboard (tests inject fakeredis-backed ones)."""
    global _dashboard
    _dashboard = dashboard


@router.get("")
async def grid_snapshot() -> dict[str, Any]:
    """Per-node latency/token/error view aggregated from live metrics."""
    try:
        return await get_dashboard().snapshot()
    except Exception as e:  # noqa: BLE001 - observability must not 500 the fleet view
        log.warning("grid_snapshot_failed", error=str(e))
        return {
            "generated_at": 0.0,
            "nodes": [],
            "totals": {
                "nodes_total": 0,
                "nodes_reachable": 0,
                "nodes_draining": 0,
                "http_requests": 0,
                "http_errors": 0,
                "llm_tokens": 0,
                "llm_calls": 0,
            },
            "error": f"{type(e).__name__}: {e}",
        }
