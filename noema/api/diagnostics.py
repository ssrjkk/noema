"""Self-diagnostics endpoint — runs all health checks."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import APIRouter, Request

from noema.config.settings import get_settings
from noema.db.engine import get_db

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


async def _check_component(
    name: str, check_fn: Callable[[], Awaitable[dict[str, Any]]], timeout: float = 5.0
) -> dict[str, Any]:
    """Run a component health check with timeout."""
    start = time.monotonic()
    try:
        result = await asyncio.wait_for(check_fn(), timeout=timeout)
        elapsed = (time.monotonic() - start) * 1000
        return {
            "component": name,
            "status": "ok" if result.get("status") == "ok" else "error",
            "detail": result,
            "duration_ms": round(elapsed, 1),
        }
    except TimeoutError:
        return {
            "component": name,
            "status": "error",
            "detail": "timeout",
            "duration_ms": timeout * 1000,
        }
    except Exception as e:
        return {
            "component": name,
            "status": "error",
            "detail": str(e),
            "duration_ms": round((time.monotonic() - start) * 1000, 1),
        }


@router.get("/diagnostics")
async def diagnostics(request: Request) -> dict[str, Any]:
    """Run all system diagnostics and return comprehensive report."""
    from noema.api.server import _get_noema, _start_time

    noema = _get_noema()

    async def _llm_check() -> dict[str, Any]:
        return {"status": "ok", "provider": noema.llm.name, "model": noema.llm.model_name}

    async def _db_check() -> dict[str, Any]:
        db = get_db()
        healthy = await db.health_check()
        return {"status": "ok" if healthy else "error"}

    async def _redis_check() -> dict[str, Any]:
        settings = get_settings()
        try:
            import redis.asyncio as aioredis

            r = aioredis.from_url(settings.redis.url, socket_timeout=3.0)
            await r.ping()
            await r.aclose()
            return {"status": "ok"}
        except Exception:
            return {"status": "error"}

    async def _sandbox_check() -> dict[str, Any]:
        return {"status": "ok", "available": noema.sandbox is not None}

    async def _memory_check() -> dict[str, Any]:
        stats = noema.memory_stats()
        return {"status": "ok", **stats}

    checks = await asyncio.gather(
        _check_component("llm", _llm_check),
        _check_component("database", _db_check),
        _check_component("redis", _redis_check, timeout=3.0),
        _check_component("sandbox", _sandbox_check),
        _check_component("memory", _memory_check),
    )

    all_ok = all(c["status"] == "ok" for c in checks)
    uptime_s = round(time.monotonic() - _start_time, 1) if _start_time else 0

    return {
        "healthy": all_ok,
        "uptime_s": uptime_s,
        "checks": checks,
        "version": "1.0.0",
        "timestamp": time.time(),
    }
