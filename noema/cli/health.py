"""CLI command for system health check."""

from __future__ import annotations

import asyncio
import json

import structlog
import typer

from noema.cli.ui import STATUS_DOT, data_table, panel
from noema.config.settings import get_settings

logger = structlog.get_logger(__name__)
health_app = typer.Typer(help="System health checks", rich_markup_mode="rich")


async def _check_llm() -> dict:
    """Check LLM provider availability."""
    settings = get_settings()
    try:
        from noema.llm.providers import create_llm_provider

        llm = create_llm_provider(settings.llm.provider, settings.llm.ollama_model)
        return {"status": "ok", "provider": settings.llm.provider, "model": llm.model_name}
    except Exception as e:
        return {"status": "error", "provider": settings.llm.provider, "error": str(e)}


async def _check_db() -> dict:
    """Check database connectivity."""
    try:
        from noema.db.engine import get_db

        db = get_db()
        healthy = await db.health_check()
        return {
            "status": "ok" if healthy else "error",
            "message": "Database reachable" if healthy else "Database unreachable",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def _check_redis() -> dict:
    """Check Redis connectivity."""
    settings = get_settings()
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.redis.url, socket_timeout=5.0)
        await r.ping()
        await r.aclose()
        return {"status": "ok", "message": "Redis reachable"}
    except ImportError:
        return {"status": "skipped", "message": "redis not installed"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def _check_sandbox() -> dict:
    """Check sandbox engine availability."""
    try:
        from noema.sandbox.engine import SandboxConfig, SandboxEngine

        engine = SandboxEngine(SandboxConfig(enabled=True, lint_enabled=True, run_enabled=False))
        available = await asyncio.to_thread(engine.is_available)
        return {"status": "ok", "message": "Sandbox engine available", "docker": available}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _status_mark(result: dict) -> str:
    status = result.get("status")
    if status == "ok":
        return f"[ok]{STATUS_DOT}[/ok]"
    if status == "error":
        return f"[err]{STATUS_DOT}[/err]"
    return f"[warn]{STATUS_DOT}[/warn]"


def _status_label(result: dict) -> str:
    status = str(result.get("status") or "skipped")
    style = {"ok": "ok", "error": "err", "skipped": "warn"}.get(status, "warn")
    return f"[{style}]{status}[/{style}]"


@health_app.command()
def check(output: str = "text") -> None:
    """Run all health checks and print results."""

    async def _run() -> bool:
        results = {
            "llm": await _check_llm(),
            "database": await _check_db(),
            "redis": await _check_redis(),
            "sandbox": await _check_sandbox(),
        }
        all_ok = all(r.get("status") == "ok" for r in results.values())
        if output == "json":
            print(json.dumps({"healthy": all_ok, "checks": results}, indent=2))
        else:
            if all_ok:
                panel("All systems operational", title="System Health", border="ok")
            else:
                panel("Issues detected", title="System Health", border="err")
            rows = []
            for name, result in results.items():
                detail = result.get("message") or result.get("error") or ""
                rows.append([_status_mark(result), name, _status_label(result), str(detail)])
            data_table("Checks", ["", "Component", "Status", "Detail"], rows)
        return all_ok

    ok_result = asyncio.run(_run())
    raise typer.Exit(code=0 if ok_result else 1)


@health_app.command()
def check_llm() -> None:
    """Check LLM provider only."""
    result = asyncio.run(_check_llm())
    print(json.dumps(result, indent=2))
