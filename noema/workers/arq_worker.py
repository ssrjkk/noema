"""Arq worker for background noema tasks."""

from __future__ import annotations

from typing import Any

import structlog

from noema.core.engine import NoemaEngine
from noema.core.types import Task, TaskComplexity

logger = structlog.get_logger(__name__)


noema_instance: NoemaEngine | None = None


async def startup(ctx: dict) -> None:
    ctx["noema"] = NoemaEngine()
    await ctx["noema"].initialize()
    logger.info("arq_worker_startup_complete")


async def shutdown(ctx: dict) -> None:
    if ctx.get("noema"):
        await ctx["noema"].shutdown()
    logger.info("arq_worker_shutdown_complete")


async def think_task(ctx: dict, task_data: dict) -> dict:
    """Run noema.think() as a background job."""
    noema: NoemaEngine = ctx["noema"]
    task = Task(
        title=task_data["title"],
        description=task_data.get("description", ""),
        complexity=TaskComplexity(task_data.get("complexity", "moderate")),
        tags=task_data.get("tags", []),
        requirements=task_data.get("requirements", []),
    )
    try:
        solution, thought = await noema.think(task)
        return {
            "status": "completed",
            "solution_id": solution.id,
            "task_id": solution.task_id,
            "quality": solution.quality.value,
            "confidence": solution.confidence,
            "duration_ms": thought.duration_ms,
        }
    except Exception as e:
        logger.error("arq_think_task_failed", error=str(e))
        return {"status": "failed", "error": str(e)}


async def fix_incident_task(ctx: dict, payload: dict) -> dict:
    """Run the incident→PR autonomy loop as a background job (T2.1)."""
    from noema.autonomy.fixer import IncidentFixer, build_github_client_from_settings

    fixer = IncidentFixer(github=build_github_client_from_settings())
    try:
        return await fixer.handle_incident(payload)
    finally:
        await fixer.close()


class NoemaWorkerSettings:
    functions = [think_task, fix_incident_task]
    on_startup = startup
    on_shutdown = shutdown
    poll_delay = 1.0
    max_retries = 3
    retry_delay = 5.0


async def create_worker(redis_url: str | None = None) -> Any:
    """Create and return an Arq worker."""
    from arq import Worker as ArqWorker
    from arq.connections import RedisSettings as ArqRedisSettings

    from noema.config.settings import get_settings

    if redis_url is None:
        settings = get_settings()
        redis_url = settings.redis.url

    redis_settings = ArqRedisSettings.from_dsn(redis_url)
    worker = ArqWorker(
        redis_settings=redis_settings,
        functions=NoemaWorkerSettings.functions,
        on_startup=NoemaWorkerSettings.on_startup,
        on_shutdown=NoemaWorkerSettings.on_shutdown,
        poll_delay=NoemaWorkerSettings.poll_delay,
        max_retries=NoemaWorkerSettings.max_retries,
        retry_delay=NoemaWorkerSettings.retry_delay,
    )
    return worker


async def enqueue_think(redis_url: str, task_data: dict) -> str | None:
    """Enqueue a think task and return job ID."""
    from arq import create_pool
    from arq.connections import RedisSettings as ArqRedisSettings

    pool = await create_pool(ArqRedisSettings.from_dsn(redis_url))
    job = await pool.enqueue_job("think_task", task_data)
    await pool.close()
    return job.job_id if job else None


async def enqueue_fix_incident(redis_url: str, payload: dict) -> str | None:
    """Enqueue an incident→PR fix task and return job ID."""
    from arq import create_pool
    from arq.connections import RedisSettings as ArqRedisSettings

    pool = await create_pool(ArqRedisSettings.from_dsn(redis_url))
    job = await pool.enqueue_job("fix_incident_task", payload)
    await pool.close()
    return job.job_id if job else None
