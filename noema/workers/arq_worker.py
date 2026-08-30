"""Arq worker for background noema tasks.

Includes liveness (heartbeat) and node identity so an operator or autoscaler
can observe worker fleet size and drain a node gracefully before shutdown.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import time
import uuid
from typing import TYPE_CHECKING, Any

import structlog
from redis.asyncio import Redis

from noema.core.engine import NoemaEngine
from noema.core.types import Task, TaskComplexity

if TYPE_CHECKING:
    from redis.asyncio import Redis as AsyncRedis

logger = structlog.get_logger(__name__)

HEARTBEAT_PREFIX = "noema:workers:"
HEARTBEAT_TTL = 15
HEARTBEAT_INTERVAL = 5

noema_instance: NoemaEngine | None = None


def make_node_id() -> str:
    """Stable per-process worker identity: ``host:pid:token``."""
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


class NodeHeartbeat:
    """Publishes worker liveness to Redis under an auto-expiring key.

    Absence of a key (TTL expiry) means the node died without draining. A
    ``draining=1`` flag is set before a graceful shutdown so observers can
    stop routing new work to this node. The advertised ``metrics_port`` is
    how the grid dashboard finds this node's ``/metrics`` endpoint.
    """

    def __init__(
        self,
        node_id: str,
        redis_url: str = "",
        redis: AsyncRedis | None = None,
        metrics_port: int = 0,
    ) -> None:
        self.node_id = node_id
        self.redis_url = redis_url
        self.metrics_port = metrics_port
        self._redis: AsyncRedis | None = redis
        self._task: asyncio.Task | None = None
        self.draining = False
        self.started_at = int(time.time())

    async def start(self) -> None:
        if self._redis is None:
            self._redis = Redis.from_url(self.redis_url, decode_responses=True)
        await self._beat()
        self._task = asyncio.create_task(self._loop())
        logger.info("worker_heartbeat_started", node_id=self.node_id)

    async def stop(self) -> None:
        """Publish draining, delete the liveness key, and cancel the loop."""
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._redis is not None:
            await self._redis.delete(self._key())
            await self._redis.aclose()
            self._redis = None

    async def mark_draining(self) -> None:
        """Flag the node as draining before the final heartbeat dies."""
        self.draining = True
        await self._beat()

    def _key(self) -> str:
        return f"{HEARTBEAT_PREFIX}{self.node_id}"

    async def _beat(self) -> None:
        if self._redis is None:
            return
        await self._redis.hset(
            self._key(),
            mapping={
                "node_id": self.node_id,
                "hostname": socket.gethostname(),
                "pid": str(os.getpid()),
                "started_at": str(self.started_at),
                "last_heartbeat": str(int(time.time())),
                "draining": "1" if self.draining else "0",
                "metrics_port": str(self.metrics_port),
            },
        )
        await self._redis.expire(self._key(), HEARTBEAT_TTL)

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            try:
                await self._beat()
            except Exception as e:
                # Liveness must never kill the worker; just warn and retry.
                logger.warning("worker_heartbeat_failed", error=str(e))


async def startup(ctx: dict) -> None:
    from noema.billing.ledger import ContributionLedger
    from noema.config.settings import get_settings

    settings = get_settings()
    ctx["noema"] = NoemaEngine()
    await ctx["noema"].initialize()
    node_id = make_node_id()
    ctx["node_id"] = node_id
    ctx["heartbeat"] = NodeHeartbeat(
        node_id,
        settings.redis.url,
        metrics_port=settings.obs.metrics_port if settings.obs.metrics_enabled else 0,
    )
    await ctx["heartbeat"].start()
    # Per-node contribution ledger (T3.3): durable JSONL when configured.
    ctx["ledger"] = ContributionLedger(path=settings.worker.ledger_path)
    logger.info("arq_worker_startup_complete", node_id=node_id)


async def drain(ctx: dict) -> None:
    """Graceful drain: mark draining, drop the heartbeat, stop the engine.

    Arq itself stops polling and lets the in-flight job complete on SIGTERM;
    this publishes that state so external observers stop routing work here.
    """
    heartbeat: NodeHeartbeat | None = ctx.get("heartbeat")
    if heartbeat is not None:
        await heartbeat.mark_draining()
        await heartbeat.stop()
    noema: NoemaEngine | None = ctx.get("noema")
    if noema is not None:
        await noema.shutdown()
    logger.info("arq_worker_drained", node_id=ctx.get("node_id", "?"))


async def shutdown(ctx: dict) -> None:
    await drain(ctx)
    logger.info("arq_worker_shutdown_complete")


async def think_task(ctx: dict, task_data: dict) -> dict:
    """Run noema.think() as a background job."""
    from noema.billing.cost_tracker import calculate_cost

    noema: NoemaEngine = ctx["noema"]
    ledger = ctx.get("ledger")
    task = Task(
        title=task_data["title"],
        description=task_data.get("description", ""),
        complexity=TaskComplexity(task_data.get("complexity", "moderate")),
        tags=task_data.get("tags", []),
        requirements=task_data.get("requirements", []),
    )
    before = noema.tracer.get_stats()
    try:
        solution, thought = await noema.think(task)
        after = noema.tracer.get_stats()
        if ledger is not None:
            model = task_data.get("model", "") or noema.llm.model_name
            tokens_input = int(after.get("tokens_input", 0) - before.get("tokens_input", 0))
            tokens_output = int(after.get("tokens_output", 0) - before.get("tokens_output", 0))
            ledger.record(
                node_id=ctx.get("node_id", ""),
                task_id=solution.task_id,
                kind="solution",
                provider=task_data.get("provider", ""),
                model=model,
                input_tokens=tokens_input,
                output_tokens=tokens_output,
                cost_usd=calculate_cost(model, tokens_input, tokens_output) if model else 0.0,
                artifact_ref=solution.id,
                meta={"duration_ms": thought.duration_ms, "quality": solution.quality.value},
            )
        return {
            "status": "completed",
            "solution_id": solution.id,
            "task_id": solution.task_id,
            "quality": solution.quality.value,
            "confidence": solution.confidence,
            "duration_ms": thought.duration_ms,
        }
    except Exception as e:
        logger.error("arq_think_task_failed", error=str(e), exc_info=e)
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


async def create_worker(redis_url: str | None = None, burst: bool = False) -> Any:
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
        burst=burst,
    )
    return worker


async def run_worker(redis_url: str | None = None, burst: bool = False) -> None:
    """Blocking entrypoint: run the arq worker until signalled.

    The worker drains gracefully on SIGINT/SIGTERM (in-flight job completes,
    liveness key removed).
    """
    worker = await create_worker(redis_url, burst=burst)
    await worker.async_run()


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


async def list_active_workers(
    redis_url: str, redis: AsyncRedis | None = None
) -> list[dict[str, str]]:
    """List live worker nodes from their Redis heartbeat keys."""
    r: AsyncRedis = redis or Redis.from_url(redis_url, decode_responses=True)
    owned = redis is None
    try:
        keys = await r.keys(f"{HEARTBEAT_PREFIX}*")
        workers: list[dict[str, str]] = []
        for key in keys:
            data = await r.hgetall(key)
            if not data:
                continue
            decoded = {
                str(k, errors="replace") if isinstance(k, bytes) else str(k): _as_str(v)
                for k, v in data.items()
            }
            decoded["key"] = key if isinstance(key, str) else str(key, errors="replace")
            workers.append(decoded)
        workers.sort(key=lambda w: w.get("started_at", "0"))
        return workers
    finally:
        if owned:
            await r.aclose()


def _as_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)
