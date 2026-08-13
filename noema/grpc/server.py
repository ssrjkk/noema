"""gRPC server for NoemaEngine service."""

from __future__ import annotations

import asyncio
import contextlib
import time
from concurrent import futures
from typing import TYPE_CHECKING, Any

import grpc

from noema.core.types import Requirement, Task, TaskComplexity
from noema.grpc.noema_engine_pb2 import (
    CancelResponse,
    EvolveResponse,
    HealthResponse,
    MetricsResponse,
    ThinkResponse,
    ThinkStatus,
)
from noema.grpc.noema_engine_pb2_grpc import (
    NoemaEngineServiceServicer,
    add_NoemaEngineServiceServicer_to_server,
)
from noema.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from noema.core.engine import NoemaEngine

logger = get_logger(__name__)


class NoemaEngineServicer(NoemaEngineServiceServicer):
    """gRPC servicer implementing NoemaEngineService."""

    def __init__(self, noema: NoemaEngine, cancellation_mgr: Any = None) -> None:
        self.noema = noema
        self._cancellation_mgr = cancellation_mgr
        self._start_time = time.monotonic()

    async def Think(self, request: Any, context: grpc.aio.ServicerContext) -> ThinkResponse:  # noqa: N802
        """Unary Think — generate solution."""
        try:
            task = self._request_to_task(request)
            solution, thought = await self.noema.think(task)
            return ThinkResponse(
                solution_id=solution.id,
                task_id=solution.task_id,
                title=solution.title,
                summary=solution.summary,
                quality=solution.quality.value,
                confidence=solution.confidence,
                code_blocks_count=len(solution.code_blocks),
                duration_ms=thought.duration_ms,
            )
        except Exception as e:
            logger.error("grpc_think_failed", error=str(e), exc_info=e)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
            raise  # pragma: no cover - abort() always raises

    async def ThinkStream(  # noqa: N802
        self, request: Any, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[ThinkStatus]:
        """Streaming Think — yield real per-step engine progress.

        The engine's ``on_step_start``/``on_step_end`` callbacks feed an
        asyncio queue which is drained into the gRPC response stream, so a
        client sees each reasoning phase as it happens rather than a single
        fabricated update.
        """
        task = self._request_to_task(request)
        queue: asyncio.Queue[ThinkStatus] = asyncio.Queue(maxsize=64)
        correlation_id = request.task_id or task.id

        async def on_step_start(name: str, label: str, done: int, total: int) -> None:
            await queue.put(
                ThinkStatus(
                    stage=name,
                    status="running",
                    progress=done / total if total else 0.0,
                    message=label,
                    correlation_id=correlation_id,
                )
            )

        async def on_step_end(name: str, result: str, done: int, total: int) -> None:
            await queue.put(
                ThinkStatus(
                    stage=name,
                    status="ok" if not result.startswith("FAILED") else "failed",
                    progress=done / total if total else 1.0,
                    message=result[:200],
                    correlation_id=correlation_id,
                )
            )

        run = asyncio.create_task(
            self._run_think(task, queue, on_step_start, on_step_end, correlation_id)
        )
        try:
            while True:
                status = await queue.get()
                yield status
                if status.stage in ("completed", "error"):
                    break
        finally:
            run.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await run

    async def _run_think(
        self,
        task: Task,
        queue: asyncio.Queue[ThinkStatus],
        on_step_start: Any,
        on_step_end: Any,
        correlation_id: str,
    ) -> None:
        try:
            solution, _ = await self.noema.think(
                task, on_step_start=on_step_start, on_step_end=on_step_end
            )
            await queue.put(
                ThinkStatus(
                    stage="completed",
                    status="ok",
                    progress=1.0,
                    message=f"Solution: {solution.id}",
                    correlation_id=correlation_id,
                )
            )
        except Exception as e:
            logger.error("grpc_think_stream_failed", error=str(e), exc_info=e)
            await queue.put(
                ThinkStatus(
                    stage="error",
                    status="failed",
                    progress=1.0,
                    message=str(e),
                    correlation_id=correlation_id,
                )
            )

    async def Health(self, request: Any, context: grpc.aio.ServicerContext) -> HealthResponse:  # noqa: N802
        return HealthResponse(
            status="ok",
            version="1.0.0",
            uptime_s=round(time.monotonic() - self._start_time, 1),
        )

    async def GetMetrics(self, request: Any, context: grpc.aio.ServicerContext) -> MetricsResponse:  # noqa: N802
        metrics = self.noema.neurosymbolic.get_metrics() if self.noema.neurosymbolic else {}
        return MetricsResponse(
            tasks_processed=metrics.get("tasks_processed", 0),
            tasks_successful=metrics.get("tasks_successful", 0),
            tasks_failed=metrics.get("tasks_failed", 0),
            success_rate=metrics.get("success_rate", 0.0),
            total_llm_calls=metrics.get("total_llm_calls", 0),
            total_refinements=metrics.get("total_refinements", 0),
        )

    async def CancelTask(self, request: Any, context: grpc.aio.ServicerContext) -> CancelResponse:  # noqa: N802
        if self._cancellation_mgr is None:
            return CancelResponse(
                cancelled=False,
                task_id=request.task_id,
                status="not_found",
            )
        cancelled = self._cancellation_mgr.cancel(request.task_id)
        return CancelResponse(
            cancelled=cancelled,
            task_id=request.task_id,
            status="cancelled" if cancelled else "not_found",
        )

    async def Evolve(self, request: Any, context: grpc.aio.ServicerContext) -> EvolveResponse:  # noqa: N802
        result = await self.noema.evolve()
        return EvolveResponse(
            patches_generated=result.get("patches_generated", 0),
            patches_applied=result.get("patches_applied", 0),
            summary=result.get("summary", ""),
            improvements=result.get("improvements", []),
        )

    def _request_to_task(self, request: Any) -> Task:
        return Task(
            title=request.title,
            description=request.description,
            complexity=TaskComplexity(request.complexity)
            if request.complexity
            else TaskComplexity.MODERATE,
            tags=list(request.tags),
            requirements=[
                Requirement(category=r.category, description=r.description, priority=r.priority)
                for r in request.requirements
            ],
        )


async def serve_grpc(
    noema: NoemaEngine, host: str = "[::]", port: int = 50051, cancellation_mgr: Any = None
) -> grpc.aio.Server:
    """Start a gRPC server bound to ``host:port`` (returns the running server).

    Call :func:`stop_grpc` for a graceful drain on shutdown.
    """
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    add_NoemaEngineServiceServicer_to_server(NoemaEngineServicer(noema, cancellation_mgr), server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    logger.info("grpc_server_started", host=host, port=port)
    return server


async def stop_grpc(server: grpc.aio.Server, grace: float = 5.0) -> None:
    """Gracefully stop a gRPC server, letting in-flight RPCs drain."""
    await server.stop(grace)
