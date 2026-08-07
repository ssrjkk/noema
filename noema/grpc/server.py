"""gRPC server for NoemaEngine service."""

from __future__ import annotations

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
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ThinkResponse(error=str(e))

    async def ThinkStream(  # noqa: N802
        self, request: Any, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[ThinkStatus]:
        """Streaming Think — yield status updates."""
        task = self._request_to_task(request)
        try:
            solution, thought = await self.noema.think(task)
            yield ThinkStatus(
                stage="completed", status="ok", progress=1.0, message=f"Solution: {solution.id}"
            )
        except Exception as e:
            yield ThinkStatus(stage="error", status="failed", message=str(e))

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
    """Start gRPC server."""
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    add_NoemaEngineServiceServicer_to_server(NoemaEngineServicer(noema, cancellation_mgr), server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    logger.info("grpc_server_started", host=host, port=port)
    return server
