"""Worker service — wraps WorkerPool + WorkerHierarchy with events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from noema.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from noema.core.events import EventBus
    from noema.workers.hierarchy import WorkerHierarchy
    from noema.workers.pool import WorkerPool

log = get_logger(__name__)


class WorkerService:
    """Manages worker pool and infinite sub-worker hierarchy."""

    def __init__(
        self,
        pool: WorkerPool,
        hierarchy: WorkerHierarchy,
        event_bus: EventBus | None = None,
    ) -> None:
        self.pool = pool
        self.hierarchy = hierarchy
        self.event_bus = event_bus

    async def start(self) -> None:
        await self.pool.start()
        log.info("worker_service_started")

    async def shutdown(self) -> None:
        await self.pool.shutdown()
        log.info("worker_service_shutdown")

    @property
    def stats(self) -> dict[str, Any]:
        return self.pool.stats

    async def execute_hierarchical(
        self,
        description: str,
        decomposer: Callable | None = None,
        executor: Callable | None = None,
        aggregator: Callable | None = None,
    ) -> dict[str, Any]:
        if self.event_bus:
            await self.event_bus.emit(
                "worker.hierarchical_start",
                {"description": description[:200]},
                source="worker_service",
            )
        task = await self.hierarchy.execute(
            description,
            decomposer=decomposer,
            executor=executor,
            aggregator=aggregator,
        )
        result = {
            "task_id": task.id,
            "state": task.state.value,
            "result": task.result,
            "subtasks": len(task.subtasks),
            "depth": task.depth,
        }
        if self.event_bus:
            await self.event_bus.emit(
                "worker.hierarchical_done",
                {"task_id": task.id, "subtasks": len(task.subtasks)},
                source="worker_service",
            )
        return result

    def hierarchy_stats(self) -> dict[str, Any]:
        return self.hierarchy.get_stats()
