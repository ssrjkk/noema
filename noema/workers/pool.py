"""Async worker pool for bounded parallel task execution.

Architecture:
- A fixed roster of ``Worker`` coroutines drains a shared :class:`asyncio.Queue`
  and executes submitted coroutines concurrently.
- ``submit`` does not poll: each task carries an :class:`asyncio.Event` that the
  worker sets on completion, so submitters wake exactly once.

Concurrency contract:
- No blocking calls in the event loop: task functions are pure coroutines.
- Bounded concurrency by construction: at most ``max_workers`` coroutines run
  concurrently, and the queue holds at most ``max_queue_size`` pending tasks.

Complexity:
- ``submit``: ``O(1)`` work, ``O(task_wall_time)`` wait.
- ``submit_many``: ``O(N)`` tasks, wall time ``O(N / max_workers)`` on average.
- Worker loop: ``O(1)`` bookkeeping per task.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from noema.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

logger = get_logger(__name__)


class WorkerState(StrEnum):
    IDLE = "idle"
    BUSY = "busy"
    SHUTDOWN = "shutdown"


class TaskState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkerTask:
    """A queued unit of work for the pool."""

    id: str
    func: Callable[..., Coroutine]
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    state: TaskState = TaskState.PENDING
    result: Any = None
    error: Exception | None = None
    done: asyncio.Event | None = None
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    completed_at: float | None = None

    @property
    def duration_ms(self) -> float | None:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return None


@dataclass
class Worker:
    """A single worker coroutine in the pool."""

    id: int
    state: WorkerState = WorkerState.IDLE
    current_task: WorkerTask | None = None
    tasks_completed: int = 0
    total_time_ms: float = 0.0


class WorkerPool:
    """Bounded async worker pool draining a shared task queue.

    Args:
        max_workers: Maximum number of concurrently executing task coroutines.
        max_queue_size: Back-pressure cap on the pending-task queue.
    """

    def __init__(self, max_workers: int = 4, max_queue_size: int = 100) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.workers: list[Worker] = []
        self._queue: asyncio.Queue[WorkerTask] = asyncio.Queue(maxsize=max_queue_size)
        self._tasks: dict[str, WorkerTask] = {}
        self._done_events: dict[str, asyncio.Event] = {}
        self._worker_tasks: list[asyncio.Task] = []
        self._running = False
        self._stats = {
            "total_submitted": 0,
            "total_completed": 0,
            "total_failed": 0,
        }

    async def start(self) -> None:
        """Spawn the worker coroutines. Idempotent."""
        if self._running:
            return

        self._running = True
        for i in range(self.max_workers):
            worker = Worker(id=i)
            self.workers.append(worker)
            task = asyncio.create_task(self._worker_loop(worker))
            self._worker_tasks.append(task)

        logger.info(f"WorkerPool запущен с {self.max_workers} воркерами")

    async def shutdown(self) -> None:
        """Cancel worker coroutines and release any pending submitters.

        Complexity: ``O(W)`` for W workers; pending ``submit`` calls finish
        gracefully instead of hanging.
        """
        self._running = False

        for worker in self.workers:
            worker.state = WorkerState.SHUTDOWN

        for worker_task in self._tasks.values():
            if worker_task.done is not None:
                worker_task.done.set()

        for task in self._worker_tasks:
            task.cancel()

        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        self._tasks.clear()
        self._done_events.clear()
        logger.info("WorkerPool остановлен")

    async def submit(
        self,
        func: Callable[..., Coroutine],
        *args,
        **kwargs,
    ) -> Any:
        """Enqueue and await a task, returning its result.

        Waits on the task's completion event rather than polling.

        Complexity: ``O(1)`` bookkeeping; waits for the worker to finish.
        Raises the task's original exception if it failed.
        """
        import uuid

        if not self._running:
            raise RuntimeError("WorkerPool is not running; call await pool.start() before submit()")
        task_id = uuid.uuid4().hex[:8]
        done = asyncio.Event()
        worker_task = WorkerTask(id=task_id, func=func, args=args, kwargs=kwargs, done=done)
        self._tasks[task_id] = worker_task
        self._done_events[task_id] = done
        self._stats["total_submitted"] += 1

        await self._queue.put(worker_task)
        try:
            await done.wait()
        finally:
            # Always release the task bookkeeping, including on failure or
            # shutdown cancellation, so the pool does not accumulate entries.
            self._tasks.pop(task_id, None)
            self._done_events.pop(task_id, None)

        if worker_task.state == TaskState.FAILED:
            raise worker_task.error  # type: ignore

        return worker_task.result

    async def submit_many(
        self,
        tasks: list[tuple[Callable[..., Coroutine], tuple, dict]],
    ) -> list[Any]:
        """Submit several tasks concurrently, collecting results in order.

        Complexity: ``O(N)`` tasks; wall time ``O(N / max_workers)`` on average
        when task durations are comparable. Exceptions are returned as values.
        """
        futures = [self.submit(func, *args, **kwargs) for func, args, kwargs in tasks]
        return await asyncio.gather(*futures, return_exceptions=True)

    async def _worker_loop(self, worker: Worker) -> None:
        """Main loop: take tasks off the queue and execute them one by one."""
        while self._running:
            try:
                task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except TimeoutError:
                continue

            worker.state = WorkerState.BUSY
            worker.current_task = task
            task.state = TaskState.RUNNING
            task.started_at = time.monotonic()

            try:
                result = await task.func(*task.args, **task.kwargs)
                task.result = result
                task.state = TaskState.COMPLETED
                worker.tasks_completed += 1
                self._stats["total_completed"] += 1
            except Exception as e:
                task.error = e
                task.state = TaskState.FAILED
                self._stats["total_failed"] += 1
                logger.error(f"Worker {worker.id} task {task.id} failed: {e}")
            finally:
                task.completed_at = time.monotonic()
                worker.total_time_ms += (task.completed_at - task.started_at) * 1000
                worker.state = WorkerState.IDLE
                worker.current_task = None
                self._queue.task_done()
                if task.done is not None:
                    task.done.set()

    @property
    def stats(self) -> dict[str, Any]:
        """Pool statistics (counters, worker utilization, queue depth)."""
        busy = sum(1 for w in self.workers if w.state == WorkerState.BUSY)
        return {
            **self._stats,
            "workers_total": self.max_workers,
            "workers_busy": busy,
            "workers_idle": self.max_workers - busy,
            "queue_size": self._queue.qsize(),
            "avg_task_time_ms": (
                sum(w.total_time_ms for w in self.workers)
                / max(sum(w.tasks_completed for w in self.workers), 1)
            ),
        }
