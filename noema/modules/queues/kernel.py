"""Queue Module — async task processing, message brokers, job scheduling."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class QueueBackend(StrEnum):
    IN_MEMORY = "in_memory"
    REDIS = "redis"
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    SQS = "sqs"
    CELERY = "celery"
    BULL = "bull"


class JobStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class JobPriority(int, Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Job:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    queue_name: str = "default"
    name: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    max_retries: int = 3
    retry_count: int = 0
    result: Any = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    timeout_seconds: float = 300.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueueStats:
    name: str = ""
    pending: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    avg_process_time_ms: float = 0.0
    throughput_per_second: float = 0.0


@dataclass
class ScheduleRule:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    cron: str = ""  # * * * * *
    interval_seconds: float = 0
    job_name: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_run: float = 0.0
    next_run: float = 0.0


class JobQueue:
    """In-memory priority job queue with async processing."""

    def __init__(self, name: str = "default", max_workers: int = 4) -> None:
        self.name = name
        self.max_workers = max_workers
        self._queue: asyncio.PriorityQueue[tuple[int, float, Job]] = asyncio.PriorityQueue()
        self._jobs: dict[str, Job] = {}
        self._handlers: dict[str, Callable] = {}
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._stats = QueueStats(name=name)
        self._process_times: list[float] = []

    def register_handler(self, job_name: str, handler: Callable) -> None:
        self._handlers[job_name] = handler

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker_loop(i))
            self._workers.append(task)

    async def stop(self) -> None:
        self._running = False
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def enqueue(
        self,
        job_name: str,
        payload: dict[str, Any] | None = None,
        priority: JobPriority = JobPriority.NORMAL,
        queue_name: str = "default",
        max_retries: int = 3,
        timeout: float = 300.0,
    ) -> Job:
        job = Job(
            name=job_name,
            queue_name=queue_name,
            payload=payload or {},
            priority=priority,
            max_retries=max_retries,
            timeout_seconds=timeout,
        )
        self._jobs[job.id] = job
        priority_val = -priority.value  # higher priority = lower number for min-heap
        await self._queue.put((priority_val, job.created_at, job))
        self._stats.pending += 1
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def get_jobs_by_status(self, status: JobStatus) -> list[Job]:
        return [j for j in self._jobs.values() if j.status == status]

    def get_stats(self) -> dict[str, Any]:
        pending = sum(1 for j in self._jobs.values() if j.status == JobStatus.PENDING)
        processing = sum(1 for j in self._jobs.values() if j.status == JobStatus.PROCESSING)
        completed = sum(1 for j in self._jobs.values() if j.status == JobStatus.COMPLETED)
        failed = sum(1 for j in self._jobs.values() if j.status == JobStatus.FAILED)
        avg_time = sum(self._process_times) / max(len(self._process_times), 1)
        return {
            "queue": self.name,
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "total": len(self._jobs),
            "avg_process_time_ms": avg_time,
            "workers": self.max_workers,
            "running": self._running,
        }

    async def _worker_loop(self, worker_id: int) -> None:
        while self._running:
            try:
                _, _, job = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except TimeoutError:
                continue

            job.status = JobStatus.PROCESSING
            job.started_at = time.time()
            self._stats.pending -= 1
            self._stats.processing += 1

            handler = self._handlers.get(job.name)
            if not handler:
                job.status = JobStatus.FAILED
                job.error = f"No handler registered for '{job.name}'"
                self._stats.processing -= 1
                self._stats.failed += 1
                continue

            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await asyncio.wait_for(
                        handler(job.payload), timeout=job.timeout_seconds
                    )
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(handler, job.payload),
                        timeout=job.timeout_seconds,
                    )
                job.result = result
                job.status = JobStatus.COMPLETED
                self._stats.completed += 1
            except TimeoutError:
                job.error = "Job timed out"
                job.status = JobStatus.FAILED
                self._stats.failed += 1
            except Exception as e:
                if job.retry_count < job.max_retries:
                    job.retry_count += 1
                    job.status = JobStatus.RETRYING
                    priority_val = -job.priority.value
                    await self._queue.put((priority_val, job.created_at, job))
                    self._stats.pending += 1
                else:
                    job.error = str(e)
                    job.status = JobStatus.FAILED
                    self._stats.failed += 1
            finally:
                job.completed_at = time.time()
                self._stats.processing -= 1
                if job.started_at:
                    self._process_times.append((job.completed_at - job.started_at) * 1000)
                self._queue.task_done()


class Scheduler:
    """Cron-like job scheduler."""

    def __init__(self) -> None:
        self.rules: list[ScheduleRule] = []
        self._running = False
        self._task: asyncio.Task | None = None

    def add_rule(self, rule: ScheduleRule) -> None:
        self.rules.append(rule)

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _loop(self) -> None:
        while self._running:
            now = time.time()
            for rule in self.rules:
                if rule.enabled and now >= rule.next_run:
                    rule.last_run = now
                    rule.next_run = now + max(rule.interval_seconds, 60)
            await asyncio.sleep(1)

    def get_due_rules(self) -> list[ScheduleRule]:
        now = time.time()
        return [r for r in self.rules if r.enabled and now >= r.next_run]


class QueuesModule:
    """Standalone queue module."""

    NAME = "queues"
    DESCRIPTION = "Async job queues, message brokers, scheduling"

    def __init__(self) -> None:
        self.queues: dict[str, JobQueue] = {}
        self.scheduler = Scheduler()

    def create_queue(self, name: str = "default", max_workers: int = 4) -> JobQueue:
        queue = JobQueue(name=name, max_workers=max_workers)
        self.queues[name] = queue
        return queue

    def get_default_queue(self) -> JobQueue:
        if "default" not in self.queues:
            self.create_queue("default")
        return self.queues["default"]

    def execute(self, task: Any) -> dict[str, Any]:
        tags = getattr(task, "tags", [])
        recommended = []
        if "redis" in tags:
            recommended.append({"backend": "redis", "use_case": "General purpose queue"})
        if "kafka" in tags:
            recommended.append({"backend": "kafka", "use_case": "Event streaming, high throughput"})
        if "rabbitmq" in tags:
            recommended.append({"backend": "rabbitmq", "use_case": "Complex routing, reliability"})
        if not recommended:
            recommended = [
                {"backend": "redis", "use_case": "Default choice, fast, reliable"},
                {"backend": "rabbitmq", "use_case": "When you need complex routing"},
            ]
        return {
            "type": "queues",
            "queues_active": len(self.queues),
            "recommendations": recommended,
            "_confidence": 0.85,
        }
