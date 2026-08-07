"""Cancellation Manager — graceful task cancellation with resource cleanup."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from noema.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Coroutine

log = get_logger(__name__)


class CancelledTaskError(Exception):
    """Raised when a task is cancelled by the user."""


class CancellationManager:
    """Manages asyncio tasks with support for external cancellation.

    Usage:
        mgr = CancellationManager()
        result = await mgr.execute_with_cancellation(task_id, engine.think(task))
        mgr.cancel(task_id)  # triggers CancelledError -> CancelledTaskError
    """

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}

    async def execute_with_cancellation(self, task_id: str, coro: Coroutine[Any, Any, Any]) -> Any:
        if task_id in self._tasks:
            raise RuntimeError(f"Task {task_id} already in progress")
        task = asyncio.create_task(coro, name=task_id)
        self._tasks[task_id] = task
        try:
            return await task
        except asyncio.CancelledError:
            log.info("task_cancelled_by_user", task_id=task_id)
            raise CancelledTaskError(f"Task {task_id} was cancelled") from None
        finally:
            self._tasks.pop(task_id, None)

    def cancel(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task and not task.done():
            task.cancel()
            log.info("cancel_requested", task_id=task_id)
            return True
        return False

    def get_active(self) -> list[str]:
        return list(self._tasks.keys())

    def cancel_all(self) -> int:
        count = 0
        for task_id in list(self._tasks.keys()):
            if self.cancel(task_id):
                count += 1
        return count
