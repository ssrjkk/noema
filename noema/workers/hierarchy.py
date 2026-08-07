"""Infinite worker hierarchy — tree-structured task decomposition.

Architecture:
- :class:`WorkerNode.process` runs three phases per node: decompose the task
  into subtasks, spawn each subtask concurrently, then aggregate the results.
- :class:`WorkerHierarchy` owns the root node, a global :class:`asyncio.Semaphore`
  bounding concurrent in-flight sub-workers, and per-node callbacks.

Concurrency contract:
- All decomposition fan-out goes through :func:`asyncio.gather` under the
  ``max_concurrent`` semaphore; failures inside a subtree are returned as
  exception values, never raised across sibling boundaries.
- No blocking calls in the event loop: callbacks are pure coroutines.

Complexity:
- A task of depth ``D`` and branching factor ``B`` touches ``O(B**D)`` nodes in
  the worst case, but only ``O(max_concurrent)`` run at once; wall time is
  ``O(B**D / max_concurrent)`` for uniform subtasks.
- ``get_stats``: ``O(T)`` over the task history.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from noema.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

logger = get_logger(__name__)


class HierarchyTaskState(StrEnum):
    PENDING = "pending"
    DECOMPOSING = "decomposing"
    RUNNING = "running"
    COLLECTING = "collecting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class HierarchyTask:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    state: HierarchyTaskState = HierarchyTaskState.PENDING
    result: Any = None
    error: Exception | None = None
    subtasks: list[HierarchyTask] = field(default_factory=list)
    parent_id: str | None = None
    depth: int = 0
    created_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None
    worker_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkerNode:
    """A single node in the worker tree that can spawn sub-workers."""

    def __init__(
        self,
        node_id: int,
        decomposer: Callable[..., Coroutine] | None = None,
        executor: Callable[..., Coroutine] | None = None,
        aggregator: Callable[..., Coroutine] | None = None,
        max_depth: int = 10,
        max_children: int = 100,
    ) -> None:
        self.node_id = node_id
        self.decomposer = decomposer
        self.executor = executor
        self.aggregator = aggregator
        self.max_depth = max_depth
        self.max_children = max_children
        self.children: list[WorkerNode] = []
        self.tasks_completed = 0
        self.total_time_ms = 0.0

    async def process(
        self,
        task: HierarchyTask,
        spawn_child: Callable[..., Coroutine],
    ) -> Any:
        """Run decompose → spawn subtasks → aggregate for one tree node.

        Subtask fan-out is capped at ``max_children`` and runs concurrently
        under the caller-supplied ``spawn_child`` (which owns the semaphore).

        Complexity: ``O(B)`` subtask spawns, where ``B = min(n, max_children)``;
        wall time ``O(max_subtask)`` plus aggregation.
        """
        task.state = HierarchyTaskState.RUNNING
        task.worker_id = self.node_id
        start = time.monotonic()

        try:
            # Phase 1: Decompose if we have a decomposer and haven't hit max depth
            if self.decomposer and task.depth < self.max_depth:
                task.state = HierarchyTaskState.DECOMPOSING
                subtask_descriptions = await self.decomposer(task)
                if subtask_descriptions:
                    # Phase 2: Spawn sub-workers for each subtask
                    task.state = HierarchyTaskState.COLLECTING
                    subtasks = []
                    for desc in subtask_descriptions[: self.max_children]:
                        subtask = HierarchyTask(
                            description=desc,
                            depth=task.depth + 1,
                            parent_id=task.id,
                        )
                        subtasks.append(subtask)
                    task.subtasks = subtasks

                    # spawn children concurrently
                    subtask_results = await asyncio.gather(
                        *[spawn_child(st, self.node_id) for st in subtasks],
                        return_exceptions=True,
                    )

                    for st, res in zip(subtasks, subtask_results, strict=False):
                        if isinstance(res, BaseException):
                            st.state = HierarchyTaskState.FAILED
                            st.error = res if isinstance(res, Exception) else Exception(str(res))
                            st.completed_at = time.monotonic()
                            st.result = None

                    # Phase 3: Aggregate
                    if self.aggregator:
                        task.result = await self.aggregator(task, subtask_results)
                    else:
                        task.result = subtask_results
                else:
                    # no subtasks — execute directly
                    task.result = await self._execute(task)
            else:
                # leaf node or max depth — execute directly
                task.result = await self._execute(task)

            task.state = HierarchyTaskState.COMPLETED
            elapsed = (time.monotonic() - start) * 1000
            self.total_time_ms += elapsed
            self.tasks_completed += 1
            task.completed_at = time.monotonic()
            return task.result

        except Exception as e:
            task.error = e
            task.state = HierarchyTaskState.FAILED
            task.completed_at = time.monotonic()
            raise

    async def _execute(self, task: HierarchyTask) -> Any:
        """Run the executor callback, or report a no-executor stub.

        Complexity: ``O(executor)`` — whatever the callback does.
        """
        if self.executor:
            return await self.executor(task)
        return {"task": task.description, "status": "no_executor"}


class WorkerHierarchy:
    """Bounded tree-structured worker hierarchy.

    Tasks are decomposed recursively by decomposer functions, spawned as
    sub-workers under a global concurrency semaphore, and aggregated back.

    Args:
        max_depth: Maximum recursion depth for decomposition.
        max_children_per_node: Cap on subtasks per decomposition step.
        max_concurrent: Global cap on concurrently in-flight sub-workers.
    """

    def __init__(
        self,
        decomposer: Callable[..., Coroutine] | None = None,
        executor: Callable[..., Coroutine] | None = None,
        aggregator: Callable[..., Coroutine] | None = None,
        max_depth: int = 10,
        max_children_per_node: int = 100,
        max_concurrent: int = 50,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if max_children_per_node < 1:
            raise ValueError("max_children_per_node must be >= 1")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.decomposer = decomposer
        self.executor = executor
        self.aggregator = aggregator
        self.max_depth = max_depth
        self.max_children_per_node = max_children_per_node
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._root_node = WorkerNode(
            node_id=0,
            decomposer=decomposer,
            executor=executor,
            aggregator=aggregator,
            max_depth=max_depth,
            max_children=max_children_per_node,
        )
        self._nodes: dict[int, WorkerNode] = {0: self._root_node}
        self._task_history: list[HierarchyTask] = []
        self._node_counter = 1

    def _create_node(
        self,
        decomposer: Callable[..., Coroutine] | None = None,
        executor: Callable[..., Coroutine] | None = None,
        aggregator: Callable[..., Coroutine] | None = None,
    ) -> WorkerNode:
        node_id = self._node_counter
        self._node_counter += 1
        node = WorkerNode(
            node_id=node_id,
            decomposer=decomposer or self.decomposer,
            executor=executor or self.executor,
            aggregator=aggregator or self.aggregator,
            max_depth=self.max_depth,
            max_children=self.max_children_per_node,
        )
        self._nodes[node_id] = node
        return node

    async def execute(
        self,
        description: str,
        decomposer: Callable[..., Coroutine] | None = None,
        executor: Callable[..., Coroutine] | None = None,
        aggregator: Callable[..., Coroutine] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HierarchyTask:
        """Execute a top-level task through the hierarchy.

        Complexity: ``O(B**D)`` node visits worst case (see module docstring),
        wall time bounded by the semaphore.
        """
        root_task = HierarchyTask(
            description=description,
            metadata=metadata or {},
        )

        # create a root node with custom callbacks if provided
        root_node = WorkerNode(
            node_id=0,
            decomposer=decomposer or self._root_node.decomposer,
            executor=executor or self._root_node.executor,
            aggregator=aggregator or self._root_node.aggregator,
            max_depth=self.max_depth,
            max_children=self.max_children_per_node,
        )

        async def spawn_child(task: HierarchyTask, parent_id: int) -> Any:
            async with self._semaphore:
                child_node = self._create_node(
                    decomposer=decomposer,
                    executor=executor,
                    aggregator=aggregator,
                )
                return await child_node.process(task, spawn_child)

        try:
            result = await root_node.process(root_task, spawn_child)
            root_task.result = result
            root_task.state = HierarchyTaskState.COMPLETED
        except Exception as e:
            root_task.error = e
            root_task.state = HierarchyTaskState.FAILED

        self._task_history.append(root_task)
        return root_task

    async def execute_batch(
        self,
        descriptions: list[str],
        **kwargs: Any,
    ) -> list[HierarchyTask]:
        """Execute multiple top-level tasks concurrently.

        Complexity: ``O(N)`` top-level hierarchies, wall time bounded by the
        slowest hierarchy.
        """
        tasks = [self.execute(desc, **kwargs) for desc in descriptions]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def get_stats(self) -> dict[str, Any]:
        """Aggregate hierarchy statistics over the task history.

        Complexity: ``O(T)`` for T executed top-level tasks.
        """
        total_tasks = len(self._task_history)
        completed = sum(1 for t in self._task_history if t.state == HierarchyTaskState.COMPLETED)
        failed = sum(1 for t in self._task_history if t.state == HierarchyTaskState.FAILED)
        total_subtasks = sum(len(t.subtasks) for t in self._task_history)
        return {
            "total_tasks": total_tasks,
            "completed": completed,
            "failed": failed,
            "total_subtasks_spawned": total_subtasks,
            "nodes_created": self._node_counter,
            "max_depth_reached": self._max_depth_reached(),
            "avg_task_time_ms": self._avg_task_time(),
        }

    def _max_depth_reached(self) -> int:
        def _walk(task: HierarchyTask) -> int:
            if not task.subtasks:
                return task.depth
            return max(_walk(s) for s in task.subtasks)

        if not self._task_history:
            return 0
        return max(_walk(t) for t in self._task_history)

    def _avg_task_time(self) -> float:
        times = []
        for t in self._task_history:
            if t.created_at and t.completed_at:
                times.append((t.completed_at - t.created_at) * 1000)
        return sum(times) / max(len(times), 1)
