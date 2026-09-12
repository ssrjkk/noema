"""Base kernel class — interface for all kernels."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, cast

from noema.logging import get_logger

if TYPE_CHECKING:
    from noema.core.types import Task, TechStack

logger = get_logger(__name__)


class BaseKernel(abc.ABC):
    """
    Abstract kernel.

    Each kernel is responsible for its own area of solution generation:
    - Architecture
    - Code generation
    - Optimization
    - Security
    - Analysis
    etc.
    """

    def __init__(self, knowledge: Any = None, **kwargs: Any) -> None:
        self.knowledge = knowledge
        self.config = kwargs
        self._hooks: dict[str, list] = {}

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Kernel name."""
        ...

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Kernel description."""
        ...

    @abc.abstractmethod
    async def execute(self, task: Task, **kwargs: Any) -> dict[str, Any]:
        """Main kernel execution method."""
        ...

    async def execute_subtask(
        self, subtask: dict, stack: TechStack | None = None
    ) -> dict[str, Any]:
        """Execute a subtask (for parallel processing)."""
        return {"subtask": subtask, "status": "processed"}

    def on(self, event: str, callback: Any) -> None:
        """Register a hook for an event."""
        self._hooks.setdefault(event, []).append(callback)

    async def _emit(self, event: str, data: Any) -> None:
        """Emit an event."""
        for cb in self._hooks.get(event, []):
            if callable(cb):
                result = cb(data)
                if hasattr(result, "__await__"):
                    await result

    async def _query_knowledge(self, query: str, top_k: int = 5) -> list[dict]:
        """Query the knowledge base."""
        if self.knowledge:
            return cast("list[dict]", await self.knowledge.search(query, top_k=top_k))
        return []

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
