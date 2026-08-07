"""Scaffold service — wraps ProjectScaffolder with events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from noema.logging import get_logger
from noema.scaffolder.generator import ProjectScaffolder

if TYPE_CHECKING:
    from noema.core.events import EventBus
    from noema.core.types import Solution, Task

log = get_logger(__name__)


class ScaffoldService:
    """Generates project structures from solutions."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
    ) -> None:
        self.event_bus = event_bus

    async def scaffold(
        self, solution: Solution, task: Task, output_dir: str = "."
    ) -> dict[str, Any]:
        scaffolder = ProjectScaffolder(output_dir=output_dir)
        result = await scaffolder.scaffold(solution, task)
        if self.event_bus:
            await self.event_bus.emit(
                "scaffold.completed",
                {
                    "project_dir": result.get("project_dir", ""),
                    "files_created": result.get("files_created", 0),
                },
                source="scaffold_service",
            )
        log.info(
            "scaffold_completed",
            files=result.get("files_created", 0),
            project_dir=result.get("project_dir", ""),
        )
        return result
