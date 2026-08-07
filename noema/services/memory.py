"""Memory service — wraps MemoryStore with events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from noema.core.events import schedule_event
from noema.logging import get_logger

if TYPE_CHECKING:
    from noema.core.events import EventBus
    from noema.core.types import Solution, Task
    from noema.memory.store import MemoryStore

log = get_logger(__name__)


class MemoryService:
    """Manages three-tier memory (episodic, semantic, procedural) with events."""

    def __init__(
        self,
        store: MemoryStore,
        event_bus: EventBus | None = None,
    ) -> None:
        self.store = store
        self.event_bus = event_bus

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if not self.event_bus:
            return
        schedule_event(self.event_bus, event_type, payload, source="memory_service")

    def save(self) -> None:
        self.store.save()

    def flush(self) -> None:
        self.store.flush()

    def record_episode(
        self,
        task_description: str,
        solution_summary: str,
        tech_stack: str = "",
        outcome: str = "success",
        duration_seconds: float = 0.0,
        error_message: str = "",
        tags: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ep = self.store.record_episode(
            task_description=task_description,
            solution_summary=solution_summary,
            tech_stack=tech_stack,
            outcome=outcome,
            duration_seconds=duration_seconds,
            error_message=error_message,
            tags=tags,
            context=context,
        )
        if self.event_bus:
            self._emit_event("memory.episode_recorded", {"episode_id": ep.id, "outcome": outcome})

    def search_episodes(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return [ep.model_dump() for ep in self.store.search_episodes(query, limit)]

    def search_knowledge(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return [k.model_dump() for k in self.store.search_knowledge(query, limit)]

    def search_procedures(self, query: str) -> list[dict[str, Any]]:
        return [p.model_dump() for p in self.store.search_procedures(query)]

    def learn_fact(
        self,
        topic: str,
        fact: str,
        confidence: float = 0.8,
        source: str = "",
        tags: list[str] | None = None,
    ) -> None:
        self.store.learn_fact(
            topic=topic, fact=fact, confidence=confidence, source=source, tags=tags
        )
        if self.event_bus:
            self._emit_event("memory.fact_learned", {"topic": topic, "confidence": confidence})

    def store_procedure(
        self, procedure_name: str, steps: list[str], tags: list[str] | None = None
    ) -> None:
        self.store.store_procedure(procedure_name=procedure_name, steps=steps, tags=tags)

    def record_procedure_outcome(
        self, procedure_name: str, succeeded: bool, duration: float = 0.0
    ) -> None:
        self.store.record_procedure_outcome(procedure_name, succeeded, duration)

    def record_task_outcome(self, task: Task, solution: Solution, duration_seconds: float) -> None:
        outcome = (
            "success"
            if solution.quality.value in ("masterpiece", "excellent", "good")
            else "partial"
        )
        self.record_episode(
            task_description=f"{task.title}: {task.description}",
            solution_summary=solution.summary[:500],
            tech_stack=", ".join(solution.stack.languages + solution.stack.frameworks)
            if solution.stack
            else "",
            outcome=outcome,
            duration_seconds=duration_seconds,
            tags=task.tags,
            context={"quality": solution.quality.value, "confidence": solution.confidence},
        )
        if solution.quality.value in ("masterpiece", "excellent"):
            self.store.store_procedure(
                procedure_name=f"solution_{task.title[:30].replace(' ', '_')}",
                steps=[],
                tags=task.tags,
            )

    def search(self, query: str, kind: str = "all") -> dict[str, Any]:
        result: dict[str, Any] = {}
        if kind in ("all", "episodes"):
            result["episodes"] = self.search_episodes(query)
        if kind in ("all", "knowledge"):
            result["knowledge"] = self.search_knowledge(query)
        if kind in ("all", "procedures"):
            result["procedures"] = self.search_procedures(query)
        return result

    def get_recent_context(self, task: Task, limit: int = 5) -> str:
        query = f"{task.title} {task.description}"
        episodes = self.store.search_episodes(query, limit=limit)
        if not episodes:
            return ""
        return "\n".join(f"[Past] {ep.task_description[:100]} → {ep.outcome}" for ep in episodes)

    def stats(self) -> dict[str, Any]:
        return self.store.stats()
