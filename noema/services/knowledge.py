"""Knowledge service — wraps KnowledgeStore + KnowledgeGraph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from noema.core.types import Task, TechStack
from noema.logging import get_logger

if TYPE_CHECKING:
    from noema.core.events import EventBus
    from noema.knowledge.graph import KnowledgeGraph
    from noema.knowledge.store import KnowledgeStore

log = get_logger(__name__)


class KnowledgeService:
    """Manages knowledge base and graph operations."""

    def __init__(
        self,
        store: KnowledgeStore,
        graph: KnowledgeGraph,
        event_bus: EventBus | None = None,
    ) -> None:
        self.store = store
        self.graph = graph
        self.event_bus = event_bus

    async def initialize(self) -> None:
        await self.store.load()
        log.info("knowledge_service_initialized", stats=self.store.get_stats())

    async def shutdown(self) -> None:
        await self.store.persist()

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return await self.store.search(query, top_k=top_k)

    async def gather_context(self, task: Task) -> str:
        query = f"{task.title} {task.description} {' '.join(task.tags)}"
        results = await self.store.search(query, top_k=5)
        parts = []
        for r in results:
            title = r.get("title", r.get("name", ""))
            content = r.get("content", r.get("description", ""))
            parts.append(f"[{r.get('type', 'knowledge')}] {title}: {content[:300]}")
        return "\n".join(parts)

    def gather_graph_context(self, task: Task) -> str:
        recs = self.graph.suggest_architecture(task.tags)
        parts = []
        for comp in recs.get("components", [])[:15]:
            parts.append(f"{comp['from']} -> {comp['to']} ({comp['relationship']})")
        return "\n".join(parts) if parts else ""

    async def select_stack(self, task: Task) -> TechStack:
        if task.preferred_stack:
            return task.preferred_stack
        candidates = await self.store.find_relevant_stacks(task)
        return (
            candidates[0]
            if candidates
            else TechStack(
                languages=["Python", "TypeScript"],
                frameworks=["FastAPI", "React"],
                databases=["PostgreSQL", "Redis"],
                infrastructure=["Docker"],
            )
        )

    def get_stats(self) -> dict[str, Any]:
        stats = self.store.get_stats()
        stats["graph"] = self.graph.get_stats()
        return stats

    async def ingest_file(self, path: str) -> dict[str, Any]:
        from noema.ingestion.loader import KnowledgeLoader

        loader = KnowledgeLoader(knowledge_store=self.store)
        result = await loader.ingest_file(path, tags=["ingested"])
        if self.event_bus:
            await self.event_bus.emit(
                "knowledge.file_ingested",
                {"path": path, "entries": result.entries_ingested},
                source="knowledge_service",
            )
        return result.model_dump()

    async def ingest_directory(
        self, path: str, patterns: list[str] | None = None
    ) -> dict[str, Any]:
        from noema.ingestion.loader import KnowledgeLoader

        loader = KnowledgeLoader(knowledge_store=self.store)
        result = await loader.ingest_directory(path, patterns=patterns, tags=["ingested"])
        if self.event_bus:
            await self.event_bus.emit(
                "knowledge.directory_ingested",
                {"path": path, "entries": result.entries_ingested},
                source="knowledge_service",
            )
        return result.model_dump()

    async def ingest_text(self, text: str, source: str = "direct") -> dict[str, Any]:
        from noema.ingestion.loader import KnowledgeLoader

        loader = KnowledgeLoader(knowledge_store=self.store)
        result = await loader.ingest_text(text, source_name=source, tags=["ingested"])
        if self.event_bus:
            await self.event_bus.emit(
                "knowledge.text_ingested",
                {"source": source, "entries": result.entries_ingested},
                source="knowledge_service",
            )
        return result.model_dump()
