"""Three-tier memory system with dense vector search (HNSW) and atomic persistence."""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

from noema.context import get_tenant_id
from noema.embeddings import DenseEmbedder, HNSWIndex
from noema.logging import get_logger
from noema.utils.atomic_io import atomic_read_json, atomic_write_json

if TYPE_CHECKING:
    from collections.abc import Coroutine

log = get_logger(__name__)


def _log_save_failure(task: asyncio.Task) -> None:
    """Surface failures of the fire-and-forget debounced save instead of losing them."""
    with suppress(asyncio.CancelledError):
        exc = task.exception()
        if exc is not None:
            log.error("memory_debounced_save_failed", error=str(exc))


class EpisodicMemory(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = Field(default_factory=time.time)
    task_description: str = ""
    solution_summary: str = ""
    tech_stack: str = ""
    outcome: str = ""
    duration_seconds: float = 0.0
    error_message: str = ""
    tags: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class SemanticMemory(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    topic: str = ""
    fact: str = ""
    confidence: float = 0.0
    source: str = ""
    use_count: int = 0
    last_used: float = Field(default_factory=time.time)
    tags: list[str] = Field(default_factory=list)


class ProceduralMemory(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    procedure_name: str = ""
    steps: list[str] = Field(default_factory=list)
    success_rate: float = 1.0
    times_applied: int = 0
    times_succeeded: int = 0
    avg_duration: float = 0.0
    prerequisites: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class _DenseVectorIndex:
    """Dense vector index using HNSW (FAISS) with numpy fallback."""

    def __init__(self, dim: int = 128) -> None:
        self._embedder = DenseEmbedder(dim=dim)
        self._index: HNSWIndex | None = None
        self._dim = self._embedder.dim
        self._dirty = True

    def build(self, documents: list[tuple[str, str]]) -> None:
        self._index = HNSWIndex(self._dim)
        if not documents:
            return
        ids = [doc_id for doc_id, _ in documents]
        texts = [text for _, text in documents]
        vectors = self._embedder.embed(texts)
        self._index.add(ids, vectors)
        self._dirty = False

    def add_document(self, doc_id: str, text: str) -> None:
        """Incrementally index a single new document (no full re-embed)."""
        if self._index is None:
            self._index = HNSWIndex(self._dim)
        self._index.add([doc_id], self._embedder.embed_one(text))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self._index is None or self._index.size == 0:
            return []
        qvec = self._embedder.embed_one(query)
        return self._index.search(qvec, top_k=top_k)


class MemoryStore:
    def __init__(
        self,
        persist_dir: str = ".noema/memory",
        auto_save: bool = True,
        backup_count: int = 5,
        auto_save_interval: float = 30.0,
    ) -> None:
        self.tenant_id = get_tenant_id()
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.episodic: list[EpisodicMemory] = []
        self.semantic: list[SemanticMemory] = []
        self.procedural: list[ProceduralMemory] = []
        self._episodic_index = _DenseVectorIndex()
        self._semantic_index = _DenseVectorIndex()
        self._dirty = False
        self._auto_save = auto_save
        self._backup_count = backup_count
        self._auto_save_interval = max(0.5, float(auto_save_interval))
        self._debounce_handle: Any = None
        self._restore_from_disk()

    def _episodic_path(self) -> Path:
        return self.persist_dir / "episodic.json"

    def _semantic_path(self) -> Path:
        return self.persist_dir / "semantic.json"

    def _procedural_path(self) -> Path:
        return self.persist_dir / "procedural.json"

    def _restore_from_disk(self) -> None:
        stores: list[tuple[Path, type[BaseModel], list[BaseModel]]] = [
            (self._episodic_path(), EpisodicMemory, cast("list[BaseModel]", self.episodic)),
            (self._semantic_path(), SemanticMemory, cast("list[BaseModel]", self.semantic)),
            (self._procedural_path(), ProceduralMemory, cast("list[BaseModel]", self.procedural)),
        ]
        for path, cls, store in stores:
            data = atomic_read_json(path, default=[])
            try:
                store.extend(cls(**item) for item in data)
                log.debug("memory_loaded", path=str(path), count=len(store))
            except Exception as exc:
                log.error("memory_load_error", path=str(path), error=str(exc))

        self._rebuild_indexes()

    def _load(self) -> None:
        self._restore_from_disk()

    def _rebuild_indexes(self) -> None:
        ep_docs = [
            (
                ep.id,
                f"{ep.task_description} {ep.solution_summary} {ep.tech_stack} {' '.join(ep.tags)}",
            )
            for ep in self.episodic
        ]
        self._episodic_index.build(ep_docs)

        sem_docs = [
            (mem.id, f"{mem.topic} {mem.fact} {' '.join(mem.tags)}") for mem in self.semantic
        ]
        self._semantic_index.build(sem_docs)

    def _cancel_debounce(self) -> None:
        if self._debounce_handle is not None:
            with suppress(Exception):
                self._debounce_handle.cancel()
            self._debounce_handle = None

    def save(self) -> None:
        self._cancel_debounce()
        stores: list[tuple[Path, list[BaseModel]]] = [
            (self._episodic_path(), cast("list[BaseModel]", self.episodic)),
            (self._semantic_path(), cast("list[BaseModel]", self.semantic)),
            (self._procedural_path(), cast("list[BaseModel]", self.procedural)),
        ]
        for path, store in stores:
            data = [m.model_dump() for m in store]
            atomic_write_json(path, data, backup=True, backup_count=self._backup_count)
        self._dirty = False
        log.debug("memory_saved")

    def _mark_dirty(self) -> None:
        self._dirty = True
        if not self._auto_save:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop: persist immediately (synchronous callers).
            self.save()
            return
        self._cancel_debounce()
        self._debounce_handle = loop.call_later(self._auto_save_interval, self._debounced_save)

    def _debounced_save(self) -> None:
        self._debounce_handle = None
        if not self._dirty:
            return
        result = cast("Any", self.save())
        if inspect.isawaitable(result):
            # PostgresMemoryStore.save() is async; consume the coroutine properly.
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(cast("Coroutine[Any, Any, None]", result))
            else:
                task = asyncio.ensure_future(cast("Coroutine[Any, Any, None]", result))
                task.add_done_callback(_log_save_failure)

    def flush(self) -> None:
        if self._dirty:
            self.save()

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
    ) -> EpisodicMemory:
        ep = EpisodicMemory(
            task_description=task_description,
            solution_summary=solution_summary,
            tech_stack=tech_stack,
            outcome=outcome,
            duration_seconds=duration_seconds,
            error_message=error_message,
            tags=tags or [],
            context=context or {},
        )
        self.episodic.append(ep)
        doc_text = (
            f"{ep.task_description} {ep.solution_summary} {ep.tech_stack} {' '.join(ep.tags)}"
        )
        self._episodic_index.add_document(ep.id, doc_text)
        self._mark_dirty()
        return ep

    def search_episodes(self, query: str, limit: int = 10) -> list[EpisodicMemory]:
        results = self._episodic_index.search(query, top_k=limit)
        ep_map = {ep.id: ep for ep in self.episodic}
        return [ep_map[doc_id] for doc_id, _ in results if doc_id in ep_map]

    def get_recent_episodes(self, limit: int = 20) -> list[EpisodicMemory]:
        sorted_eps = sorted(self.episodic, key=lambda e: e.timestamp, reverse=True)
        return sorted_eps[:limit]

    def learn_fact(
        self,
        topic: str,
        fact: str,
        confidence: float = 0.8,
        source: str = "",
        tags: list[str] | None = None,
    ) -> SemanticMemory:
        mem = SemanticMemory(
            topic=topic,
            fact=fact,
            confidence=confidence,
            source=source,
            tags=tags or [],
        )
        self.semantic.append(mem)
        doc_text = f"{mem.topic} {mem.fact} {' '.join(mem.tags)}"
        self._semantic_index.add_document(mem.id, doc_text)
        self._mark_dirty()
        return mem

    def search_knowledge(self, query: str, limit: int = 10) -> list[SemanticMemory]:
        results = self._semantic_index.search(query, top_k=limit)
        mem_map = {mem.id: mem for mem in self.semantic}
        matched = []
        for doc_id, _score in results:
            if doc_id in mem_map:
                mem = mem_map[doc_id]
                mem.use_count += 1
                mem.last_used = time.time()
                matched.append(mem)
        matched.sort(key=lambda m: m.confidence * m.use_count, reverse=True)
        self._mark_dirty()
        return matched[:limit]

    def get_facts_by_topic(self, topic: str) -> list[SemanticMemory]:
        return [m for m in self.semantic if m.topic.lower() == topic.lower()]

    def store_procedure(
        self,
        procedure_name: str,
        steps: list[str],
        prerequisites: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> ProceduralMemory:
        proc = ProceduralMemory(
            procedure_name=procedure_name,
            steps=steps,
            prerequisites=prerequisites or [],
            tags=tags or [],
        )
        self.procedural.append(proc)
        self._mark_dirty()
        return proc

    def find_procedure(self, name: str) -> ProceduralMemory | None:
        for p in self.procedural:
            if p.procedure_name.lower() == name.lower():
                return p
        return None

    def record_procedure_outcome(
        self, procedure_name: str, succeeded: bool, duration: float = 0.0
    ) -> None:
        proc = self.find_procedure(procedure_name)
        if proc:
            proc.times_applied += 1
            if succeeded:
                proc.times_succeeded += 1
            proc.success_rate = proc.times_succeeded / max(proc.times_applied, 1)
            if proc.times_applied == 1:
                proc.avg_duration = duration
            else:
                proc.avg_duration = (
                    proc.avg_duration * (proc.times_applied - 1) + duration
                ) / proc.times_applied
            self._mark_dirty()

    def search_procedures(self, query: str) -> list[ProceduralMemory]:
        query_lower = query.lower()
        return [
            p
            for p in self.procedural
            if query_lower in p.procedure_name.lower()
            or any(query_lower in tag.lower() for tag in p.tags)
        ]

    def stats(self) -> dict[str, Any]:
        successes = sum(1 for e in self.episodic if e.outcome == "success")
        failures = sum(1 for e in self.episodic if e.outcome == "failure")
        avg_confidence = sum(m.confidence for m in self.semantic) / max(len(self.semantic), 1)
        avg_success_rate = sum(p.success_rate for p in self.procedural) / max(
            len(self.procedural), 1
        )
        return {
            "episodic_count": len(self.episodic),
            "semantic_count": len(self.semantic),
            "procedural_count": len(self.procedural),
            "successes": successes,
            "failures": failures,
            "success_rate": successes / max(successes + failures, 1),
            "avg_knowledge_confidence": avg_confidence,
            "avg_procedure_success_rate": avg_success_rate,
        }
