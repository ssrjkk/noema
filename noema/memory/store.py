"""Three-tier memory system with dense vector search (HNSW) and atomic persistence.

``HybridCognitiveMemory`` is layered *over* the tiers below rather than running as a
second store: the records it indexes are derived from ``episodic``/``semantic``/
``procedural``/``failures`` on demand, so the JSON files (and the Postgres subclass)
stay the only source of truth. Sparse BM25 + bi-temporal validity + Ebbinghaus
retention are what the dense-only HNSW indexes cannot express; everything else is
reused.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

from noema.context import get_tenant_id
from noema.embeddings import HNSWIndex, get_embedder
from noema.logging import get_logger
from noema.memory.hybrid_retriever import (
    FailurePattern,
    HybridCognitiveMemory,
    MemoryRecord,
    RetrievalHit,
    epoch_to_utc,
)
from noema.utils.atomic_io import atomic_read_json, atomic_write_json

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

log = get_logger(__name__)


def _log_save_failure(task: asyncio.Task) -> None:
    """Surface failures of the fire-and-forget debounced save instead of losing them."""
    with suppress(asyncio.CancelledError):
        exc = task.exception()
        if exc is not None:
            log.error("memory_debounced_save_failed", error=str(exc))


def _parse_iso_datetime(value: object) -> datetime:
    """Read an ISO timestamp as an aware datetime, falling back to "now".

    The bi-temporal comparisons in ``MemoryRecord`` mix aware datetimes; a naive
    value from an older or hand-edited JSON file must not raise mid-retrieval.
    """
    if isinstance(value, str):
        with suppress(ValueError):
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return datetime.now(UTC)


def _failure_to_dict(pattern: FailurePattern) -> dict[str, Any]:
    data: dict[str, Any] = asdict(pattern)
    data["created_at"] = pattern.created_at.isoformat()
    data["unsat_core"] = list(pattern.unsat_core)
    return data


def _failure_from_dict(data: dict[str, Any]) -> FailurePattern:
    return FailurePattern(
        pattern_id=str(data.get("pattern_id", "")),
        error_signature=str(data.get("error_signature", "")),
        unsat_core=tuple(str(t) for t in data.get("unsat_core") or ()),
        verified_patch_diff=str(data.get("verified_patch_diff", "")),
        z3_contract=str(data.get("z3_contract", "")),
        created_at=_parse_iso_datetime(data.get("created_at")),
    )


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
    # Bi-temporal bookkeeping: a corrected fact does not delete its predecessor,
    # it closes its validity interval. Defaults keep old JSON files loadable.
    valid_from: float = Field(default_factory=time.time)
    superseded_at: float | None = None
    supersedes: str = ""
    decay_lambda: float = 0.05


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
        self._embedder = get_embedder(dim)
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
        self.failures: list[FailurePattern] = []
        self._episodic_index = _DenseVectorIndex()
        self._semantic_index = _DenseVectorIndex()
        self._hybrid: HybridCognitiveMemory | None = None
        self._hybrid_built: tuple[Any, ...] | None = None
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

    def _failure_path(self) -> Path:
        return self.persist_dir / "failures.json"

    def _restore_from_disk(self) -> None:
        stores: list[tuple[Path, type[BaseModel], list[BaseModel]]] = [
            (self._episodic_path(), EpisodicMemory, cast("list[BaseModel]", self.episodic)),
            (self._semantic_path(), SemanticMemory, cast("list[BaseModel]", self.semantic)),
            (self._procedural_path(), ProceduralMemory, cast("list[BaseModel]", self.procedural)),
        ]
        for path, cls, store in stores:
            data = atomic_read_json(path, default=[])
            try:
                store.clear()
                store.extend(cls(**item) for item in data)
                log.debug("memory_loaded", path=str(path), count=len(store))
            except Exception as exc:
                log.error("memory_load_error", path=str(path), error=str(exc))

        self.failures.clear()
        for item in atomic_read_json(self._failure_path(), default=[]):
            try:
                pattern = _failure_from_dict(item)
            except Exception as exc:
                log.error("failure_pattern_load_error", error=str(exc))
                continue
            if not pattern.pattern_id:
                # An id-less row cannot be looked up, replaced or audited, and it
                # would silently collide with every other anonymous pattern.
                log.error("failure_pattern_load_error", error="missing pattern_id")
                continue
            self.failures.append(pattern)

        self._rebuild_indexes()

    def _load(self) -> None:
        self._restore_from_disk()

    def _hybrid_embed_fn(self) -> Callable[[str], list[float]]:
        """Dense projection shared with the HNSW indexes.

        ``embed_one`` returns a (1, dim) row because the HNSW callers batch against
        it; the retriever wants a flat vector, hence the reshape.
        """
        embedder = get_embedder(128)
        return lambda text: embedder.embed_one(text).reshape(-1).tolist()

    def _hybrid_records(self) -> list[MemoryRecord]:
        """Project the persisted tiers onto bi-temporal views for hybrid retrieval.

        Failure fixes are deliberately absent here: they are indexed through
        ``HybridCognitiveMemory.index_failure_fix``, which also fills the pattern
        registry that ``lookup_similar_failures`` reads.
        """
        records: list[MemoryRecord] = []
        for ep in self.episodic:
            records.append(
                MemoryRecord(
                    record_id=f"ep::{ep.id}",
                    content=(
                        f"{ep.task_description} {ep.solution_summary} {ep.tech_stack} "
                        f"outcome={ep.outcome} {ep.error_message} {' '.join(ep.tags)}"
                    ),
                    stratum="episodic",
                    tags=tuple(ep.tags),
                    valid_from=epoch_to_utc(ep.timestamp),
                    metadata={"kind": "episodic", "memory_id": ep.id},
                )
            )
        for mem in self.semantic:
            records.append(
                MemoryRecord(
                    record_id=f"sem::{mem.id}",
                    content=f"{mem.topic} {mem.fact} {' '.join(mem.tags)}",
                    stratum="semantic",
                    tags=tuple(mem.tags),
                    valid_from=epoch_to_utc(mem.valid_from),
                    superseded_at=(
                        None if mem.superseded_at is None else epoch_to_utc(mem.superseded_at)
                    ),
                    decay_lambda=mem.decay_lambda,
                    metadata={
                        "kind": "semantic",
                        "memory_id": mem.id,
                        "confidence": mem.confidence,
                    },
                )
            )
        for proc in self.procedural:
            records.append(
                MemoryRecord(
                    record_id=f"proc::{proc.id}",
                    content=(
                        f"{proc.procedure_name} {' '.join(proc.steps)} "
                        f"{' '.join(proc.prerequisites)} {' '.join(proc.tags)}"
                    ),
                    stratum="procedural",
                    tags=tuple(proc.tags),
                    decay_lambda=0.02,
                    metadata={
                        "kind": "procedural",
                        "memory_id": proc.id,
                        "success_rate": proc.success_rate,
                    },
                )
            )
        return records

    def hybrid_memory(self) -> HybridCognitiveMemory:
        """Sparse + dense + RRF view of this store, rebuilt when the tiers changed.

        The view holds no data of its own: everything in it is derived from the
        tiers above, so persisting it would create a second source of truth.
        Freshness comes from fingerprinting the projection itself, so no caller has
        to remember a flag: a row added, edited or deleted — through these methods,
        by reassigning a tier, or by the Postgres loader — moves the fingerprint and
        forces a rebuild. Without that, a memory the store no longer holds would go
        on answering queries from a cached view.
        """
        records = self._hybrid_records()
        fingerprint = (
            *(
                (
                    record.record_id,
                    record.content,
                    record.stratum,
                    record.tags,
                    record.valid_from,
                    record.superseded_at,
                    record.decay_lambda,
                    tuple(record.metadata.items()),
                )
                for record in records
            ),
            *(
                (
                    fix.pattern_id,
                    fix.error_signature,
                    fix.unsat_core,
                    fix.verified_patch_diff,
                    fix.z3_contract,
                    fix.created_at,
                )
                for fix in self.failures
            ),
        )
        if self._hybrid is None or fingerprint != self._hybrid_built:
            view = HybridCognitiveMemory(embed_fn=self._hybrid_embed_fn())
            for record in records:
                view.add(record, overwrite=True)
            for pattern in self.failures:
                view.index_failure_fix(pattern)
            self._hybrid = view
            self._hybrid_built = fingerprint
        return self._hybrid

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        at_time: datetime | None = None,
        include_superseded: bool = False,
    ) -> list[RetrievalHit]:
        """Retrieve across all tiers with BM25 + dense fusion and bi-temporal filtering."""
        return self.hybrid_memory().retrieve(
            query, top_k=top_k, at_time=at_time, include_superseded=include_superseded
        )

    def supersede_fact(
        self,
        old_id: str,
        topic: str,
        fact: str,
        confidence: float = 0.8,
        source: str = "",
        tags: list[str] | None = None,
        *,
        strict: bool = True,
    ) -> SemanticMemory:
        """Close ``old_id``'s validity interval and record its replacement.

        The old row survives: an audit trail and "what did we believe on date X"
        both need it. Both records share one timestamp so no instant ever sees two
        active facts in the same lineage.
        """
        now = time.time()
        old = next((m for m in self.semantic if m.id == old_id), None)
        if old is None and strict:
            raise KeyError(f"Cannot supersede unknown semantic id: {old_id!r}")
        if old is not None:
            old.superseded_at = now
        new = self.learn_fact(
            topic=topic, fact=fact, confidence=confidence, source=source, tags=tags
        )
        new.valid_from = now
        new.supersedes = old_id if old is not None else ""
        return new

    def index_failure_fix(self, pattern: FailurePattern) -> None:
        """Persist a verified repair, replacing any earlier record of the same pattern."""
        self.failures = [f for f in self.failures if f.pattern_id != pattern.pattern_id]
        self.failures.append(pattern)
        self._mark_dirty()

    def lookup_similar_failures(
        self,
        error_signature: str,
        unsat_core: tuple[str, ...] = (),
        top_k: int = 3,
    ) -> list[tuple[FailurePattern, float]]:
        """Past verified patches ranked against an incoming failure signature."""
        return self.hybrid_memory().lookup_similar_failures(
            error_signature, unsat_core, top_k=top_k
        )

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
        atomic_write_json(
            self._failure_path(),
            [_failure_to_dict(f) for f in self.failures],
            backup=True,
            backup_count=self._backup_count,
        )
        self._dirty = False
        log.debug("memory_saved")

    def _mark_dirty(self) -> None:
        self._dirty = True
        if not self._auto_save:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop: persist immediately (synchronous callers). A
            # subclass's save() may be async (e.g. PostgresMemoryStore); drive
            # the coroutine to completion instead of leaking an un-awaited one.
            result = cast("Any", self.save())
            if inspect.isawaitable(result):
                asyncio.run(cast("Coroutine[Any, Any, None]", result))
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
            "failure_fix_count": len(self.failures),
            "active_fact_count": sum(1 for m in self.semantic if m.superseded_at is None),
            "successes": successes,
            "failures": failures,
            "success_rate": successes / max(successes + failures, 1),
            "avg_knowledge_confidence": avg_confidence,
            "avg_procedure_success_rate": avg_success_rate,
        }
