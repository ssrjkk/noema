"""PostgreSQL-backed MemoryStore with file fallback — для масштабирования в K8s."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from typing import Any

from noema.logging import get_logger
from noema.memory.store import (
    EpisodicMemory,
    MemoryStore,
    ProceduralMemory,
    SemanticMemory,
)

log = get_logger(__name__)


class PostgresMemoryStore(MemoryStore):
    """MemoryStore with PostgreSQL backend for cross-pod persistence.

    Falls back to file-based storage when PostgreSQL is unavailable.
    """

    def __init__(
        self,
        persist_dir: str = ".noema/memory",
        auto_save: bool = True,
        backup_count: int = 5,
        tenant_id: str = "default",
        database_url: str = "",
    ) -> None:
        self._db_url = database_url
        self._pool = None
        self._has_pg = False
        self.tenant_id = tenant_id
        super().__init__(persist_dir=persist_dir, auto_save=auto_save, backup_count=backup_count)
        self.tenant_id = tenant_id

    def _try_connect(self) -> None:
        if not self._db_url:
            return
        self._has_pg = importlib.util.find_spec("asyncpg") is not None
        if self._has_pg:
            log.info(
                "pg_memory_configured",
                url=self._db_url.split("@")[-1] if "@" in self._db_url else "configured",
            )

    async def _load(self) -> None:  # type: ignore[override]
        self._try_connect()
        if self._has_pg and self._db_url:
            await self._load_from_pg()
        else:
            await asyncio.to_thread(super()._load)

    async def save(self) -> None:  # type: ignore[override]
        if self._has_pg and self._db_url:
            await self._save_to_pg()
        else:
            await asyncio.to_thread(super().save)

    async def _load_from_pg(self) -> None:
        try:
            import asyncpg

            conn = await asyncpg.connect(self._db_url)
            try:
                await self._ensure_pg_tables(conn)
                self.episodic = await self._load_episodic(conn)
                self.semantic = await self._load_semantic(conn)
                self.procedural = await self._load_procedural(conn)
                self._rebuild_indexes()
                log.info(
                    "pg_memory_loaded", episodic=len(self.episodic), semantic=len(self.semantic)
                )
            finally:
                await conn.close()
        except Exception as e:
            log.warning("pg_memory_load_failed", error=str(e))
            await asyncio.to_thread(super()._load)

    async def _save_to_pg(self) -> None:
        try:
            import asyncpg

            conn = await asyncpg.connect(self._db_url)
            try:
                await self._ensure_pg_tables(conn)
                # All three tables are replaced inside one transaction: a
                # mid-save failure must not leave the tenant's memory half
                # written (DELETE without INSERT = empty memory).
                async with conn.transaction():
                    await self._save_episodic(conn)
                    await self._save_semantic(conn)
                    await self._save_procedural(conn)
            finally:
                await conn.close()
            self._dirty = False
        except Exception as e:
            log.warning("pg_memory_save_failed", error=str(e))
            await asyncio.to_thread(super().save)

    async def _ensure_pg_tables(self, conn: Any) -> None:
        """Create tables with tenant_id for RLS-ready schema."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS noema_episodic (
                id TEXT, tenant_id TEXT, timestamp REAL,
                task_description TEXT, solution_summary TEXT,
                tech_stack TEXT, outcome TEXT, duration_seconds REAL,
                error_message TEXT, tags TEXT[], context TEXT,
                PRIMARY KEY (id, tenant_id)
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS noema_semantic (
                id TEXT, tenant_id TEXT, topic TEXT, fact TEXT,
                confidence REAL, source TEXT, use_count INT,
                last_used REAL, tags TEXT[],
                PRIMARY KEY (id, tenant_id)
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS noema_procedural (
                id TEXT, tenant_id TEXT, procedure_name TEXT,
                steps TEXT[], success_rate REAL, times_applied INT,
                times_succeeded INT, avg_duration REAL,
                prerequisites TEXT[], tags TEXT[],
                PRIMARY KEY (id, tenant_id)
            )
        """)

    async def _load_episodic(self, conn: Any) -> list[EpisodicMemory]:
        rows = await conn.fetch("SELECT * FROM noema_episodic WHERE tenant_id = $1", self.tenant_id)
        entries: list[EpisodicMemory] = []
        for r in rows:
            context: dict[str, Any] = {}
            if r["context"]:
                try:
                    context = json.loads(r["context"])
                except (TypeError, ValueError):
                    context = {}
            entries.append(
                EpisodicMemory(
                    id=r["id"],
                    timestamp=r["timestamp"],
                    task_description=r["task_description"],
                    solution_summary=r["solution_summary"],
                    tech_stack=r["tech_stack"],
                    outcome=r["outcome"],
                    duration_seconds=r["duration_seconds"],
                    error_message=r["error_message"],
                    tags=list(r["tags"]) if r["tags"] else [],
                    context=context,
                )
            )
        return entries

    async def _load_semantic(self, conn: Any) -> list[SemanticMemory]:
        rows = await conn.fetch("SELECT * FROM noema_semantic WHERE tenant_id = $1", self.tenant_id)
        return [
            SemanticMemory(
                id=r["id"],
                topic=r["topic"],
                fact=r["fact"],
                confidence=r["confidence"],
                source=r["source"],
                use_count=r["use_count"],
                last_used=r["last_used"],
                tags=list(r["tags"]) if r["tags"] else [],
            )
            for r in rows
        ]

    async def _load_procedural(self, conn: Any) -> list[ProceduralMemory]:
        rows = await conn.fetch(
            "SELECT * FROM noema_procedural WHERE tenant_id = $1", self.tenant_id
        )
        return [
            ProceduralMemory(
                id=r["id"],
                procedure_name=r["procedure_name"],
                steps=list(r["steps"]) if r["steps"] else [],
                success_rate=r["success_rate"],
                times_applied=r["times_applied"],
                times_succeeded=r["times_succeeded"],
                avg_duration=r["avg_duration"],
                prerequisites=list(r["prerequisites"]) if r["prerequisites"] else [],
                tags=list(r["tags"]) if r["tags"] else [],
            )
            for r in rows
        ]

    async def _save_episodic(self, conn: Any) -> None:
        await conn.execute("DELETE FROM noema_episodic WHERE tenant_id = $1", self.tenant_id)
        for ep in self.episodic:
            await conn.execute(
                """INSERT INTO noema_episodic
                   (id, tenant_id, timestamp, task_description, solution_summary,
                    tech_stack, outcome, duration_seconds, error_message, tags, context)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)""",
                ep.id,
                self.tenant_id,
                ep.timestamp,
                ep.task_description,
                ep.solution_summary,
                ep.tech_stack,
                ep.outcome,
                ep.duration_seconds,
                ep.error_message,
                ep.tags,
                json.dumps(ep.context),
            )

    async def _save_semantic(self, conn: Any) -> None:
        await conn.execute("DELETE FROM noema_semantic WHERE tenant_id = $1", self.tenant_id)
        for mem in self.semantic:
            await conn.execute(
                """INSERT INTO noema_semantic
                   (id, tenant_id, topic, fact, confidence, source, use_count, last_used, tags)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
                mem.id,
                self.tenant_id,
                mem.topic,
                mem.fact,
                mem.confidence,
                mem.source,
                mem.use_count,
                mem.last_used,
                mem.tags,
            )

    async def _save_procedural(self, conn: Any) -> None:
        await conn.execute("DELETE FROM noema_procedural WHERE tenant_id = $1", self.tenant_id)
        for proc in self.procedural:
            await conn.execute(
                """INSERT INTO noema_procedural
                   (id, tenant_id, procedure_name, steps, success_rate,
                    times_applied, times_succeeded, avg_duration, prerequisites, tags)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)""",
                proc.id,
                self.tenant_id,
                proc.procedure_name,
                proc.steps,
                proc.success_rate,
                proc.times_applied,
                proc.times_succeeded,
                proc.avg_duration,
                proc.prerequisites,
                proc.tags,
            )
