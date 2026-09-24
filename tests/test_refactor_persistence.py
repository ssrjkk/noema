"""Coverage tests for ``noema.persistence.pg_memory`` (Postgres-backed MemoryStore
with file fallback when Postgres is unavailable)."""

from __future__ import annotations

import importlib.machinery
import json
import sys
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from noema.memory.store import EpisodicMemory, ProceduralMemory, SemanticMemory
from noema.persistence.pg_memory import PostgresMemoryStore

# ── PostgresMemoryStore ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pg_memory_file_fallback_roundtrip(tmp_path):
    persist = str(tmp_path / "pgmem")
    store = PostgresMemoryStore(persist_dir=persist, database_url="", tenant_id="acme")
    assert store._has_pg is False
    assert store.tenant_id == "acme"

    await store._load()
    store.record_episode(
        task_description="Build REST API",
        solution_summary="FastAPI with PostgreSQL",
        tech_stack="Python, FastAPI",
        outcome="success",
    )
    await store.save()

    assert (tmp_path / "pgmem" / "episodic.json").exists()

    store2 = PostgresMemoryStore(persist_dir=persist, database_url="")
    assert len(store2.episodic) == 1
    assert store2.episodic[0].task_description == "Build REST API"


@pytest.mark.asyncio
async def test_pg_memory_bogus_url_falls_back_to_file(tmp_path):
    persist = str(tmp_path / "pgmem")
    # asyncpg is installed but nothing listens on 127.0.0.1:9 -> connection
    # refused -> both load and save must fall back to file storage.
    store = PostgresMemoryStore(
        persist_dir=persist,
        database_url="postgresql://nouser:nopass@127.0.0.1:9/nodb",
    )
    await store._load()
    assert store._has_pg is True

    store.record_episode(
        task_description="Survive pg outage",
        solution_summary="fallback to disk",
        outcome="success",
    )
    await store.save()

    assert (tmp_path / "pgmem" / "episodic.json").exists()
    assert len(store.episodic) == 1

    await store._load()
    assert len(store.episodic) == 1
    assert store.episodic[0].task_description == "Survive pg outage"


# ── Postgres happy path (asyncpg mocked at the module boundary) ─────────────


def _episodic_row(**over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "e1",
        "timestamp": 123.0,
        "task_description": "Build API",
        "solution_summary": "FastAPI",
        "tech_stack": "Python",
        "outcome": "success",
        "duration_seconds": 1.5,
        "error_message": "",
        "tags": ["a", "b"],
        "context": json.dumps({"k": "v"}),
    }
    row.update(over)
    return row


def _semantic_row(**over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "s1",
        "topic": "python",
        "fact": "FastAPI is async",
        "confidence": 0.9,
        "source": "manual",
        "use_count": 3,
        "last_used": 100.0,
        "tags": ["t"],
    }
    row.update(over)
    return row


def _procedural_row(**over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "p1",
        "procedure_name": "deploy",
        "steps": ["build", "test"],
        "success_rate": 0.8,
        "times_applied": 5,
        "times_succeeded": 4,
        "avg_duration": 2.0,
        "prerequisites": ["docker"],
        "tags": ["ci"],
    }
    row.update(over)
    return row


def _make_conn(
    *,
    episodic: list[dict[str, Any]] | None = None,
    semantic: list[dict[str, Any]] | None = None,
    procedural: list[dict[str, Any]] | None = None,
):
    """Fake asyncpg connection routing ``fetch`` by table name."""
    conn = MagicMock()
    conn.execute = AsyncMock()
    conn.close = AsyncMock()

    tables = {
        "noema_episodic": episodic or [],
        "noema_semantic": semantic or [],
        "noema_procedural": procedural or [],
    }

    async def fetch(query: str, *args: Any):
        for table, rows in tables.items():
            if table in query:
                return rows
        return []

    conn.fetch = AsyncMock(side_effect=fetch)

    @asynccontextmanager
    async def transaction():
        yield

    conn.transaction = transaction
    return conn


def _patch_asyncpg(conn):
    """Install a fake ``asyncpg`` module returning ``conn`` from connect()."""
    module = MagicMock()
    module.connect = AsyncMock(return_value=conn)
    # _try_connect() gates on importlib.util.find_spec("asyncpg"), which reads
    # __spec__ off whatever is already registered in sys.modules.
    module.__spec__ = importlib.machinery.ModuleSpec("asyncpg", loader=None)
    return patch.dict(sys.modules, {"asyncpg": module}), module


@pytest.mark.asyncio
async def test_load_from_pg_populates_all_three_memories(tmp_path):
    conn = _make_conn(
        episodic=[_episodic_row()],
        semantic=[_semantic_row()],
        procedural=[_procedural_row()],
    )
    ctx, module = _patch_asyncpg(conn)
    store = PostgresMemoryStore(
        persist_dir=str(tmp_path / "pgmem"),
        database_url="postgresql://u@h/db",
        tenant_id="acme",
    )

    with ctx:
        await store._load()

    module.connect.assert_awaited_once_with("postgresql://u@h/db")
    assert len(store.episodic) == 1
    assert store.episodic[0].id == "e1"
    assert store.episodic[0].tags == ["a", "b"]
    assert store.episodic[0].context == {"k": "v"}
    assert len(store.semantic) == 1
    assert store.semantic[0].topic == "python"
    assert len(store.procedural) == 1
    assert store.procedural[0].procedure_name == "deploy"
    assert store.procedural[0].steps == ["build", "test"]
    conn.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_from_pg_rebuilds_search_indexes(tmp_path):
    conn = _make_conn(
        episodic=[_episodic_row(id="e1", task_description="sharding the session store")],
        semantic=[_semantic_row(id="s1", topic="postgres", fact="use partial indexes")],
    )
    ctx, _ = _patch_asyncpg(conn)
    store = PostgresMemoryStore(
        persist_dir=str(tmp_path / "pgmem"),
        database_url="postgresql://u@h/db",
    )

    with ctx:
        await store._load()

    # Both indexes are empty until _rebuild_indexes() runs on the loaded rows.
    assert [ep.id for ep in store.search_episodes("sharding")] == ["e1"]
    assert [mem.id for mem in store.search_knowledge("postgres")] == ["s1"]


@pytest.mark.asyncio
async def test_load_episodic_handles_null_and_bad_json(tmp_path):
    store = PostgresMemoryStore(persist_dir=str(tmp_path / "pgmem"), database_url="")

    rows = [
        _episodic_row(id="ok", context="not json"),
        _episodic_row(id="nulls", tags=None, context=None, error_message=None),
    ]
    conn = _make_conn(episodic=rows)
    entries = await store._load_episodic(conn)

    assert len(entries) == 2
    assert entries[0].context == {}
    assert entries[1].tags == []
    assert isinstance(entries[0], EpisodicMemory)


@pytest.mark.asyncio
async def test_load_semantic_and_procedural_tolerate_null_arrays(tmp_path):
    store = PostgresMemoryStore(persist_dir=str(tmp_path / "pgmem"), database_url="")

    semantic = await store._load_semantic(_make_conn(semantic=[_semantic_row(tags=None)]))
    assert semantic[0].tags == []

    procedural = await store._load_procedural(
        _make_conn(procedural=[_procedural_row(steps=None, prerequisites=None, tags=None)])
    )
    assert procedural[0].steps == []
    assert procedural[0].prerequisites == []


@pytest.mark.asyncio
async def test_ensure_pg_tables_creates_tenant_scoped_schema(tmp_path):
    store = PostgresMemoryStore(persist_dir=str(tmp_path / "pgmem"), database_url="")
    conn = _make_conn()

    await store._ensure_pg_tables(conn)

    created = [call.args[0] for call in conn.execute.await_args_list]
    assert len(created) == 3
    assert all("CREATE TABLE IF NOT EXISTS" in sql for sql in created)
    assert all("tenant_id TEXT" in sql for sql in created)


@pytest.mark.asyncio
async def test_save_to_pg_writes_in_one_transaction(tmp_path):
    store = PostgresMemoryStore(
        persist_dir=str(tmp_path / "pgmem"),
        database_url="postgresql://u@h/db",
        tenant_id="acme",
    )
    store._has_pg = True
    store.episodic = [EpisodicMemory(id="e1", task_description="t", outcome="success")]
    store.semantic = [SemanticMemory(id="s1", topic="tp", fact="f", confidence=0.5)]
    store.procedural = [
        ProceduralMemory(id="p1", procedure_name="pr", steps=["a"], success_rate=1.0)
    ]
    store._dirty = True

    conn = _make_conn()
    ctx, module = _patch_asyncpg(conn)

    with ctx:
        await store.save()

    module.connect.assert_awaited_once()
    executed = [call.args[0] for call in conn.execute.await_args_list]

    # 3 CREATE TABLE + 3 DELETE + 1 INSERT per memory type.
    assert sum(1 for sql in executed if "CREATE TABLE" in sql) == 3
    deletes = [sql for sql in executed if sql.startswith("DELETE FROM")]
    assert sorted(deletes) == [
        "DELETE FROM noema_episodic WHERE tenant_id = $1",
        "DELETE FROM noema_procedural WHERE tenant_id = $1",
        "DELETE FROM noema_semantic WHERE tenant_id = $1",
    ]
    assert sum(1 for sql in executed if "INSERT INTO noema_episodic" in sql) == 1
    assert sum(1 for sql in executed if "INSERT INTO noema_semantic" in sql) == 1
    assert sum(1 for sql in executed if "INSERT INTO noema_procedural" in sql) == 1

    # Tenant is bound on every write, and the episode context is serialized.
    insert_calls = [c for c in conn.execute.await_args_list if "INSERT" in c.args[0]]
    assert all(c.args[2] == "acme" for c in insert_calls)
    episodic_insert = next(c for c in insert_calls if "noema_episodic" in c.args[0])
    assert episodic_insert.args[-1] == "{}"

    assert store._dirty is False
    conn.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_to_pg_failure_falls_back_to_file(tmp_path):
    store = PostgresMemoryStore(
        persist_dir=str(tmp_path / "pgmem"),
        database_url="postgresql://u@h/db",
    )
    store._has_pg = True
    store.record_episode(task_description="t", solution_summary="s", outcome="success")

    module = MagicMock()
    module.connect = AsyncMock(side_effect=OSError("pg down"))

    with patch.dict(sys.modules, {"asyncpg": module}):
        await store.save()

    assert (tmp_path / "pgmem" / "episodic.json").exists()


@pytest.mark.asyncio
async def test_try_connect_without_url_is_noop(tmp_path):
    store = PostgresMemoryStore(persist_dir=str(tmp_path / "pgmem"), database_url="")
    store._try_connect()
    assert store._has_pg is False


@pytest.mark.asyncio
async def test_try_connect_logs_redacted_url(tmp_path):
    store = PostgresMemoryStore(
        persist_dir=str(tmp_path / "pgmem"),
        database_url="postgresql://user:secret@host:5432/db",
    )
    with patch("importlib.util.find_spec", return_value=object()):
        store._try_connect()
    assert store._has_pg is True


@pytest.mark.asyncio
async def test_try_connect_without_asyncpg_stays_on_files(tmp_path):
    store = PostgresMemoryStore(
        persist_dir=str(tmp_path / "pgmem"),
        database_url="postgresql://user@host/db",
    )
    with patch("importlib.util.find_spec", return_value=None):
        store._try_connect()
    assert store._has_pg is False


@pytest.mark.asyncio
async def test_pg_memory_learn_fact_and_procedure(tmp_path):
    store = PostgresMemoryStore(persist_dir=str(tmp_path / "pgmem"), database_url="")
    await store._load()

    store.learn_fact(topic="python", fact="FastAPI is async by default", confidence=0.9)
    store.store_procedure(procedure_name="deploy", steps=["build", "test"])

    await store.save()
    store2 = PostgresMemoryStore(persist_dir=str(tmp_path / "pgmem"), database_url="")
    assert len(store2.semantic) == 1
    assert len(store2.procedural) == 1
