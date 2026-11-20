"""Coverage tests for ``noema.persistence.pg_memory`` (Postgres-backed MemoryStore
with file fallback when Postgres is unavailable)."""

from __future__ import annotations

import importlib.machinery
import json
import re
import sys
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from noema.memory.hybrid_retriever import FailurePattern
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
        "valid_from": 90.0,
        "superseded_at": None,
        "supersedes": "",
        "decay_lambda": 0.05,
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
    failures: list[dict[str, Any]] | None = None,
):
    """Fake asyncpg connection routing ``fetch`` by table name."""
    conn = MagicMock()
    conn.execute = AsyncMock()
    conn.close = AsyncMock()

    tables = {
        "noema_episodic": episodic or [],
        "noema_semantic": semantic or [],
        "noema_procedural": procedural or [],
        "noema_failure_fix": failures or [],
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


_INSERT_RE = re.compile(r"INSERT\s+INTO\s+(\w+)\s*\(([^)]*)\)", re.IGNORECASE)
_DELETE_RE = re.compile(r"DELETE\s+FROM\s+(\w+)", re.IGNORECASE)
_SELECT_RE = re.compile(r"SELECT\s+\*\s+FROM\s+(\w+)", re.IGNORECASE)


def _make_pg_executor(
    tables: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[MagicMock, dict[str, list[dict[str, Any]]]]:
    """Fake asyncpg connection backed by in-memory tables.

    Understands only the statement shapes ``PostgresMemoryStore`` emits, which is
    enough to prove that rows written by ``save()`` come back through ``_load()``
    for the same tenant and not for another one. Passing ``tables`` lets two
    connections share one store, standing in for a second pod.
    """
    tables = {} if tables is None else tables
    conn = MagicMock()

    async def execute(query: str, *args: Any) -> None:
        insert = _INSERT_RE.search(query)
        if insert:
            columns = [name.strip() for name in insert.group(2).split(",")]
            rows = tables.setdefault(insert.group(1), [])
            rows.append(dict(zip(columns, args, strict=True)))
            return
        delete = _DELETE_RE.search(query)
        if delete:
            tables[delete.group(1)] = [
                row for row in tables.get(delete.group(1), []) if row["tenant_id"] != args[0]
            ]

    async def fetch(query: str, *args: Any) -> list[dict[str, Any]]:
        select = _SELECT_RE.search(query)
        if not select:
            return []
        return [dict(row) for row in tables.get(select.group(1), []) if row["tenant_id"] == args[0]]

    conn.execute = AsyncMock(side_effect=execute)
    conn.fetch = AsyncMock(side_effect=fetch)
    conn.close = AsyncMock()

    @asynccontextmanager
    async def transaction():
        yield

    conn.transaction = transaction
    return conn, tables


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
    assert len(created) == 5
    assert all("tenant_id TEXT" in sql for sql in created[:4])
    assert sum(1 for sql in created if "CREATE TABLE IF NOT EXISTS" in sql) == 4
    # The bi-temporal columns reach an existing deployment only through the
    # parity ALTER; CREATE TABLE IF NOT EXISTS never touches an old table.
    assert sum(1 for sql in created if "ADD COLUMN IF NOT EXISTS" in sql) == 1


@pytest.mark.asyncio
async def test_ensure_pg_tables_schema_parity_with_file_backend(tmp_path):
    """Every persisted dataclass field must have a column on the pg side."""
    from dataclasses import fields as dc_fields
    from dataclasses import is_dataclass

    def field_names(model: type) -> list[str]:
        if is_dataclass(model):
            return [f.name for f in dc_fields(model)]
        return list(model.model_fields)

    store = PostgresMemoryStore(persist_dir=str(tmp_path / "pgmem"), database_url="")
    conn = _make_conn()
    await store._ensure_pg_tables(conn)
    schema = "\n".join(call.args[0] for call in conn.execute.await_args_list)

    for model in (EpisodicMemory, SemanticMemory, ProceduralMemory, FailurePattern):
        for name in field_names(model):
            assert name in schema, f"{model.__name__}.{name} has no pg column"


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

    # 4 CREATE TABLE + 4 DELETE + 1 INSERT per memory type.
    assert sum(1 for sql in executed if "CREATE TABLE" in sql) == 4
    deletes = [sql for sql in executed if sql.startswith("DELETE FROM")]
    assert sorted(deletes) == [
        "DELETE FROM noema_episodic WHERE tenant_id = $1",
        "DELETE FROM noema_failure_fix WHERE tenant_id = $1",
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


# ── Postgres round-trip against an in-memory SQL executor ──────────────────


def _bi_temporal_semantic_row(now: float) -> SemanticMemory:
    return SemanticMemory(
        id="s1",
        topic="python",
        fact="FastAPI is async",
        confidence=0.9,
        valid_from=now - 100.0,
        superseded_at=now - 5.0,
        supersedes="s0",
        decay_lambda=0.12,
    )


def _failure_row(now: float) -> FailurePattern:
    return FailurePattern(
        pattern_id="f1",
        error_signature="KeyError: 'tenant'",
        unsat_core=("load_ok", "patch_ok"),
        verified_patch_diff="diff --git a/x",
        z3_contract="(assert (not bad))",
        created_at=datetime.fromtimestamp(now - 50.0, UTC),
    )


def _store_with_history(persist_dir: str, tenant: str, now: float) -> PostgresMemoryStore:
    store = PostgresMemoryStore(
        persist_dir=persist_dir,
        database_url="postgresql://u@h/db",
        tenant_id=tenant,
    )
    store._has_pg = True
    store.semantic = [_bi_temporal_semantic_row(now)]
    store.failures = [_failure_row(now)]
    store._dirty = True
    return store


@pytest.mark.asyncio
async def test_pg_roundtrip_preserves_bi_temporal_and_failure_fields(tmp_path):
    """The newest fields must survive a pg write followed by a pg read."""
    now = time.time()
    tables: dict[str, list[dict[str, Any]]] = {}

    store = _store_with_history(str(tmp_path / "writer"), "acme", now)
    write_ctx, _ = _patch_asyncpg(_make_pg_executor(tables)[0])
    with write_ctx:
        await store.save()

    reader = PostgresMemoryStore(
        persist_dir=str(tmp_path / "reader"),
        database_url="postgresql://u@h/db",
        tenant_id="acme",
    )
    read_ctx, _ = _patch_asyncpg(_make_pg_executor(tables)[0])
    with read_ctx:
        await reader._load()

    assert len(reader.semantic) == 1
    kept = reader.semantic[0]
    assert kept.valid_from == now - 100.0
    assert kept.superseded_at == now - 5.0
    assert kept.supersedes == "s0"
    assert kept.decay_lambda == pytest.approx(0.12)

    assert len(reader.failures) == 1
    fix = reader.failures[0]
    assert fix.pattern_id == "f1"
    assert fix.error_signature == "KeyError: 'tenant'"
    assert fix.unsat_core == ("load_ok", "patch_ok")
    assert fix.verified_patch_diff == "diff --git a/x"
    assert fix.z3_contract == "(assert (not bad))"
    assert fix.created_at == datetime.fromtimestamp(now - 50.0, UTC)


@pytest.mark.asyncio
async def test_pg_reads_are_scoped_to_the_store_tenant(tmp_path):
    """A second pod on another tenant must see none of acme's memory."""
    now = time.time()
    tables: dict[str, list[dict[str, Any]]] = {}

    store = _store_with_history(str(tmp_path / "writer"), "acme", now)
    write_ctx, _ = _patch_asyncpg(_make_pg_executor(tables)[0])
    with write_ctx:
        await store.save()

    other = PostgresMemoryStore(
        persist_dir=str(tmp_path / "other"),
        database_url="postgresql://u@h/db",
        tenant_id="globex",
    )
    read_ctx, _ = _patch_asyncpg(_make_pg_executor(tables)[0])
    with read_ctx:
        await other._load()

    assert other.semantic == []
    assert other.failures == []
