"""Coverage tests for ``noema.persistence.pg_memory`` (Postgres-backed MemoryStore
with file fallback when Postgres is unavailable)."""

import pytest

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
