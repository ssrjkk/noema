"""Coverage for noema.db.engine — Database lifecycle without a live PostgreSQL."""

import pytest

import noema.db.engine as dbmod
from noema.config.settings import get_settings


def _isolated_db(monkeypatch) -> dbmod.Database:
    settings_copy = get_settings().model_copy(deep=True)
    settings_copy.db.url = "postgresql+asyncpg://noema:noema@127.0.0.1:9/nodb"
    monkeypatch.setattr(dbmod, "get_settings", lambda: settings_copy)
    return dbmod.Database()


@pytest.mark.asyncio
async def test_session_before_init_raises(monkeypatch):
    db = _isolated_db(monkeypatch)
    with pytest.raises(RuntimeError, match="not initialized"):
        async with db.session():
            pass


@pytest.mark.asyncio
async def test_create_tables_before_init_raises(monkeypatch):
    db = _isolated_db(monkeypatch)
    with pytest.raises(RuntimeError, match="not initialized"):
        await db.create_tables()


@pytest.mark.asyncio
async def test_drop_tables_before_init_raises(monkeypatch):
    db = _isolated_db(monkeypatch)
    with pytest.raises(RuntimeError, match="not initialized"):
        await db.drop_tables()


@pytest.mark.asyncio
async def test_health_check_before_init_returns_false(monkeypatch):
    db = _isolated_db(monkeypatch)
    assert await db.health_check() is False


@pytest.mark.asyncio
async def test_close_without_init_is_noop(monkeypatch):
    db = _isolated_db(monkeypatch)
    await db.close()


@pytest.mark.asyncio
async def test_init_creates_engine_and_session(monkeypatch):
    db = _isolated_db(monkeypatch)
    await db.init()
    assert db._engine is not None
    assert db._session_factory is not None
    async with db.session() as session:
        assert session is not None
    await db.close()


@pytest.mark.asyncio
async def test_health_check_after_init_returns_false(monkeypatch):
    db = _isolated_db(monkeypatch)
    await db.init()
    assert await db.health_check() is False
    await db.close()


@pytest.mark.asyncio
async def test_session_rollback_on_error(monkeypatch):
    db = _isolated_db(monkeypatch)
    await db.init()
    with pytest.raises(RuntimeError, match="boom"):
        async with db.session():
            raise RuntimeError("boom")
    await db.close()


@pytest.mark.asyncio
async def test_get_db_singleton():
    first = dbmod.get_db()
    second = dbmod.get_db()
    assert first is second
    assert isinstance(first, dbmod.Database)
    await dbmod.close_db()
