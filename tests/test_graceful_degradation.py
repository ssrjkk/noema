"""Comprehensive tests for noema.resilience.graceful_degradation.

Covers every branch: health checks, recovery detection, cache get/set/delete
with both healthy and degraded Redis, db execute/fetch with both healthy and
degraded PostgreSQL, fallback file writing (success and failure), and status
reporting.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from noema.resilience.graceful_degradation import GracefulDegradation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_redis(healthy: bool = True) -> AsyncMock:
    """Return a mock redis client.

    When *healthy* is True the standard methods succeed.  When False every
    awaited call raises.
    """
    redis = AsyncMock()
    redis.ping = AsyncMock(side_effect=None if healthy else Exception("ping failed"))
    redis.get = AsyncMock(side_effect=None if healthy else Exception("get failed"))
    redis.set = AsyncMock(side_effect=None if healthy else Exception("set failed"))
    redis.setex = AsyncMock(side_effect=None if healthy else Exception("setex failed"))
    redis.delete = AsyncMock(side_effect=None if healthy else Exception("delete failed"))
    return redis


def _make_pg(healthy: bool = True) -> AsyncMock:
    """Return a mock asyncpg-style pool."""
    pg = AsyncMock()
    pg.fetchval = AsyncMock(
        side_effect=None if healthy else Exception("fetchval failed"),
        return_value=1,
    )
    pg.execute = AsyncMock(
        side_effect=None if healthy else Exception("execute failed"),
        return_value="OK",
    )
    pg.fetch = AsyncMock(
        side_effect=None if healthy else Exception("fetch failed"),
        return_value=[],
    )
    return pg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fallback_dir(tmp_path: Path) -> Path:
    """Isolated fallback directory for each test."""
    d = tmp_path / "fallback"
    d.mkdir()
    return d


@pytest.fixture()
def gd_no_deps(fallback_dir: Path, monkeypatch: pytest.MonkeyPatch) -> GracefulDegradation:
    """GracefulDegradation with *no* external clients."""
    monkeypatch.chdir(fallback_dir.parent)
    gd = GracefulDegradation()
    gd._fallback_dir = fallback_dir
    return gd


@pytest.fixture()
def gd_healthy(fallback_dir: Path, monkeypatch: pytest.MonkeyPatch) -> GracefulDegradation:
    """GracefulDegradation with healthy redis + pg."""
    monkeypatch.chdir(fallback_dir.parent)
    gd = GracefulDegradation(redis_client=_make_redis(), pg_pool=_make_pg())
    gd._fallback_dir = fallback_dir
    return gd


@pytest.fixture()
def gd_degraded(fallback_dir: Path, monkeypatch: pytest.MonkeyPatch) -> GracefulDegradation:
    """GracefulDegradation whose services are present but marked unhealthy."""
    monkeypatch.chdir(fallback_dir.parent)
    gd = GracefulDegradation(
        redis_client=_make_redis(healthy=False),
        pg_pool=_make_pg(healthy=False),
    )
    gd._redis_healthy = False
    gd._pg_healthy = False
    gd._fallback_dir = fallback_dir
    return gd


# ===================================================================
# 1. __init__
# ===================================================================


class TestInit:
    def test_init_with_both_clients(self, tmp_path: Path):
        with patch("noema.resilience.graceful_degradation.Path.mkdir"):
            gd = GracefulDegradation(redis_client="r", pg_pool="p")
        assert gd.redis == "r"
        assert gd.pg == "p"
        assert gd._redis_healthy is True
        assert gd._pg_healthy is True
        assert gd._memory_cache == {}

    def test_init_with_no_clients(self, tmp_path: Path):
        with patch("noema.resilience.graceful_degradation.Path.mkdir"):
            gd = GracefulDegradation()
        assert gd.redis is None
        assert gd.pg is None
        assert gd._redis_healthy is False
        assert gd._pg_healthy is False


# ===================================================================
# 2. check_health
# ===================================================================


class TestCheckHealth:
    # -- Redis ---------------------------------------------------------

    @pytest.mark.asyncio
    async def test_redis_stays_healthy(self, gd_healthy):
        """Redis ping succeeds and it was already healthy -> no log."""
        await gd_healthy.check_health()
        assert gd_healthy._redis_healthy is True

    @pytest.mark.asyncio
    async def test_redis_recovered(self, gd_degraded):
        """Lines 36-38: Redis was unhealthy, ping succeeds -> recovery logged."""
        # Replace the broken mock with a working one
        gd_degraded.redis = _make_redis(healthy=True)
        gd_degraded._redis_healthy = False

        await gd_degraded.check_health()
        assert gd_degraded._redis_healthy is True

    @pytest.mark.asyncio
    async def test_redis_goes_down(self, gd_healthy):
        """Redis was healthy, ping fails -> marked unhealthy."""
        gd_healthy.redis.ping = AsyncMock(side_effect=Exception("boom"))
        await gd_healthy.check_health()
        assert gd_healthy._redis_healthy is False

    @pytest.mark.asyncio
    async def test_redis_stays_down(self, gd_degraded):
        """Redis already unhealthy, ping still fails -> stays unhealthy, no extra log."""
        await gd_degraded.check_health()
        assert gd_degraded._redis_healthy is False

    # -- PostgreSQL ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_pg_stays_healthy(self, gd_healthy):
        await gd_healthy.check_health()
        assert gd_healthy._pg_healthy is True

    @pytest.mark.asyncio
    async def test_pg_recovered(self, gd_degraded):
        """Lines 47-49: PG was unhealthy, fetchval succeeds -> recovery logged."""
        gd_degraded.pg = _make_pg(healthy=True)
        gd_degraded._pg_healthy = False

        await gd_degraded.check_health()
        assert gd_degraded._pg_healthy is True

    @pytest.mark.asyncio
    async def test_pg_goes_down(self, gd_healthy):
        gd_healthy.pg.fetchval = AsyncMock(side_effect=Exception("boom"))
        await gd_healthy.check_health()
        assert gd_healthy._pg_healthy is False

    @pytest.mark.asyncio
    async def test_pg_stays_down(self, gd_degraded):
        await gd_degraded.check_health()
        assert gd_degraded._pg_healthy is False

    # -- No clients at all ---------------------------------------------

    @pytest.mark.asyncio
    async def test_check_health_no_clients(self, gd_no_deps):
        """Neither redis nor pg configured -> nothing happens."""
        await gd_no_deps.check_health()
        assert gd_no_deps._redis_healthy is False
        assert gd_no_deps._pg_healthy is False


# ===================================================================
# 3. cache_get
# ===================================================================


class TestCacheGet:
    @pytest.mark.asyncio
    async def test_cache_get_redis_hit(self, gd_healthy):
        """Lines 57-58: Redis healthy, get returns value."""
        gd_healthy.redis.get.return_value = "redis_val"
        result = await gd_healthy.cache_get("k")
        assert result == "redis_val"
        gd_healthy.redis.get.assert_awaited_once_with("k")

    @pytest.mark.asyncio
    async def test_cache_get_redis_fail_fallback_memory(self, gd_healthy):
        """Lines 59-61: Redis get raises -> mark unhealthy, fall back to memory."""
        gd_healthy.redis.get = AsyncMock(side_effect=Exception("timeout"))
        gd_healthy._memory_cache["k"] = "mem_val"

        result = await gd_healthy.cache_get("k")
        assert result == "mem_val"
        assert gd_healthy._redis_healthy is False

    @pytest.mark.asyncio
    async def test_cache_get_redis_healthy_miss(self, gd_healthy):
        """Redis returns None (miss) -> no memory fallback needed."""
        gd_healthy.redis.get.return_value = None
        result = await gd_healthy.cache_get("missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_get_degraded_memory_hit(self, gd_degraded):
        """Redis unhealthy -> go straight to memory cache."""
        gd_degraded._memory_cache["k"] = "mem"
        result = await gd_degraded.cache_get("k")
        assert result == "mem"

    @pytest.mark.asyncio
    async def test_cache_get_degraded_memory_miss(self, gd_degraded):
        result = await gd_degraded.cache_get("nope")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_get_no_redis_client(self, gd_no_deps):
        gd_no_deps._memory_cache["x"] = 42
        result = await gd_no_deps.cache_get("x")
        assert result == 42


# ===================================================================
# 4. cache_set
# ===================================================================


class TestCacheSet:
    @pytest.mark.asyncio
    async def test_cache_set_memory_always_updated(self, gd_degraded):
        """Memory cache is always written regardless of redis state."""
        await gd_degraded.cache_set("k", "v")
        assert gd_degraded._memory_cache["k"] == "v"

    @pytest.mark.asyncio
    async def test_cache_set_redis_no_ttl(self, gd_healthy):
        """Lines 70-71: Redis healthy, no TTL -> redis.set called."""
        await gd_healthy.cache_set("k", "v")
        gd_healthy.redis.set.assert_awaited_once_with("k", "v")
        assert gd_healthy._memory_cache["k"] == "v"

    @pytest.mark.asyncio
    async def test_cache_set_redis_with_ttl(self, gd_healthy):
        """Lines 68-69: Redis healthy, TTL provided -> redis.setex called."""
        await gd_healthy.cache_set("k", "v", ttl=60)
        gd_healthy.redis.setex.assert_awaited_once_with("k", 60, "v")

    @pytest.mark.asyncio
    async def test_cache_set_redis_fails(self, gd_healthy):
        """Lines 72-74: Redis set raises -> mark unhealthy."""
        gd_healthy.redis.set = AsyncMock(side_effect=Exception("fail"))
        await gd_healthy.cache_set("k", "v")
        assert gd_healthy._redis_healthy is False
        # Memory cache still updated
        assert gd_healthy._memory_cache["k"] == "v"

    @pytest.mark.asyncio
    async def test_cache_set_redis_setex_fails(self, gd_healthy):
        """TTL path: redis.setex raises -> mark unhealthy."""
        gd_healthy.redis.setex = AsyncMock(side_effect=Exception("fail"))
        await gd_healthy.cache_set("k", "v", ttl=30)
        assert gd_healthy._redis_healthy is False

    @pytest.mark.asyncio
    async def test_cache_set_no_redis_client(self, gd_no_deps):
        await gd_no_deps.cache_set("k", "v")
        assert gd_no_deps._memory_cache["k"] == "v"

    @pytest.mark.asyncio
    async def test_cache_set_redis_unhealthy_skips_remote(self, gd_degraded):
        """When redis is already unhealthy, only memory is written."""
        gd_degraded.redis = _make_redis(healthy=True)  # client exists
        gd_degraded._redis_healthy = False
        await gd_degraded.cache_set("k", "v")
        gd_degraded.redis.set.assert_not_awaited()
        assert gd_degraded._memory_cache["k"] == "v"


# ===================================================================
# 5. cache_delete
# ===================================================================


class TestCacheDelete:
    @pytest.mark.asyncio
    async def test_cache_delete_memory_always(self, gd_degraded):
        gd_degraded._memory_cache["k"] = "v"
        await gd_degraded.cache_delete("k")
        assert "k" not in gd_degraded._memory_cache

    @pytest.mark.asyncio
    async def test_cache_delete_redis_success(self, gd_healthy):
        """Lines 79-80: Redis healthy, delete succeeds."""
        await gd_healthy.cache_delete("k")
        gd_healthy.redis.delete.assert_awaited_once_with("k")

    @pytest.mark.asyncio
    async def test_cache_delete_redis_fails(self, gd_healthy):
        """Lines 81-82: Redis delete raises -> mark unhealthy."""
        gd_healthy.redis.delete = AsyncMock(side_effect=Exception("fail"))
        await gd_healthy.cache_delete("k")
        assert gd_healthy._redis_healthy is False

    @pytest.mark.asyncio
    async def test_cache_delete_no_redis_client(self, gd_no_deps):
        gd_no_deps._memory_cache["k"] = "v"
        await gd_no_deps.cache_delete("k")
        assert "k" not in gd_no_deps._memory_cache

    @pytest.mark.asyncio
    async def test_cache_delete_missing_key_no_error(self, gd_degraded):
        """Deleting a key not in memory should not raise."""
        await gd_degraded.cache_delete("nonexistent")


# ===================================================================
# 6. db_execute
# ===================================================================


class TestDbExecute:
    @pytest.mark.asyncio
    async def test_db_execute_pg_healthy(self, gd_healthy):
        """Lines 88-89: PG healthy, execute succeeds."""
        result = await gd_healthy.db_execute("INSERT INTO t VALUES ($1)", 1)
        assert result == "OK"
        gd_healthy.pg.execute.assert_awaited_once_with("INSERT INTO t VALUES ($1)", 1)

    @pytest.mark.asyncio
    async def test_db_execute_pg_fails_writes_fallback(self, gd_healthy, fallback_dir: Path):
        """Lines 90-94: PG execute raises -> mark unhealthy, write fallback."""
        gd_healthy.pg.execute = AsyncMock(side_effect=Exception("conn lost"))
        result = await gd_healthy.db_execute("INSERT INTO t VALUES ($1)", 42)
        assert result is None
        assert gd_healthy._pg_healthy is False
        # Fallback file should exist
        files = list(fallback_dir.glob("query_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["query"] == "INSERT INTO t VALUES ($1)"
        assert data["arg_count"] == 1
        assert data["arg_types"] == ["int"]

    @pytest.mark.asyncio
    async def test_db_execute_pg_unhealthy_writes_fallback(self, gd_degraded, fallback_dir: Path):
        """Lines 85-87: PG already unhealthy -> write fallback, return None."""
        result = await gd_degraded.db_execute("SELECT 1")
        assert result is None
        files = list(fallback_dir.glob("query_*.json"))
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_db_execute_no_pg_client(self, gd_no_deps, fallback_dir: Path):
        result = await gd_no_deps.db_execute("SELECT 1")
        assert result is None

    @pytest.mark.asyncio
    async def test_db_execute_multiple_args_types(self, gd_degraded, fallback_dir: Path):
        """Fallback records arg count and types correctly."""
        await gd_degraded.db_execute("INSERT INTO t VALUES ($1,$2,$3)", "a", 1, 3.14)
        files = list(fallback_dir.glob("query_*.json"))
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["arg_count"] == 3
        assert data["arg_types"] == ["str", "int", "float"]


# ===================================================================
# 7. db_fetch
# ===================================================================


class TestDbFetch:
    @pytest.mark.asyncio
    async def test_db_fetch_pg_healthy(self, gd_healthy):
        """Lines 98-100: PG healthy, fetch returns rows."""
        row_mock = MagicMock()
        row_mock.__iter__ = lambda self: iter([("id", 1), ("name", "a")])
        row_mock.keys = lambda: ["id", "name"]
        # dict(row) must work — use a real dict-like object
        row = {"id": 1, "name": "a"}
        gd_healthy.pg.fetch.return_value = [row]

        result = await gd_healthy.db_fetch("SELECT * FROM t")
        assert result == [{"id": 1, "name": "a"}]

    @pytest.mark.asyncio
    async def test_db_fetch_pg_healthy_empty(self, gd_healthy):
        gd_healthy.pg.fetch.return_value = []
        result = await gd_healthy.db_fetch("SELECT * FROM t")
        assert result == []

    @pytest.mark.asyncio
    async def test_db_fetch_pg_fails(self, gd_healthy):
        """Lines 101-103: PG fetch raises -> mark unhealthy, return []."""
        gd_healthy.pg.fetch = AsyncMock(side_effect=Exception("fail"))
        result = await gd_healthy.db_fetch("SELECT * FROM t")
        assert result == []
        assert gd_healthy._pg_healthy is False

    @pytest.mark.asyncio
    async def test_db_fetch_pg_unhealthy(self, gd_degraded):
        result = await gd_degraded.db_fetch("SELECT 1")
        assert result == []

    @pytest.mark.asyncio
    async def test_db_fetch_no_pg_client(self, gd_no_deps):
        result = await gd_no_deps.db_fetch("SELECT 1")
        assert result == []


# ===================================================================
# 8. _write_fallback
# ===================================================================


class TestWriteFallback:
    @pytest.mark.asyncio
    async def test_write_fallback_success(self, gd_no_deps, fallback_dir: Path):
        await gd_no_deps._write_fallback("SELECT $1", (42,))
        files = list(fallback_dir.glob("query_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["query"] == "SELECT $1"
        assert data["arg_count"] == 1
        assert data["arg_types"] == ["int"]
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_write_fallback_no_args(self, gd_no_deps, fallback_dir: Path):
        await gd_no_deps._write_fallback("SELECT 1", ())
        files = list(fallback_dir.glob("query_*.json"))
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["arg_count"] == 0
        assert data["arg_types"] == []

    @pytest.mark.asyncio
    async def test_write_fallback_file_error(self, gd_no_deps, fallback_dir: Path):
        """Lines 120-121: Writing the fallback file itself fails -> log error."""
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            # Should not raise
            await gd_no_deps._write_fallback("SELECT 1", ())

    @pytest.mark.asyncio
    async def test_write_fallback_unicode_query(self, gd_no_deps, fallback_dir: Path):
        await gd_no_deps._write_fallback("SELECT '日本語'", ())
        files = list(fallback_dir.glob("query_*.json"))
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert "日本語" in data["query"]


# ===================================================================
# 9. get_status
# ===================================================================


class TestGetStatus:
    def test_status_all_healthy(self, gd_healthy, fallback_dir):
        status = gd_healthy.get_status()
        assert status["redis"] == "healthy"
        assert status["postgresql"] == "healthy"
        assert status["memory_cache_size"] == 0
        assert status["fallback_dir"] == str(fallback_dir)

    def test_status_all_degraded(self, gd_degraded, fallback_dir):
        status = gd_degraded.get_status()
        assert status["redis"] == "degraded"
        assert status["postgresql"] == "degraded"

    def test_status_memory_cache_size(self, gd_healthy, fallback_dir):
        gd_healthy._memory_cache = {"a": 1, "b": 2}
        status = gd_healthy.get_status()
        assert status["memory_cache_size"] == 2


# ===================================================================
# 10. Multi-component / integration-style scenarios
# ===================================================================


class TestMultiComponentScenarios:
    """End-to-end flows exercising several methods in sequence."""

    @pytest.mark.asyncio
    async def test_redis_recovers_after_being_down(self, fallback_dir: Path):
        """Simulate: start healthy -> redis goes down -> recovers."""
        gd = GracefulDegradation.__new__(GracefulDegradation)
        gd.redis = _make_redis(healthy=True)
        gd.pg = None
        gd._memory_cache = {}
        gd._redis_healthy = True
        gd._pg_healthy = False
        gd._fallback_dir = fallback_dir

        # 1) Redis goes down during cache_get
        gd.redis.get = AsyncMock(side_effect=Exception("timeout"))
        await gd.cache_get("k")
        assert gd._redis_healthy is False

        # 2) check_health with working redis -> recovery
        gd.redis = _make_redis(healthy=True)
        await gd.check_health()
        assert gd._redis_healthy is True

    @pytest.mark.asyncio
    async def test_pg_recovers_after_execute_failure(self, fallback_dir: Path):
        """PG fails on execute, then recovers via check_health."""
        gd = GracefulDegradation.__new__(GracefulDegradation)
        gd.redis = None
        gd.pg = _make_pg(healthy=True)
        gd._memory_cache = {}
        gd._redis_healthy = False
        gd._pg_healthy = True
        gd._fallback_dir = fallback_dir

        # Execute fails
        gd.pg.execute = AsyncMock(side_effect=Exception("conn lost"))
        result = await gd.db_execute("INSERT INTO t VALUES ($1)", 1)
        assert result is None
        assert gd._pg_healthy is False

        # Recover
        gd.pg = _make_pg(healthy=True)
        await gd.check_health()
        assert gd._pg_healthy is True

    @pytest.mark.asyncio
    async def test_full_degradation_and_recovery_cycle(self, fallback_dir: Path):
        """Both services go down, operations fall back, then both recover."""
        gd = GracefulDegradation.__new__(GracefulDegradation)
        gd.redis = _make_redis(healthy=True)
        gd.pg = _make_pg(healthy=True)
        gd._memory_cache = {}
        gd._redis_healthy = True
        gd._pg_healthy = True
        gd._fallback_dir = fallback_dir

        # --- Redis fails on get ---
        gd.redis.get = AsyncMock(side_effect=Exception("err"))
        await gd.cache_get("k")
        assert gd._redis_healthy is False

        # --- PG fails on fetch ---
        gd.pg.fetch = AsyncMock(side_effect=Exception("err"))
        await gd.db_fetch("SELECT 1")
        assert gd._pg_healthy is False

        status = gd.get_status()
        assert status["redis"] == "degraded"
        assert status["postgresql"] == "degraded"

        # --- Both recover ---
        gd.redis = _make_redis(healthy=True)
        gd.pg = _make_pg(healthy=True)
        await gd.check_health()

        status = gd.get_status()
        assert status["redis"] == "healthy"
        assert status["postgresql"] == "healthy"

    @pytest.mark.asyncio
    async def test_cache_set_then_get_via_memory(self, gd_degraded):
        """When redis is down, set and get should round-trip through memory."""
        await gd_degraded.cache_set("key", "value")
        result = await gd_degraded.cache_get("key")
        assert result == "value"

    @pytest.mark.asyncio
    async def test_db_execute_fallback_then_fetch_returns_empty(self, gd_degraded):
        """When PG is down, execute writes fallback and fetch returns []."""
        result_exec = await gd_degraded.db_execute("INSERT INTO t VALUES ($1)", 1)
        assert result_exec is None

        result_fetch = await gd_degraded.db_fetch("SELECT * FROM t")
        assert result_fetch == []
