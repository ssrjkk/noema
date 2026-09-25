"""Tests for noema.config.feature_flags and noema.billing.quotas — pg/redis branches."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fakeredis.aioredis import FakeRedis
from structlog.testing import capture_logs

from noema.billing.quotas import QuotaExceededError, QuotaManager, TenantQuota
from noema.config.feature_flags import FeatureFlagService

TENANT = "tenant-a"


class FakePg:
    """Scripted async pg pool: records queries, serves canned answers."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple]] = []
        self.fetchval_queue: list[Any] = []
        self.fetchrow_queue: list[Any] = []
        self.fetch_queue: list[Any] = []
        self.execute_error: Exception | None = None
        self.fetchval_error: Exception | None = None
        self.fetchrow_error: Exception | None = None
        self.fetch_error: Exception | None = None

    async def execute(self, sql, *args):
        if self.execute_error:
            raise self.execute_error
        self.executed.append((sql, args))

    async def fetchval(self, sql, *args):
        if self.fetchval_error:
            raise self.fetchval_error
        if not self.fetchval_queue:
            raise AssertionError(f"unexpected fetchval: {sql}")
        return self.fetchval_queue.pop(0)

    async def fetchrow(self, sql, *args):
        if self.fetchrow_error:
            raise self.fetchrow_error
        if not self.fetchrow_queue:
            raise AssertionError(f"unexpected fetchrow: {sql}")
        return self.fetchrow_queue.pop(0)

    async def fetch(self, sql, *args):
        if self.fetch_error:
            raise self.fetch_error
        if not self.fetch_queue:
            raise AssertionError(f"unexpected fetch: {sql}")
        return self.fetch_queue.pop(0)


@pytest.fixture
def pg():
    return FakePg()


@pytest.fixture
def redis():
    return FakeRedis(decode_responses=False)


def _drain_tasks():
    """Let fire-and-forget create_task work (setex) settle before asserting."""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.sleep(0))
    loop.run_until_complete(asyncio.sleep(0))


class TestInitialize:
    @pytest.mark.asyncio
    async def test_creates_table_statements(self, pg):
        ff = FeatureFlagService(pg_pool=pg)
        await ff.initialize()
        statements = [sql for sql, _ in pg.executed]
        assert len(statements) == 1 and "CREATE TABLE IF NOT EXISTS feature_flags" in statements[0]

    @pytest.mark.asyncio
    async def test_pg_failure_is_logged_and_swallowed(self, pg):
        pg.execute_error = RuntimeError("no pg")
        ff = FeatureFlagService(pg_pool=pg)
        with capture_logs() as logs:
            await ff.initialize()
        assert any(e["event"] == "feature_flags_init_failed" for e in logs)

    @pytest.mark.asyncio
    async def test_without_pg_is_noop(self):
        await FeatureFlagService().initialize()  # must not raise

    @pytest.mark.asyncio
    async def test_quotas_table_ready(self, pg):
        qm = QuotaManager(pg_pool=pg)
        await qm.initialize()
        assert "CREATE TABLE IF NOT EXISTS tenant_quotas" in pg.executed[0][0]

    @pytest.mark.asyncio
    async def test_quotas_init_failure_logged(self, pg):
        pg.execute_error = RuntimeError("down")
        with capture_logs() as logs:
            await QuotaManager(pg_pool=pg).initialize()
        assert any(e["event"] == "quotas_table_init_failed" for e in logs)


class TestIsEnabledRedisCache:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw,expected", [(True, True), (False, False)])
    async def test_bool_cached_value(self, raw, expected):
        """Clients that deserialize to bool short-circuit the parsing logic."""

        class BoolRedis:
            def __init__(self, val):
                self._val = val

            async def get(self, key):
                return self._val

        ff = FeatureFlagService(redis_client=BoolRedis(raw))
        assert await ff.is_enabled("sandbox_execution") is expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raw,expected", [(b"true", True), (b"false", False), ("true", True), ("false", False)]
    )
    async def test_str_cached_value(self, redis, raw, expected):
        ff = FeatureFlagService(redis_client=redis)
        await redis.set(f"ff:{TENANT}:graph_rag", raw)
        assert await ff.is_enabled("graph_rag", tenant_id=TENANT) is expected

    @pytest.mark.asyncio
    async def test_redis_failure_falls_through_to_defaults(self, redis, pg):
        class Boom:
            async def get(self, key):
                raise ConnectionError("redis down")

        ff = FeatureFlagService(redis_client=Boom(), pg_pool=pg)
        with capture_logs() as logs:
            assert await ff.is_enabled("reflexion") is True
        assert any(e["event"] == "feature_flags_redis_get_failed" for e in logs)


class TestIsEnabledOverrides:
    @pytest.mark.asyncio
    async def test_in_memory_override_wins_over_default(self):
        ff = FeatureFlagService()
        await ff.set_flag(TENANT, "advanced_security", True)
        assert await ff.is_enabled("advanced_security", tenant_id=TENANT) is True
        assert await ff.is_enabled("advanced_security") is False

    @pytest.mark.asyncio
    async def test_pg_override_wins_and_is_cached(self, pg):
        pg.fetchval_queue.append(False)
        ff = FeatureFlagService(pg_pool=pg)
        assert await ff.is_enabled("reflexion", tenant_id=TENANT) is False
        assert pg.fetchval_queue == []  # second call served from cache
        assert await ff.is_enabled("reflexion", tenant_id=TENANT) is False

    @pytest.mark.asyncio
    async def test_pg_override_missing_falls_to_default(self, pg):
        pg.fetchval_queue.append(None)
        ff = FeatureFlagService(pg_pool=pg)
        assert await ff.is_enabled("pairwise_judge", tenant_id=TENANT) is True

    @pytest.mark.asyncio
    async def test_pg_failure_falls_to_default(self, pg):
        pg.fetchval_error = RuntimeError("pg gone")
        ff = FeatureFlagService(pg_pool=pg)
        with capture_logs() as logs:
            assert await ff.is_enabled("multi_modal", tenant_id=TENANT) is False
        assert any(e["event"] == "feature_flags_pg_override_failed" for e in logs)

    @pytest.mark.asyncio
    async def test_unknown_flag_defaults_false(self):
        assert await FeatureFlagService().is_enabled("no_such_flag") is False


class TestSetFlag:
    @pytest.mark.asyncio
    async def test_writes_pg_and_invalidates_caches(self, pg, redis):
        ff = FeatureFlagService(pg_pool=pg, redis_client=redis)
        await ff.set_flag(TENANT, "graph_rag", True)
        assert any("INSERT INTO feature_flags" in sql for sql, _ in pg.executed)
        key = f"ff:{TENANT}:graph_rag"
        assert key not in ff._cache
        assert await redis.exists(key) == 0
        assert await ff.is_enabled("graph_rag", tenant_id=TENANT) is True

    @pytest.mark.asyncio
    async def test_pg_failure_still_updates_memory(self, pg):
        pg.execute_error = RuntimeError("down")
        ff = FeatureFlagService(pg_pool=pg)
        with capture_logs() as logs:
            await ff.set_flag(TENANT, "graph_rag", True)
        assert any(e["event"] == "feature_flags_set_failed" for e in logs)
        assert await ff.is_enabled("graph_rag", tenant_id=TENANT) is True


class TestGetAllFlags:
    @pytest.mark.asyncio
    async def test_pg_overrides_merged_over_defaults(self, pg):
        pg.fetch_queue.append([{"flag_name": "graph_rag", "enabled": True}])
        ff = FeatureFlagService(pg_pool=pg)
        flags = await ff.get_all_flags(TENANT)
        assert flags["graph_rag"] is True
        assert flags["reflexion"] is True  # default kept

    @pytest.mark.asyncio
    async def test_pg_failure_falls_back_to_per_flag_lookup(self, pg):
        pg.fetch_error = RuntimeError("down")
        ff = FeatureFlagService(pg_pool=pg)
        with capture_logs() as logs:
            flags = await ff.get_all_flags(TENANT)
        assert any(e["event"] == "feature_flags_get_all_failed" for e in logs)
        assert flags["reflexion"] is True

    @pytest.mark.asyncio
    async def test_global_without_pg_uses_defaults(self):
        flags = await FeatureFlagService().get_all_flags()
        assert flags["sandbox_execution"] is True
        assert flags["advanced_security"] is False


class TestSetCacheSetex:
    @pytest.mark.asyncio
    async def test_cache_write_schedules_redis_setex(self, pg, redis):
        ff = FeatureFlagService(redis_client=redis)
        ff._set_cache("ff:global:x", True)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert await redis.get("ff:global:x") == b"true"
        assert await redis.ttl("ff:global:x") > 0

    @pytest.mark.asyncio
    async def test_setex_failure_is_logged(self, pg):
        class Boom:
            def setex(self, *a):
                raise ConnectionError("down")

        ff = FeatureFlagService(redis_client=Boom())
        with capture_logs() as logs:
            ff._set_cache("ff:global:x", False)
        assert any(e["event"] == "feature_flags_redis_setex_failed" for e in logs)


class TestQuotaGetSet:
    @pytest.mark.asyncio
    async def test_get_from_pg_with_json_string_features(self, pg):
        pg.fetchrow_queue.append(
            {
                "tenant_id": TENANT,
                "monthly_budget_usd": "250.50",
                "max_concurrent_tasks": 3,
                "max_tasks_per_hour": 50,
                "max_input_tokens_per_task": 5000,
                "enabled_features": '["basic", "pro"]',
            }
        )
        qm = QuotaManager(pg_pool=pg)
        quota = await qm.get_quota(TENANT)
        assert quota.monthly_budget_usd == 250.50
        assert quota.enabled_features == ["basic", "pro"]

    @pytest.mark.asyncio
    async def test_get_from_pg_with_list_features(self, pg):
        pg.fetchrow_queue.append(
            {
                "tenant_id": TENANT,
                "monthly_budget_usd": 100,
                "max_concurrent_tasks": 5,
                "max_tasks_per_hour": 100,
                "max_input_tokens_per_task": 100000,
                "enabled_features": ["basic"],
            }
        )
        quota = await QuotaManager(pg_pool=pg).get_quota(TENANT)
        assert quota.enabled_features == ["basic"]

    @pytest.mark.asyncio
    async def test_pg_failure_uses_defaults_cache(self, pg):
        pg.fetchrow_error = RuntimeError("down")
        expected = TenantQuota(tenant_id=TENANT, max_concurrent_tasks=9)
        qm = QuotaManager(pg_pool=pg)
        qm._defaults_cache[TENANT] = expected
        with capture_logs() as logs:
            assert await qm.get_quota(TENANT) is expected
        assert any(e["event"] == "quotas_fetch_failed" for e in logs)

    @pytest.mark.asyncio
    async def test_unknown_tenant_gets_default_quota(self, pg):
        pg.fetchrow_queue.append(None)
        quota = await QuotaManager(pg_pool=pg).get_quota("ghost")
        assert quota.tenant_id == "ghost"
        assert quota.monthly_budget_usd == 100.0

    @pytest.mark.asyncio
    async def test_set_quota_writes_pg_and_cache(self, pg):
        qm = QuotaManager(pg_pool=pg)
        quota = TenantQuota(tenant_id=TENANT, monthly_budget_usd=42.0)
        await qm.set_quota(TENANT, quota)
        assert any("INSERT INTO tenant_quotas" in sql for sql, _ in pg.executed)
        assert qm._defaults_cache[TENANT] is quota

    @pytest.mark.asyncio
    async def test_set_quota_pg_failure_keeps_cache(self, pg):
        pg.execute_error = RuntimeError("down")
        qm = QuotaManager(pg_pool=pg)
        quota = TenantQuota(tenant_id=TENANT)
        with capture_logs() as logs:
            await qm.set_quota(TENANT, quota)
        assert any(e["event"] == "quotas_set_failed" for e in logs)
        assert qm._defaults_cache[TENANT] is quota


class TestCheckQuota:
    @pytest.mark.asyncio
    async def test_budget_exceeded_raises(self, pg):
        pg.fetchrow_queue.append(None)  # get_quota -> default
        pg.fetchval_queue.append(95.0)  # monthly cost
        qm = QuotaManager(pg_pool=pg)
        qm._defaults_cache[TENANT] = TenantQuota(tenant_id=TENANT, monthly_budget_usd=100.0)
        with pytest.raises(QuotaExceededError, match="Monthly budget"):
            await qm.check_quota(TENANT, estimated_cost_usd=10.0)

    @pytest.mark.asyncio
    async def test_token_cap_exceeded_raises(self):
        qm = QuotaManager()
        qm._defaults_cache[TENANT] = TenantQuota(tenant_id=TENANT, max_input_tokens_per_task=1000)
        with pytest.raises(QuotaExceededError, match="per-task cap"):
            await qm.check_quota(TENANT, estimated_input_tokens=1001)

    @pytest.mark.asyncio
    async def test_zero_token_cap_disables_check(self):
        qm = QuotaManager()
        qm._defaults_cache[TENANT] = TenantQuota(tenant_id=TENANT, max_input_tokens_per_task=0)
        assert await qm.check_quota(TENANT, estimated_input_tokens=10**9) is True

    @pytest.mark.asyncio
    async def test_hourly_limit_raises(self, pg):
        qm = QuotaManager(pg_pool=pg)
        qm._defaults_cache[TENANT] = TenantQuota(tenant_id=TENANT, max_tasks_per_hour=2)
        await qm.track_active_task(TENANT, "t1")
        await qm.track_active_task(TENANT, "t2")
        with pytest.raises(QuotaExceededError, match="Hourly limit"):
            await qm.check_quota(TENANT)

    @pytest.mark.asyncio
    async def test_concurrent_limit_raises(self):
        qm = QuotaManager()
        qm._defaults_cache[TENANT] = TenantQuota(tenant_id=TENANT, max_concurrent_tasks=1)
        await qm.track_active_task(TENANT, "t1")
        with pytest.raises(QuotaExceededError, match="Concurrent tasks"):
            await qm.check_quota(TENANT)

    @pytest.mark.asyncio
    async def test_passes_within_limits(self):
        qm = QuotaManager()
        qm._defaults_cache[TENANT] = TenantQuota(tenant_id=TENANT)
        assert await qm.check_quota(TENANT, estimated_cost_usd=1.0, estimated_input_tokens=10)


class TestActiveTaskTracking:
    @pytest.mark.asyncio
    async def test_redis_track_and_untrack(self, redis):
        qm = QuotaManager(redis_client=redis)
        await qm.track_active_task(TENANT, "task-1")
        assert await redis.scard(f"tenant:{TENANT}:active_tasks") == 1
        assert await qm._get_active_task_count(TENANT) == 1
        await qm.untrack_active_task(TENANT, "task-1")
        assert await qm._get_active_task_count(TENANT) == 0

    @pytest.mark.asyncio
    async def test_redis_sadd_failure_falls_back_to_memory(self, redis):
        class Boom:
            async def sadd(self, *a):
                raise ConnectionError("down")

            async def expire(self, *a):
                raise AssertionError("expire must not run after sadd fails")

        qm = QuotaManager(redis_client=Boom())
        with capture_logs() as logs:
            await qm.track_active_task(TENANT, "t1")
        assert any(e["event"] == "redis_active_tasks_sadd_failed" for e in logs)
        assert await qm._get_active_task_count(TENANT) == 1

    @pytest.mark.asyncio
    async def test_redis_srem_failure_falls_back_to_memory(self):
        class Boom:
            async def srem(self, *a):
                raise ConnectionError("down")

        qm = QuotaManager(redis_client=Boom())
        await qm.track_active_task(TENANT, "t1")
        with capture_logs() as logs:
            await qm.untrack_active_task(TENANT, "t1")
        assert any(e["event"] == "redis_active_tasks_srem_failed" for e in logs)
        assert await qm._get_active_task_count(TENANT) == 0

    @pytest.mark.asyncio
    async def test_redis_scard_failure_counts_memory(self):
        class Boom:
            async def scard(self, *a):
                raise ConnectionError("down")

        qm = QuotaManager(redis_client=Boom())
        await qm.track_active_task(TENANT, "t1")
        await qm.track_active_task(TENANT, "t2")
        with capture_logs() as logs:
            assert await qm._get_active_task_count(TENANT) == 2
        assert any(e["event"] == "redis_active_task_count_failed" for e in logs)


class TestMonthlyCostFallbacks:
    @pytest.mark.asyncio
    async def test_pg_sum_wins(self, pg):
        pg.fetchval_queue.append(12.5)
        assert await QuotaManager(pg_pool=pg)._get_monthly_cost(TENANT) == 12.5

    @pytest.mark.asyncio
    async def test_pg_failure_uses_redis_counter(self, pg, redis):
        pg.fetchval_error = RuntimeError("down")
        month = __import__("datetime").datetime.now(__import__("datetime").UTC).strftime("%Y%m")
        await redis.set(f"cost:m:{TENANT}:{month}", 25000)
        assert await QuotaManager(pg_pool=pg, redis_client=redis)._get_monthly_cost(TENANT) == 2.5

    @pytest.mark.asyncio
    async def test_broken_returns_zero(self, pg, redis):
        pg.fetchval_error = RuntimeError("down")

        class Boom:
            async def get(self, *a):
                raise ConnectionError("down")

        with capture_logs() as logs:
            value = await QuotaManager(pg_pool=pg, redis_client=Boom())._get_monthly_cost(TENANT)
        assert value == 0.0
        assert any(e["event"] == "quota_monthly_redis_failed" for e in logs)


class TestHourlyCountFallbacks:
    @pytest.mark.asyncio
    async def test_pg_count_wins(self, pg):
        pg.fetchval_queue.append(7)
        assert await QuotaManager(pg_pool=pg)._get_hourly_task_count(TENANT) == 7

    @pytest.mark.asyncio
    async def test_pg_failure_counts_recent_timestamps(self, pg):
        pg.fetchval_error = RuntimeError("down")
        qm = QuotaManager(pg_pool=pg)
        qm._task_starts[TENANT] = [__import__("time").time() - 900, __import__("time").time() - 10]
        assert await qm._get_hourly_task_count(TENANT) == 2
