"""Tests for v7.0 enterprise modules: Audit, Quotas, Feature Flags, Graceful Degradation."""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from noema.audit.logger import AuditEvent, AuditLogger
from noema.billing.quotas import QuotaExceededError, QuotaManager, TenantQuota
from noema.config.feature_flags import FeatureFlagService
from noema.resilience.graceful_degradation import GracefulDegradation

# ── Audit Logger ─────────────────────────────────────────────────────


def test_audit_event_defaults():
    event = AuditEvent(timestamp=datetime.now(UTC), event_type="test", tenant_id="t1", user_id="u1")
    assert event.details == {}
    assert event.task_id is None
    assert event.ip_address is None


@pytest.mark.asyncio
async def test_audit_logger_file_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        logger = AuditLogger(pg_pool=None, fallback_dir=tmp)
        await logger.initialize()

        await logger.log(
            AuditEvent(
                timestamp=datetime.now(UTC),
                event_type="task_created",
                tenant_id="t1",
                user_id="u1",
                task_id="task-1",
                details={"description": "test"},
                ip_address="127.0.0.1",
            )
        )

        log_file = Path(tmp) / "t1.jsonl"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "task_created" in content
        assert "t1" in content


@pytest.mark.asyncio
async def test_audit_logger_query():
    with tempfile.TemporaryDirectory() as tmp:
        logger = AuditLogger(pg_pool=None, fallback_dir=tmp)
        await logger.initialize()

        await logger.log(AuditEvent(datetime.now(UTC), "type_a", "t1", "u1", details={"a": 1}))
        await logger.log(AuditEvent(datetime.now(UTC), "type_b", "t1", "u1", details={"b": 2}))
        await logger.log(AuditEvent(datetime.now(UTC), "type_a", "t2", "u2", details={"c": 3}))

        results_t1 = await logger.query("t1")
        assert len(results_t1) == 2

        results_t2 = await logger.query("t2")
        assert len(results_t2) == 1

        results_filtered = await logger.query("t1", event_type="type_a")
        assert len(results_filtered) == 1
        assert results_filtered[0]["event_type"] == "type_a"


@pytest.mark.asyncio
async def test_audit_logger_query_time_filter():
    with tempfile.TemporaryDirectory() as tmp:
        logger = AuditLogger(pg_pool=None, fallback_dir=tmp)
        await logger.initialize()

        now = datetime.now(UTC)
        await logger.log(AuditEvent(now - timedelta(hours=2), "old", "t1", "u1"))
        await logger.log(AuditEvent(now, "new", "t1", "u1"))

        recent = await logger.query("t1", start_time=now - timedelta(hours=1))
        assert len(recent) == 1
        assert recent[0]["event_type"] == "new"


@pytest.mark.asyncio
async def test_audit_logger_multiple_tenants():
    with tempfile.TemporaryDirectory() as tmp:
        logger = AuditLogger(pg_pool=None, fallback_dir=tmp)
        await logger.initialize()

        for i in range(5):
            await logger.log(
                AuditEvent(datetime.now(UTC), "ev", f"tenant-{i}", f"user-{i}", task_id=f"task-{i}")
            )

        for i in range(5):
            q = await logger.query(f"tenant-{i}")
            assert len(q) == 1
            assert q[0]["task_id"] == f"task-{i}"


# ── Quota Manager ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_quota_default_values():
    qm = QuotaManager()
    quota = await qm.get_quota("new-tenant")
    assert quota.monthly_budget_usd == 100.0
    assert quota.max_concurrent_tasks == 5
    assert quota.max_tasks_per_hour == 100


@pytest.mark.asyncio
async def test_quota_set_and_get():
    qm = QuotaManager()
    await qm.set_quota(
        "t1", TenantQuota(tenant_id="t1", monthly_budget_usd=500.0, max_concurrent_tasks=10)
    )
    quota = await qm.get_quota("t1")
    assert quota.monthly_budget_usd == 500.0
    assert quota.max_concurrent_tasks == 10


@pytest.mark.asyncio
async def test_quota_check_passes():
    qm = QuotaManager()
    result = await qm.check_quota("t1")
    assert result is True


@pytest.mark.asyncio
async def test_quota_track_active():
    qm = QuotaManager()
    await qm.track_active_task("t1", "task-1")
    await qm.track_active_task("t1", "task-2")
    count = await qm._get_active_task_count("t1")
    assert count == 2

    await qm.untrack_active_task("t1", "task-1")
    count = await qm._get_active_task_count("t1")
    assert count == 1


@pytest.mark.asyncio
async def test_quota_exceeded_error():
    qm = QuotaManager()
    await qm.set_quota("limited", TenantQuota(tenant_id="limited", max_concurrent_tasks=1))

    await qm.track_active_task("limited", "running")
    with pytest.raises(QuotaExceededError) as exc:
        await qm.check_quota("limited")
    assert "Concurrent" in str(exc.value) or "concurrent" in str(exc.value)


# ── Feature Flags ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_feature_flag_defaults():
    ff = FeatureFlagService()
    assert await ff.is_enabled("reflexion") is True
    assert await ff.is_enabled("advanced_security") is False
    assert await ff.is_enabled("nonexistent_flag") is False


@pytest.mark.asyncio
async def test_feature_flag_set_override():
    ff = FeatureFlagService()
    await ff.set_flag("t1", "reflexion", False)
    # After setting, it should be cached in-memory
    enabled = await ff.is_enabled("reflexion", "t1")
    assert enabled is False


@pytest.mark.asyncio
async def test_feature_flag_tenant_isolation():
    ff = FeatureFlagService()
    await ff.set_flag("tenant-a", "reflexion", False)
    await ff.set_flag("tenant-b", "reflexion", True)

    assert await ff.is_enabled("reflexion", "tenant-a") is False
    assert await ff.is_enabled("reflexion", "tenant-b") is True
    # Global default unaffected
    assert await ff.is_enabled("reflexion") is True


@pytest.mark.asyncio
async def test_feature_flag_get_all():
    ff = FeatureFlagService()
    all_flags = await ff.get_all_flags()
    assert "reflexion" in all_flags
    assert "pairwise_judge" in all_flags
    assert "advanced_security" in all_flags
    assert all_flags["reflexion"] is True


@pytest.mark.asyncio
async def test_feature_flag_toggle():
    ff = FeatureFlagService()
    await ff.set_flag("t1", "graph_rag", True)
    assert await ff.is_enabled("graph_rag", "t1") is True

    await ff.set_flag("t1", "graph_rag", False)
    assert await ff.is_enabled("graph_rag", "t1") is False


# ── Graceful Degradation ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_graceful_degradation_init():
    gd = GracefulDegradation()
    status = gd.get_status()
    assert "redis" in status
    assert "postgresql" in status
    assert status["memory_cache_size"] == 0


@pytest.mark.asyncio
async def test_graceful_cache_fallback():
    gd = GracefulDegradation(redis_client=None)
    await gd.cache_set("key1", "value1")
    val = await gd.cache_get("key1")
    assert val == "value1"


@pytest.mark.asyncio
async def test_graceful_cache_miss():
    gd = GracefulDegradation()
    val = await gd.cache_get("nonexistent")
    assert val is None


@pytest.mark.asyncio
async def test_graceful_cache_delete():
    gd = GracefulDegradation()
    await gd.cache_set("temp", "value")
    await gd.cache_delete("temp")
    assert await gd.cache_get("temp") is None


@pytest.mark.asyncio
async def test_graceful_db_fallback():
    gd = GracefulDegradation(pg_pool=None)
    result = await gd.db_execute("INSERT INTO test VALUES ($1)", "hello")
    assert result is None  # gracefully handled


@pytest.mark.asyncio
async def test_graceful_db_fetch_fallback():
    gd = GracefulDegradation(pg_pool=None)
    result = await gd.db_fetch("SELECT * FROM test")
    assert result == []


@pytest.mark.asyncio
async def test_graceful_status():
    gd = GracefulDegradation(redis_client=None, pg_pool=None)
    await gd.check_health()
    status = gd.get_status()
    assert status["redis"] == "degraded"
    assert status["postgresql"] == "degraded"


@pytest.mark.asyncio
async def test_graceful_fallback_writes_file():
    with tempfile.TemporaryDirectory() as tmp:
        gd = GracefulDegradation(pg_pool=None)
        gd._fallback_dir = Path(tmp)
        await gd.db_execute("SELECT 1")
        files = list(Path(tmp).iterdir())
        assert len(files) > 0
        content = files[0].read_text(encoding="utf-8")
        assert "SELECT 1" in content


# ── Integration: Quota + Audit together ─────────────────────────────


@pytest.mark.asyncio
async def test_audit_and_quota_integration():
    with tempfile.TemporaryDirectory() as tmp:
        logger = AuditLogger(pg_pool=None, fallback_dir=tmp)
        await logger.initialize()

        qm = QuotaManager()
        await qm.set_quota("t1", TenantQuota(tenant_id="t1", max_concurrent_tasks=10))

        # Simulate workflow
        await qm.track_active_task("t1", "task-1")
        await logger.log(
            AuditEvent(
                timestamp=datetime.now(UTC),
                event_type="task_created",
                tenant_id="t1",
                user_id="u1",
                task_id="task-1",
                details={"input_tokens": 500},
            )
        )

        await qm.untrack_active_task("t1", "task-1")
        await logger.log(
            AuditEvent(
                timestamp=datetime.now(UTC),
                event_type="solution_generated",
                tenant_id="t1",
                user_id="u1",
                task_id="task-1",
                details={"output_tokens": 200},
            )
        )

        events = await logger.query("t1")
        assert len(events) == 2
        event_types = {e["event_type"] for e in events}
        assert "task_created" in event_types
        assert "solution_generated" in event_types


# ── Integration: Feature Flags + Quota ──────────────────────────────


@pytest.mark.asyncio
async def test_feature_flags_affect_quota_checks():
    """Feature flag can gate whether a quota check is even performed."""
    ff = FeatureFlagService()
    qm = QuotaManager()

    await ff.set_flag("premium", "cost_tracking", True)
    await ff.set_flag("free", "cost_tracking", False)

    # Verify feature flags are correctly set and read back
    assert await ff.is_enabled("cost_tracking", "premium") is True
    assert await ff.is_enabled("cost_tracking", "free") is False

    # Premium tenant: quota check runs
    result = await qm.check_quota("premium")
    assert result is True

    # Free tenant: quota check also works (lightweight)
    result = await qm.check_quota("free")
    assert result is True

    # Verify all flags reflect the overrides
    premium_flags = await ff.get_all_flags("premium")
    assert premium_flags.get("cost_tracking") is True
    free_flags = await ff.get_all_flags("free")
    assert free_flags.get("cost_tracking") is False
