"""Coverage for CostTracker / calculate_cost in noema.billing.cost_tracker."""

import pytest

from noema.billing.cost_tracker import CostRecord, CostTracker, calculate_cost


class _FakeIncrRedis:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    async def incrby(self, key: str, amt: int) -> None:
        self.calls[key] = self.calls.get(key, 0) + amt

    async def expire(self, key: str, ttl: int) -> None:
        self.calls[f"expire:{key}"] = ttl


def test_calculate_cost_known_models():
    assert calculate_cost("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
    assert calculate_cost("gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.5)
    assert calculate_cost("o3", 100_000, 50_000) == pytest.approx(1.0 + 2.0)
    assert calculate_cost("claude-sonnet-4-20250514", 0, 1_000_000) == pytest.approx(15.0)


def test_calculate_cost_unknown_model_uses_default():
    assert calculate_cost("gpt-9x-future", 1_000_000, 1_000_000) == pytest.approx(12.5)


def test_calculate_cost_fallback_is_free():
    assert calculate_cost("fallback", 1_000_000, 1_000_000) == 0.0


def test_calculate_cost_zero_tokens():
    assert calculate_cost("gpt-4o-mini", 0, 0) == 0.0


@pytest.mark.asyncio
async def test_record_tracks_tenant_and_task_cost():
    tracker = CostTracker()
    rec = await tracker.record("t1", "task1", "openai", "gpt-4o-mini", 1_000_000, 1_000_000)
    assert isinstance(rec, CostRecord)
    assert rec.cost_usd == pytest.approx(0.75)
    assert tracker.get_tenant_cost("t1") == {
        "daily_usd": pytest.approx(0.75),
        "monthly_usd": pytest.approx(0.75),
    }
    assert tracker.get_task_cost("task1") == pytest.approx(0.75)
    assert tracker.get_task_cost("other") == 0.0


@pytest.mark.asyncio
async def test_record_clamps_negative_tokens():
    tracker = CostTracker()
    rec = await tracker.record("t1", "task1", "openai", "gpt-4o-mini", -5, 1_000_000)
    assert rec.input_tokens == 0
    assert rec.output_tokens == 1_000_000
    assert rec.cost_usd == pytest.approx(0.60)


@pytest.mark.asyncio
async def test_record_negative_output_tokens():
    tracker = CostTracker()
    rec = await tracker.record("t1", "task1", "openai", "gpt-4o-mini", 1_000_000, -1)
    assert rec.output_tokens == 0
    assert rec.cost_usd == pytest.approx(0.15)


@pytest.mark.asyncio
async def test_breakdown_groups_by_provider_model_step():
    tracker = CostTracker()
    await tracker.record("t1", "taskA", "openai", "gpt-4o", 1_000_000, 0, step_name="analyze")
    await tracker.record("t1", "taskA", "openai", "gpt-4o", 0, 1_000_000, step_name="generate")
    await tracker.record("t1", "taskB", "anthropic", "claude-sonnet-4-20250514", 1_000_000, 0)
    breakdown = tracker.get_breakdown("t1")
    assert breakdown["total_usd"] == pytest.approx(2.5 + 10.0 + 3.0)
    assert breakdown["by_provider"]["openai"] == pytest.approx(12.5)
    assert breakdown["by_provider"]["anthropic"] == pytest.approx(3.0)
    assert breakdown["by_model"]["gpt-4o"] == pytest.approx(12.5)
    assert breakdown["by_step"]["analyze"] == pytest.approx(2.5)
    assert breakdown["by_step"]["generate"] == pytest.approx(10.0)
    assert breakdown["by_step"]["unknown"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_breakdown_empty_for_unknown_tenant():
    tracker = CostTracker()
    await tracker.record("t1", "taskA", "openai", "gpt-4o-mini", 1_000_000, 0)
    assert tracker.get_breakdown("nope") == {
        "total_usd": 0,
        "by_provider": {},
        "by_model": {},
        "by_step": {},
    }


@pytest.mark.asyncio
async def test_stats_aggregates_totals():
    tracker = CostTracker()
    await tracker.record("t1", "taskA", "openai", "gpt-4o-mini", 1_000_000, 1_000_000)
    await tracker.record("t2", "taskB", "openai", "gpt-4o-mini", 1_000_000, 1_000_000)
    stats = tracker.stats()
    assert stats["total_records"] == 2
    assert stats["total_cost_usd"] == pytest.approx(1.5)
    assert stats["total_input_tokens"] == 2_000_000
    assert stats["total_output_tokens"] == 2_000_000


@pytest.mark.asyncio
async def test_redis_backend_gets_cost_increments():
    tracker = CostTracker()
    fake = _FakeIncrRedis()
    tracker._redis = fake
    await tracker.record("t1", "taskA", "openai", "gpt-4o-mini", 1_000_000, 1_000_000)
    assert fake.calls["cost:d:t1"] == 7500
    assert fake.calls["cost:m:t1"] == 7500
    assert fake.calls["expire:cost:d:t1"] == 86400 * 2
    assert fake.calls["expire:cost:m:t1"] == 86400 * 32
    assert tracker.get_tenant_cost("t1")["daily_usd"] == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_redis_failure_does_not_break_record():
    class _BrokenRedis:
        async def incrby(self, key: str, amt: int) -> None:
            raise RuntimeError("redis down")

        async def expire(self, key: str, ttl: int) -> None:
            raise RuntimeError("redis down")

    tracker = CostTracker()
    tracker._redis = _BrokenRedis()
    rec = await tracker.record("t1", "taskA", "openai", "gpt-4o-mini", 1_000_000, 1_000_000)
    assert rec.cost_usd == pytest.approx(0.75)
    assert tracker.get_tenant_cost("t1")["daily_usd"] == pytest.approx(0.75)
