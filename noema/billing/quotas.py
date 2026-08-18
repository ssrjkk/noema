"""Tenant Quotas & Rate Limiting — контроль затрат и лимитов per tenant."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from noema.logging import get_logger

log = get_logger(__name__)


class QuotaExceededError(Exception):
    """Raised when a tenant exceeds their quota."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


CREATE_QUOTAS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tenant_quotas (
    tenant_id VARCHAR(100) PRIMARY KEY,
    monthly_budget_usd DECIMAL(10, 2) NOT NULL DEFAULT 100.00,
    max_concurrent_tasks INT NOT NULL DEFAULT 5,
    max_tasks_per_hour INT NOT NULL DEFAULT 100,
    max_input_tokens_per_task INT NOT NULL DEFAULT 100000,
    enabled_features JSONB NOT NULL DEFAULT '["basic"]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


@dataclass
class TenantQuota:
    tenant_id: str = ""
    monthly_budget_usd: float = 100.0
    max_concurrent_tasks: int = 5
    max_tasks_per_hour: int = 100
    max_input_tokens_per_task: int = 100000
    enabled_features: list[str] = field(default_factory=lambda: ["basic"])


class QuotaManager:
    """Checks and enforces per-tenant quotas with Redis + PostgreSQL."""

    def __init__(self, pg_pool: Any = None, redis_client: Any = None) -> None:
        self.pg = pg_pool
        self.redis = redis_client
        self._defaults_cache: dict[str, TenantQuota] = {}
        self._active_tasks: dict[str, set[str]] = {}
        self._active_lock = asyncio.Lock()
        self._task_starts: dict[str, list[float]] = {}

    async def initialize(self) -> None:
        if self.pg:
            try:
                for stmt in CREATE_QUOTAS_TABLE_SQL.strip().split(";"):
                    s = stmt.strip()
                    if s:
                        await self.pg.execute(s)
                log.info("quotas_table_ready")
            except Exception as e:
                log.warning("quotas_table_init_failed", error=str(e))

    async def get_quota(self, tenant_id: str) -> TenantQuota:
        if self.pg:
            try:
                row = await self.pg.fetchrow(
                    "SELECT * FROM tenant_quotas WHERE tenant_id = $1", tenant_id
                )
                if row:
                    return TenantQuota(
                        tenant_id=row["tenant_id"],
                        monthly_budget_usd=float(row["monthly_budget_usd"]),
                        max_concurrent_tasks=row["max_concurrent_tasks"],
                        max_tasks_per_hour=row["max_tasks_per_hour"],
                        max_input_tokens_per_task=row["max_input_tokens_per_task"],
                        enabled_features=json.loads(row["enabled_features"])
                        if isinstance(row["enabled_features"], str)
                        else row["enabled_features"],
                    )
            except Exception as e:
                log.warning("quotas_fetch_failed", error=str(e))
        return self._defaults_cache.get(tenant_id, TenantQuota(tenant_id=tenant_id))

    async def set_quota(self, tenant_id: str, quota: TenantQuota) -> None:
        if self.pg:
            try:
                await self.pg.execute(
                    """INSERT INTO tenant_quotas (tenant_id, monthly_budget_usd, max_concurrent_tasks,
                       max_tasks_per_hour, max_input_tokens_per_task, enabled_features, updated_at)
                       VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW())
                       ON CONFLICT (tenant_id) DO UPDATE SET
                       monthly_budget_usd = $2, max_concurrent_tasks = $3,
                       max_tasks_per_hour = $4, max_input_tokens_per_task = $5,
                       enabled_features = $6::jsonb, updated_at = NOW()""",
                    tenant_id,
                    quota.monthly_budget_usd,
                    quota.max_concurrent_tasks,
                    quota.max_tasks_per_hour,
                    quota.max_input_tokens_per_task,
                    json.dumps(quota.enabled_features),
                )
            except Exception as e:
                log.warning("quotas_set_failed", error=str(e))
        self._defaults_cache[tenant_id] = quota

    async def check_quota(self, tenant_id: str, estimated_cost_usd: float = 0.0) -> bool:
        quota = await self.get_quota(tenant_id)

        if estimated_cost_usd > 0:
            monthly = await self._get_monthly_cost(tenant_id)
            if monthly + estimated_cost_usd > quota.monthly_budget_usd:
                raise QuotaExceededError(
                    f"Monthly budget ${monthly:.2f} + ${estimated_cost_usd:.2f} exceeds ${quota.monthly_budget_usd:.2f}"
                )

        hourly = await self._get_hourly_task_count(tenant_id)
        if hourly >= quota.max_tasks_per_hour:
            raise QuotaExceededError(f"Hourly limit {hourly} >= {quota.max_tasks_per_hour}")

        active = await self._get_active_task_count(tenant_id)
        if active >= quota.max_concurrent_tasks:
            raise QuotaExceededError(f"Concurrent tasks {active} >= {quota.max_concurrent_tasks}")
        return True

    async def track_active_task(self, tenant_id: str, task_id: str) -> None:
        now = time.time()
        async with self._active_lock:
            self._task_starts.setdefault(tenant_id, []).append(now)
            self._task_starts[tenant_id] = [
                t for t in self._task_starts[tenant_id] if t >= now - 7200
            ]
        if self.redis:
            try:
                await self.redis.sadd(f"tenant:{tenant_id}:active_tasks", task_id)
                await self.redis.expire(f"tenant:{tenant_id}:active_tasks", 7200)
                return
            except Exception as e:
                log.warning("redis_active_tasks_sadd_failed", error=str(e))
        async with self._active_lock:
            self._active_tasks.setdefault(tenant_id, set()).add(task_id)

    async def untrack_active_task(self, tenant_id: str, task_id: str) -> None:
        if self.redis:
            try:
                await self.redis.srem(f"tenant:{tenant_id}:active_tasks", task_id)
                return
            except Exception as e:
                log.warning("redis_active_tasks_srem_failed", error=str(e))
        async with self._active_lock:
            s = self._active_tasks.get(tenant_id, set())
            s.discard(task_id)

    async def _get_monthly_cost(self, tenant_id: str) -> float:
        start = datetime.now(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if self.pg:
            try:
                result = await self.pg.fetchval(
                    "SELECT COALESCE(SUM(cost_usd), 0) FROM cost_records WHERE tenant_id = $1 AND recorded_at >= $2",
                    tenant_id,
                    start,
                )
                return float(result)
            except Exception as e:
                log.warning("quota_monthly_cost_failed", error=str(e))
        # Fallback: Redis month counters written by CostTracker
        # (``cost:m:<tenant>:<YYYYMM>`` in 1/10000 USD units).
        if self.redis:
            try:
                month_key = f"cost:m:{tenant_id}:{datetime.now(UTC).strftime('%Y%m')}"
                value = await self.redis.get(month_key)
                if value:
                    return int(value) / 10000
            except Exception as e:
                log.warning("quota_monthly_redis_failed", error=str(e))
        return 0.0

    async def _get_hourly_task_count(self, tenant_id: str) -> int:
        one_hour_ago = datetime.now(UTC) - timedelta(hours=1)
        if self.pg:
            try:
                result = await self.pg.fetchval(
                    "SELECT COUNT(DISTINCT task_id) FROM cost_records WHERE tenant_id = $1 AND recorded_at >= $2",
                    tenant_id,
                    one_hour_ago,
                )
                return result or 0
            except Exception as e:
                log.warning("quota_hourly_count_failed", error=str(e))
        # Single-process fallback: timestamps tracked by track_active_task.
        cutoff = time.time() - 3600
        return sum(1 for t in self._task_starts.get(tenant_id, []) if t >= cutoff)

    async def _get_active_task_count(self, tenant_id: str) -> int:
        if self.redis:
            try:
                count = await self.redis.scard(f"tenant:{tenant_id}:active_tasks")
                return count or 0
            except Exception as e:
                log.warning("redis_active_task_count_failed", error=str(e))
        return len(self._active_tasks.get(tenant_id, set()))
