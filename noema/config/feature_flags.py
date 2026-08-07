"""Feature Flags — per-tenant overrides with Redis cache."""

from __future__ import annotations

import contextlib
from typing import Any

from noema.logging import get_logger

log = get_logger(__name__)


CREATE_FEATURE_FLAGS_SQL = """
CREATE TABLE IF NOT EXISTS feature_flags (
    tenant_id VARCHAR(100) NOT NULL,
    flag_name VARCHAR(100) NOT NULL,
    enabled BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, flag_name)
);
"""


DEFAULT_FLAGS: dict[str, bool] = {
    "reflexion": True,
    "pairwise_judge": True,
    "advanced_security": False,
    "graph_rag": False,
    "multi_modal": False,
    "sandbox_execution": True,
    "cost_tracking": True,
    "audit_logging": True,
}


class FeatureFlagService:
    """Feature flags with tenant-specific overrides (priority: tenant override > global default)."""

    def __init__(self, pg_pool: Any = None, redis_client: Any = None) -> None:
        self.pg = pg_pool
        self.redis = redis_client
        self._cache: dict[str, bool] = {}
        self._overrides: dict[str, dict[str, bool]] = {}
        self._defaults = dict(DEFAULT_FLAGS)
        self._cache_ttl = 300

    async def initialize(self) -> None:
        if self.pg:
            try:
                for stmt in CREATE_FEATURE_FLAGS_SQL.strip().split(";"):
                    s = stmt.strip()
                    if s:
                        await self.pg.execute(s)
                log.info("feature_flags_table_ready")
            except Exception as e:
                log.warning("feature_flags_init_failed", error=str(e))

    async def is_enabled(self, flag_name: str, tenant_id: str | None = None) -> bool:
        cache_key = f"ff:{tenant_id or 'global'}:{flag_name}"

        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached is not None:
                    val = (
                        cached
                        if isinstance(cached, bool)
                        else (
                            cached.lower() == b"true"
                            if isinstance(cached, bytes)
                            else cached.lower() == "true"
                        )
                    )
                    if isinstance(val, bool):
                        return val
            except Exception as e:
                log.warning("feature_flags_redis_get_failed", error=str(e))

        cached_val = self._cache.get(cache_key)
        if cached_val is not None:
            return cached_val

        if tenant_id:
            if tenant_id in self._overrides and flag_name in self._overrides[tenant_id]:
                result = self._overrides[tenant_id][flag_name]
                self._set_cache(cache_key, result)
                return result

            if self.pg:
                try:
                    override = await self.pg.fetchval(
                        "SELECT enabled FROM feature_flags WHERE tenant_id = $1 AND flag_name = $2",
                        tenant_id,
                        flag_name,
                    )
                    if override is not None:
                        result = bool(override)
                        self._set_cache(cache_key, result)
                        return result
                except Exception as e:
                    log.warning("feature_flags_pg_override_failed", error=str(e))

        default_value: bool = bool(self._defaults.get(flag_name, False))
        self._set_cache(cache_key, default_value)
        return default_value

    async def set_flag(self, tenant_id: str, flag_name: str, enabled: bool) -> None:
        self._overrides.setdefault(tenant_id, {})[flag_name] = enabled
        if self.pg:
            try:
                await self.pg.execute(
                    """INSERT INTO feature_flags (tenant_id, flag_name, enabled, updated_at)
                       VALUES ($1, $2, $3, NOW())
                       ON CONFLICT (tenant_id, flag_name)
                       DO UPDATE SET enabled = $3, updated_at = NOW()""",
                    tenant_id,
                    flag_name,
                    enabled,
                )
            except Exception as e:
                log.warning("feature_flags_set_failed", error=str(e))
        cache_key = f"ff:{tenant_id}:{flag_name}"
        self._cache.pop(cache_key, None)
        if self.redis:
            with contextlib.suppress(Exception):
                await self.redis.delete(cache_key)

    async def get_all_flags(self, tenant_id: str | None = None) -> dict[str, bool]:
        if tenant_id and self.pg:
            try:
                rows = await self.pg.fetch(
                    "SELECT flag_name, enabled FROM feature_flags WHERE tenant_id = $1",
                    tenant_id,
                )
                overrides = {r["flag_name"]: bool(r["enabled"]) for r in rows}
                result = dict(self._defaults)
                result.update(overrides)
                return result
            except Exception as e:
                log.warning("feature_flags_get_all_failed", error=str(e))
        result = {}
        for flag_name in self._defaults:
            result[flag_name] = await self.is_enabled(flag_name, tenant_id)
        return result

    def _set_cache(self, key: str, value: bool) -> None:
        self._cache[key] = value
        if self.redis:
            try:
                import asyncio

                asyncio.create_task(
                    self.redis.setex(key, self._cache_ttl, "true" if value else "false")
                )
            except Exception as e:
                log.warning("feature_flags_redis_setex_failed", error=str(e))
