"""Graceful Degradation — transparent fallback when Redis/PostgreSQL are down."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from noema.logging import get_logger

log = get_logger(__name__)


class GracefulDegradation:
    """Transparent fallback for infrastructure services.

    Redis down → in-memory cache
    PostgreSQL down → file-based storage
    """

    def __init__(self, redis_client: Any = None, pg_pool: Any = None) -> None:
        self.redis = redis_client
        self.pg = pg_pool
        self._memory_cache: dict[str, Any] = {}
        self._redis_healthy: bool = redis_client is not None
        self._pg_healthy: bool = pg_pool is not None
        self._fallback_dir = Path(".noema/fallback")
        self._fallback_dir.mkdir(parents=True, exist_ok=True)

    async def check_health(self) -> None:
        if self.redis:
            try:
                await asyncio.wait_for(self.redis.ping(), timeout=3.0)
                if not self._redis_healthy:
                    log.info("redis_recovered")
                self._redis_healthy = True
            except Exception as e:
                if self._redis_healthy:
                    log.warning("redis_down_fallback_to_memory", error=str(e))
                self._redis_healthy = False

        if self.pg:
            try:
                await asyncio.wait_for(self.pg.fetchval("SELECT 1"), timeout=3.0)
                if not self._pg_healthy:
                    log.info("postgres_recovered")
                self._pg_healthy = True
            except Exception as e:
                if self._pg_healthy:
                    log.warning("postgres_down_fallback_to_file", error=str(e))
                self._pg_healthy = False

    async def cache_get(self, key: str) -> Any | None:
        if self._redis_healthy and self.redis:
            try:
                return await self.redis.get(key)
            except Exception:
                self._redis_healthy = False
                log.warning("redis_cache_get_failed")
        return self._memory_cache.get(key)

    async def cache_set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self._memory_cache[key] = value
        if self._redis_healthy and self.redis:
            try:
                if ttl:
                    await self.redis.setex(key, ttl, value)
                else:
                    await self.redis.set(key, value)
            except Exception:
                self._redis_healthy = False
                log.warning("redis_cache_set_failed")

    async def cache_delete(self, key: str) -> None:
        self._memory_cache.pop(key, None)
        if self._redis_healthy and self.redis:
            try:
                await self.redis.delete(key)
            except Exception:
                self._redis_healthy = False

    async def db_execute(self, query: str, *args: Any) -> Any:
        if not self._pg_healthy or not self.pg:
            await self._write_fallback(query, args)
            return None
        try:
            return await self.pg.execute(query, *args)
        except Exception as e:
            self._pg_healthy = False
            log.warning("postgres_execute_failed_fallback_to_file", error=str(e))
            await self._write_fallback(query, args)
            return None

    async def db_fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        if self._pg_healthy and self.pg:
            try:
                rows = await self.pg.fetch(query, *args)
                return [dict(r) for r in rows]
            except Exception:
                self._pg_healthy = False
                log.warning("postgres_fetch_failed")
        return []

    async def _write_fallback(self, query: str, args: tuple[Any, ...]) -> None:
        ts = datetime.now(UTC).isoformat()
        filename = self._fallback_dir / f"query_{ts.replace(':', '-')}.json"
        data = {"query": query, "args": [str(a) for a in args], "timestamp": ts}
        try:
            filename.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info("fallback_query_written", path=str(filename))
        except Exception as e:
            log.error("fallback_write_failed", error=str(e))

    def get_status(self) -> dict[str, Any]:
        return {
            "redis": "healthy" if self._redis_healthy else "degraded",
            "postgresql": "healthy" if self._pg_healthy else "degraded",
            "memory_cache_size": len(self._memory_cache),
            "fallback_dir": str(self._fallback_dir),
        }
