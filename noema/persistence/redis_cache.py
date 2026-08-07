"""Redis-backed cache with in-memory fallback — для масштабирования в K8s."""

from __future__ import annotations

import contextlib
import json
from typing import Any, cast

from noema.cache import SemanticCache
from noema.logging import get_logger

log = get_logger(__name__)


class RedisBackedCache(SemanticCache):
    """SemanticCache with Redis backend for cross-pod cache sharing.

    Falls back to in-memory when Redis is unavailable.
    """

    def __init__(
        self,
        max_size: int = 500,
        similarity_threshold: float = 0.92,
        tenant_id: str = "default",
        redis_url: str = "",
        redis_prefix: str = "noema:cache:",
    ) -> None:
        super().__init__(max_size=max_size, similarity_threshold=similarity_threshold)
        self._redis_url = redis_url
        self._redis_prefix = redis_prefix
        self._redis: Any = None
        self.tenant_id = tenant_id
        self._try_connect()

    def _try_connect(self) -> None:
        if not self._redis_url:
            return
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
            log.info("redis_cache_connected", url=self._redis_url)
        except ImportError:
            log.warning("redis_cache_no_redis_driver")
        except Exception as e:
            log.warning("redis_cache_connect_failed", error=str(e))

    @property
    def is_distributed(self) -> bool:
        return self._redis is not None

    async def _redis_get(self, key: str) -> str | None:
        if not self._redis:
            return None
        try:
            value = await self._redis.get(self._redis_prefix + key)
            return cast("str | None", value)
        except Exception:
            return None

    async def _redis_set(self, key: str, value: str, ttl: int = 3600) -> None:
        if not self._redis:
            return
        with contextlib.suppress(Exception):
            await self._redis.setex(self._redis_prefix + key, ttl, value)

    async def aget(
        self, messages: list[dict[str, str]], model: str, tenant_id: str = ""
    ) -> str | None:
        result = self.get(messages, model, tenant_id=tenant_id)
        if result is not None:
            return result
        if self._redis:
            try:
                key = self._hash_prompt(messages) + f":t={tenant_id or self.tenant_id}:m={model}"
                cached = await self._redis_get(key)
                if cached:
                    self._entries[key] = self._deserialize_entry(cached)
                    return self._entries[key].response
            except Exception as e:
                log.debug("redis_cache_get_failed_falling_back_to_memory", error=str(e))
        return None

    async def aset(
        self,
        messages: list[dict[str, str]],
        response: str,
        model: str,
        tokens_used: int = 0,
        tenant_id: str = "",
        ttl: int = 3600,
    ) -> None:
        self.set(messages, response, model, tokens_used, tenant_id=tenant_id)
        if self._redis:
            try:
                key = self._hash_prompt(messages) + f":t={tenant_id or self.tenant_id}:m={model}"
                entry = self._entries.get(key)
                if entry:
                    await self._redis_set(key, self._serialize_entry(entry), ttl=ttl)
            except Exception as e:
                log.debug("redis_cache_set_failed_keeping_memory_copy", error=str(e))

    def _serialize_entry(self, entry: Any) -> str:
        return json.dumps(
            {
                "prompt_hash": entry.prompt_hash,
                "response": entry.response,
                "model": entry.model,
                "tokens_used": entry.tokens_used,
                "timestamp": entry.timestamp,
                "hit_count": entry.hit_count,
                "embedding": entry.embedding[:20],
            }
        )

    def _deserialize_entry(self, data: str) -> Any:
        from noema.cache import CacheEntry

        d = json.loads(data)
        return CacheEntry(
            prompt_hash=d["prompt_hash"],
            response=d["response"],
            model=d["model"],
            tokens_used=d.get("tokens_used", 0),
            timestamp=d.get("timestamp", 0.0),
            hit_count=d.get("hit_count", 1),
            embedding=d.get("embedding", []),
        )
