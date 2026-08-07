"""Caching Module — multi-layer caching strategies, invalidation, warming."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CacheBackend(StrEnum):
    LRU = "lru"
    REDIS = "redis"
    MEMCACHED = "memcached"
    CDN = "cdn"
    BROWSER = "browser"
    IN_MEMORY = "in_memory"


class EvictionPolicy(StrEnum):
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    FIFO = "fifo"


@dataclass
class CacheEntry:
    key: str = ""
    value: Any = None
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 3600.0
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    tags: list[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_sets: int = 0
    total_gets: int = 0
    hit_rate: float = 0.0
    total_size_bytes: int = 0
    entry_count: int = 0


class LRUCache:
    """LRU cache with TTL and tag-based invalidation."""

    def __init__(self, max_size: int = 1000, default_ttl: float = 3600.0) -> None:
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._tag_index: dict[str, set[str]] = {}
        self.stats = CacheStats()

    def get(self, key: str) -> Any | None:
        self.stats.total_gets += 1
        entry = self._cache.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        if entry.is_expired:
            self.delete(key)
            self.stats.misses += 1
            return None
        entry.access_count += 1
        entry.last_accessed = time.time()
        self._cache.move_to_end(key)
        self.stats.hits += 1
        self._update_hit_rate()
        return entry.value

    def set(
        self, key: str, value: Any, ttl: float | None = None, tags: list[str] | None = None
    ) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self.max_size:
            self._evict()

        size = (
            len(json.dumps(value, default=str).encode())
            if not isinstance(value, (str, bytes))
            else len(str(value))
        )
        entry = CacheEntry(
            key=key,
            value=value,
            ttl_seconds=ttl or self.default_ttl,
            size_bytes=size,
            tags=tags or [],
        )
        self._cache[key] = entry
        self.stats.total_sets += 1

        for tag in tags or []:
            self._tag_index.setdefault(tag, set()).add(key)

    def delete(self, key: str) -> bool:
        entry = self._cache.pop(key, None)
        if entry:
            for tag in entry.tags:
                self._tag_index.get(tag, set()).discard(key)
            return True
        return False

    def invalidate_tag(self, tag: str) -> int:
        keys = list(self._tag_index.get(tag, set()))
        for key in keys:
            self.delete(key)
        self._tag_index.pop(tag, None)
        return len(keys)

    def clear(self) -> int:
        count = len(self._cache)
        self._cache.clear()
        self._tag_index.clear()
        return count

    def get_many(self, keys: list[str]) -> dict[str, Any]:
        return {
            k: self._cache[k].value
            for k in keys
            if k in self._cache and not self._cache[k].is_expired
        }

    def get_stats(self) -> dict[str, Any]:
        self.stats.entry_count = len(self._cache)
        self.stats.total_size_bytes = sum(e.size_bytes for e in self._cache.values())
        self._update_hit_rate()
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.stats.hit_rate,
            "evictions": self.stats.evictions,
            "entries": self.stats.entry_count,
            "size_bytes": self.stats.total_size_bytes,
            "max_size": self.max_size,
        }

    def _evict(self) -> None:
        if self._cache:
            key, _ = self._cache.popitem(last=False)
            self.stats.evictions += 1

    def _update_hit_rate(self) -> None:
        total = self.stats.hits + self.stats.misses
        self.stats.hit_rate = self.stats.hits / max(total, 1)


class MultiLayerCache:
    """Multi-layer cache: L1 (in-memory LRU) -> L2 (Redis-like) -> origin."""

    def __init__(self, l1_max: int = 500, l2_max: int = 5000) -> None:
        self.l1 = LRUCache(max_size=l1_max, default_ttl=60)
        self.l2 = LRUCache(max_size=l2_max, default_ttl=3600)

    def get(self, key: str) -> Any | None:
        val = self.l1.get(key)
        if val is not None:
            return val
        val = self.l2.get(key)
        if val is not None:
            self.l1.set(key, val, ttl=60)
            return val
        return None

    def set(
        self, key: str, value: Any, ttl: float | None = None, tags: list[str] | None = None
    ) -> None:
        self.l1.set(key, value, ttl=min(ttl or 60, 60), tags=tags)
        self.l2.set(key, value, ttl=ttl, tags=tags)

    def invalidate(self, key: str) -> None:
        self.l1.delete(key)
        self.l2.delete(key)

    def invalidate_tag(self, tag: str) -> int:
        c1 = self.l1.invalidate_tag(tag)
        c2 = self.l2.invalidate_tag(tag)
        return c1 + c2

    def get_stats(self) -> dict[str, Any]:
        return {"l1": self.l1.get_stats(), "l2": self.l2.get_stats()}


class CacheKeyBuilder:
    """Build consistent cache keys from parameters."""

    @staticmethod
    def build(*parts: str, prefix: str = "") -> str:
        key = ":".join(parts)
        if prefix:
            key = f"{prefix}:{key}"
        return key

    @staticmethod
    def for_user(user_id: str, resource: str) -> str:
        return f"user:{user_id}:{resource}"

    @staticmethod
    def for_api(method: str, path: str, params: dict[str, Any] | None = None) -> str:
        param_hash = ""
        if params:
            param_hash = hashlib.md5(
                json.dumps(params, sort_keys=True, default=str).encode()
            ).hexdigest()[:8]
        return f"api:{method.lower()}:{path}:{param_hash}"


class CachingModule:
    """Standalone caching module."""

    NAME = "caching"
    DESCRIPTION = "Multi-layer caching, TTL, invalidation, warming, cache strategies"

    def __init__(self) -> None:
        self.cache = MultiLayerCache()
        self.key_builder = CacheKeyBuilder()

    def get_strategy(self, tags: list[str]) -> list[dict[str, str]]:
        strategies = []
        if "api" in tags:
            strategies.append({"layer": "L1+L2", "ttl": "60s/1h", "invalidation": "tag-based"})
        if "database" in tags or "sql" in tags:
            strategies.append({"layer": "L2", "ttl": "5m", "invalidation": "write-through"})
        if "static" in tags or "cdn" in tags:
            strategies.append({"layer": "CDN", "ttl": "24h", "invalidation": "versioned URLs"})
        if "session" in tags:
            strategies.append({"layer": "L2+Redis", "ttl": "30m", "invalidation": "on logout"})
        if not strategies:
            strategies = [
                {"layer": "L1 (LRU in-memory)", "ttl": "60s", "use": "Hot data, session"},
                {"layer": "L2 (Redis)", "ttl": "1h", "use": "Shared cache, distributed"},
                {"layer": "CDN", "ttl": "24h", "use": "Static assets, API responses"},
            ]
        return strategies

    def execute(self, task: Any) -> dict[str, Any]:
        tags = getattr(task, "tags", [])
        return {
            "type": "caching",
            "strategies": self.get_strategy(tags),
            "cache_stats": self.cache.get_stats(),
            "_confidence": 0.85,
        }
