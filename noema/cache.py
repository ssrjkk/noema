"""Semantic Cache — кэширование LLM-ответов через dense embedding similarity."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, cast

from noema.context import get_tenant_id
from noema.embeddings import DenseEmbedder
from noema.logging import get_logger

log = get_logger(__name__)


@dataclass
class CacheEntry:
    prompt_hash: str
    response: str
    model: str
    tokens_used: int = 0
    timestamp: float = 0.0
    hit_count: int = 1
    embedding: list[float] = field(default_factory=list)


class SemanticCache:
    def __init__(self, max_size: int = 500, similarity_threshold: float = 0.92) -> None:
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self._tenant_id = get_tenant_id()
        self._entries: dict[str, CacheEntry] = {}
        self._embedder = DenseEmbedder()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _hash_prompt(self, messages: list[dict[str, str]]) -> str:
        content = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _prompt_text(self, messages: list[dict[str, str]]) -> str:
        return " ".join(m.get("content", "") for m in messages)

    def _compute_embedding(self, text: str) -> list[float]:
        vec = self._embedder.embed_one(text).flatten()
        return vec.tolist()

    def get(self, messages: list[dict[str, str]], model: str, tenant_id: str = "") -> str | None:
        effective_tenant = tenant_id or get_tenant_id()
        prompt_hash = self._hash_prompt(messages) + f":t={effective_tenant}"

        if prompt_hash in self._entries:
            entry = self._entries[prompt_hash]
            entry.hit_count += 1
            self._stats["hits"] += 1
            log.debug("cache_exact_hit", key=prompt_hash[:8], model=model)
            return entry.response

        if self._entries and len(self._entries) > 10:
            best_sim = 0.0
            best_key = None
            query_text = self._prompt_text(messages)
            qvec = self._compute_embedding(query_text)
            tenant_suffix = f":t={effective_tenant}"
            for key, entry in self._entries.items():
                # Exact hits are tenant-scoped by key; semantic matches must
                # never serve a response cached for a different tenant.
                if not key.endswith(tenant_suffix):
                    continue
                if entry.embedding and entry.model == model:
                    sim = self._cosine_similarity(qvec, entry.embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_key = key

            if best_sim >= self.similarity_threshold and best_key:
                entry = self._entries[best_key]
                entry.hit_count += 1
                self._stats["hits"] += 1
                log.debug("cache_semantic_hit", key=best_key[:8], similarity=round(best_sim, 3))
                return entry.response

        self._stats["misses"] += 1
        return None

    def set(
        self,
        messages: list[dict[str, str]],
        response: str,
        model: str,
        tokens_used: int = 0,
        tenant_id: str = "",
    ) -> None:
        if len(self._entries) >= self.max_size:
            oldest = min(self._entries.keys(), key=lambda k: self._entries[k].timestamp)
            del self._entries[oldest]
            self._stats["evictions"] += 1

        effective_tenant = tenant_id or get_tenant_id()
        prompt_hash = self._hash_prompt(messages) + f":t={effective_tenant}"
        prompt_text = self._prompt_text(messages)
        embedding = self._compute_embedding(prompt_text)
        self._entries[prompt_hash] = CacheEntry(
            prompt_hash=prompt_hash,
            response=response,
            model=model,
            tokens_used=tokens_used,
            timestamp=time.time(),
            embedding=embedding,
        )

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return cast("float", dot / (norm_a * norm_b))

    def clear(self) -> None:
        self._entries.clear()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def stats(self) -> dict[str, Any]:
        total = self._stats["hits"] + self._stats["misses"]
        return {
            "size": len(self._entries),
            "max_size": self.max_size,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": round(self._stats["hits"] / total, 3) if total > 0 else 0,
            "evictions": self._stats["evictions"],
            "threshold": self.similarity_threshold,
            "semantic": self._embedder.is_semantic,
        }


_cache: SemanticCache | None = None


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
