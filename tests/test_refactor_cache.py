"""Coverage tests for ``noema.cache``, ``noema.embeddings`` and
``noema.persistence.redis_cache``.
"""

import pytest

from noema.cache import SemanticCache
from noema.embeddings import DenseEmbedder, HNSWIndex
from noema.persistence.redis_cache import RedisBackedCache

# ── DenseEmbedder ───────────────────────────────────────────────────────────


def test_embedder_dims_and_empty_input():
    emb = DenseEmbedder()
    assert emb.dim == 128
    assert emb.embed([]).shape == (0, 128)
    assert emb.embed_one("hello").shape == (1, 128)


def test_embedder_similar_prompts_cosine():
    emb = DenseEmbedder()
    a = emb.embed(["What is FastAPI?"])
    b = emb.embed(["what is fastapi?"])
    sim = DenseEmbedder.cosine_similarity(a, b)[0][0]
    assert sim > 0.99


def test_embedder_cosine_edge_cases():
    sim = DenseEmbedder.cosine_similarity(
        DenseEmbedder().embed_one("alpha beta"), DenseEmbedder().embed_one("alpha beta")
    )
    assert sim[0][0] > 0.99


# ── HNSWIndex ───────────────────────────────────────────────────────────────


def _hnsw_with_data() -> HNSWIndex:
    import numpy as np

    index = HNSWIndex(dim=4)
    vectors = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    index.add(["a", "b", "c"], vectors)
    return index


def test_hnsw_add_and_search_top_k():
    import numpy as np

    index = _hnsw_with_data()
    results = index.search(np.array([[1.0, 0.0, 0.0, 0.0]]), top_k=1)
    assert results[0][0] == "a"
    assert results[0][1] > 0.99


def test_hnsw_search_returns_all_when_k_larger():
    import numpy as np

    index = _hnsw_with_data()
    results = index.search(np.array([[0.0, 0.0, 1.0, 0.0]]), top_k=10)
    assert len(results) == 3
    assert results[0][0] == "c"


def test_hnsw_empty_and_reset():
    import numpy as np

    index = _hnsw_with_data()
    assert index.search(np.zeros((1, 4)), top_k=2) != []

    index.reset()
    assert index.size == 0
    assert index.search(np.zeros((1, 4)), top_k=2) == []

    empty = HNSWIndex(dim=4)
    assert empty.search(np.zeros((1, 4)), top_k=2) == []


# ── SemanticCache ───────────────────────────────────────────────────────────


def test_semantic_cache_exact_hit():
    cache = SemanticCache()
    messages = [{"role": "user", "content": "What is FastAPI?"}]
    assert cache.get(messages, model="m") is None
    cache.set(messages, response="FastAPI is a framework", model="m")
    assert cache.get(messages, model="m") == "FastAPI is a framework"

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_semantic_cache_tenant_isolation():
    cache = SemanticCache()
    messages = [{"role": "user", "content": "hello"}]
    cache.set(messages, response="resp", model="m", tenant_id="tenant-a")
    assert cache.get(messages, model="m", tenant_id="tenant-a") == "resp"
    assert cache.get(messages, model="m", tenant_id="tenant-b") is None


def test_semantic_cache_eviction():
    cache = SemanticCache(max_size=2)
    for i in range(3):
        cache.set([{"role": "user", "content": f"prompt {i}"}], response=f"resp {i}", model="m")

    stats = cache.stats()
    assert stats["size"] == 2
    assert stats["evictions"] == 1
    # oldest entry was evicted deterministically (min timestamp, FIFO tie-break)
    assert cache.get([{"role": "user", "content": "prompt 0"}], model="m") is None


def test_semantic_cache_semantic_hit():
    cache = SemanticCache()
    for i in range(11):
        cache.set(
            [{"role": "user", "content": f"What is fastapi feature {i}?"}],
            response=f"resp {i}",
            model="m",
        )

    # case-only change -> same embedding but different hash, triggers semantic search
    near = [{"role": "user", "content": "WHAT IS FASTAPI FEATURE 0?"}]
    assert cache.get(near, model="m") == "resp 0"

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0


def test_semantic_cache_clear():
    cache = SemanticCache()
    cache.set([{"role": "user", "content": "x"}], response="y", model="m")
    cache.clear()
    stats = cache.stats()
    assert stats["size"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["hit_rate"] == 0


def test_semantic_cache_cosine_edge_cases():
    assert SemanticCache._cosine_similarity([], []) == 0.0
    assert SemanticCache._cosine_similarity([1.0], [1.0, 2.0]) == 0.0
    assert SemanticCache._cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert SemanticCache._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


# ── RedisBackedCache ─────────────────────────────────────────────────────────


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value


@pytest.mark.asyncio
async def test_redis_cache_in_memory_fallback():
    cache = RedisBackedCache(redis_url="")
    assert cache.is_distributed is False

    messages = [{"role": "user", "content": "What is FastAPI?"}]
    assert await cache.aget(messages, model="m") is None
    await cache.aset(messages, response="FastAPI is a framework", model="m")
    assert await cache.aget(messages, model="m") == "FastAPI is a framework"


@pytest.mark.asyncio
async def test_redis_cache_async_path():
    cache = RedisBackedCache(redis_url="")
    fake = _FakeRedis()
    cache._redis = fake
    assert cache.is_distributed is True

    messages = [{"role": "user", "content": "hello from redis"}]
    await cache.aset(messages, response="redis resp", model="m")
    assert fake.store  # serialized entry landed in the fake redis store

    # evict the in-memory copy so aget must come from redis
    cache._entries.clear()
    assert await cache.aget(messages, model="m") == "redis resp"


@pytest.mark.asyncio
async def test_redis_cache_corrupt_entry_returns_none():
    cache = RedisBackedCache(redis_url="")
    fake = _FakeRedis()
    cache._redis = fake

    messages = [{"role": "user", "content": "hello from redis"}]
    await cache.aset(messages, response="redis resp", model="m")
    cache._entries.clear()

    # overwrite with garbage so deserialization fails and we fall back to None
    (k,) = fake.store.keys()
    fake.store[k] = "not-json{{{"
    assert await cache.aget(messages, model="m") is None
