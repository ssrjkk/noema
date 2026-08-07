"""Chaos: Redis failover — verify graceful degradation works."""

from noema.resilience.graceful_degradation import GracefulDegradation


class TestRedisFailover:
    async def test_graceful_degradation_detects_redis_down(self):
        """GracefulDegradation detects that Redis is unreachable."""
        deg = GracefulDegradation()

        # Simulate a broken redis client that raises on ping
        class _BrokenRedis:
            async def ping(self):
                raise ConnectionError("redis unreachable")

        deg.redis = _BrokenRedis()
        deg._redis_healthy = True
        await deg.check_health()
        status = deg.get_status()
        assert status.get("redis") == "degraded", "Should detect Redis is down"

    async def test_cache_fallback_to_memory_when_redis_down(self):
        """Cache operations fall back to in-memory dict when Redis is down."""
        deg = GracefulDegradation()
        deg._redis_healthy = False
        deg.redis = None

        # In-memory cache should still work
        await deg.cache_set("key1", "value1")
        val = await deg.cache_get("key1")
        assert val == "value1", "Should retrieve from in-memory cache"

        # get_status still works
        status = deg.get_status()
        assert "redis" in status
        assert status["redis"] == "degraded"
