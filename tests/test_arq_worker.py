"""Tests for arq worker helpers: node identity, heartbeat, liveness listing."""

from __future__ import annotations

import asyncio

import fakeredis.aioredis
import pytest

from noema.workers.arq_worker import (
    HEARTBEAT_PREFIX,
    NodeHeartbeat,
    list_active_workers,
    make_node_id,
)


@pytest.fixture
def redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


async def _hgetall(r, key: str) -> dict[str, str]:
    raw = await r.hgetall(key)
    return {
        k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
        for k, v in raw.items()
    }


class TestNodeIdentity:
    def test_make_node_id_unique(self):
        ids = {make_node_id() for _ in range(100)}
        assert len(ids) == 100
        assert all(":" in i for i in ids)


class TestNodeHeartbeat:
    async def test_heartbeat_writes_liveness(self, redis):
        hb = NodeHeartbeat("node-1", redis=redis)
        await hb.start()
        try:
            data = await _hgetall(redis, f"{HEARTBEAT_PREFIX}node-1")
            assert data["node_id"] == "node-1"
            assert data["draining"] == "0"
            assert "started_at" in data
            ttl = await redis.ttl(f"{HEARTBEAT_PREFIX}node-1")
            assert 0 < ttl <= 15
        finally:
            await hb.stop()

    async def test_mark_draining_flag(self, redis):
        hb = NodeHeartbeat("node-1", redis=redis)
        await hb.start()
        try:
            await hb.mark_draining()
            data = await _hgetall(redis, f"{HEARTBEAT_PREFIX}node-1")
            assert data["draining"] == "1"
        finally:
            await hb.stop()

    async def test_stop_removes_key(self, redis):
        hb = NodeHeartbeat("node-1", redis=redis)
        await hb.start()
        await hb.stop()
        assert await redis.exists(f"{HEARTBEAT_PREFIX}node-1") == 0

    async def test_heartbeat_refreshes_ttl(self, redis):
        hb = NodeHeartbeat("node-1", redis=redis)
        await hb.start()
        try:
            await asyncio.sleep(0.1)
            await hb._beat()
            ttl = await redis.ttl(f"{HEARTBEAT_PREFIX}node-1")
            assert ttl > 0
        finally:
            await hb.stop()

    async def test_heartbeat_loop_runs(self, redis):
        hb = NodeHeartbeat("node-1", redis=redis)
        await hb.start()
        try:
            assert hb._task is not None and not hb._task.done()
        finally:
            await hb.stop()


class TestListWorkers:
    async def test_list_active_workers(self, redis):
        hb1 = NodeHeartbeat("a", redis=redis)
        hb2 = NodeHeartbeat("b", redis=redis)
        await hb1.start()
        await hb2.start()
        try:
            workers = await list_active_workers("", redis=redis)
            assert {w["node_id"] for w in workers} == {"a", "b"}
            assert all(w["draining"] == "0" for w in workers)
        finally:
            await hb1.stop()
            await hb2.stop()

    async def test_drained_worker_not_listed(self, redis):
        hb = NodeHeartbeat("gone", redis=redis)
        await hb.start()
        await hb.stop()
        workers = await list_active_workers("", redis=redis)
        assert workers == []
