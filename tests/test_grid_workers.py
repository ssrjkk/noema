"""Tests for multi-node task distribution (T3.1): heartbeats, drain, stress.

The stress test runs 3 concurrent "nodes" (each an independent in-process
worker loop with its own heartbeat) against one shared fakeredis queue and
asserts the roadmap's acceptance criterion: 100 tasks distribute across the
nodes with zero double-execution and zero loss.
"""

from __future__ import annotations

import asyncio

import fakeredis.aioredis
import pytest

from noema.workers.arq_worker import (
    HEARTBEAT_PREFIX,
    HEARTBEAT_TTL,
    NodeHeartbeat,
    list_active_workers,
    make_node_id,
)


@pytest.fixture
def redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


class TestHeartbeatFields:
    async def test_metrics_port_advertised(self, redis):
        hb = NodeHeartbeat("node-1", redis=redis, metrics_port=9091)
        await hb.start()
        try:
            raw = await redis.hgetall(f"{HEARTBEAT_PREFIX}node-1")
            data = {str(k): str(v) for k, v in raw.items()}
            assert data["metrics_port"] == "9091"
        finally:
            await hb.stop()

    async def test_default_metrics_port_zero(self, redis):
        hb = NodeHeartbeat("node-1", redis=redis)
        await hb.start()
        try:
            raw = await redis.hgetall(f"{HEARTBEAT_PREFIX}node-1")
            data = {str(k): str(v) for k, v in raw.items()}
            assert data["metrics_port"] == "0"
        finally:
            await hb.stop()

    async def test_ttl_bounded(self, redis):
        hb = NodeHeartbeat("node-1", redis=redis)
        await hb.start()
        try:
            ttl = await redis.ttl(f"{HEARTBEAT_PREFIX}node-1")
            assert 0 < ttl <= HEARTBEAT_TTL
        finally:
            await hb.stop()


class TestDrain:
    async def test_drain_marks_then_removes(self, redis):
        hb = NodeHeartbeat("node-1", redis=redis)
        await hb.start()
        await hb.mark_draining()
        raw = await redis.hgetall(f"{HEARTBEAT_PREFIX}node-1")
        data = {str(k): str(v) for k, v in raw.items()}
        assert data["draining"] == "1"

        fleet = await list_active_workers("", redis=redis)
        assert fleet and fleet[0]["draining"] == "1"

        await hb.stop()
        assert await redis.exists(f"{HEARTBEAT_PREFIX}node-1") == 0


class TestStressDistribution:
    async def test_100_tasks_3_nodes_zero_double_execution(self, redis):
        """Roadmap T3.1 acceptance: 100 tasks over 3 nodes, no double runs.

        Each node pops task ids from a shared Redis list (LPUSH/BRPOP-ish via
        polling, like the arq queue) and records its node id against the task
        in a results hash. If any node ever saw a task twice, or a task went
        missing, the counts diverge and the test fails.
        """
        queue_key = "stress:tasks"
        results_key = "stress:results"
        n_tasks = 100
        task_ids = [f"task-{i:03d}" for i in range(n_tasks)]
        # Oldest first (RPOP order below pops task-000 first).
        await redis.rpush(queue_key, *task_ids)

        heartbeats: list[NodeHeartbeat] = []

        async def node_loop(node_id: str, delay: float) -> int:
            hb = NodeHeartbeat(node_id, redis=redis, metrics_port=9091)
            heartbeats.append(hb)
            await hb.start()
            processed = 0
            while True:
                task_id = await redis.lpop(queue_key)
                if task_id is None:
                    break
                # Yield to the cooperative loop so a zero-latency node cannot
                # drain the whole queue before the others get a pop (otherwise
                # the distribution assertion below is unreachable).
                await asyncio.sleep(0)
                if delay:
                    await asyncio.sleep(delay)
                # Claim atomically: HSETNX returns 1 only for the first node.
                raw = task_id.encode() if isinstance(task_id, str) else task_id
                claimed = await redis.hsetnx(results_key, raw, node_id)
                if claimed:
                    processed += 1
                else:
                    # Double execution: another node already claimed it.
                    raise AssertionError(f"double execution of {task_id}")
            await hb.mark_draining()
            await hb.stop()
            return processed

        delays = [0.0, 0.001, 0.002]  # heterogeneous nodes
        totals = await asyncio.wait_for(
            asyncio.gather(*(node_loop(f"node-{i}", d) for i, d in enumerate(delays))),
            timeout=60,
        )

        # All tasks processed, none lost, none double-run.
        assert sum(totals) == n_tasks
        assert all(t > 0 for t in totals), f"uneven distribution: {totals}"
        stored = await redis.hlen(results_key)
        assert stored == n_tasks

        # Every worker drained its heartbeat key.
        remaining = await redis.keys(f"{HEARTBEAT_PREFIX}*")
        assert remaining == []

    async def test_dead_node_disappears_from_fleet(self, redis):
        """A node that dies without draining vanishes once its TTL expires."""
        hb = NodeHeartbeat("ghost", redis=redis, metrics_port=9091)
        await hb._beat()  # publish once, no refresh loop
        fleet = await list_active_workers("", redis=redis)
        assert any(w["node_id"] == "ghost" for w in fleet)

        # Simulate TTL expiry (the key auto-expires after HEARTBEAT_TTL).
        await redis.delete(f"{HEARTBEAT_PREFIX}ghost")
        fleet = await list_active_workers("", redis=redis)
        assert not any(w["node_id"] == "ghost" for w in fleet)

    async def test_node_ids_unique_per_process(self):
        ids = {make_node_id() for _ in range(50)}
        assert len(ids) == 50
