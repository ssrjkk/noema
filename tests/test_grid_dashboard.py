"""Tests for the grid dashboard (T3.4): heartbeats → scrape → aggregate."""

from __future__ import annotations

import asyncio

import fakeredis.aioredis
import pytest

from noema.observability.grid import (
    GridDashboard,
    NodeView,
    _fold_samples,
    parse_exposition,
)
from noema.workers.arq_worker import HEARTBEAT_PREFIX

METRICS_TEXT = """
# HELP noema_http_requests_total Total HTTP requests
# TYPE noema_http_requests_total counter
noema_http_requests_total{endpoint="/v1/think",method="POST",status="200"} 120.0
noema_http_requests_total{endpoint="/v1/think",method="POST",status="500"} 3.0
noema_llm_tokens_total{provider="openai"} 45678.0
noema_llm_tokens_total{provider="fallback"} 10.0
noema_llm_request_duration_seconds_count{provider="openai"} 42.0
noema_llm_request_duration_seconds_sum{provider="openai"} 8.4
"""


@pytest.fixture
def redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


async def _register_node(
    redis, node_id: str, hostname: str, port: int, draining: bool = False
) -> None:
    """Seed a live node's heartbeat hash directly (same fields ``_beat``
    writes) so the node stays visible for the dashboard."""
    import time as _time

    await redis.hset(
        f"{HEARTBEAT_PREFIX}{node_id}",
        mapping={
            "node_id": node_id,
            "hostname": hostname,
            "pid": "1",
            "started_at": "0",
            "last_heartbeat": str(int(_time.time())),
            "draining": "1" if draining else "0",
            "metrics_port": str(port),
        },
    )


async def _ok_fetch(url: str) -> str:
    return METRICS_TEXT


class TestParsing:
    def test_parse_exposition_counts_and_labels(self):
        samples = parse_exposition(METRICS_TEXT)
        req = samples["noema_http_requests_total"]
        assert len(req) == 2
        assert req[0]["labels"]["status"] == "200"
        assert req[0]["value"] == 120.0

    def test_fold_samples_view(self):
        view = NodeView()
        _fold_samples(view, parse_exposition(METRICS_TEXT))
        assert view.http_requests == 123
        assert view.http_errors == 3
        assert view.llm_tokens == 45688.0
        assert view.llm_calls == 42
        assert view.llm_latency_sum_s == pytest.approx(8.4)
        assert view.llm_latency_avg_ms == pytest.approx(8.4 / 42 * 1000, rel=1e-3)
        assert view.error_rate == pytest.approx(3 / 123, rel=1e-3)

    def test_parse_empty_and_garbage(self):
        assert parse_exposition("") == {}
        assert parse_exposition("# only a comment") == {}


class TestSnapshot:
    async def test_snapshot_aggregates_nodes_and_totals(self, redis):
        await _register_node(redis, "node-1", "host1", 9091)
        await _register_node(redis, "node-2", "host2", 9092, draining=True)

        dashboard = GridDashboard(redis=redis, fetch=_ok_fetch)
        try:
            snap = await dashboard.snapshot()
        finally:
            await dashboard.aclose()

        assert snap["totals"]["nodes_total"] == 2
        assert snap["totals"]["nodes_reachable"] == 2
        assert snap["totals"]["nodes_draining"] == 1
        assert snap["totals"]["http_requests"] == 123 * 2
        nodes = {n["node_id"]: n for n in snap["nodes"]}
        assert nodes["node-1"]["reachable"] is True
        assert nodes["node-2"]["draining"] is True
        assert nodes["node-1"]["metrics_url"] == "http://host1:9091/metrics"

    async def test_unreachable_node_reported_not_raised(self, redis):
        await _register_node(redis, "node-1", "host1", 9091)

        async def boom(url: str) -> str:
            raise OSError("connection refused")

        dashboard = GridDashboard(redis=redis, fetch=boom)
        try:
            snap = await dashboard.snapshot()
        finally:
            await dashboard.aclose()

        assert snap["totals"]["nodes_reachable"] == 0
        node = snap["nodes"][0]
        assert node["reachable"] is False
        assert "connection refused" in node["error"]

    async def test_node_without_metrics_port(self, redis):
        await _register_node(redis, "node-1", "host1", 0)
        dashboard = GridDashboard(redis=redis, fetch=_ok_fetch)
        try:
            snap = await dashboard.snapshot()
        finally:
            await dashboard.aclose()
        assert snap["nodes"][0]["reachable"] is False
        assert "no metrics endpoint" in snap["nodes"][0]["error"]

    async def test_dead_nodes_expire_out_of_view(self, redis):
        await _register_node(redis, "node-1", "host1", 9091)
        # Simulate TTL expiry: the heartbeat key disappears.
        await redis.delete(f"{HEARTBEAT_PREFIX}node-1")
        dashboard = GridDashboard(redis=redis, fetch=_ok_fetch)
        try:
            snap = await dashboard.snapshot()
        finally:
            await dashboard.aclose()
        assert snap["nodes"] == []


class TestRouterRegistry:
    async def test_iter_snapshots_yields(self, redis):
        await _register_node(redis, "node-1", "host1", 9091)
        dashboard = GridDashboard(redis=redis, fetch=_ok_fetch)

        async def take_two() -> list[dict]:
            from noema.observability.grid import iter_grid_snapshots

            out = []
            async for snap in iter_grid_snapshots(dashboard, interval=0.01):
                out.append(snap)
                if len(out) == 2:
                    break
            return out

        try:
            snaps = await asyncio.wait_for(take_two(), timeout=5)
        finally:
            await dashboard.aclose()
        assert len(snaps) == 2
        assert snaps[0]["totals"]["nodes_total"] == 1
