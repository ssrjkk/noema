"""Tests for the grid API endpoint (T3.4): snapshot wiring + failure shape."""

from __future__ import annotations

import fakeredis.aioredis
from fastapi.testclient import TestClient

from noema.api.grid import set_dashboard
from noema.observability.grid import GridDashboard

METRICS_TEXT = (
    'noema_http_requests_total{endpoint="/v1/think",method="POST",status="200"} 10.0\n'
    'noema_http_requests_total{endpoint="/v1/think",method="POST",status="404"} 1.0\n'
    'noema_llm_tokens_total{provider="openai"} 500.0\n'
    'noema_llm_request_duration_seconds_count{provider="openai"} 5.0\n'
    'noema_llm_request_duration_seconds_sum{provider="openai"} 1.0\n'
)


async def _ok_fetch(url: str) -> str:
    return METRICS_TEXT


def _client() -> TestClient:
    from noema.api.server import app

    return TestClient(app)


async def _seed_node(redis, node_id: str, hostname: str, port: int) -> None:
    import time

    await redis.hset(
        f"noema:workers:{node_id}",
        mapping={
            "node_id": node_id,
            "hostname": hostname,
            "pid": "1",
            "started_at": "0",
            "last_heartbeat": str(int(time.time())),
            "draining": "0",
            "metrics_port": str(port),
        },
    )


class TestGridEndpoint:
    def test_grid_snapshot_returns_nodes_and_totals(self):
        client = _client()
        redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        import asyncio

        async def _seed() -> None:
            await _seed_node(redis, "node-1", "host1", 9091)
            await _seed_node(redis, "node-2", "host2", 9092)

        asyncio.run(_seed())
        dashboard = GridDashboard(redis=redis, fetch=_ok_fetch)
        set_dashboard(dashboard)
        try:
            resp = client.get("/grid")
            assert resp.status_code == 200
            body = resp.json()
            assert body["totals"]["nodes_total"] == 2
            assert body["totals"]["nodes_reachable"] == 2
            nodes = {n["node_id"]: n for n in body["nodes"]}
            assert nodes["node-1"]["llm_tokens"] == 500.0
            assert nodes["node-1"]["http_errors"] == 1
        finally:
            set_dashboard(None)

    def test_grid_endpoint_never_500s(self):
        client = _client()

        class BrokenDashboard:
            async def snapshot(self) -> dict:
                raise RuntimeError("redis exploded")

        set_dashboard(BrokenDashboard())  # type: ignore[arg-type]
        try:
            resp = client.get("/grid")
            assert resp.status_code == 200
            body = resp.json()
            assert body["nodes"] == []
            assert "error" in body
        finally:
            set_dashboard(None)
