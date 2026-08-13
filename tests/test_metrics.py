"""Tests for the Prometheus metrics exporter and gauge updates."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from noema.observability.metrics import (
    KNOWLEDGE_ENTRIES,
    WORKER_ACTIVE,
    WORKER_QUEUE_SIZE,
    build_metrics_app,
    update_system_gauges,
)


def _gauge_value(metric) -> float:
    return metric.collect()[0].samples[0].value


class TestMetricsApp:
    async def test_metrics_endpoint_returns_prometheus_text(self):
        app = build_metrics_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        assert b"noema_http_requests_total" in resp.content

    async def test_health_endpoint(self):
        app = build_metrics_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    async def test_unknown_route_404(self):
        app = build_metrics_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/nope")
        assert resp.status_code == 404


class TestUpdateSystemGauges:
    def test_updates_worker_and_knowledge_gauges(self):
        update_system_gauges(
            worker_stats={"workers_busy": 3, "queue_size": 7},
            knowledge_stats={"entries": 42},
        )
        assert _gauge_value(WORKER_ACTIVE) == 3
        assert _gauge_value(WORKER_QUEUE_SIZE) == 7
        assert _gauge_value(KNOWLEDGE_ENTRIES) == 42

    def test_empty_stats_are_safe(self):
        update_system_gauges(None, None)
        update_system_gauges({}, {})

    def test_noop_when_missing_keys(self):
        update_system_gauges({"unrelated": 1})


@pytest.fixture(autouse=True)
def _reset_gauges():
    yield
    for g in (WORKER_ACTIVE, WORKER_QUEUE_SIZE, KNOWLEDGE_ENTRIES):
        g.set(0)
