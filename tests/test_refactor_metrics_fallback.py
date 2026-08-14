"""Coverage for the metrics graceful-fallback path in noema.observability.metrics.

``prometheus_client`` is installed in this environment, so the ``_NoopMetric``
branch is normally dead code. It is exercised here by re-importing the module
with the ``prometheus_client`` import blocked.
"""

import builtins
import importlib
import sys

import pytest
from httpx import ASGITransport, AsyncClient

import noema.observability.metrics as metrics


def _fresh_metrics():
    return importlib.import_module("noema.observability.metrics")


@pytest.fixture
def no_prometheus(monkeypatch):
    original = sys.modules.pop("noema.observability.metrics", None)
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "prometheus_client":
            raise ImportError("prometheus_client blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    yield
    sys.modules["noema.observability.metrics"] = original


def test_fallback_disables_prometheus(no_prometheus):
    m2 = _fresh_metrics()

    assert m2._HAS_PROMETHEUS is False
    assert m2.Counter is m2._NoopMetric
    assert m2.generate_latest() == b""
    assert m2.CONTENT_TYPE_LATEST.startswith("text/plain")


def test_noop_metric_surface(no_prometheus):
    m2 = _fresh_metrics()

    counter = m2._NoopMetric()
    counter.inc()
    counter.inc(2)
    counter.labels("a", "b").inc(3)
    counter.set(5)
    counter.observe(1.0)
    counter.info({"key": "value"})
    histogram = m2._NoopMetric()
    histogram.observe(0.5)
    gauge = m2._NoopMetric()
    gauge.set(42)
    gauge.labels("x").set(0)


def test_fallback_defines_all_metric_constants(no_prometheus):
    m2 = _fresh_metrics()

    names = [
        "APP_INFO",
        "REQUEST_COUNT",
        "REQUEST_LATENCY",
        "INFLIGHT_REQUESTS",
        "LLM_REQUEST_COUNT",
        "LLM_LATENCY",
        "LLM_TOKENS_USED",
        "LLM_CIRCUIT_STATE",
        "WORKER_ACTIVE",
        "WORKER_QUEUE_SIZE",
        "WORKER_TASK_COUNT",
        "WORKER_TASK_LATENCY",
        "MEMORY_EPISODIC_COUNT",
        "MEMORY_SEMANTIC_COUNT",
        "MEMORY_PROCEDURAL_COUNT",
        "KNOWLEDGE_ENTRIES",
        "EVOLUTION_PATCHES",
        "MODULE_EXECUTION_COUNT",
        "MODULE_LATENCY",
    ]
    for name in names:
        assert isinstance(getattr(m2, name), m2._NoopMetric)


def test_fallback_constant_chaining_is_noop(no_prometheus):
    m2 = _fresh_metrics()

    m2.REQUEST_COUNT.labels(method="GET", endpoint="/x", status="200").inc()
    m2.MODULE_EXECUTION_COUNT.labels(module="auth", status="success").inc()
    m2.WORKER_ACTIVE.set(3)
    m2.APP_INFO.info({"version": "1.0.0"})


def test_fallback_keeps_services_modules_importable(no_prometheus):
    import importlib

    modules = importlib.import_module("noema.services.modules")
    assert modules is not None


@pytest.mark.asyncio
async def test_fallback_metrics_app_returns_empty_payload(no_prometheus):
    m2 = _fresh_metrics()

    app = m2.build_metrics_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        metrics_resp = await client.get("/metrics")
        health_resp = await client.get("/health")
        nope = await client.get("/nope")
    assert metrics_resp.status_code == 200
    assert metrics_resp.content == b""
    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "ok"}
    assert nope.status_code == 404


def test_spawn_metrics_server_returns_none_without_prometheus(no_prometheus):
    m2 = _fresh_metrics()

    assert m2.spawn_metrics_server(9090) is None


def test_update_system_gauges_is_noop_without_prometheus(no_prometheus):
    m2 = _fresh_metrics()

    m2.update_system_gauges({"workers_busy": 3, "queue_size": 7}, {"entries": 42})


def test_prometheus_path_active_in_default_env():
    assert metrics._HAS_PROMETHEUS is True
    assert metrics.generate_latest() != b""
