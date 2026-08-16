"""Prometheus metrics for Noema — with graceful fallback if not installed."""

from __future__ import annotations

from typing import Any

from noema.logging import get_logger

log = get_logger(__name__)


def _fallback_generate_latest(registry: Any = None, escaping: Any = None) -> bytes:
    """Return an empty payload when ``prometheus_client`` is unavailable."""
    return b""


try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

    class _NoopMetric:
        """Null-object metric used when ``prometheus_client`` is unavailable.

        Implements the same surface as the Prometheus ``Counter``/``Gauge``/
        ``Histogram``/``Info`` classes so callers need no feature detection.
        Every method returns a well-typed value; nothing is silently dropped.
        """

        def labels(self, *args: Any, **kwargs: Any) -> _NoopMetric:
            return self

        def inc(self, amount: float = 1, *args: Any, **kwargs: Any) -> None:
            return None

        def set(self, value: float, *args: Any, **kwargs: Any) -> None:
            return None

        def observe(self, amount: float, *args: Any, **kwargs: Any) -> None:
            return None

        def info(self, *args: Any, **kwargs: Any) -> None:
            return None

    Counter = Histogram = Gauge = _NoopMetric  # type: ignore[misc, assignment]
    Info = _NoopMetric  # type: ignore[misc, assignment]
    generate_latest = _fallback_generate_latest

    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    # Null-object metric constants so ``from noema.observability import ...``
    # and ``noema.services.modules`` keep importing when Prometheus is absent.
    APP_INFO: Any = _NoopMetric()
    REQUEST_COUNT: Any = _NoopMetric()
    REQUEST_LATENCY: Any = _NoopMetric()
    INFLIGHT_REQUESTS: Any = _NoopMetric()
    LLM_REQUEST_COUNT: Any = _NoopMetric()
    LLM_LATENCY: Any = _NoopMetric()
    LLM_TOKENS_USED: Any = _NoopMetric()
    LLM_CIRCUIT_STATE: Any = _NoopMetric()
    WORKER_ACTIVE: Any = _NoopMetric()
    WORKER_QUEUE_SIZE: Any = _NoopMetric()
    WORKER_TASK_COUNT: Any = _NoopMetric()
    WORKER_TASK_LATENCY: Any = _NoopMetric()
    MEMORY_EPISODIC_COUNT: Any = _NoopMetric()
    MEMORY_SEMANTIC_COUNT: Any = _NoopMetric()
    MEMORY_PROCEDURAL_COUNT: Any = _NoopMetric()
    KNOWLEDGE_ENTRIES: Any = _NoopMetric()
    EVOLUTION_PATCHES: Any = _NoopMetric()
    MODULE_EXECUTION_COUNT: Any = _NoopMetric()
    MODULE_LATENCY: Any = _NoopMetric()
    PR_COST_USD: Any = _NoopMetric()
    CODE_COST_PER_MODULE: Any = _NoopMetric()

    log.info("prometheus_client_not_installed_metrics_disabled")


if _HAS_PROMETHEUS:
    APP_INFO = Info("noema", "Noema metadata")
    APP_INFO.info({"version": "1.0.0", "python": "3.11+"})

    REQUEST_COUNT = Counter(
        "noema_http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
    )
    REQUEST_LATENCY = Histogram(
        "noema_http_request_duration_seconds",
        "Request latency in seconds",
        ["method", "endpoint"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    INFLIGHT_REQUESTS = Gauge("noema_http_inflight_requests", "Currently processing requests")
    LLM_REQUEST_COUNT = Counter(
        "noema_llm_requests_total", "Total LLM requests", ["provider", "status"]
    )
    LLM_LATENCY = Histogram(
        "noema_llm_request_duration_seconds",
        "LLM request latency",
        ["provider"],
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    LLM_TOKENS_USED = Counter("noema_llm_tokens_total", "Total LLM tokens consumed", ["provider"])
    LLM_CIRCUIT_STATE = Gauge(
        "noema_llm_circuit_state",
        "Circuit breaker state (0=closed, 1=open, 2=half_open)",
        ["provider"],
    )
    WORKER_ACTIVE = Gauge("noema_worker_active", "Active workers")
    WORKER_QUEUE_SIZE = Gauge("noema_worker_queue_size", "Tasks waiting in queue")
    WORKER_TASK_COUNT = Counter("noema_worker_tasks_total", "Total tasks processed", ["status"])
    WORKER_TASK_LATENCY = Histogram(
        "noema_worker_task_duration_seconds",
        "Task processing latency",
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
    )
    MEMORY_EPISODIC_COUNT = Gauge("noema_memory_episodic_count", "Episodic memories")
    MEMORY_SEMANTIC_COUNT = Gauge("noema_memory_semantic_count", "Semantic memories")
    MEMORY_PROCEDURAL_COUNT = Gauge("noema_memory_procedural_count", "Procedural memories")
    KNOWLEDGE_ENTRIES = Gauge("noema_knowledge_entries", "Knowledge base entries")
    EVOLUTION_PATCHES = Counter(
        "noema_evolution_patches_total", "Total evolution patches applied", ["status"]
    )
    MODULE_EXECUTION_COUNT = Counter(
        "noema_module_executions_total", "Module executions", ["module", "status"]
    )
    MODULE_LATENCY = Histogram(
        "noema_module_execution_duration_seconds",
        "Module execution latency",
        ["module"],
        buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
    )
    PR_COST_USD = Counter(
        "noema_pr_cost_usd",
        "USD spent generating a pull request, attributed per changed module",
        ["repo", "pr", "module"],
    )
    CODE_COST_PER_MODULE = Gauge(
        "noema_code_cost_per_module",
        "Latest generation cost per code module (line-weighted attribution)",
        ["repo", "module"],
    )


def build_metrics_app() -> Any:
    """Standalone ASGI app exposing ``/metrics`` and ``/health``.

    Served on ``settings.obs.metrics_port`` (default 9090) so the Prometheus
    scrape target matches the advertised port.
    """
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    async def metrics(_: Any) -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    async def health(_: Any) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    return Starlette(routes=[Route("/metrics", metrics), Route("/health", health)])


def spawn_metrics_server(port: int, host: str = "0.0.0.0") -> Any | None:
    """Start the Prometheus exporter on its own port in a daemon thread.

    Best-effort: returns ``None`` when ``prometheus_client`` is unavailable;
    the caller must never crash because metrics could not start.
    """
    if not _HAS_PROMETHEUS:
        return None
    import threading

    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(
            build_metrics_app(),
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
    )
    thread = threading.Thread(
        target=server.run,
        daemon=True,
        name="noema-metrics",
    )
    thread.start()
    log.info("metrics_server_started", port=port)
    return server


def update_system_gauges(
    worker_stats: dict[str, Any] | None = None,
    knowledge_stats: dict[str, Any] | None = None,
) -> None:
    """Refresh aggregate gauges from live engine state (best-effort)."""
    if not _HAS_PROMETHEUS:
        return
    try:
        if worker_stats:
            WORKER_ACTIVE.set(int(worker_stats.get("workers_busy", 0) or 0))
            WORKER_QUEUE_SIZE.set(int(worker_stats.get("queue_size", 0) or 0))
        if knowledge_stats:
            total = knowledge_stats.get("total_entries") or knowledge_stats.get("entries") or 0
            KNOWLEDGE_ENTRIES.set(int(total))
    except Exception as e:
        log.debug("metrics_update_failed", error=str(e))
