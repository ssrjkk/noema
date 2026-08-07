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
