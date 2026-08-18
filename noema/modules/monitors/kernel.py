"""Monitoring & Observability Module — metrics, alerting, tracing, health checks."""

from __future__ import annotations

import contextlib
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class MetricType(StrEnum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertState(StrEnum):
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class Metric:
    name: str
    metric_type: MetricType
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    history: list[tuple[float, float]] = field(default_factory=list)

    def record(self, value: float, labels: dict[str, str] | None = None) -> None:
        if self.metric_type == MetricType.COUNTER:
            self.value += value
        elif self.metric_type == MetricType.GAUGE:
            self.value = value
        elif self.metric_type == MetricType.HISTOGRAM:
            self.value = (self.value + value) / 2 if self.value else value
        elif self.metric_type == MetricType.TIMER:
            self.value = value
        self.history.append((time.time(), self.value))
        if len(self.history) > 1000:
            self.history = self.history[-500:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "value": self.value,
            "labels": self.labels,
            "last_updated": self.timestamp,
        }


@dataclass
class AlertRule:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    metric_name: str = ""
    condition: str = ""  # gt, lt, eq, gte, lte
    threshold: float = 0.0
    severity: AlertSeverity = AlertSeverity.WARNING
    enabled: bool = True
    cooldown_seconds: float = 300.0
    last_fired: float = 0.0
    message_template: str = ""


@dataclass
class Alert:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    rule_id: str = ""
    rule_name: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    state: AlertState = AlertState.FIRING
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    resolved_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    name: str = ""
    status: str = "healthy"  # healthy, degraded, unhealthy
    latency_ms: float = 0.0
    message: str = ""
    last_check: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collect, query, and export metrics."""

    def __init__(self) -> None:
        self.metrics: dict[str, Metric] = {}
        self._counters: dict[str, float] = defaultdict(float)

    def counter(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        if key not in self.metrics:
            self.metrics[key] = Metric(
                name=name, metric_type=MetricType.COUNTER, labels=labels or {}
            )
        self.metrics[key].record(value)

    def gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        if key not in self.metrics:
            self.metrics[key] = Metric(name=name, metric_type=MetricType.GAUGE, labels=labels or {})
        self.metrics[key].record(value)

    def histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        if key not in self.metrics:
            self.metrics[key] = Metric(
                name=name, metric_type=MetricType.HISTOGRAM, labels=labels or {}
            )
        self.metrics[key].record(value)

    def timer(self, name: str, duration_ms: float, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        if key not in self.metrics:
            self.metrics[key] = Metric(name=name, metric_type=MetricType.TIMER, labels=labels or {})
        self.metrics[key].record(duration_ms)

    def get(self, name: str) -> Metric | None:
        return self.metrics.get(name)

    def get_all(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self.metrics.values()]

    def summary(self) -> dict[str, Any]:
        counters = sum(1 for m in self.metrics.values() if m.metric_type == MetricType.COUNTER)
        gauges = sum(1 for m in self.metrics.values() if m.metric_type == MetricType.GAUGE)
        histograms = sum(1 for m in self.metrics.values() if m.metric_type == MetricType.HISTOGRAM)
        timers = sum(1 for m in self.metrics.values() if m.metric_type == MetricType.TIMER)
        return {
            "total_metrics": len(self.metrics),
            "counters": counters,
            "gauges": gauges,
            "histograms": histograms,
            "timers": timers,
        }

    def _key(self, name: str, labels: dict[str, str] | None) -> str:
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items())) if labels else ""
        return f"{name}{{{label_str}}}" if label_str else name


class AlertEngine:
    """Evaluate alert rules against metrics, fire alerts."""

    def __init__(self) -> None:
        self.rules: list[AlertRule] = []
        self.alerts: list[Alert] = []
        self._handlers: list[Callable] = []

    def add_rule(self, rule: AlertRule) -> None:
        self.rules.append(rule)

    def on_alert(self, handler: Callable) -> None:
        self._handlers.append(handler)

    def evaluate(self, metrics: MetricsCollector) -> list[Alert]:
        fired: list[Alert] = []
        now = time.time()
        for rule in self.rules:
            if not rule.enabled:
                continue
            if now - rule.last_fired < rule.cooldown_seconds:
                continue
            metric = metrics.get(rule.metric_name)
            if not metric:
                continue
            triggered = False
            if (
                rule.condition == "gt"
                and metric.value > rule.threshold
                or rule.condition == "lt"
                and metric.value < rule.threshold
                or rule.condition == "gte"
                and metric.value >= rule.threshold
                or rule.condition == "lte"
                and metric.value <= rule.threshold
                or rule.condition == "eq"
                and metric.value == rule.threshold
            ):
                triggered = True
            if triggered:
                alert = Alert(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=rule.message_template.format(
                        metric=rule.metric_name, value=metric.value, threshold=rule.threshold
                    ),
                )
                self.alerts.append(alert)
                # Bounded history: fired alerts must never grow without limit.
                if len(self.alerts) > 1000:
                    self.alerts = self.alerts[-1000:]
                fired.append(alert)
                rule.last_fired = now
                for handler in self._handlers:
                    with contextlib.suppress(Exception):
                        handler(alert)
        return fired

    def get_active_alerts(self) -> list[Alert]:
        return [a for a in self.alerts if a.state == AlertState.FIRING]

    def resolve_alert(self, alert_id: str) -> bool:
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.state = AlertState.RESOLVED
                alert.resolved_at = time.time()
                return True
        return False

    def summary(self) -> dict[str, Any]:
        active = sum(1 for a in self.alerts if a.state == AlertState.FIRING)
        resolved = sum(1 for a in self.alerts if a.state == AlertState.RESOLVED)
        return {
            "total_rules": len(self.rules),
            "active_alerts": active,
            "resolved_alerts": resolved,
            "total_alerts": len(self.alerts),
        }


class HealthChecker:
    """Run health checks on services."""

    def __init__(self) -> None:
        self.checks: dict[str, Callable] = {}
        self.results: dict[str, HealthCheck] = {}

    def register(self, name: str, check_fn: Callable) -> None:
        self.checks[name] = check_fn

    async def run_all(self) -> dict[str, HealthCheck]:
        for name, check_fn in self.checks.items():
            start = time.time()
            try:
                result = await check_fn() if callable(check_fn) else check_fn
                latency = (time.time() - start) * 1000
                self.results[name] = HealthCheck(
                    name=name,
                    status="healthy",
                    latency_ms=latency,
                    metadata=result if isinstance(result, dict) else {},
                )
            except Exception as e:
                self.results[name] = HealthCheck(
                    name=name,
                    status="unhealthy",
                    latency_ms=(time.time() - start) * 1000,
                    message=str(e),
                )
        return self.results

    def overall_status(self) -> str:
        if not self.results:
            return "unknown"
        statuses = [r.status for r in self.results.values()]
        if all(s == "healthy" for s in statuses):
            return "healthy"
        if any(s == "unhealthy" for s in statuses):
            return "unhealthy"
        return "degraded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall_status(),
            "checks": {
                name: {"status": hc.status, "latency_ms": hc.latency_ms, "message": hc.message}
                for name, hc in self.results.items()
            },
        }


class MonitoringModule:
    """Standalone monitoring module — metrics, alerts, health checks."""

    NAME = "monitoring"
    DESCRIPTION = "Observability: metrics, alerting, tracing, health checks"

    def __init__(self) -> None:
        self.metrics = MetricsCollector()
        self.alerts = AlertEngine()
        self.health = HealthChecker()

    def record_request(self, path: str, method: str, status: int, duration_ms: float) -> None:
        self.metrics.counter(
            "http_requests_total", labels={"path": path, "method": method, "status": str(status)}
        )
        self.metrics.timer("http_request_duration_ms", duration_ms, labels={"path": path})
        if status >= 500:
            self.metrics.counter("http_errors_total", labels={"path": path, "status": str(status)})

    def record_error(self, component: str, error_type: str) -> None:
        self.metrics.counter("errors_total", labels={"component": component, "type": error_type})

    def record_business_metric(self, name: str, value: float) -> None:
        self.metrics.gauge(f"business_{name}", value)

    def get_dashboard(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.summary(),
            "alerts": self.alerts.summary(),
            "health": self.health.to_dict(),
        }

    def execute(self, task: Any) -> dict[str, Any]:
        """Kernel-compatible execute method."""
        tags = getattr(task, "tags", [])
        return {
            "type": "monitoring",
            "dashboard": self.get_dashboard(),
            "recommended_metrics": self._suggest_metrics(tags),
            "_confidence": 0.85,
        }

    def _suggest_metrics(self, tags: list[str]) -> list[dict[str, str]]:
        suggestions = [
            {
                "name": "http_requests_total",
                "type": "counter",
                "description": "Total HTTP requests",
            },
            {"name": "http_request_duration_ms", "type": "timer", "description": "Request latency"},
            {"name": "errors_total", "type": "counter", "description": "Total errors"},
            {"name": "active_connections", "type": "gauge", "description": "Active connections"},
        ]
        if "database" in tags or "sql" in tags:
            suggestions.append(
                {"name": "db_query_duration_ms", "type": "timer", "description": "DB query latency"}
            )
            suggestions.append(
                {
                    "name": "db_connections_active",
                    "type": "gauge",
                    "description": "Active DB connections",
                }
            )
        if "queue" in tags or "async" in tags:
            suggestions.append(
                {"name": "queue_depth", "type": "gauge", "description": "Queue depth"}
            )
            suggestions.append(
                {
                    "name": "queue_processed_total",
                    "type": "counter",
                    "description": "Processed messages",
                }
            )
        return suggestions
