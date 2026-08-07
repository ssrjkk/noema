"""Module service — wraps ModuleRegistry with events + metrics."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from noema.core.events import schedule_event
from noema.logging import get_logger
from noema.observability.metrics import MODULE_EXECUTION_COUNT, MODULE_LATENCY

if TYPE_CHECKING:
    from noema.core.events import EventBus
    from noema.core.types import Task
    from noema.modules.registry import ModuleRegistry

log = get_logger(__name__)


class ModuleService:
    """Manages the 22+ pluggable domain modules with observability."""

    def __init__(
        self,
        registry: ModuleRegistry,
        event_bus: EventBus | None = None,
    ) -> None:
        self.registry = registry
        self.event_bus = event_bus

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if not self.event_bus:
            return
        schedule_event(self.event_bus, event_type, payload, source="module_service")

    def list_modules(self) -> list[dict[str, str]]:
        return self.registry.list_modules()

    def execute_module(self, module_name: str, task: Task) -> dict[str, Any]:
        t0 = time.monotonic()
        try:
            result = self.registry.execute_module(module_name, task)
            elapsed = (time.monotonic() - t0) * 1000
            MODULE_EXECUTION_COUNT.labels(module=module_name, status="success").inc()
            MODULE_LATENCY.labels(module=module_name).observe(elapsed / 1000)
            self._emit_event("module.executed", {"module": module_name, "duration_ms": elapsed})
            return result
        except Exception:
            elapsed = (time.monotonic() - t0) * 1000
            MODULE_EXECUTION_COUNT.labels(module=module_name, status="error").inc()
            MODULE_LATENCY.labels(module=module_name).observe(elapsed / 1000)
            raise

    def execute_all(
        self, task: Task, filter_tags: list[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        t0 = time.monotonic()
        results = self.registry.execute_all(task, filter_tags)
        elapsed = (time.monotonic() - t0) * 1000
        for name in results:
            MODULE_EXECUTION_COUNT.labels(module=name, status="success").inc()
        self._emit_event(
            "module.all_executed", {"modules": list(results.keys()), "duration_ms": elapsed}
        )
        return results

    def get_module(self, name: str) -> Any:
        return self.registry.get_instance(name)

    def stats(self) -> dict[str, Any]:
        return self.registry.stats()
