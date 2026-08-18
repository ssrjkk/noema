"""Module Registry — discovers, manages, and orchestrates all Noema modules."""

from __future__ import annotations

import importlib
from typing import Any, cast

from noema.logging import get_logger

logger = get_logger(__name__)


class NoemaModule:
    """Wrapper for a pluggable Noema module."""

    def __init__(
        self, name: str, module: Any, description: str = "", tags: list[str] | None = None
    ) -> None:
        self.name = name
        self.module = module
        self.description = description or getattr(module, "DESCRIPTION", "")
        self.tags = tags or []
        self._instance = None

    def get_instance(self) -> Any:
        if self._instance is None:
            if isinstance(self.module, type):
                self._instance = self.module()
            else:
                self._instance = self.module
        return self._instance

    def execute(self, task: Any) -> dict[str, Any]:
        instance = self.get_instance()
        if hasattr(instance, "execute"):
            return cast("dict[str, Any]", instance.execute(task))
        return {"module": self.name, "error": "No execute method"}

    def __repr__(self) -> str:
        return f"NoemaModule({self.name!r})"


class ModuleRegistry:
    """Central registry for all Noema modules."""

    def __init__(self) -> None:
        self.modules: dict[str, NoemaModule] = {}
        self._load_builtin_modules()

    def _load_builtin_modules(self) -> None:
        """Auto-discover and register all built-in modules."""
        builtin_specs = [
            ("monitoring", "noema.modules.monitors.kernel", "MonitoringModule"),
            ("testing", "noema.modules.testing.kernel", "TestingModule"),
            ("documentation", "noema.modules.documentation.kernel", "DocumentationModule"),
            ("database", "noema.modules.database.kernel", "DatabaseModule"),
            ("queues", "noema.modules.queues.kernel", "QueuesModule"),
            ("caching", "noema.modules.caching.kernel", "CachingModule"),
            ("auth", "noema.modules.auth.kernel", "AuthModule"),
            ("graphql", "noema.modules.graphql.kernel", "GraphQLModule"),
            ("websocket", "noema.modules.websocket.kernel", "WebSocketModule"),
            ("mobile", "noema.modules.mobile.kernel", "MobileModule"),
            ("i18n", "noema.modules.i18n.kernel", "I18nModule"),
            ("cli_generator", "noema.modules.cli_generator.kernel", "CLIGeneratorModule"),
            ("security_scanner", "noema.modules.security_scanner.kernel", "SecurityScannerModule"),
            ("performance", "noema.modules.performance.kernel", "PerformanceModule"),
            ("config", "noema.modules.config.kernel", "ConfigModule"),
            ("events", "noema.modules.events.kernel", "EventsModule"),
            ("quality", "noema.modules.quality.kernel", "QualityModule"),
            ("containers", "noema.modules.containers.kernel", "ContainersModule"),
            ("terraform", "noema.modules.terraform.kernel", "TerraformModule"),
            ("data_pipeline", "noema.modules.data_pipeline.kernel", "DataPipelineModule"),
            ("ml_ops", "noema.modules.ml_ops.kernel", "MLOpsModule"),
            ("gateway", "noema.modules.gateway.kernel", "GatewayModule"),
        ]

        for name, module_path, class_name in builtin_specs:
            try:
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name, None)
                if cls:
                    self.register(name, cls, description=getattr(cls, "DESCRIPTION", name))
            except ImportError as e:
                # A broken optional dependency must not be invisible: log it
                # so the module's absence is diagnosable.
                logger.warning("module_import_failed", name=name, path=module_path, error=str(e))

    def register(
        self, name: str, module_or_class: Any, description: str = "", tags: list[str] | None = None
    ) -> NoemaModule:
        bm = NoemaModule(name, module_or_class, description, tags)
        self.modules[name] = bm
        return bm

    def get(self, name: str) -> NoemaModule | None:
        return self.modules.get(name)

    def get_instance(self, name: str) -> Any:
        bm = self.modules.get(name)
        return bm.get_instance() if bm else None

    def execute_module(self, name: str, task: Any) -> dict[str, Any]:
        bm = self.modules.get(name)
        if not bm:
            return {"error": f"Module '{name}' not found"}
        return bm.execute(task)

    def execute_all(
        self, task: Any, filter_tags: list[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        results = {}
        for name, bm in self.modules.items():
            if filter_tags and not any(t in bm.tags for t in filter_tags):
                continue
            try:
                results[name] = bm.execute(task)
            except Exception as e:  # noqa: BLE001 - one failing module must not kill the batch
                logger.warning("module_execute_failed", name=name, error=str(e))
                results[name] = {"module": name, "error": str(e)}
        return results

    def get_relevant_modules(self, tags: list[str]) -> list[NoemaModule]:
        relevant = []
        tag_set = set(tags)
        for bm in self.modules.values():
            module_tags = set(bm.tags)
            if tag_set & module_tags:
                relevant.append(bm)
        return relevant

    def list_modules(self) -> list[dict[str, str]]:
        return [{"name": bm.name, "description": bm.description} for bm in self.modules.values()]

    def stats(self) -> dict[str, Any]:
        return {
            "total_modules": len(self.modules),
            "modules": list(self.modules.keys()),
        }


_registry: ModuleRegistry | None = None


def get_registry() -> ModuleRegistry:
    global _registry
    if _registry is None:
        _registry = ModuleRegistry()
    return _registry
