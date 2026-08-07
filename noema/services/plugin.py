"""Plugin service — wraps PluginManager with lifecycle events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from noema.logging import get_logger

if TYPE_CHECKING:
    from noema.agents.base import BaseAgent
    from noema.core.events import EventBus
    from noema.kernels.base import BaseKernel
    from noema.plugins.manager import PluginManager

log = get_logger(__name__)


class PluginService:
    """Manages plugin discovery, loading, and lifecycle."""

    def __init__(
        self,
        manager: PluginManager,
        event_bus: EventBus | None = None,
    ) -> None:
        self.manager = manager
        self.event_bus = event_bus

    async def discover(self) -> list[str]:
        return await self.manager.discover()

    async def load_all(self) -> int:
        loaded = await self.manager.load_all()
        if self.event_bus:
            await self.event_bus.emit(
                "plugin.all_loaded",
                {"count": loaded},
                source="plugin_service",
            )
        return loaded

    async def load_plugin(self, path: str) -> Any | None:
        plugin = await self.manager.load_plugin(path)
        if plugin and self.event_bus:
            await self.event_bus.emit(
                "plugin.loaded",
                {"name": plugin.meta.name, "version": plugin.meta.version},
                source="plugin_service",
            )
        return plugin

    async def unload_plugin(self, name: str) -> bool:
        result = await self.manager.unload_plugin(name)
        if result and self.event_bus:
            await self.event_bus.emit(
                "plugin.unloaded",
                {"name": name},
                source="plugin_service",
            )
        return result

    def get_all_kernels(self) -> list[BaseKernel]:
        return self.manager.get_all_kernels()

    def get_all_agents(self) -> list[BaseAgent]:
        return self.manager.get_all_agents()

    def get_stats(self) -> dict[str, Any]:
        return self.manager.get_stats()
