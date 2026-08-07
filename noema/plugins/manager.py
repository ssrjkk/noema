"""Система плагинов — расширение фреймворка кастомными ядрами и агентами."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from noema.logging import get_logger

if TYPE_CHECKING:
    from noema.agents.base import BaseAgent
    from noema.kernels.base import BaseKernel

logger = get_logger(__name__)


class PluginMeta:
    """Метаданные плагина."""

    def __init__(
        self,
        name: str,
        version: str = "0.1.0",
        author: str = "",
        description: str = "",
        dependencies: list[str] | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.author = author
        self.description = description
        self.dependencies = dependencies or []


class Plugin:
    """Базовый класс плагина."""

    def __init__(self, meta: PluginMeta) -> None:
        self.meta = meta
        self._kernels: list[BaseKernel] = []
        self._agents: list[BaseAgent] = []
        self._hooks: dict[str, list] = {}
        self._initialized = False

    async def setup(self) -> None:
        """РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ РїР»Р°РіРёРЅР° (РїРµСЂРµРѕРїСЂРµРґРµР»РёС‚СЊ)."""
        self._initialized = True

    async def teardown(self) -> None:
        """Очистка при выгрузке (переопределить)."""
        self._initialized = False

    def register_kernel(self, kernel: BaseKernel) -> None:
        self._kernels.append(kernel)

    def register_agent(self, agent: BaseAgent) -> None:
        self._agents.append(agent)

    def on(self, event: str, callback: Any) -> None:
        self._hooks.setdefault(event, []).append(callback)

    async def emit(self, event: str, data: Any) -> None:
        for cb in self._hooks.get(event, []):
            if callable(cb):
                result = cb(data)
                if hasattr(result, "__await__"):
                    await result


class PluginManager:
    """
    Менеджер плагинов.

    Управляет загрузкой, регистрацией и жизненным циклом плагинов.
    """

    def __init__(self, plugin_dirs: list[str] | None = None) -> None:
        self.plugins: dict[str, Plugin] = {}
        self._plugin_dirs = plugin_dirs or []
        self._load_paths: list[Path] = []

    async def discover(self) -> list[str]:
        """Обнаружение плагинов в указанных директориях."""
        discovered = []

        for dir_path in self._plugin_dirs:
            path = Path(dir_path)
            if not path.exists():
                continue

            for plugin_file in path.glob("**/plugin.py"):
                plugin_dir = plugin_file.parent
                meta_file = plugin_dir / "plugin_meta.json"
                if meta_file.exists():
                    discovered.append(str(plugin_dir))
                    logger.info(f"Обнаружен плагин: {plugin_dir}")

        return discovered

    async def load_plugin(self, plugin_path: str) -> Plugin | None:
        """Загрузка плагина из директории."""
        path = Path(plugin_path)
        meta_file = path / "plugin_meta.json"
        plugin_file = path / "plugin.py"

        if not plugin_file.exists():
            logger.error(f"plugin.py not found in {plugin_path}")
            return None

        # Читаем метаданные
        import json

        meta_data = {}
        if meta_file.exists():
            meta_data = json.loads(meta_file.read_text(encoding="utf-8"))

        meta = PluginMeta(
            name=meta_data.get("name", path.name),
            version=meta_data.get("version", "0.1.0"),
            author=meta_data.get("author", ""),
            description=meta_data.get("description", ""),
            dependencies=meta_data.get("dependencies", []),
        )

        # Загружаем модуль
        import sys

        sys.path.insert(0, str(path))
        try:
            module = importlib.import_module("plugin")
            plugin_class = getattr(module, "PluginImpl", None)
            if plugin_class and issubclass(plugin_class, Plugin):
                plugin = plugin_class(meta)
                await plugin.setup()
                self.plugins[meta.name] = plugin
                logger.info(f"Плагин загружен: {meta.name} v{meta.version}")
                return cast("Plugin | None", plugin)
            else:
                logger.error(f"PluginImpl not found or not a Plugin subclass in {plugin_path}")
        except Exception as e:
            logger.error(f"Failed to load plugin from {plugin_path}: {e}")
        finally:
            sys.path.pop(0)
            sys.modules.pop("plugin", None)

        return None

    async def load_all(self) -> int:
        """Загрузка всех обнаруженных плагинов."""
        discovered = await self.discover()
        loaded = 0
        for path in discovered:
            plugin = await self.load_plugin(path)
            if plugin:
                loaded += 1
        return loaded

    async def unload_plugin(self, name: str) -> bool:
        """Выгрузка плагина."""
        plugin = self.plugins.get(name)
        if plugin:
            await plugin.teardown()
            del self.plugins[name]
            logger.info(f"Плагин выгружен: {name}")
            return True
        return False

    def get_all_kernels(self) -> list[BaseKernel]:
        """Получить все ядра из плагинов."""
        kernels = []
        for plugin in self.plugins.values():
            kernels.extend(plugin._kernels)
        return kernels

    def get_all_agents(self) -> list[BaseAgent]:
        """Получить всех агентов из плагинов."""
        agents = []
        for plugin in self.plugins.values():
            agents.extend(plugin._agents)
        return agents

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_plugins": len(self.plugins),
            "plugins": {
                name: {
                    "version": p.meta.version,
                    "author": p.meta.author,
                    "kernels": len(p._kernels),
                    "agents": len(p._agents),
                }
                for name, p in self.plugins.items()
            },
        }
