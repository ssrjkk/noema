"""Tests for noema/plugins/manager.py — plugin discovery, loading and lifecycle."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from noema.plugins.manager import Plugin, PluginManager, PluginMeta

if TYPE_CHECKING:
    from pathlib import Path

VALID_SOURCE = """
from noema.plugins.manager import Plugin


class PluginImpl(Plugin):
    async def setup(self) -> None:
        await super().setup()
        self.register_kernel("kernel-marker")
        self.register_agent("agent-marker")
"""

NO_IMPL_SOURCE = "MARKER = 'no plugin implementation here'\n"

WRONG_BASE_SOURCE = """
class PluginImpl:
    pass
"""

BROKEN_SOURCE = "raise RuntimeError('plugin import exploded')\n"


def _write_plugin(
    root: Path,
    directory: str,
    source: str,
    meta: dict | None = None,
) -> Path:
    """Create ``<root>/<directory>/plugin.py`` and an optional meta file."""
    plugin_dir = root / directory
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")
    if meta is not None:
        (plugin_dir / "plugin_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return plugin_dir


def _meta(name: str = "demo", **over) -> dict:
    data = {
        "name": name,
        "version": "1.2.3",
        "author": "noema",
        "description": "a demo plugin",
        "dependencies": ["numpy"],
    }
    data.update(over)
    return data


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class TestPlugin:
    async def test_setup_and_teardown_toggle_initialized(self):
        plugin = Plugin(PluginMeta(name="p"))
        assert plugin._initialized is False

        await plugin.setup()
        assert plugin._initialized is True

        await plugin.teardown()
        assert plugin._initialized is False

    async def test_emit_runs_sync_and_async_callbacks_in_order(self):
        seen: list[str] = []
        plugin = Plugin(PluginMeta(name="p"))
        plugin.on("boot", lambda data: seen.append(f"sync:{data}"))

        async def slow(data: str) -> None:
            seen.append(f"async:{data}")

        plugin.on("boot", slow)
        await plugin.emit("boot", "x")

        assert seen == ["sync:x", "async:x"]

    async def test_emit_skips_registered_values_that_are_not_callable(self):
        plugin = Plugin(PluginMeta(name="p"))
        plugin._hooks["boot"] = [42]
        await plugin.emit("boot", None)
        assert plugin._hooks["boot"] == [42]

    async def test_emit_survives_a_failing_hook_and_keeps_the_rest(self):
        calls: list[str] = []
        plugin = Plugin(PluginMeta(name="p"))

        def boom(_data: object) -> None:
            calls.append("boom")
            raise ValueError("hook failed")

        plugin.on("boot", boom)
        plugin.on("boot", lambda _d: calls.append("after"))

        await plugin.emit("boot", None)

        assert calls == ["boom", "after"]

    async def test_emit_of_an_unknown_event_is_a_noop(self):
        plugin = Plugin(PluginMeta(name="p"))
        await plugin.emit("nothing-registered", None)
        assert plugin._hooks == {}

    def test_register_appends_in_call_order(self):
        plugin = Plugin(PluginMeta(name="p"))
        plugin.register_kernel("k1")
        plugin.register_kernel("k2")
        plugin.register_agent("a1")
        assert plugin._kernels == ["k1", "k2"]
        assert plugin._agents == ["a1"]


# ---------------------------------------------------------------------------
# PluginManager.discover
# ---------------------------------------------------------------------------


class TestDiscover:
    async def test_missing_directories_are_skipped(self, tmp_path: Path):
        manager = PluginManager(plugin_dirs=[str(tmp_path / "nope"), str(tmp_path / "gone")])
        assert await manager.discover() == []

    async def test_only_directories_with_both_files_are_found(self, tmp_path: Path):
        listed = _write_plugin(tmp_path, "listed", VALID_SOURCE, _meta())
        _write_plugin(tmp_path, "no_meta", VALID_SOURCE)
        (tmp_path / "empty").mkdir()

        manager = PluginManager(plugin_dirs=[str(tmp_path)])
        found = await manager.discover()

        assert found == [str(listed)]


# ---------------------------------------------------------------------------
# PluginManager.load_plugin
# ---------------------------------------------------------------------------


class TestLoadPlugin:
    async def test_loads_a_valid_plugin_and_runs_setup(self, tmp_path: Path):
        plugin_dir = _write_plugin(tmp_path, "demo", VALID_SOURCE, _meta())
        manager = PluginManager()

        plugin = await manager.load_plugin(str(plugin_dir))

        assert isinstance(plugin, Plugin)
        assert plugin._initialized is True
        assert plugin.meta.name == "demo"
        assert plugin.meta.version == "1.2.3"
        assert plugin.meta.author == "noema"
        assert plugin.meta.description == "a demo plugin"
        assert plugin.meta.dependencies == ["numpy"]
        assert manager.plugins == {"demo": plugin}

    async def test_missing_meta_file_falls_back_to_the_directory_name(self, tmp_path: Path):
        plugin_dir = _write_plugin(tmp_path, "bare-directory", VALID_SOURCE)

        plugin = await PluginManager().load_plugin(str(plugin_dir))

        assert plugin is not None
        assert plugin.meta.name == "bare-directory"
        assert plugin.meta.version == "0.1.0"
        assert plugin.meta.dependencies == []

    async def test_directory_without_plugin_py_returns_none(self, tmp_path: Path):
        plugin_dir = tmp_path / "hollow"
        plugin_dir.mkdir()
        (plugin_dir / "plugin_meta.json").write_text(json.dumps(_meta()), encoding="utf-8")

        manager = PluginManager()
        assert await manager.load_plugin(str(plugin_dir)) is None
        assert manager.plugins == {}

    async def test_module_without_plugin_impl_returns_none(self, tmp_path: Path):
        plugin_dir = _write_plugin(tmp_path, "noimpl", NO_IMPL_SOURCE, _meta("noimpl"))

        assert await PluginManager().load_plugin(str(plugin_dir)) is None

    async def test_plugin_impl_outside_the_plugin_hierarchy_is_rejected(self, tmp_path: Path):
        plugin_dir = _write_plugin(tmp_path, "wrong", WRONG_BASE_SOURCE, _meta("wrong"))
        manager = PluginManager()

        assert await manager.load_plugin(str(plugin_dir)) is None
        assert manager.plugins == {}

    async def test_import_error_is_contained_and_returns_none(self, tmp_path: Path):
        plugin_dir = _write_plugin(tmp_path, "broken", BROKEN_SOURCE, _meta("broken"))

        assert await PluginManager().load_plugin(str(plugin_dir)) is None

    async def test_sys_path_and_module_cache_are_restored(self, tmp_path: Path):
        import sys

        plugin_dir = _write_plugin(tmp_path, "hygienic", VALID_SOURCE, _meta("hygienic"))
        before = list(sys.path)

        await PluginManager().load_plugin(str(plugin_dir))

        assert sys.path == before
        assert "plugin" not in sys.modules


class TestLoadAll:
    async def test_loads_every_discovered_plugin(self, tmp_path: Path):
        _write_plugin(tmp_path, "one", VALID_SOURCE, _meta("one"))
        _write_plugin(tmp_path, "two", VALID_SOURCE, _meta("two"))
        _write_plugin(tmp_path, "three", NO_IMPL_SOURCE, _meta("three"))
        manager = PluginManager(plugin_dirs=[str(tmp_path)])

        assert await manager.load_all() == 2
        assert set(manager.plugins) == {"one", "two"}

    async def test_nothing_discovered_loads_nothing(self, tmp_path: Path):
        manager = PluginManager(plugin_dirs=[str(tmp_path / "absent")])
        assert await manager.load_all() == 0


# ---------------------------------------------------------------------------
# PluginManager — lifecycle and aggregation
# ---------------------------------------------------------------------------


class TestUnload:
    async def test_unload_runs_teardown_and_forgets_the_plugin(self, tmp_path: Path):
        plugin_dir = _write_plugin(tmp_path, "demo", VALID_SOURCE, _meta())
        manager = PluginManager()
        plugin = await manager.load_plugin(str(plugin_dir))
        assert plugin is not None

        assert await manager.unload_plugin("demo") is True
        assert plugin._initialized is False
        assert manager.plugins == {}

    async def test_unload_of_an_unknown_name_reports_false(self):
        assert await PluginManager().unload_plugin("ghost") is False


class TestAggregation:
    async def test_kernels_and_agents_are_pooled_across_plugins(self, tmp_path: Path):
        first = _write_plugin(tmp_path, "one", VALID_SOURCE, _meta("one"))
        second = _write_plugin(tmp_path, "two", VALID_SOURCE, _meta("two"))
        manager = PluginManager()
        await manager.load_plugin(str(first))
        await manager.load_plugin(str(second))

        assert manager.get_all_kernels() == ["kernel-marker", "kernel-marker"]
        assert manager.get_all_agents() == ["agent-marker", "agent-marker"]

    async def test_stats_summarise_each_plugin(self, tmp_path: Path):
        plugin_dir = _write_plugin(tmp_path, "demo", VALID_SOURCE, _meta())
        manager = PluginManager()
        await manager.load_plugin(str(plugin_dir))

        assert manager.get_stats() == {
            "total_plugins": 1,
            "plugins": {
                "demo": {
                    "version": "1.2.3",
                    "author": "noema",
                    "kernels": 1,
                    "agents": 1,
                }
            },
        }

    async def test_empty_manager_stats(self):
        assert PluginManager().get_stats() == {"total_plugins": 0, "plugins": {}}


@pytest.mark.parametrize("dirs", [None, []])
def test_default_plugin_dirs_leave_nothing_to_discover(dirs):
    manager = PluginManager(plugin_dirs=dirs)
    assert manager.plugins == {}
