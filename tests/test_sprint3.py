"""Tests for Sprint 3-6: event bus, sandbox, git evolution, CI/CD."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

# ═══════════════════════════════════════════════════════════════════════════
# Event Bus Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEventBus:
    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        from noema.core.events import Event, EventBus

        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe("test.event", handler)
        await bus.start()

        await bus.emit("test.event", {"key": "value"})
        await asyncio.sleep(0.2)  # Let consumer process

        assert len(received) == 1
        assert received[0].type == "test.event"
        assert received[0].payload == {"key": "value"}

        await bus.stop()

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self):
        from noema.core.events import Event, EventBus

        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event.type)

        bus.subscribe("*", handler)
        await bus.start()

        await bus.emit("type.a", {})
        await bus.emit("type.b", {})
        await asyncio.sleep(0.2)

        assert "type.a" in received
        assert "type.b" in received

        await bus.stop()

    @pytest.mark.asyncio
    async def test_no_subscribers(self):
        from noema.core.events import EventBus

        bus = EventBus()
        await bus.start()
        await bus.emit("nobody.listens", {"data": 1})
        await asyncio.sleep(0.1)
        assert bus.stats["delivered"] >= 1
        await bus.stop()

    @pytest.mark.asyncio
    async def test_handler_error_does_not_crash(self):
        from noema.core.events import Event, EventBus

        bus = EventBus()
        received = []

        async def bad_handler(event: Event):
            raise RuntimeError("oops")

        async def good_handler(event: Event):
            received.append(event.type)

        bus.subscribe("test", bad_handler)
        bus.subscribe("test", good_handler)
        await bus.start()

        await bus.emit("test", {})
        await asyncio.sleep(0.2)

        assert len(received) == 1
        assert bus.stats["failed"] >= 1

        await bus.stop()

    def test_stats(self):
        from noema.core.events import EventBus

        bus = EventBus()
        stats = bus.stats
        assert "published" in stats
        assert "delivered" in stats
        assert "failed" in stats
        assert "queue_size" in stats

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        from noema.core.events import Event, EventBus

        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(1)

        bus.subscribe("test", handler)
        bus.unsubscribe("test", handler)
        await bus.start()
        await bus.emit("test", {})
        await asyncio.sleep(0.1)
        assert len(received) == 0
        await bus.stop()

    @pytest.mark.asyncio
    async def test_emit_returns_event(self):
        from noema.core.events import EventBus

        bus = EventBus()
        await bus.start()
        event = await bus.emit("test", {"x": 1}, source="unit_test")
        assert event.type == "test"
        assert event.source == "unit_test"
        assert event.id is not None
        await bus.stop()


# ═══════════════════════════════════════════════════════════════════════════
# Sandbox Tests (schema + logic, no Docker needed)
# ═══════════════════════════════════════════════════════════════════════════


class TestSandbox:
    def test_config_defaults(self):
        from noema.sandbox.docker import SandboxConfig

        cfg = SandboxConfig()
        assert cfg.timeout > 0
        assert cfg.max_memory != ""
        assert cfg.network_disabled is True
        assert cfg.read_only_root is True

    def test_result_dataclass(self):
        from noema.sandbox.docker import SandboxResult

        r = SandboxResult(success=True, output="ok", exit_code=0, duration_ms=123.4)
        assert r.success is True
        assert r.output == "ok"
        assert r.timed_out is False

    def test_write_plugin(self, tmp_path):
        from noema.sandbox.docker import PluginSandbox

        sandbox = PluginSandbox()
        sandbox._write_plugin(
            tmp_path,
            "def main(): return 'hello'",
            "main",
            None,
        )
        assert (tmp_path / "plugin.py").is_file()
        assert (tmp_path / "runner.py").is_file()
        assert (tmp_path / "input.json").is_file()

        content = (tmp_path / "plugin.py").read_text()
        assert "def main()" in content

    def test_write_plugin_with_input(self, tmp_path):
        from noema.sandbox.docker import PluginSandbox

        sandbox = PluginSandbox()
        sandbox._write_plugin(
            tmp_path,
            "def main(x=0): return x * 2",
            "main",
            {"x": 5},
        )
        input_data = json.loads((tmp_path / "input.json").read_text())
        assert input_data == {"x": 5}


# ═══════════════════════════════════════════════════════════════════════════
# Git Evolution Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGitEvolution:
    @pytest.mark.asyncio
    async def test_is_git_repo(self, tmp_path):
        from noema.evolution.git_evolution import GitEvolution

        ge = GitEvolution(project_root=str(tmp_path))
        assert await ge.is_git_repo() is False

    @pytest.mark.asyncio
    async def test_init_repo(self, tmp_path):
        from noema.evolution.git_evolution import GitEvolution

        ge = GitEvolution(project_root=str(tmp_path))
        result = await ge.init_repo()
        assert result is True
        assert await ge.is_git_repo()

    @pytest.mark.asyncio
    async def test_stats(self, tmp_path):
        from noema.evolution.git_evolution import GitEvolution

        ge = GitEvolution(project_root=str(tmp_path))
        stats = ge.stats()
        assert stats["total_proposals"] == 0
        assert stats["auto_apply"] is False
        assert stats["test_before_apply"] is True

    @pytest.mark.asyncio
    async def test_get_log_empty(self, tmp_path):
        from noema.evolution.git_evolution import GitEvolution

        ge = GitEvolution(project_root=str(tmp_path))
        await ge.init_repo()
        log = await ge.get_log()
        assert log == []


# ═══════════════════════════════════════════════════════════════════════════
# Reasoning Service Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestReasoningService:
    def test_service_instantiation(self):
        from noema.core.chain_of_thought import ChainOfThought
        from noema.core.events import EventBus
        from noema.llm.providers import FallbackProvider
        from noema.services.reasoning import ReasoningService

        llm = FallbackProvider()
        cot = ChainOfThought(llm)
        bus = EventBus()
        svc = ReasoningService(llm=llm, cot=cot, event_bus=bus)
        assert svc.llm is llm
        assert svc.cot is cot
        assert svc.event_bus is bus


# ═══════════════════════════════════════════════════════════════════════════
# Helm Chart Tests (structure check)
# ═══════════════════════════════════════════════════════════════════════════


class TestHelmChart:
    def test_chart_yaml_exists(self):
        chart = Path(__file__).parent.parent / "deploy" / "helm" / "noema" / "Chart.yaml"
        assert chart.is_file()

    def test_values_yaml_exists(self):
        values = Path(__file__).parent.parent / "deploy" / "helm" / "noema" / "values.yaml"
        assert values.is_file()

    def test_templates_exist(self):
        templates_dir = Path(__file__).parent.parent / "deploy" / "helm" / "noema" / "templates"
        assert templates_dir.is_dir()
        templates = list(templates_dir.glob("*.yaml")) + list(templates_dir.glob("*.tpl"))
        names = [t.name for t in templates]
        assert "deployment.yaml" in names
        assert "service.yaml" in names
        assert "ingress.yaml" in names
        assert "hpa.yaml" in names
        assert "_helpers.tpl" in names


# ═══════════════════════════════════════════════════════════════════════════
# CI/CD Tests (structure check)
# ═══════════════════════════════════════════════════════════════════════════


class TestCICD:
    def test_ci_workflow_exists(self):
        ci = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"
        assert ci.is_file()

    def test_deploy_workflow_exists(self):
        deploy = Path(__file__).parent.parent / ".github" / "workflows" / "deploy.yml"
        assert deploy.is_file()

    def test_dockerfile_exists(self):
        dockerfile = Path(__file__).parent.parent / "Dockerfile"
        assert dockerfile.is_file()

    def test_dockerfile_has_healthcheck(self):
        dockerfile = Path(__file__).parent.parent / "Dockerfile"
        content = dockerfile.read_text()
        assert "HEALTHCHECK" in content

    def test_dockerfile_has_non_root(self):
        dockerfile = Path(__file__).parent.parent / "Dockerfile"
        content = dockerfile.read_text()
        assert "USER noema" in content
