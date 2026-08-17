"""Coverage tests for service wrappers in ``noema/services``:
``memory``, ``modules``, ``plugin``, ``scaffold``, ``worker``, ``evolution``.
"""

import asyncio
import json

import pytest

from noema.core.events import EventBus
from noema.core.types import Solution, SolutionQuality, Task, TechStack
from noema.evolution.engine import EvolutionEngine, EvolutionPatch
from noema.llm.providers import FallbackProvider
from noema.memory.store import MemoryStore
from noema.modules.registry import ModuleRegistry
from noema.plugins.manager import PluginManager
from noema.services.evolution import EvolutionService
from noema.services.memory import MemoryService
from noema.services.modules import ModuleService
from noema.services.plugin import PluginService
from noema.services.scaffold import ScaffoldService
from noema.services.worker import WorkerService
from noema.workers.hierarchy import WorkerHierarchy
from noema.workers.pool import WorkerPool

# ── MemoryService ───────────────────────────────────────────────────────────


def _memory_service(tmp_path) -> MemoryService:
    store = MemoryStore(persist_dir=str(tmp_path / "memory"))
    return MemoryService(store=store)


def test_memory_service_crud_and_search(tmp_path):
    svc = _memory_service(tmp_path)

    svc.record_episode(
        task_description="Build REST API",
        solution_summary="FastAPI with PostgreSQL",
        tech_stack="Python, FastAPI",
        outcome="success",
        tags=["api"],
    )
    svc.learn_fact(topic="python", fact="FastAPI is async by default", confidence=0.9)
    svc.store_procedure(procedure_name="deploy_api", steps=["build", "test", "deploy"])
    svc.record_procedure_outcome("deploy_api", succeeded=True, duration=10.0)
    svc.save()
    svc.flush()

    assert len(svc.search_episodes("REST API")) == 1
    assert len(svc.search_knowledge("fastapi")) == 1
    assert len(svc.search_procedures("deploy")) == 1
    assert svc.search("REST API", kind="episodes")["episodes"]
    assert svc.search("async", kind="knowledge")["knowledge"]
    assert "procedures" in svc.search("deploy", kind="procedures")

    stats = svc.stats()
    assert stats["episodic_count"] == 1
    assert stats["semantic_count"] == 1
    assert stats["procedural_count"] == 1


def test_memory_service_record_task_outcome(tmp_path):
    svc = _memory_service(tmp_path)
    task = Task(title="Deploy service", description="Deploy the API to prod", tags=["devops"])
    excellent = Solution(
        task_id=task.id,
        title="Deploy service",
        summary="Deployed with blue-green",
        stack=TechStack(languages=["Python"], frameworks=["FastAPI"]),
        quality=SolutionQuality.EXCELLENT,
        confidence=0.8,
    )
    svc.record_task_outcome(task, excellent, duration_seconds=12.5)
    assert len(svc.search_episodes("Deploy service")) == 1
    episodes = svc.search_episodes("Deploy service")
    assert episodes[0]["outcome"] == "success"
    # excellent/masterpiece solutions also seed a reusable procedure
    assert svc.store.find_procedure("solution_Deploy_service") is not None
    assert len(svc.search_procedures("solution_Deploy_service")) == 1

    draft = Solution(
        task_id=task.id,
        title="Deploy service",
        summary="rough sketch",
        stack=TechStack(languages=[], frameworks=[]),
        quality=SolutionQuality.DRAFT,
        confidence=0.2,
    )
    svc.record_task_outcome(task, draft, duration_seconds=1.0)
    episodes = svc.search_episodes("rough sketch")
    assert episodes[0]["outcome"] == "partial"


def test_memory_service_get_recent_context(tmp_path):
    svc = _memory_service(tmp_path)
    task = Task(title="Build an API", description="async REST backend")
    assert svc.get_recent_context(task) == ""

    svc.record_episode(
        task_description="Build an API", solution_summary="FastAPI backend", outcome="success"
    )
    ctx = svc.get_recent_context(task)
    assert ctx
    assert "[Past]" in ctx


@pytest.mark.asyncio
async def test_memory_service_emits_events(tmp_path):
    bus = EventBus()
    store = MemoryStore(persist_dir=str(tmp_path / "memory"))
    svc = MemoryService(store=store, event_bus=bus)

    svc.record_episode(task_description="t", solution_summary="s", outcome="success")
    svc.learn_fact(topic="t", fact="f", confidence=0.5)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert bus.stats["published"] >= 2


# ── ModuleService ───────────────────────────────────────────────────────────


class _BoomModule:
    NAME = "boom"

    def execute(self, task):
        raise RuntimeError("boom")


def _module_service() -> ModuleService:
    registry = ModuleRegistry()
    return ModuleService(registry=registry)


def test_module_service_list_execute_stats():
    svc = _module_service()
    task = Task(title="Build an API", description="REST API", tags=["api"])

    modules = svc.list_modules()
    assert len(modules) >= 1

    result = svc.execute_module("events", task)
    assert "error" not in result

    assert svc.get_module("events") is not None

    stats = svc.stats()
    assert stats["total_modules"] >= 1
    assert "events" in stats["modules"]


def test_module_service_unknown_module_returns_error():
    svc = _module_service()
    task = Task(title="t", description="d")
    result = svc.execute_module("does_not_exist", task)
    assert "error" in result


def test_module_service_execute_all():
    svc = _module_service()
    task = Task(title="Build an API", description="REST API", tags=["api"])
    results = svc.execute_all(task)
    assert len(results) >= 1
    assert results == {name: results[name] for name in svc.stats()["modules"]}


def test_module_service_reexecute_errors():
    svc = _module_service()
    svc.registry.register("boom", _BoomModule)
    task = Task(title="t", description="d")
    with pytest.raises(RuntimeError):
        svc.execute_module("boom", task)


@pytest.mark.asyncio
async def test_module_service_emits_event():
    bus = EventBus()
    svc = ModuleService(registry=ModuleRegistry(), event_bus=bus)
    task = Task(title="Build an API", description="REST API", tags=["api"])

    svc.execute_module("events", task)
    svc.execute_all(task)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert bus.stats["published"] >= 2


# ── PluginService ───────────────────────────────────────────────────────────

PLUGIN_PY = """\
from noema.kernels.base import BaseKernel
from noema.plugins.manager import Plugin


class _FakeKernel(BaseKernel):
    @property
    def name(self):
        return "fake"

    @property
    def description(self):
        return "fake kernel"

    async def execute(self, task, **kwargs):
        return {"type": "fake"}


class PluginImpl(Plugin):
    async def setup(self):
        await super().setup()
        self.register_kernel(_FakeKernel())
"""

PLUGIN_META = {
    "name": "demo_plugin",
    "version": "1.2.0",
    "author": "tester",
    "description": "demo plugin",
}


def _make_plugin(tmp_path) -> str:
    plugin_dir = tmp_path / "demo_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.py").write_text(PLUGIN_PY, encoding="utf-8")
    (plugin_dir / "plugin_meta.json").write_text(json.dumps(PLUGIN_META), encoding="utf-8")
    return str(plugin_dir)


@pytest.mark.asyncio
async def test_plugin_service_discover_and_load(tmp_path):
    path = _make_plugin(tmp_path)
    svc = PluginService(manager=PluginManager(plugin_dirs=[str(tmp_path)]))

    discovered = await svc.discover()
    assert path in discovered

    loaded = await svc.load_all()
    assert loaded == 1

    stats = svc.get_stats()
    assert stats["total_plugins"] == 1
    assert stats["plugins"]["demo_plugin"]["version"] == "1.2.0"
    assert stats["plugins"]["demo_plugin"]["kernels"] == 1

    assert len(svc.get_all_kernels()) == 1
    assert svc.get_all_kernels()[0].name == "fake"


@pytest.mark.asyncio
async def test_plugin_service_unload(tmp_path):
    path = _make_plugin(tmp_path)
    svc = PluginService(manager=PluginManager(plugin_dirs=[str(tmp_path)]))

    plugin = await svc.load_plugin(path)
    assert plugin is not None
    assert plugin.meta.name == "demo_plugin"

    assert await svc.unload_plugin("demo_plugin") is True
    assert svc.get_stats()["total_plugins"] == 0
    assert svc.get_all_kernels() == []


@pytest.mark.asyncio
async def test_plugin_service_empty(tmp_path):
    svc = PluginService(manager=PluginManager())
    assert await svc.discover() == []
    assert await svc.load_all() == 0
    assert svc.get_all_kernels() == []
    assert svc.get_all_agents() == []
    assert svc.get_stats()["total_plugins"] == 0


@pytest.mark.asyncio
async def test_plugin_service_emits_events(tmp_path):
    path = _make_plugin(tmp_path)
    bus = EventBus()
    svc = PluginService(manager=PluginManager(plugin_dirs=[str(tmp_path)]), event_bus=bus)

    assert await svc.load_all() == 1
    assert bus.stats["published"] == 1  # plugin.all_loaded

    plugin = await svc.load_plugin(path)
    assert plugin is not None
    assert bus.stats["published"] == 2  # plugin.loaded

    assert await svc.unload_plugin("demo_plugin") is True
    assert bus.stats["published"] == 3  # plugin.unloaded


# ── ScaffoldService ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scaffold_service_scaffold(tmp_path):
    bus = EventBus()
    svc = ScaffoldService(event_bus=bus)

    solution = Solution(
        task_id="t1",
        title="Test App",
        summary="Test",
        stack=TechStack(languages=["Python"], frameworks=["FastAPI"]),
        code_blocks=[],
    )
    task = Task(title="Test App", tags=["python"])

    result = await svc.scaffold(solution, task, str(tmp_path))

    assert result["files_created"] > 0
    assert (tmp_path / "test_app" / "README.md").exists()
    assert bus.stats["published"] == 1  # scaffold.completed


# ── WorkerService ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_worker_service_execute_hierarchical():
    bus = EventBus()
    pool = WorkerPool(max_workers=2)
    hierarchy = WorkerHierarchy(max_depth=2)
    svc = WorkerService(pool=pool, hierarchy=hierarchy, event_bus=bus)

    async def decompose(task):
        return [f"{task} part {i}" for i in range(3)]

    async def execute(task):
        return True

    await svc.start()
    try:
        result = await svc.execute_hierarchical("root task", decomposer=decompose, executor=execute)
        assert result["state"] == "completed"
        assert result["subtasks"] == 3
        assert result["depth"] == 0

        assert bus.stats["published"] == 2  # hierarchical_start + hierarchical_done

        hstats = svc.hierarchy_stats()
        assert hstats["total_tasks"] == 1
        assert hstats["completed"] == 1

        pool_stats = svc.stats
        assert isinstance(pool_stats, dict)
        assert "workers_total" in pool_stats
        assert "queue_size" in pool_stats
    finally:
        await svc.shutdown()


# ── EvolutionService ────────────────────────────────────────────────────────

APP_ORIGINAL = """\
# TODO: simplify add
def add(a, b):
    \"\"\"Add two numbers.\"\"\"
    return a + b
"""

TEST_APP = """\
from app import add


def test_add():
    assert add(1, 2) == 3
"""


def _patch_generator():
    async def _generate(file, issue):
        return EvolutionPatch(
            target="app.py",
            description="improve add",
            original_code=APP_ORIGINAL,
            patched_code=APP_ORIGINAL,
            rationale="deterministic test patch",
            confidence=1.0,
        )

    return _generate


@pytest.mark.asyncio
async def test_evolution_service_cycle(tmp_path):
    (tmp_path / "app.py").write_text(APP_ORIGINAL, encoding="utf-8")
    (tmp_path / "test_app.py").write_text(TEST_APP, encoding="utf-8")

    bus = EventBus()
    engine = EvolutionEngine(
        llm_provider=FallbackProvider(),
        project_root=str(tmp_path),
        enabled=True,
        test_before_apply=True,
        auto_apply=False,
    )
    svc = EvolutionService(engine=engine, event_bus=bus)

    analysis = await svc.analyze_self()
    assert isinstance(analysis, dict)
    assert "issues" in analysis
    assert "improvements" in analysis

    # EvolutionService.run_cycle forwards test_runner only, so stub the
    # engine's own patch generator to drive a deterministic cycle.
    engine.generate_patch = _patch_generator()
    result = await svc.run_cycle()
    assert result["patches_generated"] == 1
    assert result["patches_applied"] == 0
    assert result["patches_proposed"] == 1

    assert bus.stats["published"] == 1  # evolution.cycle_completed

    # proposed patches never mutate the worktree
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == APP_ORIGINAL

    stats = svc.get_stats()
    assert stats["total_patches"] == 1
    assert stats["total_cycles"] == 1
