"""Тесты для Noema."""

import asyncio

import pytest

from noema.agents.base import ArchitectAgent, DeveloperAgent
from noema.agents.orchestrator import AgentOrchestrator
from noema.core.engine import NoemaEngine
from noema.core.types import (
    KnowledgeEntry,
    Solution,
    Task,
    TaskComplexity,
    TechStack,
    ThoughtProcess,
)
from noema.feedback.store import FeedbackStore
from noema.kernels.ai_ml import AIMLKernel
from noema.kernels.analysis import AnalysisKernel
from noema.kernels.architecture import ArchitectureKernel
from noema.kernels.codegen import CodegenKernel
from noema.kernels.data import DataKernel
from noema.kernels.devops import DevOpsKernel
from noema.kernels.frontend import FrontendKernel
from noema.kernels.optimization import OptimizationKernel
from noema.kernels.security import SecurityKernel
from noema.knowledge.graph import KnowledgeGraph
from noema.knowledge.store import KnowledgeStore
from noema.llm.providers import FallbackProvider, LLMMessage, create_llm_provider
from noema.pipelines.engine import (
    Pipeline,
    create_quick_prototype_pipeline,
)
from noema.plugins.manager import PluginManager, PluginMeta
from noema.scaffolder.generator import ProjectScaffolder
from noema.utils.helpers import chunk_list, deep_merge, generate_id, truncate
from noema.workers.pool import WorkerPool

# ── Types Tests ────────────────────────────────────────────────────────────


def test_task_creation():
    task = Task(title="Build API", description="REST API for users")
    assert task.title == "Build API"
    assert task.id
    assert task.complexity == TaskComplexity.MODERATE


def test_tech_stack_summary():
    stack = TechStack(languages=["Python", "Go"], frameworks=["FastAPI"], databases=["PostgreSQL"])
    summary = stack.summary()
    assert "Python" in summary
    assert "FastAPI" in summary


def test_thought_process():
    tp = ThoughtProcess(task_id="test")
    tp.add_step("kernel1", "input1", "output1", 0.8)
    tp.add_step("kernel2", "input2", "output2", 0.6)
    assert len(tp.steps) == 2
    assert abs(tp.avg_confidence - 0.7) < 0.01


def test_solution_quality():
    s = Solution(task_id="t1", title="Test", summary="Test solution")
    assert s.quality.value == "draft"
    assert s.confidence == 0.5


# ── Kernel Tests ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_architecture_kernel_analyze():
    kernel = ArchitectureKernel()
    task = Task(
        title="E-commerce Platform",
        description="Микросервисная платформа для электронной коммерции",
        tags=["web", "microservice", "high-load"],
        complexity=TaskComplexity.COMPLEX,
    )
    result = await kernel.execute(task, phase="analyze")
    assert result["type"] == "analysis"
    assert "suggested_patterns" in result
    assert "_confidence" in result


@pytest.mark.asyncio
async def test_architecture_kernel_design():
    kernel = ArchitectureKernel()
    task = Task(
        title="Chat Application",
        description="Real-time чат",
        tags=["web", "real-time"],
        complexity=TaskComplexity.MODERATE,
    )
    result = await kernel.execute(task, phase="design")
    assert result["type"] == "architecture"
    assert "pattern" in result
    assert "components" in result


@pytest.mark.asyncio
async def test_codegen_kernel():
    kernel = CodegenKernel()
    task = Task(
        title="User Service",
        description="CRUD сервис для пользователей",
        tags=["python", "api"],
    )
    result = await kernel.execute(task)
    assert result["type"] == "codegen"
    assert result["block_count"] > 0
    assert len(result["blocks"]) > 0


@pytest.mark.asyncio
async def test_optimization_kernel():
    kernel = OptimizationKernel()
    task = Task(
        title="High Load API",
        tags=["high-load", "performance"],
    )
    result = await kernel.execute(task)
    assert result["type"] == "optimization"
    assert len(result["strategies"]) > 0


@pytest.mark.asyncio
async def test_security_kernel():
    kernel = SecurityKernel()
    task = Task(
        title="Auth Service",
        tags=["web", "auth", "api"],
    )
    result = await kernel.execute(task)
    assert result["type"] == "security"
    assert "risk_score" in result
    assert len(result["checks"]) > 0


@pytest.mark.asyncio
async def test_analysis_kernel():
    kernel = AnalysisKernel()
    task = Task(
        title="ML Pipeline",
        description="Построение ML пайплайна",
        tags=["ml", "data"],
        requirements=[
            {"category": "performance", "description": "Low latency inference", "priority": 8},
            {"category": "scalability", "description": "10k RPS", "priority": 7},
        ],
        complexity=TaskComplexity.COMPLEX,
    )
    result = await kernel.execute(task)
    assert result["type"] == "analysis"
    assert "complexity" in result
    assert "risks" in result
    assert "timeline" in result


# ── Worker Pool Tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_worker_pool_lifecycle():
    pool = WorkerPool(max_workers=2)
    await pool.start()
    assert len(pool.workers) == 2
    await pool.shutdown()


@pytest.mark.asyncio
async def test_worker_pool_submit():
    pool = WorkerPool(max_workers=2)
    await pool.start()

    async def add(a, b):
        return a + b

    result = await pool.submit(add, 2, 3)
    assert result == 5

    await pool.shutdown()


@pytest.mark.asyncio
async def test_worker_pool_parallel():
    pool = WorkerPool(max_workers=4)
    await pool.start()

    async def square(x):
        return x * x

    results = await pool.submit_many(
        [
            (square, (2,), {}),
            (square, (3,), {}),
            (square, (4,), {}),
        ]
    )

    assert results == [4, 9, 16]
    await pool.shutdown()


@pytest.mark.asyncio
async def test_worker_pool_stats():
    pool = WorkerPool(max_workers=2)
    await pool.start()

    async def noop():
        return True

    await pool.submit(noop)
    stats = pool.stats
    assert stats["total_completed"] == 1

    await pool.shutdown()


# ── Agent Tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_orchestrator():
    orch = AgentOrchestrator()
    await orch.initialize()
    assert len(orch.agents) >= 6

    agent = orch.get_agent_for_role("architect")
    assert agent is not None

    await orch.shutdown()


@pytest.mark.asyncio
async def test_agent_analysis():
    agent = ArchitectAgent()
    task = Task(title="Web App", tags=["web"])
    analysis = await agent.analyze(task)
    assert "domain" in analysis


@pytest.mark.asyncio
async def test_agent_contribution():
    agent = DeveloperAgent()
    task = Task(title="API Service", tags=["api"])
    solution = Solution(task_id="t1", title="API", summary="API service")
    contrib = await agent.contribute(task, solution, {})
    assert contrib["layer"] == "implementation"


# ── Knowledge Store Tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_knowledge_store_search():
    store = KnowledgeStore()
    await store.load()

    results = await store.search("fastapi web api")
    assert len(results) > 0
    assert results[0].get("score", 0) > 0


@pytest.mark.asyncio
async def test_knowledge_store_stats():
    store = KnowledgeStore()
    await store.load()
    stats = store.get_stats()
    assert stats["total_entries"] > 0
    assert stats["total_patterns"] > 0


@pytest.mark.asyncio
async def test_knowledge_add_entry():
    store = KnowledgeStore()
    await store.load()
    initial_count = len(store.entries)

    entry = KnowledgeEntry(
        category="test",
        title="Test Entry",
        content="Test content for unit testing",
        tags=["test"],
    )
    await store.add_entry(entry)
    assert len(store.entries) == initial_count + 1


# ── Noema Engine Integration Test ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_noema_full_cycle():
    noema = NoemaEngine(worker_count=2, llm_provider="fallback")
    await noema.initialize()

    task = Task(
        title="Task Management API",
        description="REST API для управления задачами",
        tags=["python", "web", "api"],
        complexity=TaskComplexity.MODERATE,
        requirements=[
            {"category": "auth", "description": "JWT authentication", "priority": 8},
            {"category": "crud", "description": "CRUD для задач", "priority": 9},
            {"category": "performance", "description": "Кэширование", "priority": 5},
        ],
    )

    solution, thought = await noema.think(task)

    assert solution.task_id == task.id
    assert solution.quality.value in ("draft", "acceptable", "good", "excellent", "masterpiece")
    assert solution.confidence > 0
    assert len(thought.steps) > 0
    assert thought.duration_ms > 0

    await noema.shutdown()


# ── Utils Tests ────────────────────────────────────────────────────────────


def test_truncate():
    assert truncate("hello", 10) == "hello"
    assert truncate("hello world", 8) == "hello..."
    assert truncate("hi", 10) == "hi"
    long_text = "a" * 50
    assert len(truncate(long_text, 20)) <= 23  # 20 chars + "..."


def test_chunk_list():
    assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunk_list([], 3) == []


def test_deep_merge():
    base = {"a": 1, "b": {"c": 2}}
    override = {"b": {"d": 3}, "e": 4}
    result = deep_merge(base, override)
    assert result == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}


def test_generate_id():
    id1 = generate_id("test")
    id2 = generate_id("test")
    id3 = generate_id("other")
    assert id1 == id2
    assert id1 != id3
    assert len(id1) == 12


# ── New Kernels Tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_frontend_kernel():
    kernel = FrontendKernel()
    task = Task(
        title="Dashboard App",
        description="React dashboard",
        tags=["react", "nextjs", "tailwind", "dashboard"],
    )
    result = await kernel.execute(task)
    assert result["type"] == "frontend"
    assert "framework" in result
    assert "components" in result
    assert result["framework"]["name"] == "Next.js 14"


@pytest.mark.asyncio
async def test_devops_kernel():
    kernel = DevOpsKernel()
    task = Task(
        title="Deploy Microservice",
        tags=["docker", "kubernetes", "ci-cd"],
    )
    result = await kernel.execute(task)
    assert result["type"] == "devops"
    assert "containerization" in result
    assert "ci_cd" in result
    assert "monitoring" in result


@pytest.mark.asyncio
async def test_data_kernel():
    kernel = DataKernel()
    task = Task(
        title="Analytics Platform",
        tags=["data", "analytics", "user", "order"],
    )
    result = await kernel.execute(task)
    assert result["type"] == "data"
    assert "data_model" in result
    assert len(result["data_model"]["tables"]) > 0


@pytest.mark.asyncio
async def test_ai_ml_kernel():
    kernel = AIMLKernel()
    task = Task(
        title="Fraud Detection",
        tags=["ml", "anomaly", "real-time"],
    )
    result = await kernel.execute(task)
    assert result["type"] == "ai_ml"
    assert "ml_pipeline" in result
    assert "model" in result


# ── Knowledge Graph Tests ──────────────────────────────────────────────────


def test_knowledge_graph_stats():
    kg = KnowledgeGraph()
    stats = kg.get_stats()
    assert stats["total_nodes"] > 20
    assert stats["total_edges"] > 20


def test_knowledge_graph_compatible():
    kg = KnowledgeGraph()
    compat = kg.get_compatible_technologies("python")
    assert "fastapi" in compat.get("supports", []) or "django" in compat.get("supports", [])


def test_knowledge_graph_suggest():
    kg = KnowledgeGraph()
    result = kg.suggest_architecture(["python", "fastapi", "redis"])
    assert result["total_suggestions"] > 0


# ── LLM Provider Tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fallback_llm():
    provider = FallbackProvider()
    assert provider.name == "fallback"
    response = await provider.complete([LLMMessage(role="user", content="test")])
    assert response.content
    assert response.model == "fallback"


@pytest.mark.asyncio
async def test_llm_generate_code():
    provider = FallbackProvider()
    code = await provider.generate_code("Create a hello world", language="python")
    assert isinstance(code, str)
    assert len(code) > 0


def test_create_llm_provider():
    p = create_llm_provider("fallback")
    assert p.name == "fallback"
    p2 = create_llm_provider("ollama")
    assert p2.name == "ollama"
    p3 = create_llm_provider()
    assert p3.name in ("ollama", "fallback")  # depends on config


# ── Pipeline Tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_execution():
    pipeline = create_quick_prototype_pipeline()
    noema = NoemaEngine(worker_count=2, llm_provider="fallback")
    await noema.initialize()

    task = Task(
        title="Simple API",
        tags=["python", "api"],
    )
    result = await pipeline.execute(task, noema=noema)
    assert result.completed_steps > 0
    assert result.total_duration_ms > 0

    await noema.shutdown()


def test_pipeline_fluent_api():
    pipeline = (
        Pipeline("test")
        .add_step("step1", kernel_name="analysis")
        .add_step("step2", kernel_name="codegen")
    )
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0].name == "step1"
    assert pipeline.steps[1].name == "step2"


# ── Feedback Tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_feedback_store():
    import os
    import tempfile

    tmp = os.path.join(tempfile.gettempdir(), "noema_test_feedback.json")

    store = FeedbackStore(persist_path=tmp)
    await store.load()

    solution = Solution(task_id="t1", title="Test", summary="Test")
    task = Task(title="Test Task", tags=["python"])

    entry = await store.record_feedback(
        solution=solution,
        task=task,
        rating=5,
        comments="Great solution!",
    )
    assert entry.rating == 5

    analysis = store.analyze_patterns()
    assert analysis["total_feedback"] == 1
    assert analysis["avg_rating"] == 5.0

    os.unlink(tmp)


# ── Plugin Tests ───────────────────────────────────────────────────────────


def test_plugin_manager():
    pm = PluginManager()
    stats = pm.get_stats()
    assert stats["total_plugins"] == 0


def test_plugin_meta():
    meta = PluginMeta(name="test-plugin", version="1.0.0", author="tester")
    assert meta.name == "test-plugin"
    assert meta.version == "1.0.0"


# ── Scaffolder Tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scaffolder():
    import os
    import tempfile

    tmp = tempfile.mkdtemp()

    scaffolder = ProjectScaffolder(output_dir=tmp)
    solution = Solution(
        task_id="t1",
        title="Test App",
        summary="Test",
        stack=TechStack(languages=["Python"], frameworks=["FastAPI"]),
        code_blocks=[],
    )
    task = Task(title="Test App", tags=["python"])

    result = await scaffolder.scaffold(solution, task)
    assert result["files_created"] > 0
    assert os.path.exists(os.path.join(tmp, "test_app", "README.md"))

    import shutil

    shutil.rmtree(tmp)


# ── Extended Noema Engine Tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_noema_with_llm():
    noema = NoemaEngine(worker_count=2, llm_provider="fallback")
    await noema.initialize()
    assert noema.llm.name == "fallback"
    await noema.shutdown()


@pytest.mark.asyncio
async def test_noema_feedback():
    noema = NoemaEngine(worker_count=2, llm_provider="fallback")
    await noema.initialize()

    task = Task(title="Test", tags=["python"])
    solution = Solution(task_id=task.id, title="Test", summary="Test")

    entry = await noema.record_feedback(solution, task, rating=4, comments="Good")
    assert entry.rating == 4

    analysis = noema.get_feedback_analysis()
    assert analysis["total_feedback"] >= 1

    await noema.shutdown()


@pytest.mark.asyncio
async def test_noema_scaffold():
    import tempfile

    tmp = tempfile.mkdtemp()

    noema = NoemaEngine(worker_count=2, llm_provider="fallback")
    await noema.initialize()

    task = Task(
        title="Scaffold Test",
        description="Test scaffold",
        tags=["python", "api"],
    )
    solution, _ = await noema.think(task)

    result = await noema.scaffold_project(solution, task, tmp)
    assert result["files_created"] > 0

    await noema.shutdown()
    import shutil

    shutil.rmtree(tmp)


@pytest.mark.asyncio
async def test_noema_kernels_count():
    noema = NoemaEngine(worker_count=2, llm_provider="fallback")
    await noema.initialize()
    assert len(noema.kernels) >= 9
    await noema.shutdown()


# ── Memory Tests ────────────────────────────────────────────────────────────


def test_memory_store_episodic():
    import tempfile

    tmp = tempfile.mkdtemp()
    from noema.memory.store import MemoryStore

    store = MemoryStore(persist_dir=tmp)
    ep = store.record_episode(
        task_description="Build REST API",
        solution_summary="FastAPI with PostgreSQL",
        tech_stack="Python, FastAPI",
        outcome="success",
    )
    assert ep.task_description == "Build REST API"
    assert ep.outcome == "success"
    assert len(store.episodic) == 1
    results = store.search_episodes("REST API")
    assert len(results) == 1
    import shutil

    shutil.rmtree(tmp)


def test_memory_store_semantic():
    import tempfile

    tmp = tempfile.mkdtemp()
    from noema.memory.store import MemoryStore

    store = MemoryStore(persist_dir=tmp)
    mem = store.learn_fact(
        topic="python",
        fact="FastAPI is async by default",
        confidence=0.9,
        source="docs",
    )
    assert mem.topic == "python"
    assert mem.confidence == 0.9
    results = store.search_knowledge("fastapi")
    assert len(results) == 1
    import shutil

    shutil.rmtree(tmp)


def test_memory_store_procedural():
    import tempfile

    tmp = tempfile.mkdtemp()
    from noema.memory.store import MemoryStore

    store = MemoryStore(persist_dir=tmp)
    proc = store.store_procedure(
        procedure_name="deploy_api",
        steps=["build", "test", "deploy"],
        tags=["devops"],
    )
    assert proc.procedure_name == "deploy_api"
    assert len(proc.steps) == 3
    found = store.find_procedure("deploy_api")
    assert found is not None
    store.record_procedure_outcome("deploy_api", succeeded=True, duration=10.0)
    found = store.find_procedure("deploy_api")
    assert found.times_applied == 1
    assert found.success_rate == 1.0
    import shutil

    shutil.rmtree(tmp)


def test_memory_stats():
    import tempfile

    tmp = tempfile.mkdtemp()
    from noema.memory.store import MemoryStore

    store = MemoryStore(persist_dir=tmp)
    store.record_episode(
        task_description="test", solution_summary="test solution", outcome="success"
    )
    store.learn_fact(topic="t", fact="f", confidence=0.8)
    stats = store.stats()
    assert stats["episodic_count"] == 1
    assert stats["semantic_count"] == 1
    import shutil

    shutil.rmtree(tmp)


# ── Discovery Tests ─────────────────────────────────────────────────────────


def test_discovery_resources():
    from noema.discovery.keys import KeyDiscovery

    disc = KeyDiscovery()
    resources = disc.discover_resources()
    assert len(resources) >= 3
    kinds = {r.kind for r in resources}
    assert "cpu" in kinds
    assert "ram" in kinds


def test_discovery_all():
    from noema.discovery.keys import KeyDiscovery

    disc = KeyDiscovery()
    result = disc.discover_all()
    assert "keys" in result
    assert "providers_available" in result
    assert "resources" in result


# ── Evolution Tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_evolution_analyze():
    from noema.evolution.engine import EvolutionEngine
    from noema.llm.providers import FallbackProvider

    engine = EvolutionEngine(llm_provider=FallbackProvider(), project_root=".")
    analysis = await engine.analyze_self()
    assert "issues" in analysis
    assert "improvements" in analysis


@pytest.mark.asyncio
async def test_evolution_stats():
    from noema.evolution.engine import EvolutionEngine
    from noema.llm.providers import FallbackProvider

    engine = EvolutionEngine(llm_provider=FallbackProvider(), project_root=".")
    stats = engine.get_stats()
    assert stats["total_cycles"] == 0
    assert stats["total_patches"] == 0


# ── Healer Tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_healer_success():
    from noema.healer.engine import SelfHealer

    healer = SelfHealer()

    async def ok_func():
        return 42

    result = await healer.execute_with_healing(ok_func)
    assert result == 42
    stats = healer.get_stats()
    assert stats["successes"] == 1


@pytest.mark.asyncio
async def test_healer_retry_then_succeed():
    from noema.healer.engine import HealingStrategy, SelfHealer

    strategy = HealingStrategy(
        max_retries=2,
        backoff_base=0.01,
        actions=["retry", "fallback"],
    )
    healer = SelfHealer(strategy=strategy)
    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("fail")
        return "ok"

    result = await healer.execute_with_healing(flaky, fallback="fallback_value")
    assert result == "ok"


@pytest.mark.asyncio
async def test_healer_fallback():
    from noema.healer.engine import HealingStrategy, SelfHealer

    strategy = HealingStrategy(
        max_retries=1,
        backoff_base=0.01,
        actions=["fallback"],
    )
    healer = SelfHealer(strategy=strategy)

    async def always_fail():
        raise ValueError("boom")

    result = await healer.execute_with_healing(always_fail, fallback="fallback")
    assert result == "fallback"


# ── Worker Hierarchy Tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hierarchy_basic():
    from noema.workers.hierarchy import WorkerHierarchy

    async def execute(task):
        return {"done": True, "depth": task.depth}

    hierarchy = WorkerHierarchy(max_depth=2)
    result = await hierarchy.execute("test task", executor=execute)
    assert result.state.value == "completed"
    assert result.result["done"] is True


@pytest.mark.asyncio
async def test_hierarchy_decompose():
    from noema.workers.hierarchy import WorkerHierarchy

    async def decompose(task):
        if task.depth >= 2:
            return []
        return ["sub1", "sub2", "sub3"]

    async def execute(task):
        return task.description

    hierarchy = WorkerHierarchy(max_depth=3, max_concurrent=10)
    result = await hierarchy.execute(
        "root task",
        decomposer=decompose,
        executor=execute,
    )
    assert result.state.value == "completed"
    assert len(result.subtasks) == 3


@pytest.mark.asyncio
async def test_hierarchy_stats():
    from noema.workers.hierarchy import WorkerHierarchy

    async def execute(task):
        return True

    hierarchy = WorkerHierarchy()
    await hierarchy.execute("task1", executor=execute)
    stats = hierarchy.get_stats()
    assert stats["total_tasks"] == 1
    assert stats["completed"] == 1


# ── Ingestion Tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingestion_text():
    import tempfile

    tmp = tempfile.mkdtemp()
    from noema.ingestion.loader import KnowledgeLoader
    from noema.memory.store import MemoryStore

    store = MemoryStore(persist_dir=tmp)
    loader = KnowledgeLoader(knowledge_store=store)
    result = await loader.ingest_text(
        "FastAPI should be used for high-performance APIs. Always use async endpoints.",
        source_name="test_doc",
    )
    assert result.entries_ingested > 0
    assert result.source_type == "text"
    import shutil

    shutil.rmtree(tmp)


@pytest.mark.asyncio
async def test_ingestion_file():
    import os
    import tempfile

    from noema.ingestion.loader import KnowledgeLoader
    from noema.memory.store import MemoryStore

    tmp_dir = tempfile.mkdtemp()
    tmp_file = os.path.join(tmp_dir, "test.txt")
    with open(tmp_file, "w") as f:
        f.write("You should always validate input data. Never trust user input directly.")

    store = MemoryStore(persist_dir=tmp_dir)
    loader = KnowledgeLoader(knowledge_store=store)
    result = await loader.ingest_file(tmp_file, tags=["test"])
    assert result.entries_ingested > 0
    import shutil

    shutil.rmtree(tmp_dir)


# ── Noema Extended Tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_noema_memory_integration():
    noema = NoemaEngine(worker_count=2, llm_provider="fallback")
    await noema.initialize()
    stats = noema.memory_stats()
    assert "episodic_count" in stats
    assert "semantic_count" in stats
    await noema.shutdown()


@pytest.mark.asyncio
async def test_noema_discover():
    noema = NoemaEngine(worker_count=2, llm_provider="fallback")
    await noema.initialize()
    result = noema.discover_resources()
    assert "providers_available" in result
    assert "resources" in result
    await noema.shutdown()


# ── Module Registry Tests ───────────────────────────────────────────────────


def test_module_registry_loads():
    from noema.modules.registry import get_registry

    registry = get_registry()
    assert len(registry.modules) >= 10
    stats = registry.stats()
    assert stats["total_modules"] >= 10


def test_module_registry_list():
    from noema.modules.registry import get_registry

    registry = get_registry()
    modules = registry.list_modules()
    names = [m["name"] for m in modules]
    assert "monitoring" in names
    assert "testing" in names
    assert "auth" in names
    assert "database" in names
    assert "graphql" in names
    assert "mobile" in names
    assert "i18n" in names


# ── Monitoring Module Tests ────────────────────────────────────────────────


def test_monitoring_metrics():
    from noema.modules.monitors.kernel import MonitoringModule

    mod = MonitoringModule()
    mod.record_request("/api/users", "GET", 200, 45.2)
    mod.record_error("auth", "timeout")
    mod.record_business_metric("active_users", 42)
    dashboard = mod.get_dashboard()
    assert dashboard["metrics"]["total_metrics"] >= 2
    assert dashboard["alerts"]["total_rules"] == 0


def test_monitoring_alerts():
    from noema.modules.monitors.kernel import (
        AlertEngine,
        AlertRule,
        AlertSeverity,
        MetricsCollector,
    )

    mc = MetricsCollector()
    ae = AlertEngine()
    ae.add_rule(
        AlertRule(
            name="high_errors",
            metric_name="errors",
            condition="gt",
            threshold=5,
            severity=AlertSeverity.CRITICAL,
        )
    )
    mc.counter("errors", 10)
    fired = ae.evaluate(mc)
    assert len(fired) == 1
    assert fired[0].severity == AlertSeverity.CRITICAL


def test_monitoring_health_checks():
    from noema.modules.monitors.kernel import HealthChecker

    hc = HealthChecker()
    hc.register("db", lambda: {"status": "ok"})
    hc.register("cache", lambda: {"status": "ok"})
    assert hc.overall_status() == "unknown"
    hc.results["db"] = type("HC", (), {"status": "healthy", "latency_ms": 1.0, "message": ""})()
    hc.results["cache"] = type("HC", (), {"status": "healthy", "latency_ms": 0.5, "message": ""})()
    assert hc.overall_status() == "healthy"


# ── Testing Module Tests ───────────────────────────────────────────────────


def test_testing_generate_unit_tests():
    from noema.modules.testing.kernel import TestGenerator

    gen = TestGenerator()
    code = "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b"
    suite = gen.generate_unit_tests(code, "python")
    assert len(suite.test_cases) == 2
    assert suite.framework.value == "pytest"


def test_testing_analyze_codebase():
    from noema.modules.testing.kernel import TestingModule

    mod = TestingModule()
    files = {
        "main.py": "def foo(): pass\ndef bar(): pass\nclass MyClass:\n    def method(self): pass"
    }
    result = mod.analyze_codebase(files)
    assert result["total_files"] == 1
    assert result["total_functions"] == 3


# ── Documentation Module Tests ─────────────────────────────────────────────


def test_docs_readme_generation():
    from noema.modules.documentation.kernel import DocGenerator

    gen = DocGenerator()
    readme = gen.generate_readme("TestProject", "A test project", features=["Fast", "Secure"])
    assert "# TestProject" in readme
    assert "Fast" in readme


def test_docs_openapi():
    from noema.modules.documentation.kernel import DocGenerator

    gen = DocGenerator()
    endpoints = [{"path": "/users", "method": "GET", "summary": "List users"}]
    spec = gen.generate_openapi_spec("MyAPI", "1.0.0", endpoints)
    assert spec["openapi"] == "3.0.3"
    assert "/users" in spec["paths"]


# ── Database Module Tests ──────────────────────────────────────────────────


def test_db_schema_design():
    from noema.modules.database.kernel import DatabaseModule

    mod = DatabaseModule()
    result = mod.design_and_generate(["user management", "authentication"])
    assert len(result["tables"]) >= 2
    assert "CREATE TABLE" in result["sql"]


def test_db_query_optimizer():
    from noema.modules.database.kernel import QueryOptimizer

    qo = QueryOptimizer()
    opts = qo.analyze_query("SELECT * FROM users WHERE name LIKE '%test%'")
    assert len(opts) >= 1
    assert opts[0].impact in ("high", "medium", "low")


# ── Queue Module Tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_queue_job_creation():
    from noema.modules.queues.kernel import JobQueue

    queue = JobQueue("test", max_workers=1)
    job = await queue.enqueue("test_job", {"key": "value"})
    assert job.name == "test_job"
    assert job.status.value == "pending"


@pytest.mark.asyncio
async def test_queue_handler():
    from noema.modules.queues.kernel import JobQueue

    queue = JobQueue("test", max_workers=1)

    async def hello_handler(payload):
        return f"Hello {payload.get('name', 'world')}"

    queue.register_handler("hello", hello_handler)
    await queue.start()
    await queue.enqueue("hello", {"name": "Noema"})
    await asyncio.sleep(0.3)
    stats = queue.get_stats()
    assert stats["completed"] >= 1
    await queue.stop()


# ── Caching Module Tests ───────────────────────────────────────────────────


def test_lru_cache():
    from noema.modules.caching.kernel import LRUCache

    cache = LRUCache(max_size=3)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") == 1
    cache.set("d", 4)  # should evict "b"
    assert cache.get("b") is None
    stats = cache.get_stats()
    assert stats["evictions"] == 1


def test_multilayer_cache():
    from noema.modules.caching.kernel import MultiLayerCache

    cache = MultiLayerCache(l1_max=10, l2_max=100)
    cache.set("key1", "value1", ttl=60)
    assert cache.get("key1") == "value1"
    cache.invalidate("key1")
    assert cache.get("key1") is None


# ── Auth Module Tests ──────────────────────────────────────────────────────


def test_auth_register_and_login():
    from noema.modules.auth.kernel import AuthModule

    mod = AuthModule()
    user = mod.register_user("test@example.com", "testuser", "password123")
    assert user.email == "test@example.com"
    tokens = mod.authenticate("test@example.com", "password123")
    assert tokens is not None
    assert tokens.access_token
    bad = mod.authenticate("test@example.com", "wrong")
    assert bad is None


def test_auth_rbac():
    from noema.modules.auth.kernel import AuthModule

    mod = AuthModule()
    user = mod.register_user("a@b.com", "user1", "pass")
    assert mod.authorize(user.id, "read")
    assert not mod.authorize(user.id, "admin")
    mod.rbac.assign_role(user.id, "admin")
    assert mod.authorize(user.id, "admin")


def test_auth_rate_limiter():
    from noema.modules.auth.kernel import RateLimiter, RateLimitRule

    rl = RateLimiter()
    rl.add_rule(RateLimitRule(name="api", max_requests=3, window_seconds=1.0))
    r1 = rl.check("api", "user1")
    rl.check("api", "user1")
    rl.check("api", "user1")
    r4 = rl.check("api", "user1")
    assert r1["allowed"]
    assert r4["allowed"] is False


def test_auth_password_hashing():
    from noema.modules.auth.kernel import PasswordHasher

    ph = PasswordHasher()
    h = ph.hash_password("secret")
    assert ph.verify_password("secret", h)
    assert not ph.verify_password("wrong", h)


# ── GraphQL Module Tests ───────────────────────────────────────────────────


def test_graphql_schema_generation():
    from noema.modules.graphql.kernel import GraphQLModule

    mod = GraphQLModule()
    result = mod.execute(type("Task", (), {"tags": ["api"]})())
    assert result["type"] == "graphql"
    assert "schema_preview" in result
    assert result["types_count"] >= 3


# ── WebSocket Module Tests ────────────────────────────────────────────────


def test_websocket_rooms():
    from noema.modules.websocket.kernel import RoomManager

    rm = RoomManager()
    rm.join("general", "client1")
    rm.join("general", "client2")
    assert len(rm.get_room_clients("general")) == 2
    stats = rm.get_stats()
    assert stats["rooms"] == 1


@pytest.mark.asyncio
async def test_websocket_event_bus():
    from noema.modules.websocket.kernel import EventBus

    bus = EventBus()
    received = []
    bus.subscribe("test_event", lambda msg: received.append(msg.data))
    await bus.publish("test_event", "hello")
    assert len(received) == 1
    assert received[0] == "hello"


# ── Mobile Module Tests ───────────────────────────────────────────────────


def test_mobile_generate():
    from noema.modules.mobile.kernel import MobileModule

    mod = MobileModule()
    task = type("Task", (), {"tags": ["flutter", "auth", "ecommerce"], "title": "ShopApp"})()
    result = mod.execute(task)
    assert result["framework"] == "flutter"
    assert len(result["screens"]) >= 3


# ── I18n Module Tests ─────────────────────────────────────────────────────


def test_i18n_translations():
    from noema.modules.i18n.kernel import I18nModule

    mod = I18nModule()
    mod.setup_common_translations()
    assert mod.store.get("auth.login", "en") == "Log In"
    assert mod.store.get("auth.login", "ru") == "Войти"
    missing = mod.store.missing_keys("en", "ru")
    assert isinstance(missing, list)


def test_i18n_number_format():
    from noema.modules.i18n.kernel import NumberFormatter

    assert NumberFormatter.format(1234567.89, "en") == "1,234,567.89"
    assert NumberFormatter.format_currency(99.99, "en") == "$99.99"


# ── CLI Generator Module Tests ────────────────────────────────────────────


def test_cli_generator():
    from noema.modules.cli_generator.kernel import CLIGeneratorModule

    mod = CLIGeneratorModule()
    task = type("Task", (), {"tags": ["typer", "deploy"], "title": "deploy-tool"})()
    result = mod.execute(task)
    assert result["framework"] == "typer"
    assert "import typer" in result["code_preview"]


# ── Noema + Modules Integration Tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_noema_modules_list():
    noema = NoemaEngine(worker_count=2, llm_provider="fallback")
    await noema.initialize()
    modules = noema.list_modules()
    assert len(modules) >= 10
    names = [m["name"] for m in modules]
    assert "monitoring" in names
    assert "auth" in names
    await noema.shutdown()


@pytest.mark.asyncio
async def test_noema_execute_module():
    noema = NoemaEngine(worker_count=2, llm_provider="fallback")
    await noema.initialize()
    task = Task(title="Test", tags=["monitoring"])
    result = noema.execute_module("monitoring", task)
    assert result["type"] == "monitoring"
    await noema.shutdown()


@pytest.mark.asyncio
async def test_noema_modules_stats():
    noema = NoemaEngine(worker_count=2, llm_provider="fallback")
    await noema.initialize()
    stats = noema.modules_stats()
    assert stats["total_modules"] >= 10
    await noema.shutdown()


# ── Wave 2: New Module Tests ────────────────────────────────────────────────


def test_module_registry_22_modules():
    from noema.modules.registry import get_registry

    registry = get_registry()
    assert len(registry.modules) >= 22
    names = set(registry.modules.keys())
    assert "security_scanner" in names
    assert "performance" in names
    assert "config" in names
    assert "events" in names
    assert "quality" in names
    assert "containers" in names
    assert "terraform" in names
    assert "data_pipeline" in names
    assert "ml_ops" in names
    assert "gateway" in names


# ── Security Scanner Tests ─────────────────────────────────────────────────


def test_security_scan_code():
    from noema.modules.security_scanner.kernel import SecurityScanner

    scanner = SecurityScanner()
    code = 'query = f"SELECT * FROM users WHERE id = {user_id}"'
    result = scanner.scan_code(code, "python")
    assert result.score <= 100


def test_security_scan_secrets():
    from noema.modules.security_scanner.kernel import SecurityScanner

    scanner = SecurityScanner()
    code = 'API_KEY = "sk-1234567890abcdef"\npassword = "hardcoded123"'
    result = scanner.scan_secrets(code)
    assert len(result.vulnerabilities) > 0


def test_security_module_execute():
    from noema.modules.security_scanner.kernel import SecurityScannerModule

    mod = SecurityScannerModule()
    task = type("Task", (), {"tags": ["python", "api"], "title": "Secure API"})()
    result = mod.execute(task)
    assert "scan_type" in result
    assert "_confidence" in result


# ── Performance Tests ──────────────────────────────────────────────────────


def test_performance_analyze():
    from noema.modules.performance.kernel import Profiler

    profiler = Profiler()
    code = """
for i in range(10000):
    for j in range(10000):
        pass
"""
    bottlenecks = profiler.analyze_code_performance(code)
    assert len(bottlenecks) > 0


def test_performance_module_execute():
    from noema.modules.performance.kernel import PerformanceModule

    mod = PerformanceModule()
    task = type("Task", (), {"tags": ["python", "performance"], "title": "Optimize"})()
    result = mod.execute(task)
    assert "action" in result
    assert "_confidence" in result


# ── Config Tests ───────────────────────────────────────────────────────────


def test_config_generate_schema():
    from noema.modules.config.kernel import ConfigManager

    cm = ConfigManager()
    schema = cm.generate_config_schema(
        {"database": {"required": True}, "cache": {"required": True}, "auth": {"required": True}}
    )
    assert len(schema) >= 3


def test_config_feature_flags():
    from noema.modules.config.kernel import FeatureFlag

    flag = FeatureFlag(name="dark_mode", enabled=True, rollout_percentage=50)
    assert flag.name == "dark_mode"
    assert flag.rollout_percentage == 50


def test_config_module_execute():
    from noema.modules.config.kernel import ConfigModule

    mod = ConfigModule()
    task = type("Task", (), {"tags": ["12factor", "config"], "title": "Config"})()
    result = mod.execute(task)
    assert "action" in result


# ── Events Tests ───────────────────────────────────────────────────────────


def test_event_store():
    from noema.modules.events.kernel import Event, EventStore

    store = EventStore()
    event = Event(type="UserCreated", payload={"user_id": "123"}, aggregate_id="user-1")
    store.append(event)
    events = store.get_events("user-1")
    assert len(events) == 1
    assert events[0].type == "UserCreated"


def test_command_bus():
    from noema.modules.events.kernel import CommandBus

    bus = CommandBus()
    results = []
    bus.register("CreateUser", lambda cmd: results.append(cmd))
    bus.dispatch("CreateUser", {"name": "John"})
    assert len(results) == 1


def test_events_module_execute():
    from noema.modules.events.kernel import EventsModule

    mod = EventsModule()
    task = type("Task", (), {"tags": ["event-sourcing", "cqrs"], "title": "Events"})()
    result = mod.execute(task)
    assert "action" in result


# ── Quality Tests ──────────────────────────────────────────────────────────


def test_quality_analyze():
    from noema.modules.quality.kernel import CodeAnalyzer

    analyzer = CodeAnalyzer()
    code = """
def simple():
    return 42

def complex_func(data):
    result = []
    for item in data:
        if item > 0:
            if item % 2 == 0:
                for i in range(item):
                    if i > 5:
                        result.append(i * 2)
                    else:
                        result.append(i)
            else:
                result.append(item)
        else:
            if item < -10:
                result.append(0)
            else:
                result.append(item * -1)
    return result
"""
    report = analyzer.analyze(code, "python")
    assert report.grade in ("A", "B", "C", "D", "E", "F")
    assert len(report.smells) >= 0


def test_quality_module_execute():
    from noema.modules.quality.kernel import QualityModule

    mod = QualityModule()
    task = type("Task", (), {"tags": ["quality", "code-analysis"], "title": "Quality"})()
    result = mod.execute(task)
    assert "action" in result


# ── Containers Tests ──────────────────────────────────────────────────────


def test_dockerfile_generation():
    from noema.modules.containers.kernel import DockerfileGenerator

    gen = DockerfileGenerator()
    dockerfile = gen.generate("python")
    assert "FROM" in dockerfile
    assert "python" in dockerfile.lower()


def test_docker_compose_generation():
    from noema.modules.containers.kernel import DockerComposeGenerator

    gen = DockerComposeGenerator()
    compose = gen.generate([{"name": "api", "image": "python:3.12", "ports": ["8000:8000"]}])
    assert "services" in compose or "api" in compose


def test_kubernetes_generation():
    from noema.modules.containers.kernel import KubernetesGenerator

    gen = KubernetesGenerator()
    deployment = gen.generate_deployment("myapp", "myapp:latest", 3)
    assert "myapp" in deployment


def test_containers_module_execute():
    from noema.modules.containers.kernel import ContainersModule

    mod = ContainersModule()
    task = type("Task", (), {"tags": ["docker", "kubernetes"], "title": "Deploy"})()
    result = mod.execute(task)
    assert "action" in result


# ── Terraform Tests ────────────────────────────────────────────────────────


def test_terraform_generate():
    from noema.modules.terraform.kernel import TerraformGenerator

    gen = TerraformGenerator()
    provider = gen.generate_provider("aws", {"region": "us-east-1"})
    assert "provider" in provider
    assert "aws" in provider
    resource = gen.generate_resource(
        "aws_instance", "web", {"ami": "ami-123", "instance_type": "t3.micro"}
    )
    assert "resource" in resource
    assert "aws_instance" in resource


def test_terraform_module_execute():
    from noema.modules.terraform.kernel import TerraformModule

    mod = TerraformModule()
    task = type("Task", (), {"tags": ["aws", "terraform", "infrastructure"], "title": "IaC"})()
    result = mod.execute(task)
    assert "action" in result


# ── Data Pipeline Tests ───────────────────────────────────────────────────


def test_data_pipeline():
    from noema.modules.data_pipeline.kernel import DataPipeline, PipelineStep

    pipeline = DataPipeline(name="etl")
    pipeline.add_step(PipelineStep(name="extract", type="extract", config={"source": "s3"}))
    pipeline.add_step(
        PipelineStep(name="transform", type="transform", config={"ops": ["filter", "map"]})
    )
    pipeline.add_step(PipelineStep(name="load", type="load", config={"target": "postgres"}))
    assert len(pipeline._steps) == 3
    validation = pipeline.validate()
    assert isinstance(validation, dict)


def test_data_pipeline_module_execute():
    from noema.modules.data_pipeline.kernel import DataPipelineModule

    mod = DataPipelineModule()
    task = type("Task", (), {"tags": ["etl", "data"], "title": "Pipeline"})()
    result = mod.execute(task)
    assert "action" in result


# ── ML Ops Tests ──────────────────────────────────────────────────────────


def test_mlops_pipeline():
    from noema.modules.ml_ops.kernel import ExperimentConfig, MLPipeline

    pipeline = MLPipeline()
    ExperimentConfig(
        name="exp1",
        model="random_forest",
        dataset="iris.csv",
        hyperparameters={"n_estimators": 100},
    )
    pipeline.define_pipeline(
        [
            {"name": "data_load", "type": "extract"},
            {"name": "preprocess", "type": "transform"},
            {"name": "train", "type": "train"},
            {"name": "evaluate", "type": "evaluate"},
            {"name": "deploy", "type": "deploy"},
        ]
    )
    assert len(pipeline._steps) == 5


def test_mlops_module_execute():
    from noema.modules.ml_ops.kernel import MLOpsModule

    mod = MLOpsModule()
    task = type(
        "Task", (), {"tags": ["ml", "classification", "training"], "title": "ML Pipeline"}
    )()
    result = mod.execute(task)
    assert "action" in result


# ── Gateway Tests ──────────────────────────────────────────────────────────


def test_gateway_config():
    from noema.modules.gateway.kernel import GatewayConfig, GatewayMiddleware, GatewayRoute

    config = GatewayConfig()
    route = GatewayRoute(path="/api/users", method="GET", upstream="user-service:8080")
    config.add_route(route)
    middleware = GatewayMiddleware(name="auth", type="auth", config={"jwt": True})
    config.add_middleware(middleware)
    assert len(config._routes) == 1
    assert len(config._middleware) == 1


def test_gateway_nginx():
    from noema.modules.gateway.kernel import GatewayConfig, GatewayRoute

    config = GatewayConfig()
    config.add_route(GatewayRoute(path="/api", method="GET", upstream="backend:8000"))
    nginx = config.generate_nginx_config()
    assert "upstream" in nginx.lower() or "location" in nginx.lower()


def test_gateway_module_execute():
    from noema.modules.gateway.kernel import GatewayModule

    mod = GatewayModule()
    task = type("Task", (), {"tags": ["api", "gateway", "routing"], "title": "Gateway"})()
    result = mod.execute(task)
    assert "action" in result


# ── All 22 Modules Execute Test ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_all_22_modules_execute():
    noema = NoemaEngine(worker_count=2, llm_provider="fallback")
    await noema.initialize()
    task = Task(title="Test All Modules", tags=["python", "api", "web"])
    results = noema.execute_all_modules(task)
    assert len(results) >= 22
    for name, result in results.items():
        assert "_confidence" in result, f"Module {name} missing '_confidence'"
    await noema.shutdown()


@pytest.mark.asyncio
async def test_noema_modules_count():
    noema = NoemaEngine(worker_count=2, llm_provider="fallback")
    await noema.initialize()
    stats = noema.modules_stats()
    assert stats["total_modules"] >= 22
    await noema.shutdown()
