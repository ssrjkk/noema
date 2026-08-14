"""Coverage tests for the service layer: ``noema/services/reasoning.py`` and ``noema/services/knowledge.py``."""

import json

import pytest

from noema.core.chain_of_thought import ChainOfThought
from noema.core.events import EventBus
from noema.core.types import (
    SolutionQuality,
    Task,
    TaskComplexity,
    TechStack,
)
from noema.knowledge.graph import KnowledgeGraph
from noema.knowledge.store import KnowledgeStore
from noema.llm.providers import BaseLLMProvider, LLMResponse
from noema.services.knowledge import KnowledgeService
from noema.services.reasoning import ReasoningService

# ── ReasoningService ────────────────────────────────────────────────────────


class _ScriptedLLM(BaseLLMProvider):
    """Returns step-specific JSON so solution assembly can extract structures."""

    def __init__(self, empty: bool = False) -> None:
        super().__init__()
        self.empty = empty

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def model_name(self) -> str:
        return "scripted"

    async def _complete(self, messages, temperature=0.7, max_tokens=4096):
        content = messages[-1].content if messages else ""
        if self.empty:
            return LLMResponse(content="{}", model="scripted", tokens_used=10)
        if "production-ready code" in content:
            return LLMResponse(
                content=json.dumps(
                    {
                        "files": [
                            {
                                "path": "main.py",
                                "language": "python",
                                "content": "print('hi')\n",
                            }
                        ]
                    }
                ),
                model="scripted",
                tokens_used=100,
            )
        if "Design the optimal architecture" in content:
            return LLMResponse(
                content=json.dumps(
                    {"pattern": {"name": "microservices", "description": "service split"}}
                ),
                model="scripted",
                tokens_used=100,
            )
        if "Perform final review" in content:
            return LLMResponse(
                content=json.dumps({"final_summary": "Everything checks out"}),
                model="scripted",
                tokens_used=100,
            )
        return LLMResponse(content='{"ok": true}', model="scripted", tokens_used=10)


@pytest.mark.asyncio
async def test_reasoning_service_think_extracts_solution():
    llm = _ScriptedLLM()
    cot = ChainOfThought(llm)
    bus = EventBus()
    svc = ReasoningService(llm=llm, cot=cot, event_bus=bus)
    task = Task(
        title="Build an API",
        description="REST API",
        tags=["api"],
        complexity=TaskComplexity.MODERATE,
    )

    solution, thought = await svc.think(task)

    assert solution.quality == SolutionQuality.GOOD
    assert len(solution.code_blocks) == 1
    assert solution.code_blocks[0].filename == "main.py"
    assert solution.architecture is not None
    assert solution.architecture.name == "microservices"
    assert solution.summary == "Everything checks out"
    assert solution.confidence == 0.7
    assert solution.metadata["llm_provider"] == "scripted"
    assert solution.metadata["reasoning_steps"] == len(cot.get_steps())

    assert thought.task_id == task.id
    assert thought.duration_ms >= 0
    assert '"final_summary"' in thought.final_reasoning
    assert len(thought.steps) == len(cot.get_steps())
    assert all(s.output_summary for s in thought.steps)

    # the reasoning.completed event was published onto the bus
    assert bus.stats["published"] == 1


@pytest.mark.asyncio
async def test_reasoning_service_think_fallback_to_draft():
    llm = _ScriptedLLM(empty=True)
    cot = ChainOfThought(llm)
    svc = ReasoningService(llm=llm, cot=cot, event_bus=None)
    task = Task(title="Vague task", description="No details", tags=[])

    solution, thought = await svc.think(task)

    assert solution.quality == SolutionQuality.DRAFT
    assert solution.code_blocks == []
    assert solution.architecture is None
    assert solution.confidence == 0.4
    assert solution.summary == "No details"  # fallback to the task description
    assert thought.final_reasoning == "{}"


# ── KnowledgeService ────────────────────────────────────────────────────────


def _svc(tmp_path) -> KnowledgeService:
    store = KnowledgeStore(persist_path=str(tmp_path / "knowledge.json"))
    graph = KnowledgeGraph()
    return KnowledgeService(store=store, graph=graph)


@pytest.mark.asyncio
async def test_knowledge_service_initialize_and_stats(tmp_path):
    svc = _svc(tmp_path)
    await svc.initialize()
    stats = svc.get_stats()
    assert stats["total_entries"] > 0
    assert stats["total_patterns"] > 0
    assert "graph" in stats


@pytest.mark.asyncio
async def test_knowledge_service_search_and_context(tmp_path):
    svc = _svc(tmp_path)
    await svc.initialize()

    results = await svc.search("fastapi", top_k=3)
    assert isinstance(results, list)
    assert all("score" in r for r in results)

    task = Task(title="Build a fastapi service", description="async api", tags=["api", "python"])
    ctx = await svc.gather_context(task)
    assert isinstance(ctx, str)
    assert ctx  # builtin corpus always returns matches

    graph_ctx = svc.gather_graph_context(task)
    assert isinstance(graph_ctx, str)


@pytest.mark.asyncio
async def test_knowledge_service_select_stack(tmp_path):
    svc = _svc(tmp_path)
    task = Task(title="Build an api", tags=["python", "api"])
    stack = await svc.select_stack(task)
    assert isinstance(stack, TechStack)

    preferred = TechStack(languages=["Rust"], frameworks=["Axum"])
    pref_task = Task(title="x", preferred_stack=preferred)
    assert await svc.select_stack(pref_task) is preferred


@pytest.mark.asyncio
async def test_knowledge_service_ingest_and_shutdown(tmp_path):
    svc = _svc(tmp_path)
    result = await svc.ingest_text(
        "Always use FastAPI for high-performance async backends.", source="manual"
    )
    assert result["entries_ingested"] > 0
    assert svc.store.persist_path.exists()

    await svc.shutdown()
    assert svc.store.persist_path.exists()
