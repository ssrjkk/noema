"""Tests for noema.services.knowledge — knowledge service wrapper."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from structlog.testing import capture_logs

from noema.core.types import Task, TechStack
from noema.ingestion.loader import IngestionResult
from noema.services.knowledge import KnowledgeService


def _event(logs, name):
    """Fetch the single structlog entry with this event name."""
    matches = [e for e in logs if e["event"] == name]
    assert len(matches) == 1, [e["event"] for e in logs]
    return matches[0]


def _patch_loader(monkeypatch, method, *, entries):
    """Replace the lazily-imported KnowledgeLoader with a real-result recording stub.

    ``IngestionResult`` is the genuine pydantic model, so ``model_dump()`` output
    is exercised rather than a hand-rolled dict.
    """
    result = IngestionResult(entries_ingested=entries)
    loader = AsyncMock(name=method)
    getattr(loader, method).return_value = result
    ctor = MagicMock(name="KnowledgeLoader", return_value=loader)
    monkeypatch.setattr("noema.ingestion.loader.KnowledgeLoader", ctor)
    return SimpleNamespace(ctor=ctor, loader=loader, result=result)


@pytest.fixture
def mock_store():
    store = AsyncMock()
    store.load = AsyncMock()
    store.persist = AsyncMock()
    store.search = AsyncMock(return_value=[])
    store.find_relevant_stacks = AsyncMock(return_value=[])
    store.get_stats = MagicMock(return_value={"entries": 10, "categories": 5})
    return store


@pytest.fixture
def mock_graph():
    graph = MagicMock()
    graph.suggest_architecture = MagicMock(return_value={"components": []})
    graph.get_stats = MagicMock(return_value={"nodes": 20, "edges": 30})
    return graph


@pytest.fixture
def mock_event_bus():
    bus = AsyncMock()
    bus.emit = AsyncMock()
    return bus


@pytest.fixture
def service(mock_store, mock_graph, mock_event_bus):
    return KnowledgeService(mock_store, mock_graph, mock_event_bus)


class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_loads_store(self, service, mock_store):
        with capture_logs() as logs:
            await service.initialize()
        mock_store.load.assert_awaited_once()
        entry = _event(logs, "knowledge_service_initialized")
        assert entry["stats"] == {"entries": 10, "categories": 5}


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_persists_store(self, service, mock_store):
        await service.shutdown()
        mock_store.persist.assert_awaited_once()
        mock_store.load.assert_not_awaited()


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_delegates_to_store(self, service, mock_store):
        mock_store.search.return_value = [{"title": "result1"}]
        results = await service.search("query", top_k=3)
        mock_store.search.assert_awaited_once_with("query", top_k=3)
        assert results == [{"title": "result1"}]

    @pytest.mark.asyncio
    async def test_search_default_top_k(self, service, mock_store):
        await service.search("query")
        mock_store.search.assert_awaited_once_with("query", top_k=5)


class TestGatherContext:
    @pytest.mark.asyncio
    async def test_gather_context_builds_query(self, service, mock_store):
        task = Task(title="Build API", description="FastAPI service", tags=["python", "api"])
        mock_store.search.return_value = [
            {"title": "FastAPI Guide", "content": "How to use FastAPI", "type": "guide"}
        ]
        context = await service.gather_context(task)
        assert "FastAPI Guide" in context
        assert "How to use FastAPI" in context
        assert "[guide]" in context
        mock_store.search.assert_awaited_once_with("Build API FastAPI service python api", top_k=5)

    @pytest.mark.asyncio
    async def test_gather_context_defaults_type_when_missing(self, service, mock_store):
        task = Task(title="Task")
        mock_store.search.return_value = [{"title": "Item", "content": "Body"}]
        assert await service.gather_context(task) == "[knowledge] Item: Body"

    @pytest.mark.asyncio
    async def test_gather_context_handles_missing_fields(self, service, mock_store):
        task = Task(title="Task")
        mock_store.search.return_value = [{"name": "Item", "description": "Desc"}]
        context = await service.gather_context(task)
        assert context == "[knowledge] Item: Desc"

    @pytest.mark.asyncio
    async def test_gather_context_truncates_content(self, service, mock_store):
        task = Task(title="Task")
        mock_store.search.return_value = [{"title": "Item", "content": "x" * 500}]
        context = await service.gather_context(task)
        assert context == f"[knowledge] Item: {'x' * 300}"

    @pytest.mark.asyncio
    async def test_gather_context_joins_multiple_results(self, service, mock_store):
        task = Task(title="Task")
        mock_store.search.return_value = [
            {"title": "A", "content": "1"},
            {"title": "B", "content": "2"},
        ]
        assert await service.gather_context(task) == "[knowledge] A: 1\n[knowledge] B: 2"

    @pytest.mark.asyncio
    async def test_gather_context_empty_results(self, service, mock_store):
        assert await service.gather_context(Task(title="Task")) == ""


class TestGatherGraphContext:
    def test_gather_graph_context_formats_components(self, service, mock_graph):
        task = Task(title="Task", tags=["api"])
        mock_graph.suggest_architecture.return_value = {
            "components": [
                {"from": "Client", "to": "API", "relationship": "calls"},
                {"from": "API", "to": "DB", "relationship": "queries"},
            ]
        }
        context = service.gather_graph_context(task)
        assert context == "Client -> API (calls)\nAPI -> DB (queries)"
        mock_graph.suggest_architecture.assert_called_once_with(["api"])

    def test_gather_graph_context_limits_components(self, service, mock_graph):
        task = Task(title="Task", tags=["api"])
        components = [{"from": f"A{i}", "to": f"B{i}", "relationship": "rel"} for i in range(20)]
        mock_graph.suggest_architecture.return_value = {"components": components}
        context = service.gather_graph_context(task)
        assert context.split("\n") == [f"A{i} -> B{i} (rel)" for i in range(15)]

    def test_gather_graph_context_empty(self, service, mock_graph):
        task = Task(title="Task", tags=[])
        mock_graph.suggest_architecture.return_value = {"components": []}
        context = service.gather_graph_context(task)
        assert context == ""

    def test_gather_graph_context_missing_components_key(self, service, mock_graph):
        mock_graph.suggest_architecture.return_value = {}
        assert service.gather_graph_context(Task(title="Task", tags=["api"])) == ""


class TestSelectStack:
    @pytest.mark.asyncio
    async def test_select_stack_prefers_task_stack(self, service, mock_store):
        preferred = TechStack(languages=["Go"], frameworks=["Gin"])
        task = Task(title="Task", preferred_stack=preferred)
        result = await service.select_stack(task)
        assert result == preferred
        mock_store.find_relevant_stacks.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_select_stack_finds_relevant(self, service, mock_store):
        task = Task(title="Task")
        candidate = TechStack(languages=["Python"], frameworks=["Django"])
        mock_store.find_relevant_stacks.return_value = [candidate, TechStack(languages=["Rust"])]
        result = await service.select_stack(task)
        assert result == candidate
        mock_store.find_relevant_stacks.assert_awaited_once_with(task)

    @pytest.mark.asyncio
    async def test_select_stack_default_fallback(self, service, mock_store):
        task = Task(title="Task")
        mock_store.find_relevant_stacks.return_value = []
        result = await service.select_stack(task)
        assert result == TechStack(
            languages=["Python", "TypeScript"],
            frameworks=["FastAPI", "React"],
            databases=["PostgreSQL", "Redis"],
            infrastructure=["Docker"],
        )


class TestGetStats:
    def test_get_stats_combines_store_and_graph(self, service, mock_store, mock_graph):
        stats = service.get_stats()
        assert stats == {
            "entries": 10,
            "categories": 5,
            "graph": {"nodes": 20, "edges": 30},
        }
        mock_store.get_stats.assert_called_once_with()
        mock_graph.get_stats.assert_called_once_with()


class TestIngestFile:
    @pytest.mark.asyncio
    async def test_ingest_file_calls_loader(self, service, mock_store, mock_event_bus, monkeypatch):
        patch_ = _patch_loader(monkeypatch, "ingest_file", entries=5)

        result = await service.ingest_file("/path/to/file.md")
        patch_.ctor.assert_called_once_with(knowledge_store=mock_store)
        patch_.loader.ingest_file.assert_awaited_once_with("/path/to/file.md", tags=["ingested"])
        assert result["entries_ingested"] == 5
        mock_event_bus.emit.assert_awaited_once_with(
            "knowledge.file_ingested",
            {"path": "/path/to/file.md", "entries": 5},
            source="knowledge_service",
        )

    @pytest.mark.asyncio
    async def test_ingest_file_without_event_bus(self, mock_store, mock_graph, monkeypatch):
        service = KnowledgeService(mock_store, mock_graph, event_bus=None)
        patch_ = _patch_loader(monkeypatch, "ingest_file", entries=3)

        result = await service.ingest_file("/path/to/file.md")
        assert result["entries_ingested"] == 3
        patch_.loader.ingest_file.assert_awaited_once_with("/path/to/file.md", tags=["ingested"])


class TestIngestDirectory:
    @pytest.mark.asyncio
    async def test_ingest_directory_calls_loader(self, service, mock_event_bus, monkeypatch):
        patch_ = _patch_loader(monkeypatch, "ingest_directory", entries=10)

        result = await service.ingest_directory("/path/to/dir", patterns=["*.md"])
        assert result["entries_ingested"] == 10
        patch_.loader.ingest_directory.assert_awaited_once_with(
            "/path/to/dir", patterns=["*.md"], tags=["ingested"]
        )
        mock_event_bus.emit.assert_awaited_once_with(
            "knowledge.directory_ingested",
            {"path": "/path/to/dir", "entries": 10},
            source="knowledge_service",
        )

    @pytest.mark.asyncio
    async def test_ingest_directory_defaults_patterns(self, service, monkeypatch):
        patch_ = _patch_loader(monkeypatch, "ingest_directory", entries=0)
        await service.ingest_directory("/path/to/dir")
        patch_.loader.ingest_directory.assert_awaited_once_with(
            "/path/to/dir", patterns=None, tags=["ingested"]
        )


class TestIngestText:
    @pytest.mark.asyncio
    async def test_ingest_text_calls_loader(self, service, mock_event_bus, monkeypatch):
        patch_ = _patch_loader(monkeypatch, "ingest_text", entries=1)

        result = await service.ingest_text("Some text", source="test")
        assert result["entries_ingested"] == 1
        patch_.loader.ingest_text.assert_awaited_once_with(
            "Some text", source_name="test", tags=["ingested"]
        )
        mock_event_bus.emit.assert_awaited_once_with(
            "knowledge.text_ingested",
            {"source": "test", "entries": 1},
            source="knowledge_service",
        )

    @pytest.mark.asyncio
    async def test_ingest_text_default_source(self, service, monkeypatch):
        patch_ = _patch_loader(monkeypatch, "ingest_text", entries=1)
        await service.ingest_text("Some text")
        patch_.loader.ingest_text.assert_awaited_once_with(
            "Some text", source_name="direct", tags=["ingested"]
        )

    @pytest.mark.asyncio
    async def test_ingest_text_result_is_serialized_model(self, service, monkeypatch):
        """The wrapper must hand back the loader's model_dump(), not a partial dict."""
        patch_ = _patch_loader(monkeypatch, "ingest_text", entries=2)
        result = await service.ingest_text("Some text")
        assert result == patch_.result.model_dump()
        assert set(result) == set(IngestionResult.model_fields)
