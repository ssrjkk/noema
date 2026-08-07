"""Integration tests for noema API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from noema.api.server import app
from noema.core.types import (
    ArchitecturePattern,
    CodeBlock,
    Solution,
    SolutionQuality,
    TechStack,
    ThoughtProcess,
)
from noema.resilience.cancellation import CancellationManager

# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_solution(task_id: str = "task-1", title: str = "Test Solution") -> Solution:
    return Solution(
        id="sol-1",
        task_id=task_id,
        title=title,
        summary="A test solution summary",
        architecture=ArchitecturePattern(
            name="Microservices",
            description="Microservice architecture",
        ),
        stack=TechStack(
            languages=["Python"],
            frameworks=["FastAPI"],
            databases=["PostgreSQL"],
        ),
        code_blocks=[
            CodeBlock(filename="main.py", language="python", content="print('hello')"),
        ],
        deployment={"type": "docker"},
        performance_notes=["Fast response time"],
        security_notes=["Use HTTPS"],
        quality=SolutionQuality.GOOD,
        confidence=0.85,
        metadata={"version": "1.0"},
    )


def _make_thought(task_id: str = "task-1") -> ThoughtProcess:
    tp = ThoughtProcess(task_id=task_id)
    tp.add_step("analyze", "input", "output", 0.9)
    tp.add_step("codegen", "input", "output", 0.85)
    tp.duration_ms = 1234.5
    return tp


def _make_noema_mock():
    noema = MagicMock()
    noema.worker_pool.stats = {"active_workers": 1, "max_workers": 4, "queue_size": 0}
    noema.neurosymbolic = MagicMock()
    noema.neurosymbolic.get_metrics.return_value = {
        "tasks_processed": 42,
        "tasks_successful": 40,
        "tasks_failed": 2,
        "success_rate": 0.9524,
        "total_llm_calls": 500,
        "total_refinements": 10,
    }
    noema.knowledge.get_stats.return_value = {"total_entries": 100, "categories": 5}
    noema.knowledge.search = AsyncMock(return_value=[{"id": "1", "title": "API Design"}])
    noema.kernels = {
        "arch": MagicMock(name="Architecture"),
        "code": MagicMock(name="Codegen"),
    }
    (noema.kernels["arch"]).name = "Architecture"
    (noema.kernels["arch"]).description = "Architecture kernel"
    (noema.kernels["code"]).name = "Codegen"
    (noema.kernels["code"]).description = "Codegen kernel"
    noema.orchestrator.agents = {
        "arch": MagicMock(name="Architect"),
    }
    (noema.orchestrator.agents["arch"]).name = "Architect"
    (noema.orchestrator.agents["arch"]).role = MagicMock()
    (noema.orchestrator.agents["arch"]).role.value = "architect"
    (noema.orchestrator.agents["arch"]).expertise = "system design"
    noema.memory_stats = MagicMock(
        return_value={
            "episodic_count": 10,
            "procedural_count": 5,
            "knowledge_count": 20,
        }
    )
    return noema


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_state():
    import noema.api.server as srv

    srv._cancellation_mgr = CancellationManager()
    srv._noema = None
    srv._start_time = 1000.0
    import noema.api.webhooks as wh

    wh._dispatcher = None
    app.state.noema = _make_noema_mock()
    app.state.audit_logger = AsyncMock()
    app.state.quota_manager = AsyncMock()
    app.state.feature_flags = AsyncMock()
    app.state.degradation = MagicMock()
    app.state.webhook_dispatcher = MagicMock()
    yield


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_noema():
    noema = _make_noema_mock()
    with patch("noema.api.server._get_noema", return_value=noema):
        yield noema


@pytest.fixture
def mock_think():
    solution = _make_solution()
    thought = _make_thought()

    async def _execute(task_id, coro):
        return solution, thought

    import noema.api.server as srv

    with patch.object(srv._cancellation_mgr, "execute_with_cancellation", side_effect=_execute):
        yield


@pytest.fixture
def mock_cancel():
    import noema.api.server as srv

    with patch.object(srv._cancellation_mgr, "cancel", return_value=False):
        yield


valid_task = {
    "title": "Build a REST API",
    "description": "Build a REST API for user management",
    "complexity": "simple",
    "tags": ["api", "rest"],
}


# ─── Health & Readiness ─────────────────────────────────────────────────────


class TestHealth:
    def test_health_endpoint(self, client, mock_noema):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["version"] == "1.0.0"
        assert "uptime_s" in body
        assert body["db"] == "ok"

    def test_ready_endpoint(self, client, mock_noema):
        resp = client.get("/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert "ready" in body
        assert "checks" in body


# ─── Think Endpoints ────────────────────────────────────────────────────────


class TestThink:
    def test_think_simple_task(self, client, mock_noema, mock_think):
        resp = client.post("/think", json=valid_task)
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "sol-1"
        assert body["task_id"] == "task-1"
        assert body["title"] == "Test Solution"
        assert body["quality"] == "good"
        assert body["confidence"] == 0.85

    def test_think_detail(self, client, mock_noema, mock_think):
        resp = client.post("/think/detail", json=valid_task)
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "sol-1"
        assert body["architecture"] is not None
        assert body["architecture"]["name"] == "Microservices"
        assert isinstance(body["code_blocks"], list)
        assert len(body["code_blocks"]) == 1
        assert isinstance(body["stack"], dict)
        assert isinstance(body["deployment"], dict)
        assert isinstance(body["metadata"], dict)

    def test_think_invalid_title(self, client):
        resp = client.post("/think", json={"title": "", "description": "test"})
        assert resp.status_code == 422


# ─── Admin Endpoints ────────────────────────────────────────────────────────


class TestAdmin:
    def test_admin_metrics(self, client, mock_noema):
        resp = client.get("/admin/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "tasks" in body
        assert body["tasks"]["total"] == 42
        assert body["tasks"]["success_rate"] == 0.9524

    def test_admin_tasks_history(self, client):
        app.state.audit_logger.query = AsyncMock(return_value=[{"id": "evt-1"}, {"id": "evt-2"}])
        resp = client.get("/admin/tasks/history")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)

    def test_admin_neurosymbolic_stats(self, client, mock_noema):
        resp = client.get("/admin/neurosymbolic/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert "enabled" in body
        assert "settings" in body
        assert "metrics" in body or not body["enabled"]


# ─── Webhook Endpoints ──────────────────────────────────────────────────────


class TestWebhooks:
    def test_register_webhook(self, client):
        resp = client.post(
            "/webhooks/register",
            json={
                "url": "https://example.com/hook",
                "events": ["task.completed"],
                "retry_count": 3,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "webhook_id" in body
        assert body["url"] == "https://example.com/hook"
        assert body["events"] == ["task.completed"]

    def test_list_webhooks(self, client):
        client.post("/webhooks/register", json={"url": "https://example.com/hook"})
        resp = client.get("/webhooks")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, dict)
        for _wid, info in body.items():
            assert "***" in info["url"]

    def test_unregister_webhook(self, client):
        reg_resp = client.post("/webhooks/register", json={"url": "https://example.com/hook"})
        wid = reg_resp.json()["webhook_id"]
        resp = client.delete(f"/webhooks/{wid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unregistered"

    def test_unregister_nonexistent(self, client):
        resp = client.delete("/webhooks/nonexistent")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unregistered"


# ─── Knowledge ──────────────────────────────────────────────────────────────


class TestKnowledge:
    def test_knowledge_stats(self, client, mock_noema):
        resp = client.get("/knowledge/stats")
        assert resp.status_code == 200
        assert resp.json() == {"total_entries": 100, "categories": 5}

    def test_knowledge_search(self, client, mock_noema):
        resp = client.get("/knowledge/search?q=api")
        assert resp.status_code == 200
        assert resp.json() == [{"id": "1", "title": "API Design"}]


# ─── Kernels & Agents ───────────────────────────────────────────────────────


class TestKernelsAndAgents:
    def test_list_kernels(self, client, mock_noema):
        resp = client.get("/kernels")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        names = [k["name"] for k in body]
        assert "Architecture" in names

    def test_list_agents(self, client, mock_noema):
        resp = client.get("/agents")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        names = [a["name"] for a in body]
        assert "Architect" in names


# ─── Cancellation ───────────────────────────────────────────────────────────


class TestCancellation:
    def test_cancel_nonexistent(self, client, mock_cancel):
        resp = client.delete("/think/nonexistent")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_found_or_completed"


# ─── Headers & CORS ─────────────────────────────────────────────────────────


class TestHeadersAndCORS:
    def test_cors_headers(self, client, mock_noema):
        resp = client.get("/health", headers={"Origin": "http://example.com"})
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "*"


# ─── Error Handling ─────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_404(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404

    def test_method_not_allowed(self, client):
        resp = client.put("/health")
        assert resp.status_code == 405
