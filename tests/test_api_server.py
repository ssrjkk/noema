"""Tests for API server endpoints — /workers/stats, /tasks/active, /health/infra, etc."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest.fixture
def mock_noema():
    """Mock NoemaEngine with minimal attributes."""
    engine = MagicMock()
    engine.worker_pool.stats = {"workers": 2, "active": 1, "queued": 0}

    # Create proper kernel mock
    kernel = MagicMock()
    kernel.name = "test_kernel"
    kernel.description = "Test kernel"
    engine.kernels = {"test_kernel": kernel}

    # Create proper agent mock
    agent = MagicMock()
    agent.name = "test_agent"
    agent.role = MagicMock()
    agent.role.value = "assistant"
    agent.expertise = ["python", "testing"]
    engine.orchestrator.agents = {"test_agent": agent}

    return engine


@pytest.fixture
def mock_cancellation_mgr():
    """Mock CancellationManager."""
    mgr = MagicMock()
    mgr.get_active.return_value = ["task-1", "task-2"]
    mgr.cancel.return_value = True
    return mgr


@pytest.fixture
def mock_degradation():
    """Mock GracefulDegradation."""
    deg = MagicMock()
    deg.check_health = AsyncMock()
    deg.get_status = MagicMock(return_value={"redis": "ok", "postgresql": "ok"})
    return deg


@pytest.fixture
def mock_feature_flags():
    """Mock FeatureFlagService."""
    ff = AsyncMock()
    ff.get_all_flags.return_value = {"feature_x": True, "feature_y": False}
    return ff


@pytest.fixture
def app(mock_noema, mock_cancellation_mgr, mock_degradation, mock_feature_flags):
    """FastAPI app with mocked dependencies."""
    import noema.api.server as server_mod

    # Store original values
    original_noema = server_mod._noema
    original_mgr = server_mod._cancellation_mgr

    # Set mocks - both global _noema and app.state
    server_mod._noema = mock_noema
    server_mod.app.state.noema = mock_noema
    server_mod.app.state.degradation = mock_degradation
    server_mod.app.state.feature_flags = mock_feature_flags

    # Mock cancellation manager
    server_mod._cancellation_mgr = mock_cancellation_mgr

    yield server_mod.app

    # Restore
    server_mod._noema = original_noema
    server_mod._cancellation_mgr = original_mgr


@pytest.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestWorkerStats:
    async def test_worker_stats_returns_pool_statistics(self, client: AsyncClient):
        resp = await client.get("/workers/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "workers" in data
        assert "active" in data
        assert data["workers"] == 2
        assert data["active"] == 1


class TestListActiveTasks:
    async def test_list_active_tasks_returns_running_ids(self, client: AsyncClient):
        resp = await client.get("/tasks/active")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_tasks" in data
        assert data["active_tasks"] == ["task-1", "task-2"]


class TestHealthInfra:
    async def test_health_infra_returns_degradation_status(self, client: AsyncClient):
        resp = await client.get("/health/infra")
        assert resp.status_code == 200
        data = resp.json()
        assert "redis" in data
        assert "postgresql" in data
        assert data["redis"] == "ok"
        assert data["postgresql"] == "ok"

    async def test_health_infra_without_degradation(self, app: FastAPI):
        """When degradation service is not available, return unknown status."""
        app.state.degradation = None
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/infra")
        assert resp.status_code == 200
        data = resp.json()
        assert data["redis"] == "unknown"
        assert data["postgresql"] == "unknown"


class TestGetFeatures:
    async def test_get_features_returns_all_flags(self, client: AsyncClient):
        resp = await client.get("/features")
        assert resp.status_code == 200
        data = resp.json()
        assert "features" in data
        assert data["features"]["feature_x"] is True
        assert data["features"]["feature_y"] is False

    async def test_get_features_without_service(self, app: FastAPI):
        """When feature flag service is not available, return empty dict."""
        app.state.feature_flags = None
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/features")
        assert resp.status_code == 200
        data = resp.json()
        assert data["features"] == {}


class TestListKernels:
    async def test_list_kernels_returns_registered_kernels(self, client: AsyncClient):
        resp = await client.get("/kernels")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["name"] == "test_kernel"
        assert data[0]["description"] == "Test kernel"


class TestListAgents:
    async def test_list_agents_returns_registered_agents(self, client: AsyncClient):
        resp = await client.get("/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["name"] == "test_agent"
        assert data[0]["role"] == "assistant"
        assert data[0]["expertise"] == ["python", "testing"]


class TestCancelThink:
    async def test_cancel_think_returns_cancelled_when_task_exists(
        self, client: AsyncClient, mock_cancellation_mgr
    ):
        mock_cancellation_mgr.cancel.return_value = True
        resp = await client.delete("/think/task-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"
        assert data["task_id"] == "task-123"

    async def test_cancel_think_returns_not_found_when_task_missing(
        self, client: AsyncClient, mock_cancellation_mgr
    ):
        mock_cancellation_mgr.cancel.return_value = False
        resp = await client.delete("/think/nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_found_or_completed"
        assert data["task_id"] == "nonexistent"
