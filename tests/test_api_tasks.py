"""Tests for api/tasks.py — enqueue endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def app():
    import noema.api.server as server_mod

    server_mod.app.state.noema = MagicMock()
    return server_mod.app


@pytest.mark.asyncio
async def test_enqueue_task(app):
    with (
        patch("noema.api.tasks.enqueue_think", new_callable=AsyncMock, return_value="job-123"),
        patch("noema.api.tasks.get_settings") as mock_settings,
    ):
        settings = MagicMock()
        settings.redis.url = "redis://localhost:6379/0"
        mock_settings.return_value = settings

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/tasks/enqueue",
                json={
                    "title": "Build API",
                    "description": "REST API with auth",
                    "complexity": "moderate",
                    "tags": ["python", "fastapi"],
                },
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == "job-123"
    assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_enqueue_task_minimal(app):
    with (
        patch("noema.api.tasks.enqueue_think", new_callable=AsyncMock, return_value="job-456"),
        patch("noema.api.tasks.get_settings") as mock_settings,
    ):
        settings = MagicMock()
        settings.redis.url = "redis://localhost:6379/0"
        mock_settings.return_value = settings

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/tasks/enqueue",
                    json={"title": "Simple task"},
                )
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == "job-456"


@pytest.mark.asyncio
async def test_enqueue_task_empty_title(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/tasks/enqueue", json={"title": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_enqueue_task_invalid_complexity(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/tasks/enqueue",
            json={"title": "Task", "complexity": "impossible"},
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_enqueue_task_redis_down(app):
    with (
        patch("noema.api.tasks.enqueue_think", new_callable=AsyncMock, side_effect=ConnectionError("redis down")),
        patch("noema.api.tasks.get_settings") as mock_settings,
    ):
        settings = MagicMock()
        settings.redis.url = "redis://localhost:6379/0"
        mock_settings.return_value = settings

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/tasks/enqueue",
                json={"title": "Task"},
            )
    assert resp.status_code == 503
    assert "Failed to enqueue" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_enqueue_too_many_tags(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/tasks/enqueue",
            json={"title": "Task", "tags": [f"tag{i}" for i in range(25)]},
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_enqueue_title_too_long(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/tasks/enqueue",
            json={"title": "x" * 201},
        )
    assert resp.status_code == 422
