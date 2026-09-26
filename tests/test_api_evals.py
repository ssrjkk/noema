"""Tests for eval API leaderboard endpoint."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from noema.api.evals import router


def test_leaderboard_endpoint():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    mock_rows = [
        {"provider": "openai", "model": "gpt-4", "score": 95.5},
        {"provider": "anthropic", "model": "claude-3", "score": 92.0},
    ]

    with (
        patch("noema.api.evals.compute_leaderboard", return_value=mock_rows),
        patch("noema.api.evals.to_dict", return_value=mock_rows),
    ):
        response = client.get("/eval/leaderboard")

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert data["leaderboard"] == mock_rows


def test_leaderboard_with_custom_top():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    mock_rows = [{"provider": "openai", "model": "gpt-4", "score": 95.5}]

    with (
        patch("noema.api.evals.compute_leaderboard", return_value=mock_rows) as mock_compute,
        patch("noema.api.evals.to_dict", return_value=mock_rows),
    ):
        response = client.get("/eval/leaderboard?top=10")

    assert response.status_code == 200
    mock_compute.assert_called_once()
    call_args = mock_compute.call_args
    assert call_args[1]["top_n"] == 10


def test_leaderboard_empty_results():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with (
        patch("noema.api.evals.compute_leaderboard", return_value=[]),
        patch("noema.api.evals.to_dict", return_value=[]),
    ):
        response = client.get("/eval/leaderboard")

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["leaderboard"] == []
