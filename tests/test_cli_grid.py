"""Tests for CLI grid commands — status and federate."""

from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from noema.cli.grid import grid_app

runner = CliRunner()


@pytest.fixture
def mock_dashboard_snapshot():
    return {
        "nodes": [
            {
                "node_id": "node-1",
                "address": "10.0.0.1:6379",
                "reachable": True,
                "draining": False,
                "llm_tokens": 1500,
                "llm_calls": 42,
                "llm_latency_avg_ms": 120.5,
                "http_errors": 0,
            },
            {
                "node_id": "node-2",
                "address": None,
                "reachable": False,
                "error": "Connection refused",
                "draining": True,
                "llm_tokens": 0,
                "llm_calls": 0,
                "llm_latency_avg_ms": 0,
                "http_errors": 3,
            },
        ],
        "totals": {"total_tokens": 1500, "total_calls": 42, "reachable_nodes": 1},
    }


def test_grid_status(mock_dashboard_snapshot):
    mock_dashboard = MagicMock()
    mock_dashboard.snapshot = AsyncMock(return_value=mock_dashboard_snapshot)
    mock_dashboard.aclose = AsyncMock()

    mock_mod = MagicMock()
    mock_mod.GridDashboard = MagicMock(return_value=mock_dashboard)

    with patch.dict(sys.modules, {"noema.observability.grid": mock_mod}):
        result = runner.invoke(grid_app, ["status"])
        assert result.exit_code == 0
        output = result.stdout
        assert "node-1" in output
        assert "node-2" in output
        lines = output.strip().split("\n")
        json_line = lines[-1]
        parsed = json.loads(json_line)
        assert parsed["totals"]["total_tokens"] == 1500


def test_grid_status_unreachable_node_shows_error():
    snap = {
        "nodes": [
            {
                "node_id": "broken",
                "address": "10.0.0.9:6379",
                "reachable": False,
                "error": "timeout after 30s",
                "draining": False,
                "llm_tokens": 0,
                "llm_calls": 0,
                "llm_latency_avg_ms": 0,
                "http_errors": 5,
            },
        ],
        "totals": {},
    }
    mock_dashboard = MagicMock()
    mock_dashboard.snapshot = AsyncMock(return_value=snap)
    mock_dashboard.aclose = AsyncMock()

    mock_mod = MagicMock()
    mock_mod.GridDashboard = MagicMock(return_value=mock_dashboard)

    with patch.dict(sys.modules, {"noema.observability.grid": mock_mod}):
        result = runner.invoke(grid_app, ["status"])
        assert result.exit_code == 0
        assert "timeout after 30s" in result.stdout


def test_grid_federate_local():
    mock_solution = MagicMock()
    mock_solution.id = "sol-123"
    mock_solution.quality = MagicMock()
    mock_solution.quality.value = "good"

    mock_engine = MagicMock()
    mock_engine.initialize = AsyncMock()
    mock_engine.shutdown = AsyncMock()
    mock_engine.think = AsyncMock(return_value=(mock_solution, MagicMock()))

    mock_router = MagicMock()
    mock_router.execute = AsyncMock(
        return_value={
            "delegated": 1,
            "local": 2,
            "failed": 0,
            "duration_ms": 450.0,
        }
    )
    mock_router.aclose = AsyncMock()

    mock_settings = MagicMock()
    mock_settings.federation.peers = []

    with (
        patch("noema.config.settings.get_settings", return_value=mock_settings),
        patch("noema.core.engine.NoemaEngine", return_value=mock_engine),
        patch("noema.federation.router.FederationRouter", return_value=mock_router),
        patch("noema.billing.ledger.ContributionLedger"),
        patch("noema.workers.arq_worker.make_node_id", return_value="test-node"),
    ):
        result = runner.invoke(
            grid_app,
            ["federate", "--title", "Build app", "--subtask", "Auth", "--subtask", "API"],
        )
        assert result.exit_code == 0
        output = result.stdout
        assert "1 delegated" in output
        assert "2 local" in output


def test_grid_federate_with_peers():
    mock_solution = MagicMock()
    mock_solution.id = "sol-456"
    mock_solution.quality = MagicMock()
    mock_solution.quality.value = "excellent"

    mock_engine = MagicMock()
    mock_engine.initialize = AsyncMock()
    mock_engine.shutdown = AsyncMock()
    mock_engine.think = AsyncMock(return_value=(mock_solution, MagicMock()))

    mock_router = MagicMock()
    mock_router.execute = AsyncMock(
        return_value={
            "delegated": 3,
            "local": 0,
            "failed": 0,
            "duration_ms": 1200.0,
        }
    )
    mock_router.aclose = AsyncMock()

    mock_settings = MagicMock()
    mock_settings.federation.peers = []

    with (
        patch("noema.config.settings.get_settings", return_value=mock_settings),
        patch("noema.core.engine.NoemaEngine", return_value=mock_engine),
        patch("noema.federation.router.FederationRouter", return_value=mock_router),
        patch("noema.billing.ledger.ContributionLedger"),
        patch("noema.workers.arq_worker.make_node_id", return_value="test-node"),
    ):
        result = runner.invoke(
            grid_app,
            [
                "federate",
                "--title",
                "Distributed task",
                "--subtask",
                "Part A",
                "--peer",
                "10.0.0.1:8080",
                "--peer",
                "10.0.0.2:8080",
                "--timeout",
                "10.0",
            ],
        )
        assert result.exit_code == 0
        assert "3 delegated" in result.stdout
