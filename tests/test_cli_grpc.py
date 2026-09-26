"""Tests for CLI gRPC commands."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from noema.cli.grpc import grpc_app

runner = CliRunner()


def test_grpc_serve_starts_server():
    """serve() should initialize engine, start gRPC server, and handle shutdown."""
    mock_engine = AsyncMock()
    mock_server = AsyncMock()
    mock_server.wait_for_termination = AsyncMock()
    mock_serve_grpc = AsyncMock(return_value=mock_server)
    mock_stop_grpc = AsyncMock()

    with (
        patch("noema.core.engine.NoemaEngine", return_value=mock_engine),
        patch("noema.grpc.server.serve_grpc", mock_serve_grpc),
        patch("noema.grpc.server.stop_grpc", mock_stop_grpc),
    ):
        result = runner.invoke(grpc_app, ["serve", "--host", "127.0.0.1", "--port", "50052"])

    assert result.exit_code == 0
    assert "Starting gRPC server" in result.stdout
    assert "127.0.0.1:50052" in result.stdout
    mock_engine.initialize.assert_called_once()
    mock_serve_grpc.assert_called_once()
    mock_server.wait_for_termination.assert_called_once()
    mock_stop_grpc.assert_called_once_with(mock_server)
    mock_engine.shutdown.assert_called_once()


def test_grpc_health_checks_liveness():
    """health() should connect, check health, and print status."""
    mock_client = AsyncMock()
    mock_client.health = AsyncMock(return_value={"status": "ok", "version": "1.0.0"})
    mock_client.connect = AsyncMock()
    mock_client.close = AsyncMock()

    with patch("noema.grpc.client.NoemaGRPCClient", return_value=mock_client):
        result = runner.invoke(grpc_app, ["health", "--host", "localhost", "--port", "50051"])

    assert result.exit_code == 0
    mock_client.connect.assert_called_once()
    mock_client.health.assert_called_once()
    mock_client.close.assert_called_once()
    assert "ok" in result.stdout
    assert "1.0.0" in result.stdout
    parsed = json.loads(result.stdout.split("\n")[0])
    assert parsed["status"] == "ok"
