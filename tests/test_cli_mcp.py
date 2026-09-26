"""Tests for CLI MCP commands."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from noema.cli.mcp import mcp_app

runner = CliRunner()


def test_mcp_serve_calls_read_loop():
    """serve() should setup logging and enter the MCP read loop."""
    mock_setup = MagicMock()
    mock_read_loop = MagicMock()

    with (
        patch("noema.logging.setup_logging", mock_setup),
        patch("noema.mcp.server.read_loop", mock_read_loop),
    ):
        result = runner.invoke(mcp_app, ["serve"])

    assert result.exit_code == 0
    mock_setup.assert_called_once()
    mock_read_loop.assert_called_once()


def test_mcp_list_tools_prints_schemas():
    """list_tools() should print server info and JSON tool schemas."""
    mock_info = {"name": "noema", "version": "1.0.0"}
    mock_schemas = [
        {
            "name": "noema_think",
            "description": "Generate a solution",
            "inputSchema": {"type": "object", "properties": {"task": {"type": "string"}}},
        }
    ]

    with patch("noema.mcp.server.MCP_SERVER_INFO", mock_info), patch(
        "noema.mcp.server.TOOL_SCHEMAS", mock_schemas
    ):
        result = runner.invoke(mcp_app, ["list-tools"])

    assert result.exit_code == 0
    assert "noema v1.0.0" in result.stdout
    parsed = json.loads(result.stdout.split("\n", 1)[1])
    assert parsed == mock_schemas
    assert len(parsed) == 1
    assert parsed[0]["name"] == "noema_think"
