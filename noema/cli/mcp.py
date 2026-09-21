"""CLI commands for the MCP server: serve over stdio and inspect tools."""

from __future__ import annotations

import json

import typer

from noema.cli.ui import ok

mcp_app = typer.Typer(
    help="MCP server — expose Noema to AI clients (Claude Desktop, etc.)",
    rich_markup_mode="rich",
)


@mcp_app.command("serve")
def serve() -> None:
    """Run the MCP server over stdio (line-delimited JSON-RPC).

    Point an MCP client at ``noema mcp serve``. Tool ``noema_think`` generates
    a full technical solution; ``noema_stats`` reports reasoning statistics.
    Works with the built-in fallback provider (no API keys) out of the box.
    """
    from noema.logging import setup_logging
    from noema.mcp.server import read_loop

    setup_logging()  # logs to stderr; stdout stays reserved for the protocol
    read_loop()


@mcp_app.command("list-tools")
def list_tools() -> None:
    """Print the tool schemas this server exposes (JSON)."""
    from noema.mcp.server import MCP_SERVER_INFO, TOOL_SCHEMAS

    ok(f"{MCP_SERVER_INFO['name']} v{MCP_SERVER_INFO['version']}")
    print(json.dumps(TOOL_SCHEMAS, indent=2))
