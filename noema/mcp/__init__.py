"""MCP — Model Context Protocol integration.

Serves Noema to MCP clients (Claude Desktop, Codespaces, custom clients) over
the standard **stdio** transport: newline-delimited JSON-RPC 2.0 on stdin/
stdout, zero new dependencies. The engine's fallback ``think`` pipeline works
out of the box, so an MCP client gets real solution generation with no API
keys.

Tools:
- ``noema_think``  — generate a technical solution for a task.
- ``noema_stats``  — current reasoning/token statistics.
"""

from noema.mcp.server import (
    DEFAULT_PROTOCOL_VERSION,
    MCP_SERVER_INFO,
    TOOL_SCHEMAS,
    handle_payload,
    read_loop,
    run_tool,
)

__all__ = [
    "DEFAULT_PROTOCOL_VERSION",
    "MCP_SERVER_INFO",
    "TOOL_SCHEMAS",
    "handle_payload",
    "read_loop",
    "run_tool",
]
