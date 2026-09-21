"""MCP server implementation over stdio (line-delimited JSON-RPC 2.0).

The protocol envelope::

    request:  {"jsonrpc":"2.0","id":N,"method":"initialize","params":{...}}
    response: {"jsonrpc":"2.0","id":N,"result":{...}}
    error:    {"jsonrpc":"2.0","id":N,"error":{"code":..,"message":".."}}
    notify:   {"jsonrpc":"2.0","method":"notifications/initialized"}  # no reply

The engine runs the ``fallback`` provider by default (no API keys) and every
``tools/call`` spins an isolated engine in its own event loop, mirroring the
experiment runner's loop hygiene: no loop is shared across calls.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from noema import __version__
from noema.core.types import Task
from noema.logging import get_logger

log = get_logger(__name__)

DEFAULT_PROTOCOL_VERSION = "2025-06-18"
MCP_SERVER_INFO = {"name": "noema", "version": __version__}

MAX_TEXT = 48_000
_MAX_TITLE = 500
_MAX_DESC = 100_000
_MAX_TAGS = 100

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "noema_think",
        "description": (
            "Generate a complete technical solution (architecture, stack, code "
            "files, deployment, security notes) for a software task."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short task title, e.g. 'Real-time chat app'.",
                },
                "description": {
                    "type": "string",
                    "description": "Optional detailed requirements.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional domain tags, e.g. ['api','auth'].",
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "noema_stats",
        "description": "Current reasoning statistics (spans, LLM calls, tokens, errors).",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

_CODES = {"parse": -32700, "invalid_request": -32600, "method": -32601, "params": -32602}


# ── JSON-RPC envelope helpers ───────────────────────────────────────────────


def _reply(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}


def _tool_error(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "isError": True}


# ── Tools ───────────────────────────────────────────────────────────────────


def _think_sync(args: dict[str, Any]) -> dict[str, Any]:
    """Run the engine once for ``noema_think`` in a dedicated event loop."""
    from noema.core.engine import NoemaEngine

    provider = os.environ.get("NOEMA_MCP_PROVIDER", "fallback")
    title = str(args.get("title", "")).strip() or "Untitled task"
    description = str(args.get("description", ""))[: _MAX_DESC]
    raw_tags = args.get("tags") or []
    if not isinstance(raw_tags, list):
        raise ValueError("tags must be a list of strings")
    tags = [str(t)[:200] for t in raw_tags][: _MAX_TAGS]

    async def _go() -> tuple:
        engine = NoemaEngine(llm_provider=provider)
        await engine.initialize()
        try:
            return await engine.think(Task(title=title[: _MAX_TITLE], description=description, tags=tags))
        finally:
            await engine.shutdown()

    solution, _ = asyncio.run(_go())
    files = [
        {
            "filename": b.filename,
            "language": b.language,
            "description": b.description,
            "lines": b.content.count("\n") + 1,
            "content_preview": b.content[:2000],
        }
        for b in solution.code_blocks[:10]
    ]
    return {
        "solution_id": solution.id,
        "title": solution.title,
        "quality": solution.quality.value,
        "confidence": solution.confidence,
        "summary": solution.summary,
        "architecture": solution.architecture.name if solution.architecture else "",
        "files": files,
        "deployment": solution.deployment,
        "performance_notes": solution.performance_notes[:10],
        "security_notes": solution.security_notes[:10],
    }


def _stats_sync() -> dict[str, Any]:
    from noema.tracing.tracer import get_tracer

    stats = get_tracer().get_stats()
    prompts = get_tracer().prompt_stats()
    return {
        "tracer": stats,
        "prompts": prompts,
        "services": {"version": __version__},
        "hint": "Set NOEMA_LLM__PROVIDER=ollama|openai|anthropic for real LLM reasoning.",
    }


def _render_think(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['title']}",
        f"Quality: **{result['quality']}** · Confidence: {result['confidence'] * 100:.0f}% · "
        f"Solution: `{result['solution_id']}`",
        "",
        (result.get("summary") or "").strip(),
    ]
    if result.get("architecture"):
        lines += ["", f"Architecture: {result['architecture']}"]
    files = result.get("files") or []
    if files:
        lines += ["", f"## Files ({len(files)})"]
        for f in files:
            lines.append(
                f"- `{f['filename']}` — {f['description'] or 'generated module'} "
                f"({f['language']}, {f['lines']} lines)"
            )
    if result.get("deployment"):
        lines += ["", "## Deployment"] + [f"- `{k}`: {v}" for k, v in list(result["deployment"].items())[:8]]
    return "\n".join(lines)[:MAX_TEXT]


def _render_stats(result: dict[str, Any]) -> str:
    tracer = result["tracer"]
    lines = [
        "## Noema statistics",
        f"- version: {result['services']['version']}",
        f"- total spans: {tracer['total_spans']}",
        f"- LLM calls: {tracer['llm_calls']}",
        f"- total tokens: {tracer['total_tokens']}",
        f"- total LLM latency: {tracer['total_llm_latency_ms']} ms",
        f"- errors: {tracer['errors']}",
        "",
        result["hint"],
    ]
    return "\n".join(lines)


def run_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute an MCP tool, returning the ``tools/call`` result envelope."""
    if name == "noema_think":
        try:
            result = _think_sync(arguments or {})
        except Exception as exc:  # noqa: BLE001 - report into the tool result
            log.warning("mcp_tool_failed", tool=name, error=str(exc))
            return _tool_error(f"{type(exc).__name__}: {exc}")
        return {"content": [{"type": "text", "text": _render_think(result)}], "isError": False}
    if name == "noema_stats":
        return {
            "content": [{"type": "text", "text": _render_stats(_stats_sync())}],
            "isError": False,
        }
    return _tool_error(f"Unknown tool: {name}")


# ── Protocol dispatch ───────────────────────────────────────────────────────


def handle_payload(raw: str) -> str | None:
    """Handle one stdin line; return the reply line, or None for notifications."""
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return json.dumps(_error(None, _CODES["parse"], "Parse error: invalid JSON"))
    if not isinstance(msg, dict):
        return json.dumps(_error(None, _CODES["parse"], "Parse error: expected object"))

    method = msg.get("method")
    msg_id = msg.get("id")
    if not isinstance(method, str):
        return json.dumps(_error(msg_id, _CODES["invalid_request"], "Invalid request"))
    if msg_id is None:
        return None  # notification (e.g. notifications/initialized)

    try:
        if method == "initialize":
            server_protocol = (msg.get("params") or {}).get("protocolVersion", DEFAULT_PROTOCOL_VERSION)
            return json.dumps(
                _reply(
                    msg_id,
                    {
                        "protocolVersion": server_protocol,
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": dict(MCP_SERVER_INFO),
                    },
                )
            )
        if method == "ping":
            return json.dumps(_reply(msg_id, {}))
        if method == "tools/list":
            return json.dumps(_reply(msg_id, {"tools": TOOL_SCHEMAS}))
        if method == "tools/call":
            return json.dumps(_reply(msg_id, _handle_call(msg.get("params"))))
        return json.dumps(_error(msg_id, _CODES["method"], f"Method not found: {method}"))
    except Exception as exc:  # noqa: BLE001 - a handler bug must still answer
        log.warning("mcp_dispatch_failed", method=method, error=str(exc))
        return json.dumps(_error(msg_id, -32603, f"Internal error: {exc}"))


def _handle_call(params: Any) -> dict[str, Any]:
    if not isinstance(params, dict):
        return _tool_error("tools/call params must be an object")
    name = params.get("name")
    if not isinstance(name, str):
        return _tool_error("Missing tool name")
    arguments = params.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return _tool_error("Tool arguments must be an object")
    return run_tool(name, arguments)


def read_loop() -> None:
    """Blocking stdio loop: one JSON line in, one reply line out. Logs to stderr."""
    inp = sys.stdin
    out = sys.stdout
    for encoding_attr in ("reconfigure",):
        try:
            getattr(inp, encoding_attr)(encoding="utf-8")
            getattr(out, encoding_attr)(encoding="utf-8")
        except (AttributeError, OSError, ValueError):  # pragma: no cover - platform quirks
            break
    log.info("mcp_server_started", protocol=DEFAULT_PROTOCOL_VERSION)
    out.flush()
    for line in inp:
        reply = handle_payload(line)
        if reply is not None:
            out.write(reply + "\n")
            out.flush()


if __name__ == "__main__":  # pragma: no cover
    read_loop()
