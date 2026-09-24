"""Tests for the MCP server (stdio JSON-RPC layer)."""

from __future__ import annotations

import json

from noema.mcp.server import (
    DEFAULT_PROTOCOL_VERSION,
    MCP_SERVER_INFO,
    TOOL_SCHEMAS,
    handle_payload,
)


def _call(msg: dict) -> dict:
    reply = handle_payload(json.dumps(msg, ensure_ascii=False))
    assert reply is not None
    return json.loads(reply)


def _result(msg: dict) -> dict:
    out = _call(msg)
    assert "error" not in out, out
    return out["result"]


# ── Protocol handshake ─────────────────────────────────────────────────────


def test_initialize_handshake():
    out = _result({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert out["protocolVersion"] == DEFAULT_PROTOCOL_VERSION
    assert out["capabilities"]["tools"]["listChanged"] is False
    assert out["serverInfo"]["name"] == "noema"
    assert out["serverInfo"]["version"]


def test_initialize_honors_client_protocol_version():
    out = _result(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        }
    )
    assert out["protocolVersion"] == "2024-11-05"


def test_initialized_notification_gets_no_reply():
    reply = handle_payload(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}))
    assert reply is None


def test_ping_roundtrip():
    out = _result({"jsonrpc": "2.0", "id": 3, "method": "ping", "params": {}})
    assert out == {}


# ── Tools ───────────────────────────────────────────────────────────────────


def test_tools_list_schema():
    out = _result({"jsonrpc": "2.0", "id": 4, "method": "tools/list", "params": {}})
    tools = out["tools"]
    names = {t["name"] for t in tools}
    assert "noema_think" in names
    assert "noema_stats" in names
    think = next(t for t in tools if t["name"] == "noema_think")
    assert think["inputSchema"]["required"] == ["title"]
    assert isinstance(think["inputSchema"]["properties"], dict)


def test_tools_call_stats():
    out = _result(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "noema_stats", "arguments": {}},
        }
    )
    assert out["isError"] is False
    text = out["content"][0]["text"]
    assert "Noema statistics" in text
    assert "- total spans:" in text


def test_tools_call_think():
    out = _result(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "noema_think",
                "arguments": {"title": "Small toy", "description": "Build a tiny thing."},
            },
        }
    )
    assert out["isError"] is False
    text = out["content"][0]["text"]
    assert "Small toy" in text
    assert "Files" in text


def test_tools_call_unknown_tool():
    out = _result(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        }
    )
    assert out["isError"] is True
    assert "Unknown tool" in out["content"][0]["text"]


def test_tools_call_bad_arguments():
    out = _result(
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {"name": "noema_think", "arguments": "not-an-object"},
        }
    )
    assert out["isError"] is True


def test_missing_tool_name():
    out = _result(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"arguments": {}},
        }
    )
    assert out["isError"] is True


# ── Errors / robustness ────────────────────────────────────────────────────


def test_malformed_json_reply():
    reply = handle_payload("{not json")
    assert reply is not None
    out = json.loads(reply)
    assert out["error"]["code"] == -32700


def test_non_object_message():
    out = json.loads(handle_payload("[1,2,3]"))
    assert out["error"]["code"] == -32700


def test_unknown_method():
    out = json.loads(handle_payload(json.dumps({"jsonrpc": "2.0", "id": 10, "method": "nope"})))
    assert out["error"]["code"] == -32601
    assert out["id"] == 10


def test_handlers_tolerate_null_id_requests():
    reply = handle_payload(json.dumps({"jsonrpc": "2.0", "method": "tools/list"}))
    assert reply is None  # treated as a notification


def test_schema_metadata():
    assert MCP_SERVER_INFO["name"] == "noema"
    assert {t["name"] for t in TOOL_SCHEMAS} == {"noema_think", "noema_stats"}


def test_internal_error_answered():
    reply = handle_payload(json.dumps({"jsonrpc": "2.0", "id": 11, "method": "tools/call"}))
    assert reply is not None
    out = json.loads(reply)
    assert out["id"] == 11
