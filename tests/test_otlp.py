"""Tests for the OTLP/HTTP trace exporter."""

from __future__ import annotations

import base64
import time

import pytest

from noema.observability.otlp import (
    KIND_TO_OTLP,
    build_otlp_span,
    build_otlp_trace_request,
    emit_otlp_span,
    flush_otlp_exporter,
    get_otlp_exporter,
    is_otlp_active,
    set_post_fn,
    start_otlp_exporter,
    stop_otlp_exporter,
)
from noema.tracing.tracer import TraceConfig, Tracer, reset_tracer


@pytest.fixture(autouse=True)
def _clean_exporter():
    reset_tracer()
    stop_otlp_exporter()
    set_post_fn(None)
    yield
    stop_otlp_exporter()
    set_post_fn(None)
    reset_tracer()


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _fake_recorder(records: list[dict]):
    async def fake_post(endpoint: str, payload: dict) -> bool:
        records.append({"endpoint": endpoint, "payload": payload})
        return True

    return fake_post


# ── Payload builder ────────────────────────────────────────────────────────


def test_build_otlp_span_mapping():
    span = {
        "trace_id": "0123456789abcdef",
        "span_id": "deadbeefdeadbeef",
        "parent_id": "1234567890abcdef",
        "name": "llm.openai",
        "kind": "llm",
        "duration_ms": 150.0,
        "status": "ok",
        "attributes": {"llm.provider": "openai", "llm.tokens": 42, "retry": True},
        "events": [{"name": "cache.hit", "attributes": {"key": "user:1"}}],
        "error": "",
    }

    otlp = build_otlp_span(span)

    # traceId is 128-bit: a 16-hex-char id is left-zero-padded to 16 bytes.
    assert (
        otlp["traceId"] == base64.b64encode(bytes.fromhex(span["trace_id"].rjust(32, "0"))).decode()
    )
    assert otlp["spanId"] == base64.b64encode(bytes.fromhex(span["span_id"])).decode()
    assert otlp["parentSpanId"] == base64.b64encode(bytes.fromhex(span["parent_id"])).decode()
    assert otlp["name"] == "llm.openai"
    assert otlp["kind"] == KIND_TO_OTLP["llm"]
    assert otlp["status"] == {"code": 1}
    assert otlp["startTimeUnixNano"].isdigit()
    assert otlp["endTimeUnixNano"].isdigit()

    attrs = {a["key"]: a["value"] for a in otlp["attributes"]}
    assert attrs["llm.provider"] == {"stringValue": "openai"}
    assert attrs["llm.tokens"] == {"intValue": "42"}
    assert attrs["retry"] == {"boolValue": True}

    assert otlp["events"][0]["name"] == "cache.hit"


def test_build_otlp_span_error_status():
    otlp = build_otlp_span({"name": "x", "status": "error", "error": "ValueError: boom"})
    assert otlp["status"]["code"] == 2
    assert "boom" in otlp["status"]["message"]


def test_build_otlp_span_uses_epoch_times():
    span = {"duration_ms": 10.0, "start_epoch_ns": 500, "end_epoch_ns": 900}
    otlp = build_otlp_span(span)
    assert otlp["startTimeUnixNano"] == "500"
    assert otlp["endTimeUnixNano"] == "900"


def test_build_otlp_span_falls_back_to_duration():
    otlp = build_otlp_span({"duration_ms": 123.0})
    start_ns = int(otlp["startTimeUnixNano"])
    end_ns = int(otlp["endTimeUnixNano"])
    assert end_ns - start_ns == 123_000_000


def test_trace_request_nesting():
    payload = build_otlp_trace_request([{"name": "a"}, {"name": "b"}], service_name="my-svc")
    resources = payload["resourceSpans"]
    assert len(resources) == 1
    attrs = resources[0]["resource"]["attributes"]
    assert {"key": "service.name", "value": {"stringValue": "my-svc"}} in attrs
    spans = resources[0]["scopeSpans"][0]["spans"]
    assert [s["name"] for s in spans] == ["a", "b"]


def test_base64_id_padding():
    from noema.observability.otlp import _base64_id

    assert _base64_id("", 8) == base64.b64encode(b"\x00" * 8).decode()
    assert _base64_id("ff", 8) == base64.b64encode(b"\x00" * 7 + b"\xff").decode()
    assert _base64_id("zz!", 8) == base64.b64encode(b"\x00" * 8).decode()


# ── Exporter transport ─────────────────────────────────────────────────────


def test_exporter_delivers_batch():
    records: list[dict] = []
    set_post_fn(_fake_recorder(records))
    start_otlp_exporter("http://collector:4318", flush_interval=0.05)
    exporter = get_otlp_exporter()
    assert exporter is not None

    emit_otlp_span(
        {"span_id": "0123456789abcdef", "name": "one", "status": "ok", "trace_id": "t" * 16}
    )
    emit_otlp_span(
        {"span_id": "fedcba9876543210", "name": "two", "status": "ok", "trace_id": "t" * 16}
    )

    assert _wait_until(
        lambda: records and records[0]["payload"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
    )

    assert records[0]["endpoint"] == "http://collector:4318"
    sent_names = [
        s["name"]
        for r in records
        for s in r["payload"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
    ]
    assert "one" in sent_names
    assert "two" in sent_names


def test_exporter_survives_transport_failure():
    async def fail_post(endpoint: str, payload: dict) -> bool:
        raise OSError("connection refused")

    set_post_fn(fail_post)
    start_otlp_exporter("http://down:4318", flush_interval=0.05)
    emit_otlp_span({"span_id": "s" * 16, "name": "fails", "status": "ok", "trace_id": "t" * 16})
    time.sleep(0.2)

    assert is_otlp_active()
    assert get_otlp_exporter() is not None
    emit_otlp_span(
        {"span_id": "a" * 16, "name": "still-works", "status": "ok", "trace_id": "t" * 16}
    )


def test_emit_without_exporter_is_noop():
    emit_otlp_span({"span_id": "x", "name": "orphan"})
    assert not is_otlp_active()


def test_start_exporter_is_idempotent():
    start_otlp_exporter("http://a:4318", flush_interval=0.05)
    first = get_otlp_exporter()
    start_otlp_exporter("http://a:4318", flush_interval=0.05)
    assert get_otlp_exporter() is first

    start_otlp_exporter("http://b:4318", flush_interval=0.05)
    assert get_otlp_exporter() is not first
    assert get_otlp_exporter().endpoint == "http://b:4318"


def test_flush_sync_drains_queue():
    records: list[dict] = []
    set_post_fn(_fake_recorder(records))
    start_otlp_exporter("http://c:4318", flush_interval=60.0)
    emit_otlp_span({"span_id": "f" * 16, "name": "flushed", "status": "ok", "trace_id": "t" * 16})

    flush_otlp_exporter()

    names = [
        s["name"]
        for r in records
        for s in r["payload"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
    ]
    assert "flushed" in names


def test_tracer_pushes_spans_to_exporter():
    records: list[dict] = []
    set_post_fn(_fake_recorder(records))
    start_otlp_exporter("http://d:4318", flush_interval=0.05)

    config = TraceConfig(export_endpoint="http://d:4318")
    tracer = Tracer(config=config)
    span = tracer.start_span("watch-op", attributes={"env": "test"})
    tracer.end_span(span)

    def delivered() -> bool:
        for r in records:
            for s in r["payload"]["resourceSpans"][0]["scopeSpans"][0]["spans"]:
                if s["name"] == "watch-op":
                    return True
        return False

    assert _wait_until(delivered)


def test_tracer_without_endpoint_does_not_export():
    tracer = Tracer()
    tracer.start_span("local-only")
    tracer.end_span()
    time.sleep(0.05)
    assert not is_otlp_active()


def test_span_epoch_fields_populated():
    tracer = Tracer()
    span = tracer.start_span("timed")
    tracer.end_span(span)
    d = span.to_dict()
    assert d["start_epoch_ns"] > 0
    assert d["end_epoch_ns"] >= d["start_epoch_ns"]


def test_manual_span_dict_unchanged():
    from noema.tracing.tracer import TraceSpan

    span = TraceSpan(span_id="abc123", parent_id="p", name="n", kind="internal")
    assert "start_epoch_ns" not in span.to_dict()
    assert "end_epoch_ns" not in span.to_dict()
