"""Tests for Tracer — OpenTelemetry-compatible tracing + Prompt Version Control."""

import json

from noema.tracing.tracer import (
    TraceConfig,
    Tracer,
    TraceSpan,
    get_tracer,
    reset_tracer,
)


def test_start_end_span_basic():
    tracer = Tracer()
    span = tracer.start_span("test-op")
    assert span.name == "test-op"
    assert span.span_id
    assert span.parent_id == ""
    assert span.start_time > 0

    result = tracer.end_span(span)
    assert result.duration_ms >= 0
    assert result.status == "ok"
    assert result.error == ""


def test_span_stack_nesting():
    tracer = Tracer()
    parent = tracer.start_span("parent")
    child = tracer.start_span("child")
    assert child.parent_id == parent.span_id

    tracer.end_span(child)
    tracer.end_span(parent)

    trace = tracer.get_trace()
    assert len(trace) == 2
    span_ids = [s["span_id"] for s in trace]
    assert parent.span_id in span_ids
    assert child.span_id in span_ids


def test_llm_call_logging():
    tracer = Tracer()
    messages = [{"role": "user", "content": "hello"}]
    span = tracer.trace_llm_call(
        provider="openai",
        model="gpt-4",
        messages=messages,
        response="hi there",
        tokens_used=42,
        latency_ms=150.5,
    )
    assert span.kind == "llm"
    assert span.status == "ok"
    assert span.attributes["llm.provider"] == "openai"
    assert span.attributes["llm.model"] == "gpt-4"
    assert span.attributes["llm.tokens"] == 42
    assert span.attributes["llm.latency_ms"] == 150.5
    assert "llm.prompt" in span.attributes
    assert "llm.response" in span.attributes


def test_llm_call_with_error():
    tracer = Tracer()
    span = tracer.trace_llm_call(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "hello"}],
        response="",
        tokens_used=0,
        latency_ms=5000,
        error="rate_limit_exceeded",
    )
    assert span.status == "error"
    assert span.error == "rate_limit_exceeded"
    assert "llm.response" not in span.attributes


def test_attributes_propagation():
    tracer = Tracer()
    span = tracer.start_span("data-process", attributes={"source": "s3", "format": "csv"})
    assert span.attributes["source"] == "s3"
    assert span.attributes["format"] == "csv"

    tracer.end_span(span)
    trace = tracer.get_trace()
    exported = trace[0]
    assert exported["attributes"]["source"] == "s3"
    assert exported["attributes"]["format"] == "csv"


def test_attributes_defaults_to_empty():
    tracer = Tracer()
    span = tracer.start_span("no-attrs")
    assert span.attributes == {}


def test_error_status_on_exception():
    tracer = Tracer()
    span = tracer.start_span("failing-op")
    tracer.end_span(span, status="error", error="ValueError: invalid input")

    trace = tracer.get_trace()
    assert trace[0]["status"] == "error"
    assert "ValueError" in trace[0]["error"]


def test_multiple_concurrent_spans():
    tracer = Tracer()
    a = tracer.start_span("A")
    b = tracer.start_span("B")
    c = tracer.start_span("C")

    assert b.parent_id == a.span_id
    assert c.parent_id == b.span_id

    tracer.end_span(c)
    tracer.end_span(b)
    tracer.end_span(a)

    assert len(tracer.get_trace()) == 3


def test_context_manager_usage():
    tracer = Tracer()
    span = tracer.start_span("ctx-op")
    tracer.end_span(span)
    assert span.status == "ok"


def test_context_manager_nesting():
    tracer = Tracer()
    outer = tracer.start_span("outer")
    inner = tracer.start_span("inner")
    tracer.end_span(inner)
    tracer.end_span(outer)

    trace = tracer.get_trace()
    assert len(trace) == 2
    assert trace[0]["parent_id"] == outer.span_id
    assert trace[1]["span_id"] == outer.span_id


def test_end_span_uses_stack_when_no_arg():
    tracer = Tracer()
    tracer.start_span("top")
    tracer.start_span("middle")
    tracer.start_span("bottom")

    b = tracer.end_span()
    assert b.name == "bottom"
    m = tracer.end_span()
    assert m.name == "middle"
    t = tracer.end_span()
    assert t.name == "top"


def test_end_span_empty_stack_returns_empty():
    tracer = Tracer()
    result = tracer.end_span()
    assert isinstance(result, TraceSpan)
    assert result.span_id == ""


def test_add_event():
    tracer = Tracer()
    tracer.start_span("with-events")
    tracer.add_event("cache.hit", {"key": "user:1"})
    tracer.add_event("cache.miss", {"key": "user:2"})

    span = tracer.end_span()
    assert len(span.events) == 2
    assert span.events[0]["name"] == "cache.hit"
    assert span.events[1]["attributes"]["key"] == "user:2"


def test_add_event_no_active_span():
    tracer = Tracer()
    tracer.add_event("orphan-event")  # should not raise


def test_trace_export_format():
    tracer = Tracer()
    span = tracer.start_span("export-test", attributes={"env": "prod"})
    tracer.end_span(span)

    trace = tracer.get_trace()
    assert len(trace) == 1
    exported = trace[0]
    assert "span_id" in exported
    assert "parent_id" in exported
    assert "name" in exported
    assert exported["name"] == "export-test"
    assert "start_time" in exported
    assert "duration_ms" in exported
    assert "status" in exported
    assert exported["status"] == "ok"
    assert "attributes" in exported
    assert exported["attributes"]["env"] == "prod"
    assert "events" in exported
    assert "error" in exported
    json.dumps(exported)  # ensure JSON-serializable


def test_get_stats():
    tracer = Tracer()
    assert tracer.get_stats()["total_spans"] == 0

    tracer.trace_llm_call("openai", "gpt-4", [{"role": "user", "content": "hi"}], "ok", 10, 100)
    tracer.trace_llm_call(
        "anthropic", "claude-3", [{"role": "user", "content": "hi"}], "ok", 20, 200, error="timeout"
    )
    tracer.start_span("plain")
    tracer.end_span()

    stats = tracer.get_stats()
    assert stats["total_spans"] == 3
    assert stats["llm_calls"] == 2
    assert stats["total_tokens"] == 30
    assert stats["errors"] == 1


def test_reset_tracer():
    reset_tracer()
    t1 = get_tracer()
    t2 = get_tracer()
    assert t1 is t2
    reset_tracer()
    t3 = get_tracer()
    assert t3 is not t1


def test_trace_config_defaults():
    config = TraceConfig()
    assert config.enabled is True
    assert config.trace_llm_calls is True
    assert config.service_name == "noema"


def test_trace_span_to_dict():
    span = TraceSpan(
        span_id="abc123",
        parent_id="parent1",
        name="test-span",
        kind="internal",
        start_time=1000.0,
        end_time=1001.5,
        duration_ms=1500.0,
        attributes={"key": "value"},
        events=[{"name": "event1"}],
        status="ok",
        error="",
    )
    d = span.to_dict()
    assert d["span_id"] == "abc123"
    assert d["duration_ms"] == 1500.0
    assert d["attributes"]["key"] == "value"
    assert len(d["events"]) == 1


def test_llm_call_redaction():
    tracer = Tracer()
    # sk- followed by 20+ alphanumeric chars triggers OpenAI key redaction
    messages = [{"role": "user", "content": "my key is sk-proj-A1B2C3D4E5F6G7H8I9J0K1L2M3N4O"}]
    span = tracer.trace_llm_call("openai", "gpt-4", messages, "response ok", 5, 10)
    assert "sk-proj-A1B2C3D4E5F6G7H8I9J0K1L2M3N4O" not in span.attributes.get("llm.prompt", "")
    assert "[REDACTED-OPENAI-KEY]" in span.attributes.get("llm.prompt", "")
    assert "response ok" in span.attributes.get("llm.response", "")


def test_llm_call_trace_disabled():
    config = TraceConfig(trace_prompts=False, trace_responses=False)
    tracer = Tracer(config=config)
    messages = [{"role": "user", "content": "hello"}]
    span = tracer.trace_llm_call("openai", "gpt-4", messages, "world", 10, 50)
    assert "llm.prompt" not in span.attributes
    assert "llm.response" not in span.attributes


def test_tracer_prompt_version_control():
    tracer = Tracer()
    pv = tracer.register_prompt("greeting", "Hello, {name}!", version="1.0.0")
    assert pv.name == "greeting"
    assert pv.version == "1.0.0"
    assert pv.previous_text == ""

    pv2 = tracer.register_prompt("greeting", "Hi, {name}!", version="2.0.0")
    assert pv2.previous_text == "Hello, {name}!"

    retrieved = tracer.get_prompt_version("greeting")
    assert retrieved is not None
    assert retrieved.text == "Hi, {name}!"


def test_shadow_prompt_mode():
    tracer = Tracer()
    pv = tracer.register_prompt("test-shadow", "original", shadow_mode=True)
    assert pv.shadow_mode is True

    for _ in range(5):
        tracer.record_shadow_result("test-shadow", 0.9, ["grammar"])
    assert len(pv.shadow_results) == 5

    promoted = tracer.promote_shadow_prompt("test-shadow", min_score=0.05)
    assert promoted is True
    assert pv.shadow_mode is False


def test_shadow_promotion_not_enough_data():
    tracer = Tracer()
    tracer.register_prompt("shadow2", "text", shadow_mode=True)
    tracer.record_shadow_result("shadow2", 0.9, [])
    assert tracer.promote_shadow_prompt("shadow2") is False


def test_prompt_stats():
    tracer = Tracer()
    tracer.register_prompt("p1", "hello", shadow_mode=True)
    tracer.register_prompt("p2", "world")
    stats = tracer.prompt_stats()
    assert "p1" in stats
    assert "p2" in stats
    assert stats["p1"]["shadow"] is True
    assert stats["p2"]["shadow"] is False
