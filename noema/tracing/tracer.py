"""Tracer — OpenTelemetry-совместимая трассировка + Prompt Version Control."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from noema.logging import get_logger
from noema.security.redactor import redact_text

log = get_logger(__name__)


@dataclass
class PromptVersion:
    name: str = ""
    version: str = "1.0.0"
    text: str = ""
    previous_text: str = ""
    shadow_mode: bool = False
    shadow_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TraceConfig:
    enabled: bool = True
    trace_llm_calls: bool = True
    trace_prompts: bool = True
    trace_responses: bool = True
    max_prompt_length: int = 2000
    max_response_length: int = 2000
    export_endpoint: str = ""
    service_name: str = "noema"


@dataclass
class TraceSpan:
    span_id: str = ""
    parent_id: str = ""
    name: str = ""
    kind: str = "internal"
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "start_time": round(self.start_time, 3),
            "duration_ms": round(self.duration_ms, 1),
            "status": self.status,
            "attributes": {k: str(v)[:500] for k, v in self.attributes.items()},
            "events": self.events[-10:],
            "error": self.error[:500] if self.error else "",
        }


class Tracer:
    def __init__(self, config: TraceConfig | None = None) -> None:
        self.config = config or TraceConfig()
        self._spans: list[TraceSpan] = []
        self._stack: list[TraceSpan] = []
        self._prompt_registry: dict[str, PromptVersion] = {}

    # ── Prompt Version Control ────────────────────────────────────

    @property
    def current_span_id(self) -> str:
        """ID of the innermost active span, or empty string."""
        return self._stack[-1].span_id if self._stack else ""

    def register_prompt(
        self,
        name: str,
        text: str,
        version: str = "1.0.0",
        shadow_mode: bool = False,
    ) -> PromptVersion:
        prev = self._prompt_registry.get(name)
        pv = PromptVersion(
            name=name,
            version=version,
            text=text,
            previous_text=prev.text if prev else "",
            shadow_mode=shadow_mode,
        )
        self._prompt_registry[name] = pv
        log.info("prompt_registered", name=name, version=version, shadow=shadow_mode)
        return pv

    def get_prompt_version(self, name: str) -> PromptVersion | None:
        return self._prompt_registry.get(name)

    def record_shadow_result(self, name: str, judge_score: float, weaknesses: list[str]) -> None:
        pv = self._prompt_registry.get(name)
        if pv and pv.shadow_mode:
            pv.shadow_results.append(
                {
                    "judge_score": judge_score,
                    "weaknesses": weaknesses,
                    "timestamp": time.time(),
                }
            )

    def promote_shadow_prompt(self, name: str, min_score: float = 0.05) -> bool:
        """Promote shadow prompt to production if statistically better (p < 0.05 simulation)."""
        pv = self._prompt_registry.get(name)
        if not pv or not pv.shadow_mode:
            return False
        if len(pv.shadow_results) < 5:
            return False
        scores = [r["judge_score"] for r in pv.shadow_results]
        avg = sum(scores) / len(scores)
        if avg >= min_score:
            pv.shadow_mode = False
            log.info("prompt_promoted", name=name, avg_score=round(avg, 3))
            return True
        return False

    def prompt_stats(self) -> dict[str, Any]:
        return {
            name: {
                "version": pv.version,
                "shadow": pv.shadow_mode,
                "results_count": len(pv.shadow_results),
            }
            for name, pv in self._prompt_registry.items()
        }

    def start_span(
        self,
        name: str,
        kind: str = "internal",
        attributes: dict[str, Any] | None = None,
    ) -> TraceSpan:
        parent_id = self._stack[-1].span_id if self._stack else ""
        span = TraceSpan(
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent_id,
            name=name,
            kind=kind,
            start_time=time.monotonic(),
            attributes=attributes or {},
        )
        self._stack.append(span)
        return span

    def end_span(
        self,
        span: TraceSpan | None = None,
        status: str = "ok",
        error: str = "",
    ) -> TraceSpan:
        if not self._stack:
            return TraceSpan()
        s = span or self._stack.pop()
        s.end_time = time.monotonic()
        s.duration_ms = (s.end_time - s.start_time) * 1000
        s.status = status
        s.error = error
        self._spans.append(s)
        return s

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        if self._stack:
            self._stack[-1].events.append(
                {
                    "name": name,
                    "timestamp": time.monotonic(),
                    "attributes": attributes or {},
                }
            )

    def trace_llm_call(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        response: str,
        tokens_used: int,
        latency_ms: float,
        error: str = "",
    ) -> TraceSpan:
        attrs: dict[str, Any] = {
            "llm.provider": provider,
            "llm.model": model,
            "llm.tokens": tokens_used,
            "llm.latency_ms": round(latency_ms, 1),
        }
        if self.config.trace_prompts:
            raw = json.dumps(messages)
            attrs["llm.prompt"] = redact_text(raw)[: self.config.max_prompt_length]
        if self.config.trace_responses and not error:
            attrs["llm.response"] = redact_text(response)[: self.config.max_response_length]

        span = self.start_span(f"llm.{provider}", kind="llm", attributes=attrs)
        self.end_span(span, status="error" if error else "ok", error=error)
        return span

    def get_trace(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._spans]

    def get_stats(self) -> dict[str, Any]:
        llm_spans = [s for s in self._spans if s.kind == "llm"]
        total_tokens = sum(s.attributes.get("llm.tokens", 0) for s in llm_spans)
        total_latency = sum(s.duration_ms for s in llm_spans)
        return {
            "total_spans": len(self._spans),
            "llm_calls": len(llm_spans),
            "total_tokens": total_tokens,
            "total_llm_latency_ms": round(total_latency, 1),
            "errors": sum(1 for s in self._spans if s.status == "error"),
        }


_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def reset_tracer() -> None:
    global _tracer
    _tracer = None
