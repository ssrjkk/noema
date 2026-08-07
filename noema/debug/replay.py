"""Replay & Debug Mode — повторение трейсов с измененными параметрами."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from noema.llm.providers import LLMMessage, create_llm_provider
from noema.tracing.tracer import get_tracer


@dataclass
class ReplayStepDiff:
    step_name: str = ""
    original_preview: str = ""
    new_preview: str = ""
    identical: bool = True


@dataclass
class ReplayResult:
    trace_id: str = ""
    original_steps: int = 0
    new_steps: int = 0
    diffs: list[ReplayStepDiff] = field(default_factory=list)
    total_cost_original: float = 0.0
    total_cost_new: float = 0.0
    error: str = ""


class ReplayEngine:
    """Replays a traced LLM session with parameter overrides for debugging."""

    async def replay_trace(
        self,
        trace_id: str,
        overrides: dict[str, Any] | None = None,
    ) -> ReplayResult:
        overrides = overrides or {}
        tracer = get_tracer()
        if tracer is None:
            return ReplayResult(trace_id=trace_id, error="Tracer not initialized")
        trace = tracer.get_trace()
        if not trace:
            return ReplayResult(trace_id=trace_id, error="No trace data available")

        llm_spans = [s for s in trace if s.get("kind") == "llm"]
        diffs: list[ReplayStepDiff] = []
        result = ReplayResult(trace_id=trace_id, original_steps=len(llm_spans))

        for span in llm_spans:
            attrs = span.get("attributes", {})
            provider = overrides.get("provider", attrs.get("llm.provider", ""))
            model = overrides.get("model", attrs.get("llm.model", ""))
            raw_prompt = attrs.get("llm.prompt", "")
            original_response = attrs.get("llm.response", "")

            new_provider = create_llm_provider(provider, model)
            try:
                messages_data = []
                if isinstance(raw_prompt, str):
                    try:
                        messages_data = json.loads(raw_prompt)
                    except json.JSONDecodeError:
                        messages_data = [{"role": "user", "content": raw_prompt}]
                elif isinstance(raw_prompt, list):
                    messages_data = raw_prompt
                messages = [
                    LLMMessage(role=m.get("role", "user"), content=m.get("content", ""))
                    for m in messages_data
                ]
                new_response = await new_provider.complete(
                    messages,
                    temperature=overrides.get("temperature", 0.3),
                    max_tokens=overrides.get("max_tokens", 2048),
                )
                new_preview = new_response.content[:200]
            except Exception as e:
                new_preview = f"[ERROR: {e}]"

            original_preview = original_response[:200] if original_response else "(no response)"
            diffs.append(
                ReplayStepDiff(
                    step_name=span.get("name", ""),
                    original_preview=original_preview,
                    new_preview=new_preview,
                    identical=original_preview == new_preview,
                )
            )

        result.new_steps = len(diffs)
        result.diffs = diffs
        return result
