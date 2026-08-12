"""Incident modeling — normalize Sentry alerts / webhook payloads into fix tasks."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Incident:
    """A normalized production incident that needs a fix."""

    id: str
    title: str
    description: str
    stack_trace: str = ""
    source: str = "webhook"
    tags: list[str] = field(default_factory=list)
    occurred_at: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


def _sentry_title(payload: dict[str, Any]) -> str:
    event = payload.get("event") or payload
    if isinstance(event, str):
        try:
            event = json.loads(event)
        except (TypeError, json.JSONDecodeError):
            event = {}
    title = (
        event.get("title")
        or event.get("message")
        or event.get("exception", {}).get("values", [{}])[0].get("type", "")
        or "Production incident"
    )
    return str(title)[:200]


def _sentry_stack(payload: dict[str, Any]) -> str:
    event = payload.get("event") or payload
    if isinstance(event, str):
        try:
            event = json.loads(event)
        except (TypeError, json.JSONDecodeError):
            event = {}
    frames: list[str] = []
    for entry in event.get("exception", {}).get("values", []):
        value = str(entry.get("value", ""))
        stack = str(entry.get("stacktrace", ""))
        frames.append(value)
        for frame in entry.get("stacktrace", {}).get("frames", [])[:20]:
            filename = str(frame.get("filename", ""))
            function = str(frame.get("function", ""))
            line = frame.get("lineno", "")
            frames.append(f"  File {filename!r}, line {line}, in {function}")
        if stack:
            frames.append(stack)
    return "\n".join(frames) if frames else str(event.get("stacktrace", ""))


def _sentry_context(payload: dict[str, Any]) -> str:
    event = payload.get("event") or payload
    if isinstance(event, str):
        try:
            event = json.loads(event)
        except (TypeError, json.JSONDecodeError):
            event = {}
    parts: list[str] = []
    for key in ("environment", "release", "server_name"):
        if event.get(key):
            parts.append(f"{key}={event[key]}")
    culprit = event.get("culprit")
    if culprit:
        parts.append(f"culprit={culprit}")
    return "; ".join(parts)


def parse_incident(payload: dict[str, Any]) -> Incident:
    """Normalize a Sentry webhook or a generic incident webhook payload.

    Understands:
    - Sentry ``issue``/``event`` alerts (``{"event": {...}}`` envelope),
    - generic ``{"incident": {...}}`` or flat ``{"title", "description"}``,
    - and the project's own incident schema
      ``{"payload": {"title", "description", "stack_trace"}}``.
    """
    body = payload.get("payload")
    if not isinstance(body, dict):
        body = payload
    event = body.get("event")
    incident = body.get("incident")
    source = str(body.get("source", "sentry" if event else "webhook"))

    if incident and isinstance(incident, dict):
        return Incident(
            id=str(incident.get("id") or uuid.uuid4().hex[:12]),
            title=str(incident.get("title") or "Incident")[:200],
            description=str(incident.get("description", ""))[:2000],
            stack_trace=str(incident.get("stack_trace", "")),
            source=source,
            tags=[str(t) for t in incident.get("tags", [])][:20],
            occurred_at=float(incident.get("occurred_at", time.time())),
            raw=body,
        )

    if event or source == "sentry":
        event_obj = event if isinstance(event, dict) else (body if isinstance(body, dict) else {})
        event_id = str(event_obj.get("id") or body.get("id") or uuid.uuid4().hex[:12])[:32]
        stack = _sentry_stack(body)
        ctx = _sentry_context(body)
        description = str(
            event_obj.get("logentry", {}).get("formatted", "") or event_obj.get("message", "") or ""
        )
        if ctx:
            description = f"{description} ({ctx})" if description else ctx
        title = _sentry_title(body)
        return Incident(
            id=event_id,
            title=title,
            description=description[:2000],
            stack_trace=stack,
            source="sentry",
            tags=list(_sentry_tags(body))[:20],
            occurred_at=time.time(),
            raw=body,
        )

    title = str(body.get("title") or body.get("message") or "Incident")[:200]
    return Incident(
        id=str(body.get("id") or uuid.uuid4().hex[:12]),
        title=title,
        description=str(body.get("description", ""))[:2000],
        stack_trace=str(body.get("stack_trace", "")),
        source=source,
        tags=[str(t) for t in body.get("tags", [])][:20],
        occurred_at=float(body.get("occurred_at", time.time())),
        raw=body,
    )


def _sentry_tags(body: dict[str, Any]) -> list[str]:
    event = body.get("event")
    if isinstance(event, dict):
        tags = event.get("tags", [])
        if isinstance(tags, list):
            return [str(t[1]) for t in tags if isinstance(t, (list, tuple)) and len(t) == 2]
    return []


def incident_to_task(incident: Incident) -> dict[str, Any]:
    """Project an incident onto the fix-task contract consumed by NoemaEngine."""
    title = f"Fix: {incident.title}"
    description = incident.description or f"Reproduce and fix the incident: {incident.title}"
    if incident.stack_trace:
        description = f"{description}\n\nStack trace:\n{incident.stack_trace[:3000]}"
    tags = list(incident.tags or [])
    for tag in ("bug", "incident"):
        if tag not in tags:
            tags.append(tag)
    return {
        "title": title,
        "description": description,
        "complexity": "complex",
        "tags": tags[:20],
    }


def incident_branch_name(incident: Incident) -> str:
    """Derive a stable, git-safe branch name from an incident."""
    slug = re.sub(r"[^a-z0-9]+", "-", incident.title.lower()).strip("-")
    slug = slug[:40] or "incident"
    return f"noema-fix/{incident.id[:8]}-{slug}"
