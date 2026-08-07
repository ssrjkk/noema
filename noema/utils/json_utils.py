"""Canonical JSON serialization and markdown-fenced JSON extraction helpers."""

from __future__ import annotations

import json
from typing import Any


def serialize_to_bytes(data: Any) -> bytes:
    """Canonical JSON bytes used for hashing and commitments.

    Deterministic across runs: keys sorted, ASCII-safe, non-serializable
    values stringified via ``default=str``.
    """
    return json.dumps(data, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")


def strip_fences(text: str) -> str:
    """Extract the inner content of the first markdown code block, if any."""
    t = text.strip()
    if "```json" in t:
        start = t.index("```json") + len("```json")
        end = t.find("```", start)
        if end == -1:
            return t[start:].strip()
        return t[start:end].strip()
    if "```" in t:
        start = t.index("```") + 3
        end = t.find("```", start)
        if end == -1:
            return t[start:].strip()
        return t[start:end].strip()
    return t


def extract_fenced_json(text: str, default: Any = None) -> Any:
    """Parse JSON out of an LLM response that may be markdown-fenced."""
    try:
        return json.loads(strip_fences(text))
    except (json.JSONDecodeError, ValueError):
        # Last resort: pull the largest balanced {…} or […] span out of the text.
        for open_char, close_char in (("{", "}"), ("[", "]")):
            first = text.find(open_char)
            last = text.rfind(close_char)
            if first != -1 and last > first:
                try:
                    return json.loads(text[first : last + 1])
                except (json.JSONDecodeError, ValueError):
                    continue
        return default
