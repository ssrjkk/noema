"""Tests for noema.utils.json_utils — JSON serialization and extraction."""

import json

from noema.utils.json_utils import extract_fenced_json, serialize_to_bytes, strip_fences


class TestSerializeToBytes:
    def test_deterministic_key_order(self):
        data = {"z": 1, "a": 2, "m": 3}
        result = serialize_to_bytes(data)
        assert result == serialize_to_bytes(data)
        parsed = json.loads(result)
        assert list(parsed.keys()) == ["a", "m", "z"]

    def test_ascii_safe(self):
        data = {"emoji": "🔥", "cyrillic": "Привет"}
        result = serialize_to_bytes(data)
        assert isinstance(result, bytes)
        assert b"\\u" in result

    def test_non_serializable_stringified(self):
        class Custom:
            def __str__(self):
                return "custom_value"

        data = {"obj": Custom()}
        result = serialize_to_bytes(data)
        parsed = json.loads(result)
        assert parsed["obj"] == "custom_value"

    def test_nested_structures(self):
        data = {"list": [1, 2, {"nested": True}], "dict": {"a": [3, 4]}}
        result = serialize_to_bytes(data)
        parsed = json.loads(result)
        assert parsed == data

    def test_empty_dict(self):
        assert serialize_to_bytes({}) == b"{}"

    def test_empty_list(self):
        assert serialize_to_bytes([]) == b"[]"


class TestStripFences:
    def test_no_fences(self):
        text = '{"key": "value"}'
        assert strip_fences(text) == text

    def test_json_fence(self):
        text = '```json\n{"key": "value"}\n```'
        assert strip_fences(text) == '{"key": "value"}'

    def test_plain_fence(self):
        text = '```\n{"key": "value"}\n```'
        assert strip_fences(text) == '{"key": "value"}'

    def test_unclosed_json_fence(self):
        text = '```json\n{"key": "value"}'
        result = strip_fences(text)
        assert result == '{"key": "value"}'

    def test_unclosed_plain_fence(self):
        text = '```\n{"key": "value"}'
        result = strip_fences(text)
        assert result == '{"key": "value"}'

    def test_fence_with_surrounding_text(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        assert strip_fences(text) == '{"key": "value"}'

    def test_multiple_fences_takes_first(self):
        text = '```json\n{"first": 1}\n```\n```json\n{"second": 2}\n```'
        assert strip_fences(text) == '{"first": 1}'

    def test_empty_fence(self):
        text = "```json\n```"
        assert strip_fences(text) == ""

    def test_whitespace_handling(self):
        text = '  ```json\n  {"key": "value"}  \n```  '
        assert strip_fences(text) == '{"key": "value"}'


class TestExtractFencedJson:
    def test_valid_json_no_fences(self):
        text = '{"key": "value"}'
        assert extract_fenced_json(text) == {"key": "value"}

    def test_valid_json_with_fence(self):
        text = '```json\n{"key": "value"}\n```'
        assert extract_fenced_json(text) == {"key": "value"}

    def test_invalid_json_returns_default(self):
        text = "not json at all"
        assert extract_fenced_json(text) is None
        assert extract_fenced_json(text, default={}) == {}

    def test_fallback_extracts_object(self):
        text = 'Some text {"key": "value"} more text'
        result = extract_fenced_json(text)
        assert result == {"key": "value"}

    def test_fallback_extracts_array(self):
        text = "Some text [1, 2, 3] more text"
        result = extract_fenced_json(text)
        assert result == [1, 2, 3]

    def test_fallback_fails_gracefully(self):
        text = "Text with {unclosed bracket"
        assert extract_fenced_json(text) is None

    def test_invalid_object_span_falls_through_to_array_span(self):
        text = "Payload {not: json} tail [1, 2] end"
        assert extract_fenced_json(text) == [1, 2]

    def test_both_spans_invalid_returns_default(self):
        text = "Payload {not: json} tail [1, 2 end"
        assert extract_fenced_json(text, default=[]) == []

    def test_fallback_nested_structures(self):
        text = 'Data: {"outer": {"inner": [1, 2, 3]}} end'
        result = extract_fenced_json(text)
        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_fallback_prefers_object_over_array(self):
        text = '{"obj": 1} [2, 3]'
        result = extract_fenced_json(text)
        assert result == {"obj": 1}

    def test_empty_string(self):
        assert extract_fenced_json("") is None

    def test_complex_nested_json(self):
        data = {
            "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            "metadata": {"count": 2, "active": True},
        }
        text = f"```json\n{json.dumps(data)}\n```"
        assert extract_fenced_json(text) == data
