"""Tests for noema.utils.helpers — utility functions."""

import pytest

from noema.utils.helpers import chunk_list, deep_merge, generate_id, timer, truncate


class TestTimer:
    @pytest.mark.asyncio
    async def test_timer_decorator_measures_time(self):
        @timer
        async def sample_func():
            return "result"

        result = await sample_func()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_timer_decorator_with_args(self):
        @timer
        async def add(a: int, b: int) -> int:
            return a + b

        result = await add(2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_timer_decorator_with_kwargs(self):
        @timer
        async def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = await greet("Alice", greeting="Hi")
        assert result == "Hi, Alice!"

    @pytest.mark.asyncio
    async def test_timer_decorator_preserves_exceptions(self):
        @timer
        async def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await failing_func()


class TestGenerateId:
    def test_deterministic(self):
        assert generate_id("test") == generate_id("test")

    def test_different_inputs_different_ids(self):
        assert generate_id("a") != generate_id("b")

    def test_length_is_12(self):
        result = generate_id("test_data")
        assert len(result) == 12

    def test_hex_characters(self):
        result = generate_id("test")
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_string(self):
        result = generate_id("")
        assert len(result) == 12


class TestDeepMerge:
    def test_flat_dicts(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_dicts(self):
        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"b": 3, "c": 4}}
        result = deep_merge(base, override)
        assert result == {"outer": {"a": 1, "b": 3, "c": 4}}

    def test_deeply_nested(self):
        base = {"l1": {"l2": {"l3": {"value": 1}}}}
        override = {"l1": {"l2": {"l3": {"extra": 2}}}}
        result = deep_merge(base, override)
        assert result == {"l1": {"l2": {"l3": {"value": 1, "extra": 2}}}}

    def test_override_replaces_non_dict_with_dict(self):
        base = {"key": "string"}
        override = {"key": {"nested": True}}
        result = deep_merge(base, override)
        assert result == {"key": {"nested": True}}

    def test_override_replaces_dict_with_non_dict(self):
        base = {"key": {"nested": True}}
        override = {"key": "string"}
        result = deep_merge(base, override)
        assert result == {"key": "string"}

    def test_empty_base(self):
        assert deep_merge({}, {"a": 1}) == {"a": 1}

    def test_empty_override(self):
        assert deep_merge({"a": 1}, {}) == {"a": 1}

    def test_both_empty(self):
        assert deep_merge({}, {}) == {}

    def test_does_not_mutate_base(self):
        base = {"a": 1}
        override = {"b": 2}
        deep_merge(base, override)
        assert base == {"a": 1}


class TestTruncate:
    def test_short_text_unchanged(self):
        assert truncate("hello", 100) == "hello"

    def test_exact_length_unchanged(self):
        assert truncate("hello", 5) == "hello"

    def test_long_text_truncated(self):
        result = truncate("hello world", 8)
        assert result == "hello..."
        assert len(result) == 8

    def test_custom_suffix(self):
        result = truncate("hello world", 8, suffix="~")
        assert result == "hello w~"

    def test_empty_suffix(self):
        result = truncate("hello world", 5, suffix="")
        assert result == "hello"

    def test_empty_string(self):
        assert truncate("", 10) == ""

    def test_default_max_len(self):
        long_text = "x" * 200
        result = truncate(long_text)
        assert len(result) == 100
        assert result.endswith("...")


class TestChunkList:
    def test_even_split(self):
        assert chunk_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

    def test_chunk_larger_than_list(self):
        assert chunk_list([1, 2], 5) == [[1, 2]]

    def test_chunk_size_one(self):
        assert chunk_list([1, 2, 3], 1) == [[1], [2], [3]]

    def test_empty_list(self):
        assert chunk_list([], 3) == []

    def test_preserves_order(self):
        result = chunk_list([1, 2, 3, 4, 5, 6], 3)
        assert result == [[1, 2, 3], [4, 5, 6]]
