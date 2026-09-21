"""Additional tests for neurosymbolic/symbolic.py — covering utility functions and edge cases."""

from __future__ import annotations

import pytest

from noema.neurosymbolic.symbolic import (
    _to_num,
    _coerce_int,
    _candidate_name,
    _render_bounds,
)


# ── _to_num ──────────────────────────────────────────────────────────────────


def test_to_num_bool_returns_none():
    assert _to_num(True) is None
    assert _to_num(False) is None


def test_to_num_int_returns_float():
    assert _to_num(42) == 42.0
    assert _to_num(0) == 0.0
    assert _to_num(-10) == -10.0


def test_to_num_float_returns_float():
    assert _to_num(3.14) == 3.14
    assert _to_num(0.0) == 0.0


def test_to_num_string_valid():
    assert _to_num("42") == 42.0
    assert _to_num("3.14") == 3.14
    assert _to_num("  100  ") == 100.0
    assert _to_num("-5.5") == -5.5


def test_to_num_string_invalid():
    assert _to_num("not a number") is None
    assert _to_num("") is None
    assert _to_num("abc123") is None


def test_to_num_none_returns_none():
    assert _to_num(None) is None


def test_to_num_list_returns_none():
    assert _to_num([1, 2, 3]) is None


def test_to_num_dict_returns_none():
    assert _to_num({"value": 42}) is None


# ── _coerce_int ──────────────────────────────────────────────────────────────


def test_coerce_int_valid_int():
    assert _coerce_int(42, 0) == 42
    assert _coerce_int(0, 10) == 0
    assert _coerce_int(-5, 0) == -5


def test_coerce_int_string_int():
    assert _coerce_int("42", 0) == 42
    assert _coerce_int("0", 10) == 0


def test_coerce_int_invalid_string():
    assert _coerce_int("not a number", 99) == 99
    assert _coerce_int("", 50) == 50


def test_coerce_int_none():
    assert _coerce_int(None, 100) == 100


def test_coerce_int_float_truncates():
    assert _coerce_int(3.7, 0) == 3
    assert _coerce_int(-2.9, 0) == -2


def test_coerce_int_overflow():
    assert _coerce_int(float("inf"), 77) == 77


# ── _candidate_name ──────────────────────────────────────────────────────────


def test_candidate_name_dict_with_name():
    req = {"name": "my_var"}
    assert _candidate_name(req, 0) == "my_var"


def test_candidate_name_dict_with_category():
    req = {"category": "performance"}
    assert _candidate_name(req, 0) == "performance"


def test_candidate_name_dict_name_preferred_over_category():
    req = {"name": "specific", "category": "general"}
    assert _candidate_name(req, 0) == "specific"


def test_candidate_name_dict_empty():
    req = {}
    assert _candidate_name(req, 5) == ""


def test_candidate_name_non_dict_with_model_dump():
    class MockReq:
        def model_dump(self):
            return {"name": "dumped_var"}

    req = MockReq()
    assert _candidate_name(req, 0) == "dumped_var"


def test_candidate_name_non_dict_without_model_dump():
    req = "just a string"
    assert _candidate_name(req, 3) == "req_3"


def test_candidate_name_none():
    assert _candidate_name(None, 7) == "req_7"


# ── _render_bounds ───────────────────────────────────────────────────────────


def test_render_bounds_both():
    assert _render_bounds("x", 0.0, 10.0) == "x in [0, 10]"
    assert _render_bounds("y", -5.5, 5.5) == "y in [-5.5, 5.5]"


def test_render_bounds_lower_only():
    assert _render_bounds("x", 0.0, None) == "x >= 0"
    assert _render_bounds("y", -10.5, None) == "y >= -10.5"


def test_render_bounds_upper_only():
    assert _render_bounds("x", None, 100.0) == "x <= 100"
    assert _render_bounds("y", None, 3.14) == "y <= 3.14"


def test_render_bounds_none():
    assert _render_bounds("x", None, None) == ""


def test_render_bounds_integer_formatting():
    assert _render_bounds("count", 1.0, 100.0) == "count in [1, 100]"


def test_render_bounds_float_formatting():
    result = _render_bounds("ratio", 0.1, 0.9)
    assert "ratio in [0.1, 0.9]" == result
