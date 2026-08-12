"""Tests for T1.3: natural-language requirement -> symbolic contract extraction.

Covers ``noema/neurosymbolic/symbolic.py``:
- the extended ``_extract_bounds`` (phrases, percentages, units),
- the done-when criterion: a corpus of requirement sentences yields a
  non-vacuous contract for at least 90% of them,
- description rendering for parsed requirements.
"""

import pytest

from noema.neurosymbolic.symbolic import _extract_bounds, _render_bounds, _unit_scale

#: Corpus of natural-language requirement sentences (T1.3 done-when).
CORPUS: list[str] = [
    "response time must stay below 200ms",
    "latency must not exceed 500 ms",
    "at least 10 requests per second",
    "throughput must be at most 1000 rps",
    "no more than 3 retries",
    "no less than 2 replicas",
    "memory usage must stay under 256MB",
    "cpu utilization must stay above 0.3",
    "failure rate must stay below 1%",
    "accuracy should be at least 0.9",
    "error rate must not exceed 5%",
    "max 8 workers",
    "min 2 nodes",
    "x must be between 2 and 5",
    "y in [1, 10]",
    "z >= 0.5",
    "maximum 8 cores",
    "minimum 2 nodes",
    "must be at least 80%",
    "must be at most 25 MB",
    "up to 64 connections",
    "below 10 seconds",
    "exactly 3 attempts",
    "must stay over 5",
    "greater than or equal to 42",
    "less than or equal to 99",
    "at most 2 hours",
    "must stay below 150ms",
    "must not go above 7",
    "kept above 0",
]


def _non_vacuous(sentence: str) -> bool:
    lower, upper = _extract_bounds({"description": sentence})
    return lower is not None or upper is not None


def test_corpus_non_vacuous_rate_meets_done_when() -> None:
    """T1.3 done-when: >= 90% of the corpus yields a non-vacuous contract."""
    hits = sum(1 for sentence in CORPUS if _non_vacuous(sentence))
    rate = hits / len(CORPUS)
    misses = [s for s in CORPUS if not _non_vacuous(s)]
    assert rate >= 0.9, f"corpus pass rate {rate:.0%} < 90%; missed: {misses}"


def test_corpus_bounds_are_sensible() -> None:
    """Every non-vacuous corpus entry produces sane, finite bounds."""
    for sentence in CORPUS:
        lower, upper = _extract_bounds({"description": sentence})
        if lower is not None:
            assert lower == lower  # not NaN
        if upper is not None:
            assert upper == upper  # not NaN
        if lower is not None and upper is not None:
            assert lower <= upper, f"{sentence!r}: lower {lower} > upper {upper}"


def test_ms_scales_to_seconds() -> None:
    lower, upper = _extract_bounds({"description": "response time must stay below 200ms"})
    assert lower is None
    assert upper == pytest.approx(0.2)


def test_percentage_extracts_number() -> None:
    lower, upper = _extract_bounds({"description": "error rate must not exceed 5%"})
    assert upper == pytest.approx(5.0)


def test_at_least_is_lower_bound() -> None:
    lower, upper = _extract_bounds({"description": "at least 10 requests per second"})
    assert lower == pytest.approx(10.0)
    assert upper is None


def test_at_most_is_upper_bound() -> None:
    lower, upper = _extract_bounds({"description": "at most 1000 rps"})
    assert upper == pytest.approx(1000.0)
    assert lower is None


def test_exactly_sets_both_bounds() -> None:
    lower, upper = _extract_bounds({"description": "exactly 3 attempts"})
    assert lower == pytest.approx(3.0)
    assert upper == pytest.approx(3.0)


def test_between_and_range_still_work() -> None:
    lower, upper = _extract_bounds({"description": "x must be between 2 and 5"})
    assert lower == pytest.approx(2.0)
    assert upper == pytest.approx(5.0)
    lower, upper = _extract_bounds({"description": "y in [1, 10]"})
    assert lower == pytest.approx(1.0)
    assert upper == pytest.approx(10.0)


def test_explicit_min_max_win_over_phrases() -> None:
    lower, upper = _extract_bounds({"min": 0, "max": 100, "description": "at least 10"})
    assert lower == pytest.approx(10.0)  # phrase narrows the lower bound
    assert upper == pytest.approx(100.0)


def test_mixed_constraints_narrow_range() -> None:
    lower, upper = _extract_bounds(
        {
            "constraints": [
                "at least 5",
                "no more than 50",
                "between 10 and 40",
            ]
        }
    )
    assert lower == pytest.approx(10.0)  # max of all lower bounds
    assert upper == pytest.approx(40.0)  # min of all upper bounds


def test_scanning_text_fields_besides_constraints() -> None:
    lower, upper = _extract_bounds({"requirement": "must not exceed 12 ms"})
    assert upper == pytest.approx(0.012)
    lower, upper = _extract_bounds({"statement": "at least 3 workers"})
    assert lower == pytest.approx(3.0)


def test_unit_scale() -> None:
    assert _unit_scale("ms") == 0.001
    assert _unit_scale("us") == 1e-6
    assert _unit_scale("mb") == 1.0
    assert _unit_scale("") == 1.0


def test_render_bounds() -> None:
    assert _render_bounds("x", 0, 100) == "x in [0, 100]"
    assert _render_bounds("x", 5, None) == "x >= 5"
    assert _render_bounds("x", None, 9) == "x <= 9"
    assert _render_bounds("x", None, None) == ""
