"""Hypothesis property tests for the refactored NoemaEngine.

Covers ``noema/core/noema.py``:
- zero-trust ``think()`` input validation (rejects malformed tasks pre-init),
- quality-tier mapping from average step confidence (boundary-exhaustive),
- reasoning-summary invariants,
- defensive ``_safe_parse`` behavior on arbitrary inputs.
"""

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from noema.core.engine import NoemaEngine
from noema.core.types import (
    Solution,
    SolutionQuality,
    Task,
    ThoughtProcess,
)
from noema.utils.json_utils import extract_fenced_json

FLAT_FLOAT = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
SMALL_TEXT = st.text(max_size=80)
# Printable characters safe to embed inside a JSON string literal.
JSON_SAFE = st.text(
    alphabet=st.characters(blacklist_categories=("C", "Z"), blacklist_characters='"\\'),
    min_size=1,
    max_size=20,
)


# Arbitrary text guaranteed NOT to be valid standalone JSON and not to contain
# a balanced {…}/[…] fragment that extract_fenced_json would recover.
def _is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


NON_JSON_TEXT = st.text(min_size=1, max_size=300).filter(
    lambda s: not _is_valid_json(s) and extract_fenced_json(s, default=None) is None
)

# Built once at import: construction is expensive (~2s of component wiring),
# and the methods under test are stateless w.r.t. the engine, so a single
# shared instance is safe and keeps every hypothesis example fast.
_ENGINE = NoemaEngine(worker_count=1)


def _noema() -> NoemaEngine:
    return _ENGINE


# ── think(): zero-trust input validation ──────────────────────────────


@pytest.mark.asyncio
async def test_think_rejects_huge_description() -> None:
    noema = NoemaEngine(worker_count=1)
    task = Task(title="ok", description="x" * 100_001)
    with pytest.raises(ValueError, match="description"):
        await noema.think(task)


@pytest.mark.asyncio
async def test_think_rejects_too_many_tags() -> None:
    noema = NoemaEngine(worker_count=1)
    task = Task(title="ok", tags=[f"t{i}" for i in range(101)])
    with pytest.raises(ValueError, match="tags"):
        await noema.think(task)


@pytest.mark.asyncio
@settings(max_examples=25)
@given(
    title=st.text(max_size=10),
    description=SMALL_TEXT,
    tags=st.lists(SMALL_TEXT, max_size=5),
)
async def test_think_validation_never_initializes_on_rejection(title, description, tags) -> None:
    """Resource-exhaustion guards raise before any component initialization."""
    noema = NoemaEngine(worker_count=1)
    invalid = len(description) > 100_000 or len(tags) > 100
    if not invalid:
        pytest.skip("valid input")
    task = Task(title=title, description=description, tags=tags)
    with pytest.raises(ValueError):
        await noema.think(task)
    assert noema._initialized is False


# ── _evaluate_quality: boundary-exhaustive tier mapping ───────────────


def _thought_with_confidence(confidence: float) -> ThoughtProcess:
    thought = ThoughtProcess(task_id="t1")
    thought.add_step("analysis", "in", "out", confidence)
    return thought


@settings(max_examples=200)
@given(score=FLAT_FLOAT)
def test_evaluate_quality_tier_boundaries(score: float) -> None:
    noema = _noema()
    solution = Solution(task_id="t1", title="x", summary="y")
    noema._evaluate_quality(solution, _thought_with_confidence(score))

    if score >= 0.9:
        expected = SolutionQuality.MASTERPIECE
    elif score >= 0.75:
        expected = SolutionQuality.EXCELLENT
    elif score >= 0.6:
        expected = SolutionQuality.GOOD
    elif score >= 0.4:
        expected = SolutionQuality.ACCEPTABLE
    else:
        expected = SolutionQuality.DRAFT
    assert solution.quality == expected
    assert solution.confidence == pytest.approx(score, abs=1e-9)


@settings(max_examples=50)
@given(scores=st.lists(FLAT_FLOAT, min_size=1, max_size=8))
def test_evaluate_quality_uses_average_confidence(scores: list[float]) -> None:
    noema = _noema()
    solution = Solution(task_id="t1", title="x", summary="y")
    thought = ThoughtProcess(task_id="t1")
    for s in scores:
        thought.add_step("analysis", "in", "out", s)
    avg = sum(scores) / len(scores)
    noema._evaluate_quality(solution, thought)
    assert solution.confidence == pytest.approx(avg, abs=1e-9)


@settings(max_examples=50)
@given(scores=st.lists(FLAT_FLOAT, max_size=8))
def test_evaluate_quality_empty_steps_map_to_draft(scores: list[float]) -> None:
    """Empty thought (no steps) must map to DRAFT / confidence 0.0."""
    noema = _noema()
    solution = Solution(task_id="t1", title="x", summary="y")
    thought = ThoughtProcess(task_id="t1")
    noema._evaluate_quality(solution, thought)
    assert solution.quality == SolutionQuality.DRAFT
    assert solution.confidence == 0.0


# ── _extract_summary: invariants ──────────────────────────────────────


@settings(max_examples=100)
@given(
    design=SMALL_TEXT,
    review=SMALL_TEXT,
)
def test_extract_summary_includes_artifacts(design: str, review: str) -> None:
    noema = _noema()
    reasoning = {
        "architecture": {"high_level_design": design},
        "review": {"final_summary": review},
    }
    summary = noema._extract_summary(reasoning)
    assert isinstance(summary, str)
    assert len(summary) > 0
    if design:
        assert design[:200] in summary


@settings(max_examples=100)
@given(reasoning=st.dictionaries(st.text(max_size=10), st.text(max_size=40), max_size=5))
def test_extract_summary_never_empty_or_raises(reasoning: dict) -> None:
    noema = _noema()
    summary = noema._extract_summary(reasoning)
    assert isinstance(summary, str)
    assert len(summary) > 0


@settings(max_examples=100)
@given(field=st.text(max_size=500))
def test_extract_summary_truncates_artifact_fields(field: str) -> None:
    """Artifact fields are capped at 200 chars, so summary stays bounded."""
    noema = _noema()
    summary = noema._extract_summary({"architecture": {"high_level_design": field}})
    if field:
        assert field[:200] in summary
        assert len(summary) <= len(field[:200]) + len(" | ")


# ── _safe_parse: defensive parsing ────────────────────────────────────


@settings(max_examples=100)
@given(
    data=st.one_of(
        st.dictionaries(st.text(max_size=10), st.text(max_size=20), max_size=5),
        st.lists(st.text(max_size=10), max_size=5),
    )
)
def test_safe_parse_passthrough_structured(data: object) -> None:
    noema = _noema()
    result = noema._safe_parse(data)
    assert result == data


@settings(max_examples=100)
@given(
    data=st.one_of(
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans(),
        st.none(),
    )
)
def test_safe_parse_scalar_to_empty_dict(data: object) -> None:
    noema = _noema()
    assert noema._safe_parse(data) == {}


@settings(max_examples=100)
@given(raw=NON_JSON_TEXT)
def test_safe_parse_non_json_text_wraps_raw(raw: str) -> None:
    noema = _noema()
    result = noema._safe_parse(raw)
    assert isinstance(result, dict)
    assert "raw" in result
    assert len(result["raw"]) <= 500


@settings(max_examples=50)
@given(
    key=JSON_SAFE,
    value=JSON_SAFE,
)
def test_safe_parse_extracts_fenced_json(key: str, value: str) -> None:
    noema = _noema()
    payload = f'{{\n  "{key}": "{value}"\n}}'
    result = noema._safe_parse(f"```json\n{payload}\n```")
    assert isinstance(result, dict)
    assert result.get(key) == value
