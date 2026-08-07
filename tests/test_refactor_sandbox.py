"""Hypothesis property tests for the refactored SandboxEngine.

Covers ``noema/sandbox/engine.py``:
- zero-trust ``_safe_join`` (path traversal rejected, safe paths stay rooted),
- AST validation of arbitrary Python snippets (never raises, structured result),
- non-Python passthrough.
"""

import os
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from noema.sandbox.engine import SandboxEngine, _safe_join

ENGINE = SandboxEngine()
BASE = Path("C:/tmp/base") if os.name == "nt" else Path("/tmp/base")

# Safe relative components: no separators, no NUL, no drive-relative "C:"-style
# prefixes (on Windows those escape the root, so "_safe_join" rejects them).
SAFE_REL = st.text(
    alphabet=st.characters(blacklist_categories=("C", "Z"), blacklist_characters="/\\\x00:"),
    max_size=40,
)

# Concatenating two separator-free components escapes the root only when it
# yields exactly ".." (e.g. "." + "."). "_safe_join" correctly rejects that.
SAFE_COMBO = st.tuples(SAFE_REL, SAFE_REL).filter(lambda p: p[0] + p[1] != "..")


@settings(max_examples=200)
@given(combo=SAFE_COMBO)
def test_safe_join_safe_paths_stay_rooted(combo: tuple[str, str]) -> None:
    rel, name = combo
    path = _safe_join(BASE, rel + name)
    assert path.is_absolute()
    assert BASE.resolve() in path.resolve().parents or path.resolve() == BASE.resolve()


@settings(max_examples=50)
@given(escape=st.sampled_from(["../..", "a/../../b", "..", "./../x"]))
def test_safe_join_rejects_traversal(escape: str) -> None:
    with pytest.raises(ValueError, match="Unsafe path"):
        _safe_join(BASE, escape)


@settings(max_examples=50)
@given(abs_path=st.sampled_from(["/etc/passwd", "\\windows\\system32", "///abs"]))
def test_safe_join_neutralizes_leading_separators(abs_path: str) -> None:
    """Leading separators are stripped, so absolute paths become rooted."""
    path = _safe_join(BASE, abs_path)
    assert BASE.resolve() in path.resolve().parents


@settings(max_examples=100)
@given(code=st.text(alphabet="defg x():\\n\\t=+*01", max_size=200))
async def test_validate_code_block_never_raises(code: str) -> None:
    result = await ENGINE.validate_code_block(code)
    assert result.ast_valid is False or result.ast_valid is True
    assert isinstance(result.ast_errors, list)


@settings(max_examples=50)
@given(code=st.text(alphabet="abc x = 1\\n\\t\\\"'+", max_size=300))
async def test_validate_code_block_has_no_ast_errors_when_valid(code: str) -> None:
    result = await ENGINE.validate_code_block(code)
    if result.ast_valid:
        assert result.ast_errors == []


async def test_validate_code_block_valid_python() -> None:
    result = await ENGINE.validate_code_block("x = 1\ny = x + 1\n")
    assert result.ast_valid is True
    assert result.ast_errors == []


async def test_validate_code_block_invalid_python() -> None:
    result = await ENGINE.validate_code_block("def broken(:\n    pass\n")
    assert result.ast_valid is False
    assert len(result.ast_errors) >= 1


@settings(max_examples=50)
@given(code=st.text(max_size=200))
async def test_validate_code_block_non_python_skips_ast(code: str) -> None:
    """Non-Python languages bypass the AST stage entirely."""
    result = await ENGINE.validate_code_block(code, language="typescript")
    assert result.ast_valid is True
    assert result.ast_errors == []
