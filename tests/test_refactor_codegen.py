"""Hypothesis property tests for the refactored CodegenKernel.

Covers ``noema/kernels/codegen.py``:
- identifier sanitization invariants (alnum/underscore only, bounded, non-empty),
- module stub invariants (extension matches language, non-empty body),
- stack inference from tags.
"""

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from noema.core.types import Task
from noema.kernels.codegen import CodegenKernel

KERNEL = CodegenKernel()
TEXT = st.text(max_size=200)
LANG = st.sampled_from(["python", "typescript", "go", "rust", "java"])


@settings(max_examples=200)
@given(raw=TEXT, max_len=st.integers(min_value=1, max_value=64))
def test_sanitize_identifier_invariants(raw: str, max_len: int) -> None:
    ident = CodegenKernel._sanitize_identifier(raw, max_len=max_len)
    assert isinstance(ident, str)
    assert len(ident) > 0
    assert len(ident) <= max_len
    assert re.fullmatch(r"[a-z0-9_]*", ident)
    assert ident == ident.strip("_")


@settings(max_examples=100)
@given(raw=TEXT)
def test_sanitize_identifier_preserves_safe_chars(raw: str) -> None:
    ident = CodegenKernel._sanitize_identifier(raw)
    # A clean lowercase alnum input must round-trip up to the length cap.
    # Truncation can cut a run of separators mid-way, leaving an underscore at
    # the boundary; the implementation strips it AFTER truncation, so the
    # expectation must do the same.
    clean = re.sub(r"[^a-z0-9_]+", "_", raw.lower().strip()).strip("_")
    if clean:
        assert ident == clean[:20].strip("_")


@settings(max_examples=100)
@given(
    lang=LANG,
    requirement=TEXT,
    filename=st.text(max_size=60),
)
async def test_execute_subtask_extension_matches_language(lang, requirement, filename) -> None:
    ext_map = {"python": ".py", "typescript": ".ts", "go": ".go", "rust": ".rs", "java": ".java"}
    result = await KERNEL.execute_subtask(
        {"requirement": requirement, "filename": filename},
        stack={"languages": [lang]},
    )
    assert result["language"] == lang
    assert result["filename"].endswith(ext_map[lang])
    assert len(result["content"]) > 0


@settings(max_examples=50)
@given(requirement=TEXT)
async def test_execute_subtask_content_escapes_requirement(requirement) -> None:
    """The requirement text is quote-escaped and backslash-doubled verbatim."""
    result = await KERNEL.execute_subtask(
        {"requirement": requirement, "filename": "x.py"}, stack={"languages": ["python"]}
    )
    expected = requirement.replace('"', '\\"').replace("\\", "\\\\")[:80]
    if requirement:
        assert expected in result["content"]
    assert len(result["content"]) > 0


@settings(max_examples=50)
@given(
    tags=st.lists(
        st.sampled_from(["go", "rust", "java", "spring", "typescript", "node", "python"]),
        max_size=6,
    )
)
async def test_infer_stack_matches_priority(tags) -> None:
    task = Task(title="t", tags=tags)
    stack = KERNEL._infer_stack(task)
    assert stack.languages
    first = stack.languages[0].lower()
    if "go" in tags:
        assert first == "go"
    elif "rust" in tags:
        assert first == "rust"
    elif "java" in tags or "spring" in tags:
        assert first == "java"
    elif "typescript" in tags or "node" in tags:
        assert first == "typescript"
    else:
        assert first == "python"


@settings(max_examples=50)
@given(tags=st.lists(st.text(max_size=12), max_size=10))
async def test_execute_always_produces_python_blocks(tags) -> None:
    """Python is the default fallback, so execute must never return empty blocks."""
    task = Task(title="User Service", description="x", tags=tags)
    result = await KERNEL.execute(task)
    assert result["type"] == "codegen"
    assert result["block_count"] > 0
    assert len(result["blocks"]) == result["block_count"]
