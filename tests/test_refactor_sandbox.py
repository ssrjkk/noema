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

from noema.sandbox.engine import SandboxConfig, SandboxEngine, _safe_join
from noema.sandbox.static_check import StaticIssue, analyze_code

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


def test_docker_flags_use_configured_limits() -> None:
    """T1.1: memory/CPU quotas come from the config, not hardcoded values."""
    cfg = SandboxConfig(max_memory_mb=128, max_cpus=0.25)
    engine = SandboxEngine(config=cfg)
    flags = engine._docker_run_flags()  # noqa: SLF001
    assert "--memory" in flags and "128m" in flags
    assert "--cpus" in flags and "0.25" in flags
    assert "--network=none" in flags


def test_configured_limits_flow_into_preexec() -> None:
    """T1.1: the direct-run path binds config values into the rlimit preexec."""
    cfg = SandboxConfig(max_memory_mb=96, max_cpu_seconds=3.0)
    engine = SandboxEngine(config=cfg)
    preexec = engine._resource_limits_preexec()  # noqa: SLF001
    if os.name == "nt":
        assert preexec is None
        return
    assert preexec is not None
    import resource

    def call() -> None:
        preexec()
        # memory limit is applied as RLIMIT_AS
        soft, _ = resource.getrlimit(resource.RLIMIT_AS)  # type: ignore[attr-defined]
        assert soft == 96 * 1024 * 1024

    # rlimits apply to a child process; the preexec sets them in the child.
    import subprocess
    import sys

    probe = subprocess.run(
        [sys.executable, "-c", "import resource; print(resource.getrlimit(resource.RLIMIT_AS)[0])"],
        capture_output=True,
        text=True,
        preexec_fn=preexec,
    )
    assert probe.returncode == 0
    assert probe.stdout.strip() == str(96 * 1024 * 1024)


# ── T1.2 static validation pre-sandbox ──────────────────────────────────────


def test_static_accepts_valid_code() -> None:
    code = "import math\n\ndef area(r: float) -> float:\n    return math.pi * r**2\n"
    assert analyze_code(code) == []


def test_static_flags_undefined_call() -> None:
    """Call-graph sanity: a call to a name never defined is rejected statically."""
    code = "def solve():\n    return helper_that_does_not_exist(1)\n"
    issues = analyze_code(code)
    assert any(issue.rule == "undefined-name" for issue in issues)


def test_static_flags_wildcard_import() -> None:
    issues = analyze_code("from os import *\n")
    assert any(issue.rule == "wildcard-import" for issue in issues)


def test_static_flags_relative_import() -> None:
    issues = analyze_code("from . import helpers\n")
    assert any(issue.rule == "relative-import" for issue in issues)


def test_static_flags_forbidden_import() -> None:
    issues = analyze_code("import numpy as np\n")
    assert any(issue.rule == "import-not-allowed" for issue in issues)


def test_static_reports_syntax() -> None:
    issues = analyze_code("def broken(:\n    pass\n")
    assert len(issues) == 1
    assert issues[0].rule == "syntax"


def test_static_defers_module_forward_references() -> None:
    """Names bound at module level after first use are still defined."""
    code = "def run():\n    return total\n\ntotal = 3\n"
    assert analyze_code(code) == []


def test_static_respects_nested_scopes() -> None:
    code = (
        "def outer(items):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "    squared = [v * v for v in items]\n"
        "    return total, len(squared)\n"
    )
    assert analyze_code(code) == []


def test_static_flags_undefined_in_closure_scope() -> None:
    code = "def outer():\n    def inner():\n        return missing\n    return inner()\n"
    issues = analyze_code(code)
    assert any(issue.rule == "undefined-name" and "missing" in issue.message for issue in issues)


def test_static_allows_sibling_import_when_passed() -> None:
    code = "from app import helper\n"
    issues = analyze_code(code)
    assert any(issue.rule == "import-not-allowed" for issue in issues)
    assert analyze_code(code, allowed_imports={"app"}) == []


def test_static_issue_renders() -> None:
    issue = StaticIssue(line=3, rule="undefined-name", message="'foo' is never defined")
    assert issue.render() == "line 3: [undefined-name] 'foo' is never defined"


async def test_validate_block_fills_static_fields() -> None:
    engine = SandboxEngine()
    bad = await engine.validate_code_block("def solve():\n    return missing()\n")
    assert bad.static_passed is False
    assert any("undefined-name" in line for line in bad.static_issues)
    good = await engine.validate_code_block("def solve():\n    return 1\n")
    assert good.static_passed is True
    assert good.static_issues == []


async def test_validate_block_static_can_be_disabled() -> None:
    engine = SandboxEngine(config=SandboxConfig(static_check_enabled=False))
    bad = await engine.validate_code_block("def solve():\n    return missing()\n")
    assert bad.static_passed is True
    assert bad.static_issues == []


async def test_static_failure_skips_sandbox_run() -> None:
    """T1.2 done-when: broken output is rejected by the static pass, not the sandbox."""
    cfg = SandboxConfig(lint_enabled=False, test_enabled=False, run_enabled=True)
    engine = SandboxEngine(config=cfg)
    result = await engine.validate_files(
        [
            {
                "path": "main.py",
                "language": "python",
                "content": "def solve():\n    return missing()\n",
            }
        ]
    )
    vr = result.files[0]
    assert vr.static_passed is False
    assert vr.run_passed is True  # run stage was short-circuited, never executed
    assert vr.run_errors == ""
    assert result.all_valid is False


async def test_sibling_imports_pass_static_in_multifile() -> None:
    cfg = SandboxConfig(lint_enabled=False, test_enabled=False, run_enabled=False)
    engine = SandboxEngine(config=cfg)
    result = await engine.validate_files(
        [
            {"path": "app.py", "language": "python", "content": "def helper():\n    return 1\n"},
            {
                "path": "main.py",
                "language": "python",
                "content": "from app import helper\nx = helper()\n",
            },
        ]
    )
    assert all(vr.static_passed for vr in result.files)
    assert result.all_valid is True
