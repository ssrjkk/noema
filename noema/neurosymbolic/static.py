"""Static-analysis verdict over a solution hypothesis.

Bridges the pure-AST checks from ``noema.sandbox.static_check`` into the
neurosymbolic pipeline: any Python-source snippet found inside a hypothesis is
analyzed for structure (syntax, import hygiene, undefined names) *before* the
candidate is verified against the symbolic contract, so the pipeline reports a
static-analysis verdict alongside the Z3 verdict.

Complexity: ``O(S · N)`` — ``S`` code snippets found in the hypothesis, each
analyzed in ``O(N)`` over its AST nodes.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from noema.sandbox.static_check import analyze_code

#: Keys that hint a hypothesis value is Python source code.
_CODE_HINT_KEYS = frozenset(
    {
        "code",
        "implementation",
        "implementation_code",
        "source",
        "snippet",
        "python",
        "solution_code",
    }
)


@dataclass
class StaticAnalysisVerdict:
    """Result of analyzing all code snippets in a solution hypothesis."""

    analyzed: bool = False
    code_snippets: int = 0
    passed: bool = True
    issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "analyzed": self.analyzed,
            "code_snippets": self.code_snippets,
            "passed": self.passed,
            "issues": self.issues,
        }


def _is_python_code(text: str) -> bool:
    """True when ``text`` looks like Python source, not a bare value.

    A bare single expression (``"ok"``, ``"42"``) is not code; anything with
    imports, definitions, statements, or multiple lines is.
    """
    stripped = text.strip()
    if len(stripped) < 4:
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
        return "\n" in text
    return True


def _find_code_strings(value: Any) -> list[str]:
    """Recursively collect Python-looking string values from a hypothesis."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and (key in _CODE_HINT_KEYS or _is_python_code(item)):
                found.append(item)
            else:
                found.extend(_find_code_strings(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            found.extend(_find_code_strings(item))
    return found


def analyze_solution_static(solution: Any) -> StaticAnalysisVerdict:
    """Run the AST static pass over every code snippet in a hypothesis.

    Returns:
        A :class:`StaticAnalysisVerdict`; ``analyzed`` is ``True`` only when at
        least one Python snippet was found. Fails open: never raises.
    """
    snippets = _find_code_strings(solution)
    verdict = StaticAnalysisVerdict(code_snippets=len(snippets))
    if not snippets:
        return verdict

    verdict.analyzed = True
    for snippet_index, snippet in enumerate(snippets):
        issues = analyze_code(snippet)
        if issues:
            verdict.passed = False
            verdict.issues.extend(
                f"snippet {snippet_index + 1}: {issue.render()}" for issue in issues
            )
    return verdict
