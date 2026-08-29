"""Pure-AST static checks for generated code, run before the sandbox stages.

Checks (no subprocess, no disk I/O):
- **syntax** — ``ast.parse``; a ``SyntaxError`` is reported as an issue,
- **import hygiene** — wildcard imports, relative imports, and imports whose
  root module is not available in the sandbox are flagged,
- **call-graph sanity** — scope-aware analysis flags names that are used but
  never defined anywhere they could resolve to (imports, assignments,
  parameters, comprehensions, loop/with/exception targets).

The analyzer is deliberately permissive: it may miss a genuine error
(a false negative — the sandbox run still catches it) but should never flag a
name that Python would resolve at runtime. On an internal error it fails open
so the analyzer can never block valid code.

Complexity: ``O(N)`` time/space for ``N`` AST nodes, single pass per phase.
"""

from __future__ import annotations

import ast
import builtins
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection

_BUILTINS = frozenset(dir(builtins))
_STDLIB_MODULES = frozenset(sys.stdlib_module_names)
# Names the interpreter injects into every module; usable as bare names.
_IMPLICIT = frozenset(
    {
        "__name__",
        "__file__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__builtins__",
        "__annotations__",
    }
)


@dataclass(frozen=True)
class StaticIssue:
    line: int
    rule: str
    message: str

    def render(self) -> str:
        return f"line {self.line}: [{self.rule}] {self.message}"


def _target_names(target: ast.expr) -> set[str]:
    """Names bound by an assignment target (supports tuples / starred)."""
    return {
        node.id
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def _bind_walrus_targets(stmt: ast.AST, bound: set[str]) -> None:
    """Bind :class:`ast.NamedExpr` (walrus ``:=``) targets owned by ``stmt``.

    Walrus expressions bind their target into the *enclosing* scope, so they
    must be recorded here regardless of which statement/expression position
    they appear in (e.g. an ``if (n := f())`` test). Traversal prunes nested
    function/class/lambda bodies and comprehensions — targets bound there
    belong to an inner scope, not the current one.
    """
    stack = list(ast.iter_child_nodes(stmt))
    while stack:
        node = stack.pop()
        if isinstance(node, ast.NamedExpr):
            bound.update(_target_names(node.target))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        elif isinstance(
            node,
            (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp),
        ):
            # Comprehension targets live in an inner scope, but walrus inside
            # the comprehension body still binds the enclosing scope; only the
            # generator targets are scoped. The body expression is reached via
            # its `elt`/`key`/`value` fields, which are not generator targets,
            # so keep traversing — targets are handled by _bind_stmt's
            # scope-aware branches in the visitor, not here.
            stack.extend(ast.iter_child_nodes(node))
        else:
            stack.extend(ast.iter_child_nodes(node))


def _bind_stmt(stmt: ast.stmt, bound: set[str]) -> None:
    """Record names bound by ``stmt`` in the current scope.

    Never descends into a nested function/class body (their bindings belong to
    their own scope); the nested definition's *name* is bound here.
    """
    _bind_walrus_targets(stmt, bound)
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        bound.add(stmt.name)
        return
    if isinstance(stmt, ast.Lambda):
        return
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        for alias in stmt.names:
            if alias.name == "*":
                continue
            bound.add(alias.asname or alias.name.split(".")[0])
        return
    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            bound.update(_target_names(target))
        return
    if isinstance(stmt, ast.AnnAssign):
        bound.update(_target_names(stmt.target))
        return
    if isinstance(stmt, ast.AugAssign):
        bound.update(_target_names(stmt.target))
        return
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        bound.update(_target_names(stmt.target))
        for child in stmt.body + stmt.orelse:
            _bind_stmt(child, bound)
        return
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        for item in stmt.items:
            if item.optional_vars is not None:
                bound.update(_target_names(item.optional_vars))
        for child in stmt.body:
            _bind_stmt(child, bound)
        return
    if isinstance(stmt, ast.ExceptHandler):
        if stmt.name:
            bound.add(stmt.name)
        for child in stmt.body:
            _bind_stmt(child, bound)
        return
    if isinstance(stmt, (ast.If, ast.While)):
        for child in stmt.body + stmt.orelse:
            _bind_stmt(child, bound)
        return
    if isinstance(stmt, ast.Try):
        for child in [*stmt.body, *stmt.orelse, *stmt.finalbody]:
            _bind_stmt(child, bound)
        for handler in stmt.handlers:
            if handler.name:
                bound.add(handler.name)
            for child in handler.body:
                _bind_stmt(child, bound)
        return
    # Bare expressions and remaining statements: walrus targets are recorded
    # by _bind_walrus_targets (called at the top of this function).
    return


def _bindings(stmts: list[ast.stmt]) -> set[str]:
    bound: set[str] = set()
    for stmt in stmts:
        _bind_stmt(stmt, bound)
    return bound


class _Scope:
    __slots__ = ("defined", "parent")

    def __init__(self, parent: _Scope | None) -> None:
        self.defined: set[str] = set()
        self.parent = parent


def _param_names(args: ast.arguments) -> set[str]:
    names = {arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs}
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return names


class _UndefinedNameFinder:
    """Scope-aware scan for names used but never defined (call-graph sanity).

    Conservative: comprehension targets, loop/with/exception bindings and
    forward references inside a function are all treated as defined, so the
    pass only fires on names that no enclosing scope could provide.
    """

    def __init__(self, issues: list[StaticIssue]) -> None:
        self._issues = issues
        self._module = _Scope(None)
        self._stack: list[_Scope] = [self._module]

    def run(self, tree: ast.Module) -> None:
        self._module.defined.update(_bindings(tree.body))
        for stmt in tree.body:
            self._visit_stmt(stmt)

    def _resolves(self, name: str) -> bool:
        for scope in reversed(self._stack):
            if name in scope.defined:
                return True
        return name in _BUILTINS or name in _IMPLICIT

    def _report(self, node: ast.Name) -> None:
        self._issues.append(
            StaticIssue(node.lineno, "undefined-name", f"'{node.id}' is never defined")
        )

    def _visit_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = _Scope(self._stack[-1])
            scope.defined.update(_bindings(stmt.body))
            scope.defined.update(_param_names(stmt.args))
            for decorator in stmt.decorator_list:
                self._visit_expr(decorator)
            for default in stmt.args.defaults + stmt.args.kw_defaults:
                if default is not None:
                    self._visit_expr(default)
            self._stack.append(scope)
            for child in stmt.body:
                self._visit_stmt(child)
            self._stack.pop()
            return
        if isinstance(stmt, ast.ClassDef):
            scope = _Scope(self._stack[-1])
            scope.defined.update(_bindings(stmt.body))
            for decorator in stmt.decorator_list:
                self._visit_expr(decorator)
            for base in stmt.bases:
                self._visit_expr(base)
            for keyword in stmt.keywords:
                self._visit_expr(keyword.value)
            self._stack.append(scope)
            for child in stmt.body:
                self._visit_stmt(child)
            self._stack.pop()
            return
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            self._visit_expr(stmt.iter)
            for child in stmt.body + stmt.orelse:
                self._visit_stmt(child)
            return
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                self._visit_expr(item.context_expr)
            for child in stmt.body:
                self._visit_stmt(child)
            return
        if isinstance(stmt, ast.While):
            self._visit_expr(stmt.test)
            for child in stmt.body + stmt.orelse:
                self._visit_stmt(child)
            return
        if isinstance(stmt, ast.If):
            self._visit_expr(stmt.test)
            for child in stmt.body + stmt.orelse:
                self._visit_stmt(child)
            return
        if isinstance(stmt, ast.Try):
            for handler in stmt.handlers:
                if handler.type is not None:
                    self._visit_expr(handler.type)
                for child in handler.body:
                    self._visit_stmt(child)
            for child in stmt.body + stmt.orelse + stmt.finalbody:
                self._visit_stmt(child)
            return
        if isinstance(
            stmt,
            (
                ast.Return,
                ast.Raise,
                ast.Assert,
                ast.Expr,
                ast.Assign,
                ast.AnnAssign,
                ast.AugAssign,
                ast.Delete,
            ),
        ):
            for sub in ast.iter_child_nodes(stmt):
                if isinstance(sub, ast.expr):
                    self._visit_expr(sub)
            return
        if isinstance(
            stmt,
            (
                ast.Import,
                ast.ImportFrom,
                ast.Global,
                ast.Nonlocal,
                ast.Pass,
                ast.Break,
                ast.Continue,
            ),
        ):
            return
        for sub in ast.iter_child_nodes(stmt):
            if isinstance(sub, ast.expr):
                self._visit_expr(sub)
            elif isinstance(sub, ast.stmt):
                self._visit_stmt(sub)

    def _visit_expr(self, expr: ast.expr) -> None:
        if isinstance(expr, ast.Name):
            if isinstance(expr.ctx, ast.Load) and not self._resolves(expr.id):
                self._report(expr)
            return
        if isinstance(expr, ast.Lambda):
            scope = _Scope(self._stack[-1])
            scope.defined.update(_param_names(expr.args))
            self._stack.append(scope)
            self._visit_expr(expr.body)
            self._stack.pop()
            return
        if isinstance(expr, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            scope = _Scope(self._stack[-1])
            for generator in expr.generators:
                scope.defined.update(_target_names(generator.target))
            self._stack.append(scope)
            for generator in expr.generators:
                self._visit_expr(generator.iter)
                for condition in generator.ifs:
                    self._visit_expr(condition)
            if isinstance(expr, ast.DictComp):
                self._visit_expr(expr.key)
                self._visit_expr(expr.value)
            else:
                self._visit_expr(expr.elt)
            self._stack.pop()
            return
        if isinstance(expr, ast.NamedExpr):
            self._visit_expr(expr.value)
            return
        for child in ast.iter_child_nodes(expr):
            if isinstance(child, ast.expr):
                self._visit_expr(child)


def _import_issues(tree: ast.Module, allowed_imports: Collection[str]) -> list[StaticIssue]:
    issues: list[StaticIssue] = []
    allowed = set(allowed_imports)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in allowed:
                    issues.append(
                        StaticIssue(
                            node.lineno,
                            "import-not-allowed",
                            f"import '{root}' is not available in the sandbox",
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                issues.append(
                    StaticIssue(node.lineno, "relative-import", "relative imports are not allowed")
                )
                continue
            for alias in node.names:
                if alias.name == "*":
                    issues.append(
                        StaticIssue(
                            node.lineno,
                            "wildcard-import",
                            "wildcard imports ('import *') are not allowed",
                        )
                    )
                    continue
                root = (node.module or "").split(".")[0]
                if root and root not in allowed:
                    issues.append(
                        StaticIssue(
                            node.lineno,
                            "import-not-allowed",
                            f"import '{root}' is not available in the sandbox",
                        )
                    )
    return issues


def analyze_code(
    code: str,
    *,
    allowed_imports: Collection[str] = _STDLIB_MODULES,
) -> list[StaticIssue]:
    """Return the static issues found in ``code``. Never raises.

    ``allowed_imports`` is the set of import root modules the code may use
    (stdlib by default; callers with sibling files should extend it).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [StaticIssue(exc.lineno or 1, "syntax", exc.msg or "invalid syntax")]

    issues: list[StaticIssue] = _import_issues(tree, allowed_imports)
    try:
        finder = _UndefinedNameFinder(issues)
        finder.run(tree)
    except Exception:  # fail open: the analyzer must never block valid code
        return issues
    return issues
