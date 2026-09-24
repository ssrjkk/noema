"""Comprehensive tests for noema.sandbox.static_check.

Drives coverage to 100% by exercising every branch:
- StaticIssue dataclass and render()
- _target_names with tuples / starred
- _bind_walrus_targets: walrus inside if-tests, skipping nested func/class/lambda,
  comprehension scoping
- _bind_stmt: every statement kind (With, AsyncWith, ExceptHandler, Try, Lambda guard)
- _param_names: *args, **kwargs
- _UndefinedNameFinder: decorators, defaults, class keywords, With, While, Try,
  Lambda expressions, comprehensions with conditions, DictComp, NamedExpr,
  fallback branches for unusual stmt types
- _import_issues: wildcard, relative, disallowed root
- analyze_code: syntax error, fail-open on internal error
"""

from __future__ import annotations

import ast
from unittest.mock import patch

import pytest

from noema.sandbox.static_check import (
    StaticIssue,
    _bind_stmt,
    _bind_walrus_targets,
    _bindings,
    _param_names,
    _target_names,
    _UndefinedNameFinder,
    analyze_code,
)

# ── StaticIssue ──────────────────────────────────────────────────────────────


class TestStaticIssue:
    def test_render(self):
        issue = StaticIssue(line=10, rule="syntax", message="bad token")
        assert issue.render() == "line 10: [syntax] bad token"

    def test_frozen(self):
        issue = StaticIssue(line=1, rule="r", message="m")
        with pytest.raises(AttributeError):
            issue.line = 2  # type: ignore[misc]


# ── _target_names ────────────────────────────────────────────────────────────


class TestTargetNames:
    def test_simple_name(self):
        tree = ast.parse("x = 1")
        assign = tree.body[0]
        assert _target_names(assign.targets[0]) == {"x"}

    def test_tuple_unpack(self):
        tree = ast.parse("x, y = 1, 2")
        assign = tree.body[0]
        assert _target_names(assign.targets[0]) == {"x", "y"}

    def test_starred(self):
        tree = ast.parse("a, *b = [1, 2, 3]")
        assign = tree.body[0]
        assert _target_names(assign.targets[0]) == {"a", "b"}

    def test_nested_tuple(self):
        tree = ast.parse("(x, (y, z)) = (1, (2, 3))")
        assign = tree.body[0]
        assert _target_names(assign.targets[0]) == {"x", "y", "z"}


# ── _bind_walrus_targets ─────────────────────────────────────────────────────


class TestBindWalrusTargets:
    def test_walrus_in_if(self):
        """Line 79: walrus target is bound into the enclosing scope."""
        tree = ast.parse("if (n := 10): pass")
        bound: set[str] = set()
        _bind_walrus_targets(tree.body[0], bound)
        assert "n" in bound

    def test_walrus_skips_nested_function(self):
        """Nested function definitions are skipped (line 80 branch).

        _bind_walrus_targets iterates child nodes of the *statement*; for a
        FunctionDef the body stmts are direct children, so the walrus is
        reached before the FunctionDef guard triggers.  To exercise the
        ``continue`` guard we nest the function one level deeper — inside an
        ``if`` body — so the FunctionDef appears as a grandchild node.
        """
        code = "if True:\n    def f():\n        (x := 1)\n"
        tree = ast.parse(code)
        bound: set[str] = set()
        _bind_walrus_targets(tree.body[0], bound)
        # x is inside f() which is inside if; the FunctionDef guard prunes it
        assert "x" not in bound

    def test_walrus_skips_nested_class(self):
        """ClassDef inside an if-body is pruned by the guard (line 80)."""
        code = "if True:\n    class C:\n        (x := 1)\n"
        tree = ast.parse(code)
        bound: set[str] = set()
        _bind_walrus_targets(tree.body[0], bound)
        assert "x" not in bound

    def test_walrus_skips_nested_lambda(self):
        """Lambda body is not traversed (line 80-81 branch)."""
        code = "f = lambda: (x := 1)\n"
        tree = ast.parse(code)
        bound: set[str] = set()
        # The assignment statement contains a Lambda; walrus inside lambda
        # should not bind at outer scope
        _bind_walrus_targets(tree.body[0], bound)
        assert "x" not in bound

    def test_walrus_inside_comprehension_body(self):
        """Walrus in comprehension body binds enclosing scope (lines 86-92)."""
        code = "result = [(y := x + 1) for x in range(5)]\n"
        tree = ast.parse(code)
        bound: set[str] = set()
        _bind_walrus_targets(tree.body[0], bound)
        # y is bound by walrus in comprehension body -> enclosing scope
        assert "y" in bound


# ── _bind_stmt ───────────────────────────────────────────────────────────────


class TestBindStmt:
    def test_with_as(self):
        """Lines 131-136: With statement binds 'as' target."""
        code = "with open('f') as fh:\n    data = fh.read()\n"
        tree = ast.parse(code)
        bound = _bindings(tree.body)
        assert "fh" in bound
        assert "data" in bound

    def test_with_multiple_items(self):
        code = "with open('a') as a, open('b') as b:\n    pass\n"
        tree = ast.parse(code)
        bound = _bindings(tree.body)
        assert "a" in bound
        assert "b" in bound

    def test_with_no_as(self):
        """With statement without 'as' clause."""
        code = "with ctx():\n    pass\n"
        tree = ast.parse(code)
        bound = _bindings(tree.body)
        # no name bound from the with itself
        assert bound == set()

    def test_async_with(self):
        code = "async def f():\n    async with bar() as x:\n        pass\n"
        tree = ast.parse(code)
        # The async with is inside async def f; _bindings on the async def body
        func_body = tree.body[0].body  # type: ignore[attr-defined]
        bound = _bindings(func_body)
        assert "x" in bound

    def test_except_handler_named(self):
        """Lines 138-142: ExceptHandler binds the exception name."""
        code = "try:\n    pass\nexcept ValueError as e:\n    pass\n"
        tree = ast.parse(code)
        bound = _bindings(tree.body)
        assert "e" in bound

    def test_except_handler_unnamed(self):
        """ExceptHandler without a name (line 138: if stmt.name is False)."""
        code = "try:\n    pass\nexcept ValueError:\n    pass\n"
        tree = ast.parse(code)
        bound = _bindings(tree.body)
        # no exception name bound
        assert bound == set()

    def test_except_handler_body_bindings(self):
        """Lines 140-141: body of ExceptHandler is recursed into."""
        code = "try:\n    pass\nexcept Exception as e:\n    result = 42\n"
        tree = ast.parse(code)
        bound = _bindings(tree.body)
        assert "e" in bound
        assert "result" in bound

    def test_try_full(self):
        """Lines 148-155: Try with body, handlers, orelse, finalbody."""
        code = (
            "try:\n"
            "    x = 1\n"
            "except TypeError as te:\n"
            "    y = 2\n"
            "else:\n"
            "    z = 3\n"
            "finally:\n"
            "    w = 4\n"
        )
        tree = ast.parse(code)
        bound = _bindings(tree.body)
        assert {"x", "y", "z", "w", "te"} <= bound

    def test_except_handler_direct_call_with_name(self):
        """Lines 138-142: ExceptHandler binds exception name and recurses body.

        ExceptHandlers are never in statement lists (they're children of Try),
        so we call _bind_stmt directly to cover this branch.
        """
        code = "try:\n    pass\nexcept ValueError as e:\n    result = 42\n"
        tree = ast.parse(code)
        try_stmt = tree.body[0]
        handler = try_stmt.handlers[0]
        assert isinstance(handler, ast.ExceptHandler)
        bound: set[str] = set()
        _bind_stmt(handler, bound)  # type: ignore[arg-type]
        assert "e" in bound
        assert "result" in bound

    def test_except_handler_direct_call_without_name(self):
        """Line 138: ExceptHandler without name (stmt.name is None)."""
        code = "try:\n    pass\nexcept ValueError:\n    pass\n"
        tree = ast.parse(code)
        try_stmt = tree.body[0]
        handler = try_stmt.handlers[0]
        bound: set[str] = set()
        _bind_stmt(handler, bound)  # type: ignore[arg-type]
        assert bound == set()

    def test_lambda_guard(self):
        """Line 108: _bind_stmt with a Lambda returns immediately.

        Lambda is an expression, not a statement, so this is a defensive guard.
        We call _bind_stmt directly to cover the branch.
        """
        tree = ast.parse("f = lambda: 1")
        assign = tree.body[0]
        # Extract the lambda node from the assignment value
        lambda_node = assign.value  # type: ignore[attr-defined]
        assert isinstance(lambda_node, ast.Lambda)
        bound: set[str] = set()
        _bind_stmt(lambda_node, bound)  # type: ignore[arg-type]
        # Lambda guard returns without binding anything
        assert bound == set()


# ── _param_names ─────────────────────────────────────────────────────────────


class TestParamNames:
    def test_regular_args(self):
        tree = ast.parse("def f(a, b, c): pass")
        func = tree.body[0]
        assert _param_names(func.args) == {"a", "b", "c"}

    def test_vararg(self):
        """Line 179: *args is bound."""
        tree = ast.parse("def f(*args): pass")
        func = tree.body[0]
        names = _param_names(func.args)
        assert "args" in names

    def test_kwarg(self):
        """Line 181: **kwargs is bound."""
        tree = ast.parse("def f(**kwargs): pass")
        func = tree.body[0]
        names = _param_names(func.args)
        assert "kwargs" in names

    def test_all_param_kinds(self):
        tree = ast.parse("def f(a, /, b, *args, c, **kwargs): pass")
        func = tree.body[0]
        names = _param_names(func.args)
        assert names == {"a", "b", "args", "c", "kwargs"}

    def test_kwonly(self):
        tree = ast.parse("def f(*, x, y): pass")
        func = tree.body[0]
        assert _param_names(func.args) == {"x", "y"}


# ── _UndefinedNameFinder ─────────────────────────────────────────────────────


class TestUndefinedNameFinder:
    def _find_undefined(self, code: str) -> list[StaticIssue]:
        tree = ast.parse(code)
        issues: list[StaticIssue] = []
        finder = _UndefinedNameFinder(issues)
        finder.run(tree)
        return issues

    def test_simple_undefined(self):
        issues = self._find_undefined("print(undefined_var)")
        assert any(i.rule == "undefined-name" and "undefined_var" in i.message for i in issues)

    def test_builtin_not_flagged(self):
        issues = self._find_undefined("print(len([1,2,3]))")
        undefined = [i for i in issues if i.rule == "undefined-name"]
        assert undefined == []

    def test_implicit_names_not_flagged(self):
        issues = self._find_undefined("x = __name__")
        undefined = [i for i in issues if i.rule == "undefined-name"]
        assert undefined == []

    def test_function_decorator_visited(self):
        """Lines 219-220: decorators are visited for undefined names."""
        code = "@unknown_decorator\ndef f(): pass\n"
        issues = self._find_undefined(code)
        assert any("unknown_decorator" in i.message for i in issues)

    def test_function_defaults_visited(self):
        """Lines 221-223: default values are visited for undefined names."""
        code = "def f(x=missing_default): pass\n"
        issues = self._find_undefined(code)
        assert any("missing_default" in i.message for i in issues)

    def test_function_defaults_none_skipped(self):
        """Line 222: None defaults (positional) are skipped."""
        code = "def f(a, b): pass\n"
        issues = self._find_undefined(code)
        undefined = [i for i in issues if i.rule == "undefined-name"]
        assert undefined == []

    def test_kw_defaults_visited(self):
        """Line 223: keyword-only defaults are visited."""
        code = "def f(*, x=missing_kw): pass\n"
        issues = self._find_undefined(code)
        assert any("missing_kw" in i.message for i in issues)

    def test_class_bases_visited(self):
        """Lines 234-235: class bases are visited."""
        code = "class C(MissingBase): pass\n"
        issues = self._find_undefined(code)
        assert any("MissingBase" in i.message for i in issues)

    def test_class_keywords_visited(self):
        """Line 237: class keyword argument values are visited."""
        code = "class C(metaclass=MissingMeta): pass\n"
        issues = self._find_undefined(code)
        assert any("MissingMeta" in i.message for i in issues)

    def test_class_decorator_visited(self):
        """Line 233: class decorators are visited for undefined names."""
        code = "@missing_class_decorator\nclass C: pass\n"
        issues = self._find_undefined(code)
        assert any("missing_class_decorator" in i.message for i in issues)

    def test_with_context_visited(self):
        """Lines 249-253: With context expressions are visited."""
        code = "with missing_ctx() as x: pass\n"
        issues = self._find_undefined(code)
        assert any("missing_ctx" in i.message for i in issues)

    def test_with_body_names_defined(self):
        """Names bound in with-as are available in the body."""
        code = "with open('f') as fh:\n    data = fh.read()\n"
        issues = self._find_undefined(code)
        undefined = [i for i in issues if i.rule == "undefined-name"]
        assert undefined == []

    def test_async_with_visited(self):
        """AsyncWith context expressions are visited."""
        code = "async def f():\n    async with missing_async() as x: pass\n"
        issues = self._find_undefined(code)
        assert any("missing_async" in i.message for i in issues)

    def test_while_test_visited(self):
        """Lines 255-258: While test expression is visited."""
        code = "while missing_flag:\n    break\n"
        issues = self._find_undefined(code)
        assert any("missing_flag" in i.message for i in issues)

    def test_while_body_and_orelse_visited(self):
        code = "while True:\n    do_thing()\nelse:\n    do_other()\n"
        issues = self._find_undefined(code)
        assert any("do_thing" in i.message for i in issues)
        assert any("do_other" in i.message for i in issues)

    def test_try_handlers_visited(self):
        """Lines 265-272: Try handlers' types and bodies are visited."""
        code = (
            "try:\n"
            "    do_stuff()\n"
            "except MissingError:\n"
            "    handle_it()\n"
            "else:\n"
            "    else_stuff()\n"
            "finally:\n"
            "    final_stuff()\n"
        )
        issues = self._find_undefined(code)
        names_flagged = {i.message for i in issues if i.rule == "undefined-name"}
        assert any("MissingError" in m for m in names_flagged)
        assert any("do_stuff" in m for m in names_flagged)
        assert any("handle_it" in m for m in names_flagged)
        assert any("else_stuff" in m for m in names_flagged)
        assert any("final_stuff" in m for m in names_flagged)

    def test_try_handler_type_none(self):
        """Line 266: bare except (handler.type is None) doesn't crash."""
        code = "try:\n    pass\nexcept:\n    pass\n"
        issues = self._find_undefined(code)
        undefined = [i for i in issues if i.rule == "undefined-name"]
        assert undefined == []

    def test_lambda_expression(self):
        """Lines 315-320: Lambda creates a new scope with its params."""
        code = "f = lambda x, y: x + y\n"
        issues = self._find_undefined(code)
        undefined = [i for i in issues if i.rule == "undefined-name"]
        assert undefined == []

    def test_lambda_undefined_body(self):
        code = "f = lambda x: x + missing\n"
        issues = self._find_undefined(code)
        assert any("missing" in i.message for i in issues)

    def test_comprehension_with_condition(self):
        """Line 329: comprehension generator conditions are visited."""
        code = "result = [x for x in data if missing_filter(x)]\n"
        issues = self._find_undefined(code)
        assert any("missing_filter" in i.message for i in issues)
        # data is also undefined
        assert any("data" in i.message for i in issues)

    def test_comprehension_target_scoped(self):
        """Comprehension targets don't leak to outer scope."""
        code = "result = [x for x in range(10)]\n"
        issues = self._find_undefined(code)
        undefined = [i for i in issues if i.rule == "undefined-name"]
        assert undefined == []

    def test_dict_comprehension(self):
        """Lines 331-332: DictComp key and value are visited."""
        code = "d = {missing_key: missing_val for x in items}\n"
        issues = self._find_undefined(code)
        names_flagged = {i.message for i in issues if i.rule == "undefined-name"}
        assert any("missing_key" in m for m in names_flagged)
        assert any("missing_val" in m for m in names_flagged)
        assert any("items" in m for m in names_flagged)

    def test_dict_comprehension_valid(self):
        code = "items = [1,2,3]\nd = {k: v for k, v in items}\n"
        issues = self._find_undefined(code)
        undefined = [i for i in issues if i.rule == "undefined-name"]
        assert undefined == []

    def test_set_comp(self):
        code = "s = {x for x in missing_set}\n"
        issues = self._find_undefined(code)
        assert any("missing_set" in i.message for i in issues)

    def test_generator_exp(self):
        code = "g = (x for x in missing_gen)\n"
        issues = self._find_undefined(code)
        assert any("missing_gen" in i.message for i in issues)

    def test_named_expr_visited(self):
        """Lines 338-339: NamedExpr value is visited."""
        code = "if (x := missing_func()):\n    pass\n"
        issues = self._find_undefined(code)
        assert any("missing_func" in i.message for i in issues)

    def test_named_expr_binds_target(self):
        """Walrus target is bound and usable."""
        code = "if (n := 10):\n    print(n)\n"
        issues = self._find_undefined(code)
        undefined = [i for i in issues if i.rule == "undefined-name"]
        assert undefined == []

    def test_fallback_stmt_branch(self):
        """Lines 303-307: fallback for unrecognized stmt types.

        ast.Match (3.10+) is handled by the fallback branch since it's not
        in any of the explicit isinstance checks.
        """
        code = "match x:\n    case 1:\n        pass\n"
        issues = self._find_undefined(code)
        # x is used but never defined
        assert any("x" in i.message for i in issues)

    def test_fallback_expr_child_stmt(self):
        """Lines 306-307: fallback visits child stmts.

        No standard Python stmt reaches this branch naturally (all stmts with
        child stmts are handled explicitly).  We synthesise a custom ast.stmt
        subclass to exercise it.
        """

        class _CustomStmt(ast.stmt):
            """A synthetic statement containing a child statement."""

            _fields = ("inner",)

            def __init__(self, inner: ast.stmt):
                self.inner = inner
                self.lineno = 1
                self.col_offset = 0
                self.end_lineno = 1
                self.end_col_offset = 0

        # Build a module with our custom stmt wrapping an Expr( Name('x') )
        inner_expr = ast.Expr(
            value=ast.Name(id="undefined_in_custom", ctx=ast.Load(), lineno=1, col_offset=0),
            lineno=1,
            col_offset=0,
        )
        custom = _CustomStmt(inner=inner_expr)
        tree = ast.Module(body=[custom], type_ignores=[])
        ast.fix_missing_locations(tree)

        issues: list[StaticIssue] = []
        finder = _UndefinedNameFinder(issues)
        finder.run(tree)
        assert any("undefined_in_custom" in i.message for i in issues)

    def test_forward_ref_in_function(self):
        """Names defined later in a function body are still found by _bindings."""
        code = "def f():\n    print(x)\n    x = 10\n"
        issues = self._find_undefined(code)
        undefined = [i for i in issues if i.rule == "undefined-name"]
        # x is bound somewhere in the function body, so it resolves
        assert undefined == []

    def test_nested_function_scope(self):
        code = "def outer():\n    def inner():\n        return undefined_inner\n"
        issues = self._find_undefined(code)
        assert any("undefined_inner" in i.message for i in issues)

    def test_class_scope(self):
        code = "class C:\n    x = undefined_class_attr\n"
        issues = self._find_undefined(code)
        assert any("undefined_class_attr" in i.message for i in issues)

    def test_for_iter_visited(self):
        code = "for x in missing_iter:\n    pass\n"
        issues = self._find_undefined(code)
        assert any("missing_iter" in i.message for i in issues)

    def test_for_target_bound(self):
        code = "for item in range(10):\n    print(item)\n"
        issues = self._find_undefined(code)
        undefined = [i for i in issues if i.rule == "undefined-name"]
        assert undefined == []

    def test_async_for(self):
        code = "async def f():\n    async for x in missing_aiter:\n        pass\n"
        issues = self._find_undefined(code)
        assert any("missing_aiter" in i.message for i in issues)

    def test_import_not_flagged_as_undefined(self):
        code = "import os\nprint(os.path)\n"
        issues = self._find_undefined(code)
        undefined = [i for i in issues if i.rule == "undefined-name"]
        assert undefined == []

    def test_delete_stmt_covered(self):
        """Delete is handled by the expression-statement branch (line 283).

        del targets use Store/Del context, not Load, so they are not flagged
        as undefined.  We just verify no crash and no false positive.
        """
        code = "x = 1\ndel x\n"
        issues = self._find_undefined(code)
        undefined = [i for i in issues if i.rule == "undefined-name"]
        assert undefined == []

    def test_assert_visited(self):
        code = "assert missing_assertion\n"
        issues = self._find_undefined(code)
        assert any("missing_assertion" in i.message for i in issues)

    def test_raise_visited(self):
        code = "raise MissingException\n"
        issues = self._find_undefined(code)
        assert any("MissingException" in i.message for i in issues)

    def test_return_visited(self):
        code = "def f():\n    return missing_return\n"
        issues = self._find_undefined(code)
        assert any("missing_return" in i.message for i in issues)

    def test_annassign_visited(self):
        code = "x: int = missing_val\n"
        issues = self._find_undefined(code)
        assert any("missing_val" in i.message for i in issues)

    def test_augassign_visited(self):
        code = "x = 1\nx += missing_aug\n"
        issues = self._find_undefined(code)
        assert any("missing_aug" in i.message for i in issues)

    def test_global_nonlocal_not_flagged(self):
        """Global and nonlocal statements are skipped (no visiting)."""
        code = "def f():\n    global x\n    nonlocal y\n"
        issues = self._find_undefined(code)
        undefined = [i for i in issues if i.rule == "undefined-name"]
        assert undefined == []

    def test_pass_break_continue_not_flagged(self):
        code = "for i in range(10):\n    pass\n    break\n    continue\n"
        issues = self._find_undefined(code)
        undefined = [i for i in issues if i.rule == "undefined-name"]
        assert undefined == []

    def test_if_test_visited(self):
        code = "if missing_cond:\n    pass\n"
        issues = self._find_undefined(code)
        assert any("missing_cond" in i.message for i in issues)

    def test_if_orelse_visited(self):
        code = "if True:\n    pass\nelse:\n    missing_else()\n"
        issues = self._find_undefined(code)
        assert any("missing_else" in i.message for i in issues)


# ── _import_issues (via analyze_code) ────────────────────────────────────────


class TestImportChecks:
    def test_wildcard_import(self):
        code = "from os import *\n"
        issues = analyze_code(code)
        assert any(i.rule == "wildcard-import" for i in issues)

    def test_relative_import(self):
        code = "from . import foo\n"
        issues = analyze_code(code)
        assert any(i.rule == "relative-import" for i in issues)

    def test_relative_import_from_dotdot(self):
        code = "from ..module import thing\n"
        issues = analyze_code(code)
        assert any(i.rule == "relative-import" for i in issues)

    def test_disallowed_import(self):
        code = "import requests\n"
        issues = analyze_code(code, allowed_imports={"os", "sys"})
        assert any(i.rule == "import-not-allowed" and "requests" in i.message for i in issues)

    def test_allowed_import(self):
        code = "import os\n"
        issues = analyze_code(code, allowed_imports={"os"})
        import_issues = [i for i in issues if i.rule == "import-not-allowed"]
        assert import_issues == []

    def test_from_import_disallowed(self):
        code = "from requests import get\n"
        issues = analyze_code(code, allowed_imports={"os"})
        assert any(i.rule == "import-not-allowed" and "requests" in i.message for i in issues)

    def test_from_import_allowed(self):
        code = "from os.path import join\n"
        issues = analyze_code(code, allowed_imports={"os"})
        import_issues = [i for i in issues if i.rule == "import-not-allowed"]
        assert import_issues == []

    def test_dotted_import_root_checked(self):
        code = "import os.path\n"
        issues = analyze_code(code, allowed_imports={"os"})
        import_issues = [i for i in issues if i.rule == "import-not-allowed"]
        assert import_issues == []

    def test_import_as_alias(self):
        code = "import numpy as np\n"
        issues = analyze_code(code, allowed_imports={"os"})
        assert any("numpy" in i.message for i in issues)

    def test_from_import_as_alias(self):
        code = "from requests import get as g\n"
        issues = analyze_code(code, allowed_imports={"os"})
        assert any("requests" in i.message for i in issues)

    def test_from_import_no_module(self):
        """from . import x — level > 0, module is None."""
        code = "from . import x\n"
        issues = analyze_code(code)
        assert any(i.rule == "relative-import" for i in issues)


# ── analyze_code ─────────────────────────────────────────────────────────────


class TestAnalyzeCode:
    def test_syntax_error(self):
        """Line 401: syntax error returns a single issue."""
        code = "def f(\n"
        issues = analyze_code(code)
        assert len(issues) == 1
        assert issues[0].rule == "syntax"

    def test_syntax_error_line_number(self):
        code = "x = 1\ndef f(\n"
        issues = analyze_code(code)
        assert issues[0].line == 2

    def test_clean_code_no_issues(self):
        code = "import os\n\nprint(os.getcwd())\n"
        issues = analyze_code(code)
        assert issues == []

    def test_default_allowed_imports_is_stdlib(self):
        code = "import os\nimport sys\n"
        issues = analyze_code(code)
        import_issues = [i for i in issues if i.rule == "import-not-allowed"]
        assert import_issues == []

    def test_fail_open_on_internal_error(self):
        """Lines 407-408: if _UndefinedNameFinder raises, we fail open.

        Import issues collected before the failure are still returned.
        """
        code = "import requests\nprint(hello)\n"
        with patch.object(_UndefinedNameFinder, "run", side_effect=RuntimeError("simulated")):
            issues = analyze_code(code, allowed_imports={"os"})
        # Import issue for 'requests' was collected before the failure
        assert any(i.rule == "import-not-allowed" for i in issues)
        # No undefined-name issues because the finder crashed (fail open)
        assert not any(i.rule == "undefined-name" for i in issues)

    def test_never_raises(self):
        """analyze_code is documented to never raise."""
        # Even completely broken code should return a list
        issues = analyze_code("def f(\n")
        assert isinstance(issues, list)
        assert len(issues) >= 1

    def test_empty_code(self):
        issues = analyze_code("")
        assert issues == []

    def test_combined_issues(self):
        """Multiple issue types in one snippet."""
        code = "from os import *\nimport requests\nprint(undefined_name)\n"
        issues = analyze_code(code, allowed_imports={"os"})
        rules = {i.rule for i in issues}
        assert "wildcard-import" in rules
        assert "import-not-allowed" in rules
        assert "undefined-name" in rules
