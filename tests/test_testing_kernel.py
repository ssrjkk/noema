"""Tests for noema.modules.testing.kernel — Testing module kernel."""

from __future__ import annotations

from types import SimpleNamespace

from noema.modules.testing.kernel import (
    CoverageReport,
    MutationResult,
    TestCase,
    TestFramework,
    TestGenerator,
    TestingModule,
    TestSuite,
    TestType,
)

# These are production symbols that happen to start with "Test"; pytest would
# otherwise try to collect them as test classes.
for _symbol in (TestCase, TestFramework, TestGenerator, TestingModule, TestSuite, TestType):
    _symbol.__test__ = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PYTHON_CLASS_CODE = """\
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
"""

PYTHON_FUNC_CODE = """\
def greet(name):
    return f"Hello, {name}"

async def async_fetch(url):
    return {"status": 200}
"""

JS_CODE = """\
function hello() {
  return "world";
}

const foo = async (x) => {
  return x + 1;
};
"""


# ---------------------------------------------------------------------------
# Lines 106-111: class method iteration in generate_unit_tests
# ---------------------------------------------------------------------------


class TestGenerateUnitTestsClasses:
    """Cover lines 106-111 — iterate over extracted class methods."""

    def test_generates_tests_for_class_methods(self):
        gen = TestGenerator()
        suite = gen.generate_unit_tests(PYTHON_CLASS_CODE, language="python", filename="calc.py")
        # Should have test cases for each method in the class
        method_names = [tc.target for tc in suite.test_cases]
        assert "Calculator.add" in method_names
        assert "Calculator.subtract" in method_names
        for tc in suite.test_cases:
            assert tc.test_type == TestType.UNIT

    def test_class_and_functions_combined(self):
        code = PYTHON_FUNC_CODE + "\n" + PYTHON_CLASS_CODE
        gen = TestGenerator()
        suite = gen.generate_unit_tests(code, language="python", filename="mixed.py")
        targets = [tc.target for tc in suite.test_cases]
        assert "greet" in targets
        assert "Calculator.add" in targets


# ---------------------------------------------------------------------------
# Lines 119-139: generate_integration_tests
# ---------------------------------------------------------------------------


class TestGenerateIntegrationTests:
    """Cover lines 119-139."""

    def test_basic_integration_tests(self):
        gen = TestGenerator()
        endpoints = [
            {"path": "/users", "method": "GET"},
            {"path": "/items", "method": "POST"},
        ]
        suite = gen.generate_integration_tests(endpoints, language="python")
        assert suite.name == "integration_tests"
        assert suite.framework == TestFramework.PYTEST
        assert len(suite.test_cases) == 2
        assert suite.test_cases[0].test_type == TestType.INTEGRATION
        assert suite.test_cases[0].assertions == 2
        assert suite.estimated_coverage == min(2 * 8, 80)

    def test_default_method_is_get(self):
        gen = TestGenerator()
        endpoints = [{"path": "/health"}]
        suite = gen.generate_integration_tests(endpoints)
        assert "get" in suite.test_cases[0].name

    def test_integration_javascript_framework(self):
        gen = TestGenerator()
        endpoints = [{"path": "/api", "method": "DELETE"}]
        suite = gen.generate_integration_tests(endpoints, language="javascript")
        assert suite.framework == TestFramework.JEST

    def test_empty_endpoints(self):
        gen = TestGenerator()
        suite = gen.generate_integration_tests([])
        assert len(suite.test_cases) == 0
        assert suite.estimated_coverage == 0


# ---------------------------------------------------------------------------
# Lines 144-158: generate_property_tests
# ---------------------------------------------------------------------------


class TestGeneratePropertyTests:
    """Cover lines 144-158."""

    def test_property_test_python(self):
        gen = TestGenerator()
        spec = {"x": "int", "y": "str"}
        suite = gen.generate_property_tests("my_func", spec, language="python")
        assert suite.name == "property_my_func"
        assert suite.framework == TestFramework.PYTEST
        assert len(suite.test_cases) == 1
        tc = suite.test_cases[0]
        assert tc.test_type == TestType.FUZZ
        assert tc.target == "my_func"
        assert tc.assertions == 50
        assert "hypothesis" in tc.code

    def test_property_test_javascript(self):
        gen = TestGenerator()
        spec = {"n": "number"}
        suite = gen.generate_property_tests("jsFunc", spec, language="javascript")
        assert suite.framework == TestFramework.JEST
        # Non-pytest framework hits the else branch
        tc = suite.test_cases[0]
        assert "// Property test" in tc.code


# ---------------------------------------------------------------------------
# Lines 176-180: JavaScript/TypeScript extraction in _extract_functions
# ---------------------------------------------------------------------------


class TestExtractFunctionsJS:
    """Cover lines 176-180 — JS/TS function extraction."""

    def test_javascript_functions(self):
        gen = TestGenerator()
        funcs = gen._extract_functions(JS_CODE, "javascript")
        names = [f[0] for f in funcs]
        assert "hello" in names
        assert "foo" in names

    def test_typescript_functions(self):
        gen = TestGenerator()
        ts_code = """\
function bar(): string {
  return "baz";
}
"""
        funcs = gen._extract_functions(ts_code, "typescript")
        names = [f[0] for f in funcs]
        assert "bar" in names

    def test_unknown_language_returns_empty(self):
        gen = TestGenerator()
        funcs = gen._extract_functions("some code", "cobol")
        assert funcs == []


# ---------------------------------------------------------------------------
# Lines 213-223: JEST and else branches in _generate_single_unit_test
# ---------------------------------------------------------------------------


class TestGenerateSingleUnitTestFrameworks:
    """Cover lines 213-223 — JEST and fallback branches."""

    def test_jest_framework(self):
        gen = TestGenerator()
        tc = gen._generate_single_unit_test("myFunc", "code", "javascript", TestFramework.JEST)
        assert "describe(" in tc.code
        assert "expect(" in tc.code
        assert tc.framework == TestFramework.JEST

    def test_fallback_framework(self):
        gen = TestGenerator()
        tc = gen._generate_single_unit_test("fn", "code", "go", TestFramework.GO_TEST)
        assert "// Test for fn" in tc.code
        assert "// TODO: implement" in tc.code


# ---------------------------------------------------------------------------
# Lines 238-254: _generate_integration_test
# ---------------------------------------------------------------------------


class TestGenerateIntegrationTest:
    """Cover lines 238-254 — integration test code gen for PYTEST, JEST, fallback."""

    def test_pytest_integration(self):
        gen = TestGenerator()
        code = gen._generate_integration_test("/users", "GET", "python", TestFramework.PYTEST)
        assert "httpx" in code
        assert "async def test_get" in code
        assert "resp.status_code" in code

    def test_jest_integration(self):
        gen = TestGenerator()
        code = gen._generate_integration_test("/items", "POST", "javascript", TestFramework.JEST)
        assert "test(" in code
        assert "expect(res.status)" in code

    def test_fallback_integration(self):
        gen = TestGenerator()
        code = gen._generate_integration_test("/data", "PUT", "go", TestFramework.GO_TEST)
        assert "// Integration test for PUT /data" in code


# ---------------------------------------------------------------------------
# Lines 259-272: _generate_property_test
# ---------------------------------------------------------------------------


class TestGeneratePropertyTest:
    """Cover lines 259-272 — property test code gen."""

    def test_pytest_property(self):
        gen = TestGenerator()
        code = gen._generate_property_test("double", {"x": "int"}, "python", TestFramework.PYTEST)
        assert "hypothesis" in code
        assert "@given" in code
        assert "double(input_val)" in code

    def test_non_pytest_property(self):
        gen = TestGenerator()
        code = gen._generate_property_test("double", {"x": "int"}, "javascript", TestFramework.JEST)
        assert "// Property test for double" in code


# ---------------------------------------------------------------------------
# Lines 302, 305, 307, 309, 312-315: _extract_code_from_task branches
# ---------------------------------------------------------------------------


class TestExtractCodeFromTask:
    """Cover various branches in _extract_code_from_task."""

    def test_context_not_dict(self):
        """Line 302 — context is not a dict → return ''."""
        task = SimpleNamespace(context="not a dict")
        assert TestGenerator._extract_code_from_task(task) == ""

    def test_files_is_dict_with_files_key(self):
        """Line 305 — files is a dict, drill into 'files' key."""
        task = SimpleNamespace(context={"code": {"files": ["print('hi')"]}})
        result = TestGenerator._extract_code_from_task(task)
        assert "print('hi')" in result

    def test_files_is_string(self):
        """Line 307 — files is a plain string."""
        task = SimpleNamespace(context={"code": "def foo(): pass"})
        result = TestGenerator._extract_code_from_task(task)
        assert result == "def foo(): pass"

    def test_files_not_list(self):
        """Line 309 — files is neither dict, str, nor list."""
        task = SimpleNamespace(context={"code": 42})
        assert TestGenerator._extract_code_from_task(task) == ""

    def test_files_list_with_dicts(self):
        """Line 312 — list items are dicts with 'content' key."""
        task = SimpleNamespace(context={"files": [{"content": "line1"}, {"content": "line2"}]})
        result = TestGenerator._extract_code_from_task(task)
        assert "line1" in result
        assert "line2" in result

    def test_files_list_with_strings(self):
        """Line 314-315 — list items are plain strings."""
        task = SimpleNamespace(context={"files": ["code_a", "code_b"]})
        result = TestGenerator._extract_code_from_task(task)
        assert "code_a" in result
        assert "code_b" in result

    def test_files_list_mixed_types(self):
        """Mix of dicts and strings, plus non-matching items."""
        task = SimpleNamespace(
            context={"files": [{"content": "a"}, "b", 999, {"no_content": True}]}
        )
        result = TestGenerator._extract_code_from_task(task)
        assert "a" in result
        assert "b" in result

    def test_no_context_attribute(self):
        """Task without context attribute at all."""
        task = SimpleNamespace()
        result = TestGenerator._extract_code_from_task(task)
        assert result == ""

    def test_context_none(self):
        """context=None should be treated as {}."""
        task = SimpleNamespace(context=None)
        result = TestGenerator._extract_code_from_task(task)
        assert result == ""

    def test_files_dict_without_files_key(self):
        """files dict without 'files' key → empty list → empty string."""
        task = SimpleNamespace(context={"code": {"other": "val"}})
        result = TestGenerator._extract_code_from_task(task)
        assert result == ""


# ---------------------------------------------------------------------------
# Lines 351-353: TestingModule.generate_full_suite
# ---------------------------------------------------------------------------


class TestTestingModuleGenerateFullSuite:
    """Cover lines 351-353."""

    def test_generate_full_suite(self):
        mod = TestingModule()
        code = "def add(a, b):\n    return a + b\n"
        suite = mod.generate_full_suite(code, language="python", filename="math.py")
        assert isinstance(suite, TestSuite)
        assert suite in mod.suites
        assert len(suite.test_cases) >= 1

    def test_generate_full_suite_appends(self):
        mod = TestingModule()
        mod.generate_full_suite("def x(): pass")
        mod.generate_full_suite("def y(): pass")
        assert len(mod.suites) == 2


# ---------------------------------------------------------------------------
# Additional coverage: execute method, TestSuite.add_case, dataclass sanity
# ---------------------------------------------------------------------------


class TestExecute:
    """TestGenerator.execute and TestingModule.execute."""

    def test_execute_with_language_tag(self):
        gen = TestGenerator()
        task = SimpleNamespace(
            tags=["javascript"],
            description="some js code",
            context={"code": "function hello() { return 1; }"},
        )
        result = gen.execute(task)
        assert result["type"] == "testing"
        assert result["framework"] == "jest"

    def test_execute_without_language_tag(self):
        gen = TestGenerator()
        task = SimpleNamespace(tags=["fast"], description="def foo(): pass", context={})
        result = gen.execute(task)
        assert result["framework"] == "pytest"

    def test_testing_module_execute_delegates(self):
        mod = TestingModule()
        task = SimpleNamespace(tags=[], description="def bar(): return 1", context={})
        result = mod.execute(task)
        assert result["type"] == "testing"


class TestDataclasses:
    """Basic sanity for dataclasses."""

    def test_test_suite_add_case(self):
        suite = TestSuite()
        tc = TestCase(assertions=3)
        suite.add_case(tc)
        assert suite.total_assertions == 3
        assert len(suite.test_cases) == 1

    def test_coverage_report_defaults(self):
        cr = CoverageReport()
        assert cr.total_statements == 0
        assert cr.uncovered_lines == []

    def test_mutation_result_defaults(self):
        mr = MutationResult()
        assert mr.killed == 0
        assert mr.mutants == []


class TestAnalyzeCodebase:
    """TestingModule.analyze_codebase."""

    def test_analyze_python_files(self):
        mod = TestingModule()
        files = {
            "main.py": "def hello():\n    pass\n\nclass Foo:\n    def bar(self):\n        pass\n",
            "util.py": "def helper():\n    return True\n",
        }
        result = mod.analyze_codebase(files, language="python")
        assert result["total_files"] == 2
        assert result["total_functions"] >= 2
        assert result["total_classes"] >= 1
        assert result["total_lines"] > 0
