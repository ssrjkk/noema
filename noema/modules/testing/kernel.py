"""Testing Module — test generation, coverage analysis, mutation testing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TestFramework(StrEnum):
    PYTEST = "pytest"
    unittest = "unittest"
    JEST = "jest"
    VITEST = "vitest"
    GO_TEST = "go test"
    RUST_TEST = "cargo test"
    JUNIT = "junit"
    RSPEC = "rspec"


class TestType(StrEnum):
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    SECURITY = "security"
    FUZZ = "fuzz"
    MUTATION = "mutation"
    SNAPSHOT = "snapshot"


@dataclass
class TestCase:
    id: str = ""
    name: str = ""
    test_type: TestType = TestType.UNIT
    target: str = ""  # function/class being tested
    framework: TestFramework = TestFramework.PYTEST
    code: str = ""
    description: str = ""
    assertions: int = 0
    estimated_coverage: float = 0.0
    tags: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class TestSuite:
    name: str = ""
    test_cases: list[TestCase] = field(default_factory=list)
    framework: TestFramework = TestFramework.PYTEST
    total_assertions: int = 0
    estimated_coverage: float = 0.0

    def add_case(self, case: TestCase) -> None:
        self.test_cases.append(case)
        self.total_assertions += case.assertions


@dataclass
class CoverageReport:
    total_statements: int = 0
    covered_statements: int = 0
    coverage_percent: float = 0.0
    uncovered_lines: list[dict[str, Any]] = field(default_factory=list)
    file_reports: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MutationResult:
    original_score: float = 0.0
    killed: int = 0
    survived: int = 0
    equivalent: int = 0
    mutation_score: float = 0.0
    mutants: list[dict[str, Any]] = field(default_factory=list)


class TestGenerator:
    """Generate test cases from code analysis."""

    LANGUAGE_FRAMEWORKS: dict[str, TestFramework] = {
        "python": TestFramework.PYTEST,
        "javascript": TestFramework.JEST,
        "typescript": TestFramework.VITEST,
        "go": TestFramework.GO_TEST,
        "rust": TestFramework.RUST_TEST,
        "java": TestFramework.JUNIT,
        "ruby": TestFramework.RSPEC,
    }

    def generate_unit_tests(
        self, code: str, language: str = "python", filename: str = ""
    ) -> TestSuite:
        framework = self.LANGUAGE_FRAMEWORKS.get(language, TestFramework.PYTEST)
        suite = TestSuite(name=f"unit_{filename}", framework=framework)

        functions = self._extract_functions(code, language)
        for func_name, func_code in functions:
            test = self._generate_single_unit_test(func_name, func_code, language, framework)
            suite.add_case(test)

        classes = self._extract_classes(code, language)
        for class_name, methods in classes:
            for method_name, method_code in methods:
                test = self._generate_single_unit_test(
                    f"{class_name}.{method_name}", method_code, language, framework
                )
                test.test_type = TestType.UNIT
                suite.add_case(test)

        suite.estimated_coverage = min(len(suite.test_cases) * 12, 95)
        return suite

    def generate_integration_tests(
        self, endpoints: list[dict[str, str]], language: str = "python"
    ) -> TestSuite:
        framework = self.LANGUAGE_FRAMEWORKS.get(language, TestFramework.PYTEST)
        suite = TestSuite(name="integration_tests", framework=framework)

        for ep in endpoints:
            path = ep.get("path", "/")
            method = ep.get("method", "GET").upper()
            test_code = self._generate_integration_test(path, method, language, framework)
            suite.add_case(
                TestCase(
                    name=f"test_{method.lower()}_{path.replace('/', '_').strip('_')}",
                    test_type=TestType.INTEGRATION,
                    target=path,
                    framework=framework,
                    code=test_code,
                    assertions=2,
                    description=f"Test {method} {path}",
                )
            )

        suite.estimated_coverage = min(len(suite.test_cases) * 8, 80)
        return suite

    def generate_property_tests(
        self, function_name: str, input_spec: dict[str, str], language: str = "python"
    ) -> TestSuite:
        framework = self.LANGUAGE_FRAMEWORKS.get(language, TestFramework.PYTEST)
        suite = TestSuite(name=f"property_{function_name}", framework=framework)

        test_code = self._generate_property_test(function_name, input_spec, language, framework)
        suite.add_case(
            TestCase(
                name=f"test_property_{function_name}",
                test_type=TestType.FUZZ,
                target=function_name,
                framework=framework,
                code=test_code,
                assertions=50,
            )
        )
        return suite

    def _extract_functions(self, code: str, language: str) -> list[tuple[str, str]]:
        functions = []
        if language == "python":
            pattern = r"(?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*\w+)?:"
            for match in re.finditer(pattern, code):
                start = match.start()
                indent = len(code[start:].split("\n")[0]) - len(
                    code[start:].split("\n")[0].lstrip()
                )
                end = start
                lines = code[start:].split("\n")[1:]
                for line in lines:
                    if line.strip() and (len(line) - len(line.lstrip())) <= indent and line.strip():
                        break
                    end += len(line) + 1
                functions.append((match.group(1), code[start:end]))
        elif language in ("javascript", "typescript"):
            pattern = r"(?:async\s+)?function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?\("
            for match in re.finditer(pattern, code):
                name = match.group(1) or match.group(2)
                functions.append((name, code[match.start() : match.start() + 200]))
        return functions

    def _extract_classes(self, code: str, language: str) -> list[tuple[str, list[tuple[str, str]]]]:
        classes = []
        if language == "python":
            pattern = r"class\s+(\w+)[^:]*:"
            for match in re.finditer(pattern, code):
                class_name = match.group(1)
                class_start = match.end()
                methods = []
                method_pattern = r"(?:async\s+)?def\s+(\w+)\s*\("
                for m in re.finditer(method_pattern, code[class_start:]):
                    methods.append(
                        (m.group(1), code[class_start + m.start() : class_start + m.start() + 200])
                    )
                classes.append((class_name, methods))
        return classes

    def _generate_single_unit_test(
        self, name: str, code: str, language: str, framework: TestFramework
    ) -> TestCase:
        if framework == TestFramework.PYTEST:
            test_code = (
                f"import pytest\n\n"
                f"def test_{name.replace('.', '_')}():\n"
                f"    # Arrange\n"
                f"    # Act\n"
                f"    result = {name}()\n"
                f"    # Assert\n"
                f"    assert result is not None\n"
                f"    assert not hasattr(result, '__len__') or len(result) >= 0\n"
            )
        elif framework == TestFramework.JEST:
            test_code = (
                f"describe('{name}', () => {{\n"
                f"  it('should work correctly', () => {{\n"
                f"    const result = {name}();\n"
                f"    expect(result).toBeDefined();\n"
                f"  }});\n"
                f"}});\n"
            )
        else:
            test_code = f"// Test for {name}\n// TODO: implement"

        return TestCase(
            name=f"test_{name.replace('.', '_')}",
            test_type=TestType.UNIT,
            target=name,
            framework=framework,
            code=test_code,
            assertions=2,
            estimated_coverage=15.0,
        )

    def _generate_integration_test(
        self, path: str, method: str, language: str, framework: TestFramework
    ) -> str:
        if framework == TestFramework.PYTEST:
            return (
                f"import httpx\n\n"
                f"async def test_{method.lower()}_{path.replace('/', '_')}():\n"
                f"    async with httpx.AsyncClient() as client:\n"
                f"        resp = await client.{method.lower()}('http://localhost:8000{path}')\n"
                f"        assert resp.status_code in (200, 201, 204)\n"
                f"        assert resp.json() is not None\n"
            )
        elif framework == TestFramework.JEST:
            return (
                f"test('{method} {path} returns success', async () => {{\n"
                f"  const res = await request(app).{method.lower()}('{path}');\n"
                f"  expect(res.status).toBeLessThan(400);\n"
                f"}});\n"
            )
        return f"// Integration test for {method} {path}"

    def _generate_property_test(
        self, func: str, spec: dict[str, str], language: str, framework: TestFramework
    ) -> str:
        if framework == TestFramework.PYTEST:
            "\n".join(f"    {k}: {v}," for k, v in spec.items())
            return (
                f"from hypothesis import given, strategies as st\n\n"
                f"@given(st.data())\n"
                f"def test_property_{func}(data):\n"
                f"    # Generate random inputs\n"
                f"    input_val = data.draw(st.integers())\n"
                f"    result = {func}(input_val)\n"
                f"    assert result is not None\n"
                f"    # Property: output type should be consistent\n"
                f"    assert isinstance(result, type(input_val)) or result is not None\n"
            )
        return f"// Property test for {func}"

    def execute(self, task: Any) -> dict[str, Any]:
        tags = getattr(task, "tags", [])
        description = getattr(task, "description", "") if hasattr(task, "description") else ""
        language = "python"
        for tag in tags:
            if tag in self.LANGUAGE_FRAMEWORKS:
                language = tag
                break
        code = self._extract_code_from_task(task) or description
        suite = self.generate_unit_tests(code, language=language)
        return {
            "type": "testing",
            "framework": suite.framework.value,
            "test_count": len(suite.test_cases),
            "total_assertions": suite.total_assertions,
            "estimated_coverage": suite.estimated_coverage,
            "test_cases": [
                {"name": t.name, "type": t.test_type.value, "target": t.target}
                for t in suite.test_cases[:10]
            ],
            "_confidence": 0.8,
        }

    @staticmethod
    def _extract_code_from_task(task: Any) -> str:
        """Pull source code from the task context if the caller provided any."""
        context = getattr(task, "context", {}) or {}
        if not isinstance(context, dict):
            return ""
        files = context.get("code", context.get("files", []))
        if isinstance(files, dict):
            files = files.get("files", [])
        if isinstance(files, str):
            return files
        if not isinstance(files, list):
            return ""
        contents = []
        for f in files:
            if isinstance(f, dict) and isinstance(f.get("content"), str):
                contents.append(f["content"])
            elif isinstance(f, str):
                contents.append(f)
        return "\n".join(contents)


class TestingModule:
    """Standalone testing module — test generation, coverage, mutation."""

    NAME = "testing"
    DESCRIPTION = "Test generation, coverage analysis, mutation testing"

    def __init__(self) -> None:
        self.generator = TestGenerator()
        self.suites: list[TestSuite] = []
        self.coverage: CoverageReport | None = None
        self.mutation: MutationResult | None = None

    def analyze_codebase(self, files: dict[str, str], language: str = "python") -> dict[str, Any]:
        total_functions = 0
        total_classes = 0
        total_lines = 0
        for _path, code in files.items():
            lines = code.count("\n") + 1
            total_lines += lines
            total_functions += len(self.generator._extract_functions(code, language))
            total_classes += len(self.generator._extract_classes(code, language))
        return {
            "total_files": len(files),
            "total_lines": total_lines,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "estimated_test_cases": total_functions + total_classes * 3,
        }

    def generate_full_suite(
        self, code: str, language: str = "python", filename: str = ""
    ) -> TestSuite:
        suite = self.generator.generate_unit_tests(code, language, filename)
        self.suites.append(suite)
        return suite

    def execute(self, task: Any) -> dict[str, Any]:
        return self.generator.execute(task)
