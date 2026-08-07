"""Code quality module — complexity analysis, code smells, metrics, grading."""

import math
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualityReport:
    grade: str = "F"
    metrics: dict[str, Any] = field(default_factory=dict)
    smells: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class CodeAnalyzer:
    def __init__(self) -> None:
        self._smell_counter = 0

    def analyze(self, code: str, language: str = "python") -> QualityReport:
        self._smell_counter = 0
        metrics = self.complexity(code)
        smells = self.smells(code)
        report = self.grade(metrics)
        report.metrics.update(metrics)
        report.smells = smells
        return report

    def complexity(self, code: str) -> dict[str, Any]:
        lines = code.split("\n")
        total_lines = len(lines)
        blank_lines = sum(1 for line in lines if not line.strip())
        comment_lines = sum(
            1
            for line in lines
            if line.strip().startswith("#")
            or line.strip().startswith("//")
            or line.strip().startswith("/*")
            or line.strip().startswith("*")
        )
        code_lines = total_lines - blank_lines - comment_lines

        functions: list[dict[str, Any]] = []
        classes: list[dict[str, Any]] = []
        current_func: dict[str, Any] | None = None

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            func_match = re.match(r"\s*def\s+(\w+)\s*\(", line)
            class_match = re.match(r"\s*class\s+(\w+)", line)

            if func_match:
                if current_func:
                    functions.append(current_func)
                current_func = {
                    "name": func_match.group(1),
                    "start": i,
                    "end": i,
                    "complexity": 1,
                    "nesting_depth": 0,
                }

            if current_func:
                current_func["end"] = i
                nesting = len(line) - len(line.lstrip()) - len("def")
                current_func["nesting_depth"] = max(current_func["nesting_depth"], nesting // 4)

                if re.search(r"\bif\b|\belif\b|\band\b|\bor\b|\bcase\b", stripped):
                    current_func["complexity"] += 1
                if re.search(r"\bfor\b|\bwhile\b", stripped):
                    current_func["complexity"] += 1
                if re.search(r"\bexcept\b", stripped):
                    current_func["complexity"] += 1
                if re.search(r"\breturn\b.*\band\b|\breturn\b.*\bor\b", stripped):
                    current_func["complexity"] += 1

            if class_match and not func_match:
                classes.append(
                    {
                        "name": class_match.group(1),
                        "line": i,
                    }
                )

        if current_func:
            functions.append(current_func)

        total_complexity = sum(f["complexity"] for f in functions) if functions else 1
        avg_complexity = total_complexity / len(functions) if functions else 0
        max_complexity = max((f["complexity"] for f in functions), default=0)

        cognitive = self._calc_cognitive_complexity(code)

        loc_per_function = code_lines / len(functions) if functions else code_lines
        duplication = self._calc_duplication(code)

        mi_raw = (
            171
            - 5.2 * math.log(max(1, code_lines))
            - 0.23 * total_complexity
            - 16.2 * math.log(max(1, duplication))
            if code_lines > 0
            else 100
        )
        maintainability_index = max(0, min(100, mi_raw))

        return {
            "total_lines": total_lines,
            "code_lines": code_lines,
            "blank_lines": blank_lines,
            "comment_lines": comment_lines,
            "comment_ratio": round(comment_lines / max(1, code_lines) * 100, 1),
            "functions": len(functions),
            "classes": len(classes),
            "cyclomatic_complexity": total_complexity,
            "avg_cyclomatic_complexity": round(avg_complexity, 2),
            "max_cyclomatic_complexity": max_complexity,
            "cognitive_complexity": cognitive,
            "maintainability_index": round(maintainability_index, 1),
            "loc_per_function": round(loc_per_function, 1),
            "duplication_ratio": round(duplication, 2),
            "avg_function_length": round(loc_per_function, 1),
            "function_details": [
                {
                    "name": f["name"],
                    "complexity": f["complexity"],
                    "length": f["end"] - f["start"] + 1,
                }
                for f in functions
            ],
        }

    def smells(self, code: str) -> list[dict[str, Any]]:
        self._smell_counter = 0
        smells = []
        lines = code.split("\n")

        func_starts: list[dict[str, Any]] = []
        current_func: dict[str, Any] | None = None
        for i, line in enumerate(lines):
            func_match = re.match(r"\s*def\s+(\w+)\s*\(", line)
            if func_match:
                if current_func:
                    func_starts.append(current_func)
                current_func = {"name": func_match.group(1), "start": i, "end": i}
            if current_func:
                current_func["end"] = i
        if current_func:
            func_starts.append(current_func)

        for func in func_starts:
            length = func["end"] - func["start"] + 1
            if length > 50:
                self._smell_counter += 1
                smells.append(
                    {
                        "id": f"SMELL-{self._smell_counter:04d}",
                        "type": "long_method",
                        "location": f"function '{func['name']}' (line {func['start'] + 1})",
                        "severity": "high" if length > 100 else "medium",
                        "description": f"Function '{func['name']}' is {length} lines long",
                        "recommendation": "Break into smaller functions; aim for <30 lines per function",
                    }
                )

        class_starts: list[dict[str, Any]] = []
        current_class: dict[str, Any] | None = None
        for i, line in enumerate(lines):
            class_match = re.match(r"\s*class\s+(\w+)", line)
            if class_match:
                if current_class:
                    class_starts.append(current_class)
                current_class = {"name": class_match.group(1), "start": i, "end": i, "methods": 0}
            if current_class:
                current_class["end"] = i
                if re.match(r"\s*def\s+", line):
                    current_class["methods"] += 1
        if current_class:
            class_starts.append(current_class)

        for cls in class_starts:
            cls_length = cls["end"] - cls["start"] + 1
            if cls_length > 200 or cls["methods"] > 15:
                self._smell_counter += 1
                smells.append(
                    {
                        "id": f"SMELL-{self._smell_counter:04d}",
                        "type": "god_class",
                        "location": f"class '{cls['name']}' (line {cls['start'] + 1})",
                        "severity": "high",
                        "description": f"Class '{cls['name']}' is {cls_length} lines with {cls['methods']} methods",
                        "recommendation": "Split into smaller, focused classes following SRP",
                    }
                )

        nesting_max = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (
                stripped.startswith("if ")
                or stripped.startswith("for ")
                or stripped.startswith("while ")
                or stripped.startswith("with ")
            ):
                indent = len(line) - len(line.lstrip())
                depth = indent // 4
                if depth > 3:
                    self._smell_counter += 1
                    smells.append(
                        {
                            "id": f"SMELL-{self._smell_counter:04d}",
                            "type": "deep_nesting",
                            "location": f"line {i + 1}",
                            "severity": "high" if depth > 4 else "medium",
                            "description": f"Nesting depth of {depth} at line {i + 1}",
                            "recommendation": "Use early returns, extract conditions, or restructure logic",
                        }
                    )
                nesting_max = max(nesting_max, depth)

        magic_numbers = re.findall(r"(?<![\w.])(\d{2,})(?![\w.])", code)
        seen_magics = set()
        for num_str in magic_numbers:
            num = int(num_str)
            if num not in (0, 1, 2, 10, 100) and num not in seen_magics:
                seen_magics.add(num)
                self._smell_counter += 1
                smells.append(
                    {
                        "id": f"SMELL-{self._smell_counter:04d}",
                        "type": "magic_number",
                        "location": "codebase",
                        "severity": "low",
                        "description": f"Magic number {num} found",
                        "recommendation": f"Extract {num} into a named constant",
                    }
                )

        for i, line in enumerate(lines):
            stripped = line.strip()
            if (
                (re.match(r"^pass\s*$", stripped) or re.match(r"^\.\.\.\s*$", stripped))
                and i > 0
                and re.match(
                    r"\s*(def|class|if|else|elif|try|except|finally|for|while)\s",
                    lines[i - 1] if i > 0 else "",
                )
            ):
                self._smell_counter += 1
                smells.append(
                    {
                        "id": f"SMELL-{self._smell_counter:04d}",
                        "type": "dead_code",
                        "location": f"line {i + 1}",
                        "severity": "low",
                        "description": f"Empty block: '{stripped}' at line {i + 1}",
                        "recommendation": "Implement or remove dead code blocks",
                    }
                )

        dup_lines = self._find_duplicate_lines(code)
        if len(dup_lines) > 3:
            self._smell_counter += 1
            smells.append(
                {
                    "id": f"SMELL-{self._smell_counter:04d}",
                    "type": "duplicate_code",
                    "location": "codebase",
                    "severity": "medium",
                    "description": f"{len(dup_lines)} groups of duplicate code detected",
                    "recommendation": "Extract duplicated logic into shared functions or utilities",
                }
            )

        for i, line in enumerate(lines):
            if "except:" in line.strip() or "except Exception" in line.strip():
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if next_line == "pass" or next_line.startswith("#"):
                    self._smell_counter += 1
                    smells.append(
                        {
                            "id": f"SMELL-{self._smell_counter:04d}",
                            "type": "swallowed_exception",
                            "location": f"line {i + 1}",
                            "severity": "medium",
                            "description": "Exception caught and silently ignored",
                            "recommendation": "Log the exception or handle it appropriately",
                        }
                    )

        return smells

    def grade(self, metrics: dict[str, Any]) -> QualityReport:
        score = 100

        cc = metrics.get("max_cyclomatic_complexity", 0)
        if cc > 25:
            score -= 25
        elif cc > 15:
            score -= 15
        elif cc > 10:
            score -= 8
        elif cc > 5:
            score -= 3

        mi = metrics.get("maintainability_index", 100)
        if mi < 20:
            score -= 25
        elif mi < 40:
            score -= 15
        elif mi < 60:
            score -= 8

        cog = metrics.get("cognitive_complexity", 0)
        if cog > 50:
            score -= 20
        elif cog > 25:
            score -= 10
        elif cog > 15:
            score -= 5

        dup = metrics.get("duplication_ratio", 0)
        if dup > 20:
            score -= 20
        elif dup > 10:
            score -= 10
        elif dup > 5:
            score -= 5

        comment_ratio = metrics.get("comment_ratio", 0)
        if comment_ratio < 5 or comment_ratio > 50:
            score -= 5

        score = max(0, min(100, score))

        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        elif score >= 40:
            grade = "E"
        else:
            grade = "F"

        recommendations = []
        if cc > 10:
            recommendations.append("Reduce cyclomatic complexity by simplifying conditional logic")
        if mi < 50:
            recommendations.append(
                "Improve maintainability by reducing code complexity and adding documentation"
            )
        if cog > 20:
            recommendations.append("Reduce cognitive complexity by flattening nested conditions")
        if dup > 10:
            recommendations.append("Extract duplicate code into shared functions")
        if comment_ratio < 10:
            recommendations.append("Add more documentation and comments")
        if metrics.get("avg_function_length", 0) > 30:
            recommendations.append("Break long functions into smaller, focused ones")

        if not recommendations:
            recommendations.append("Code quality is good. Keep up the practices!")

        return QualityReport(
            grade=grade,
            metrics={"score": score},
            recommendations=recommendations,
        )

    def _calc_cognitive_complexity(self, code: str) -> int:
        complexity = 0
        lines = code.split("\n")

        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            current_nesting = indent // 4

            if re.search(r"\bif\b|\belse\s+if\b|\belif\b", stripped) or re.search(
                r"\bfor\b|\bwhile\b", stripped
            ):
                complexity += 1 + current_nesting
            elif re.search(r"\bexcept\b", stripped):
                complexity += 1
            elif re.search(r"\belse\b", stripped):
                if re.search(r"\belse\b\s*:", stripped):
                    complexity += 1
            elif re.search(r"\band\b|\bor\b", stripped):
                complexity += stripped.count(" and ") + stripped.count(" or ")
            elif re.search(r"\bbreak\b|\bcontinue\b", stripped):
                complexity += 1

        return complexity

    def _calc_duplication(self, code: str) -> float:
        lines = [
            line.strip()
            for line in code.split("\n")
            if line.strip()
            and not line.strip().startswith("#")
            and not line.strip().startswith("//")
        ]
        if len(lines) < 3:
            return 0.0

        seen: dict[str, int] = {}
        duplicate_count = 0
        window = 3

        for i in range(len(lines) - window + 1):
            block = "\n".join(lines[i : i + window])
            if block in seen:
                duplicate_count += window
            else:
                seen[block] = i

        return (duplicate_count / max(1, len(lines))) * 100

    def _find_duplicate_lines(self, code: str) -> list[list[str]]:
        lines = [
            line.strip() for line in code.split("\n") if line.strip() and len(line.strip()) > 10
        ]
        line_counts: dict[str, int] = {}
        for line in lines:
            line_counts[line] = line_counts.get(line, 0) + 1
        return [[line] * count for line, count in line_counts.items() if count > 1]


class QualityModule:
    NAME = "quality"
    DESCRIPTION = "Code quality analysis: complexity, code smells, metrics, grading"

    def __init__(self) -> None:
        self.analyzer = CodeAnalyzer()

    def execute(self, task: Any) -> dict[str, Any]:
        task_title = getattr(task, "title", "")
        task_tags = getattr(task, "tags", [])

        code = ""
        language = "python"
        if hasattr(task, "content"):
            code = task.content
        if hasattr(task, "metadata"):
            language = task.metadata.get("language", language)

        action = "full"
        if "complexity" in str(task_title).lower() or "complexity" in task_tags:
            action = "complexity"
        elif "smell" in str(task_title).lower() or "smells" in task_tags:
            action = "smells"
        elif "grade" in str(task_title).lower() or "score" in task_tags:
            action = "grade"

        if not code:
            return {
                "action": action,
                "message": "No code provided for analysis",
                "suggestion": "Provide code in task.content",
                "_confidence": 0.30,
            }

        if action == "complexity":
            metrics = self.analyzer.complexity(code)
            return {
                "action": "complexity",
                "metrics": metrics,
                "_confidence": 0.85,
            }
        elif action == "smells":
            smells = self.analyzer.smells(code)
            return {
                "action": "smells",
                "smells": smells,
                "count": len(smells),
                "_confidence": 0.80,
            }
        else:
            report = self.analyzer.analyze(code, language)
            return {
                "action": "full_analysis",
                "grade": report.grade,
                "metrics": report.metrics,
                "smells": report.smells,
                "recommendations": report.recommendations,
                "_confidence": 0.85,
            }
