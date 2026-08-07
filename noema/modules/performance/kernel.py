"""Performance analysis module — profiling suggestions, benchmarking plans, load testing configs, bottleneck detection."""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Bottleneck:
    location: str
    type: str  # cpu, memory, io, network
    severity: str  # critical, high, medium, low
    suggestion: str


@dataclass
class LoadTestConfig:
    target: str = ""
    concurrent_users: int = 100
    duration_seconds: int = 60
    ramp_up: int = 10
    endpoints: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ProfileSuggestion:
    tool: str
    reason: str
    command: str
    output_format: str


class Profiler:
    def __init__(self) -> None:
        self._bottleneck_counter = 0

    def analyze_code_performance(self, code: str) -> dict[str, Any]:
        issues = []
        suggestions = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            if re.search(r"for\s+\w+\s+in\s+range\s*\(\s*\d{4,}", stripped):
                issues.append(
                    Bottleneck(
                        location=f"line {i}",
                        type="cpu",
                        severity="medium",
                        suggestion="Large range loop detected. Consider generator expressions or numpy vectorization.",
                    )
                )

            if (
                re.search(r"\.append\s*\(", stripped)
                and re.search(r"for\s+", stripped)
                and "list comprehension" not in code.lower()
            ):
                issues.append(
                    Bottleneck(
                        location=f"line {i}",
                        type="cpu",
                        severity="low",
                        suggestion="Consider using list comprehension instead of loop with .append()",
                    )
                )

            if re.search(r'\+\s*=\s*["\']', stripped) or re.search(
                r"\.join.*for.*in.*range", stripped
            ):
                issues.append(
                    Bottleneck(
                        location=f"line {i}",
                        type="cpu",
                        severity="medium",
                        suggestion="String concatenation in loop is O(n^2). Use str.join() or f-strings.",
                    )
                )

            if re.search(r"open\s*\(", stripped) and not re.search(r"with\s+open", stripped):
                issues.append(
                    Bottleneck(
                        location=f"line {i}",
                        type="io",
                        severity="medium",
                        suggestion="File opened without context manager. Risk of resource leak and unflushed writes.",
                    )
                )

            if re.search(r"requests\.(get|post)\s*\(", stripped):
                issues.append(
                    Bottleneck(
                        location=f"line {i}",
                        type="network",
                        severity="low",
                        suggestion="Synchronous HTTP call. Consider async (aiohttp/httpx) for concurrent requests.",
                    )
                )

            if re.search(r"time\.sleep\s*\(", stripped):
                issues.append(
                    Bottleneck(
                        location=f"line {i}",
                        type="cpu",
                        severity="medium",
                        suggestion="Blocking sleep detected. Consider async alternatives.",
                    )
                )

            if re.search(r"json\.loads?\s*\(", stripped) or re.search(
                r"json\.dumps?\s*\(", stripped
            ):
                issues.append(
                    Bottleneck(
                        location=f"line {i}",
                        type="cpu",
                        severity="low",
                        suggestion="For high-throughput JSON, consider orjson or ujson.",
                    )
                )

            if re.search(r"SELECT\s+\*\s+FROM", stripped, re.IGNORECASE):
                issues.append(
                    Bottleneck(
                        location=f"line {i}",
                        type="io",
                        severity="high",
                        suggestion="SELECT * is inefficient. Select only needed columns.",
                    )
                )

            if re.search(
                r"\.execute\s*\(.*SELECT.*WHERE", stripped, re.IGNORECASE
            ) and not re.search(r"LIMIT|FETCH FIRST", stripped, re.IGNORECASE):
                issues.append(
                    Bottleneck(
                        location=f"line {i}",
                        type="io",
                        severity="medium",
                        suggestion="Query without LIMIT may return unbounded results.",
                    )
                )

            if re.search(r"def\s+\w+.*:\s*$", stripped) and i < len(lines):
                next_line = lines[i].strip() if i < len(lines) else ""
                func_name = re.search(r"def\s+(\w+)", stripped)
                if func_name and re.search(r"global\s+", next_line):
                    issues.append(
                        Bottleneck(
                            location=f"line {i}",
                            type="cpu",
                            severity="medium",
                            suggestion=f"Function {func_name.group(1)} uses global state. Consider parameterization.",
                        )
                    )

        if any(b.type == "cpu" for b in issues):
            suggestions.append(
                ProfileSuggestion(
                    tool="cProfile",
                    reason="CPU-bound issues detected",
                    command="python -m cProfile -s cumtime your_script.py",
                    output_format="text",
                )
            )
            suggestions.append(
                ProfileSuggestion(
                    tool="line_profiler",
                    reason="Line-level CPU profiling recommended",
                    command="@profile decorator + kernprof -l -v your_script.py",
                    output_format="text",
                )
            )

        if any(b.type == "memory" for b in issues):
            suggestions.append(
                ProfileSuggestion(
                    tool="memory_profiler",
                    reason="Memory-related issues detected",
                    command="python -m memory_profiler your_script.py",
                    output_format="text",
                )
            )

        if any(b.type == "io" for b in issues):
            suggestions.append(
                ProfileSuggestion(
                    tool="py-spy",
                    reason="I/O blocking detected — sampling profiler recommended",
                    command="py-spy top --pid <PID>",
                    output_format="text",
                )
            )

        total_lines = len(lines)
        complexity_score = min(100, max(0, 100 - len(issues) * 8))

        return {
            "bottlenecks": [
                {
                    "location": b.location,
                    "type": b.type,
                    "severity": b.severity,
                    "suggestion": b.suggestion,
                }
                for b in issues
            ],
            "profiling_suggestions": [
                {"tool": s.tool, "reason": s.reason, "command": s.command} for s in suggestions
            ],
            "total_lines": total_lines,
            "issues_found": len(issues),
            "performance_score": complexity_score,
        }

    def suggest_optimizations(self, metrics: dict[str, Any]) -> list[dict[str, str]]:
        optimizations = []
        cpu_time = metrics.get("cpu_time", 0)
        memory_mb = metrics.get("memory_mb", 0)
        io_wait = metrics.get("io_wait", 0)
        response_time_ms = metrics.get("response_time_ms", 0)
        throughput = metrics.get("throughput_rps", 0)

        if cpu_time > 80:
            optimizations.append(
                {
                    "area": "cpu",
                    "current": f"{cpu_time}%",
                    "suggestion": "High CPU usage. Profile with cProfile, consider Cython or multiprocessing.",
                    "priority": "high",
                }
            )
        elif cpu_time > 50:
            optimizations.append(
                {
                    "area": "cpu",
                    "current": f"{cpu_time}%",
                    "suggestion": "Moderate CPU usage. Review hot loops and optimize algorithmic complexity.",
                    "priority": "medium",
                }
            )

        if memory_mb > 1024:
            optimizations.append(
                {
                    "area": "memory",
                    "current": f"{memory_mb}MB",
                    "suggestion": "High memory usage. Check for leaks, use generators, profile with tracemalloc.",
                    "priority": "high",
                }
            )
        elif memory_mb > 512:
            optimizations.append(
                {
                    "area": "memory",
                    "current": f"{memory_mb}MB",
                    "suggestion": "Moderate memory usage. Consider streaming processing or pagination.",
                    "priority": "medium",
                }
            )

        if io_wait > 30:
            optimizations.append(
                {
                    "area": "io",
                    "current": f"{io_wait}%",
                    "suggestion": "High I/O wait. Use async I/O, connection pooling, or caching.",
                    "priority": "high",
                }
            )

        if response_time_ms > 1000:
            optimizations.append(
                {
                    "area": "latency",
                    "current": f"{response_time_ms}ms",
                    "suggestion": "Slow response. Add caching (Redis), optimize DB queries, use CDN.",
                    "priority": "high",
                }
            )
        elif response_time_ms > 200:
            optimizations.append(
                {
                    "area": "latency",
                    "current": f"{response_time_ms}ms",
                    "suggestion": "Acceptable but improvable latency. Consider response compression and caching.",
                    "priority": "medium",
                }
            )

        if throughput > 0 and response_time_ms > 0:
            theoretical_max = 1000 / response_time_ms * 1000
            if throughput < theoretical_max * 0.5:
                optimizations.append(
                    {
                        "area": "throughput",
                        "current": f"{throughput} rps",
                        "suggestion": f"Theoretical max ~{theoretical_max:.0f} rps. Optimize bottlenecks or scale horizontally.",
                        "priority": "medium",
                    }
                )

        return optimizations

    def generate_benchmark_plan(self, task: dict[str, Any]) -> dict[str, Any]:
        task_type = task.get("type", "api")
        endpoints = task.get("endpoints", ["/"])
        target = task.get("target", "localhost:8000")

        def _ep_path(ep: Any) -> str:
            if isinstance(ep, dict):
                return str(ep.get("path") or ep.get("url") or "/")
            return str(ep) if ep else "/"

        phases = []
        for ep in endpoints:
            path = _ep_path(ep)
            ep_label = path.replace("/", "_").strip("_") or "root"
            phases.append(
                {
                    "name": f"baseline_{ep_label}",
                    "target": f"{target}{path}",
                    "method": "GET",
                    "concurrent_users": 10,
                    "duration_seconds": 30,
                    "ramp_up": 5,
                }
            )
            phases.append(
                {
                    "name": f"load_{ep_label}",
                    "target": f"{target}{path}",
                    "method": "GET",
                    "concurrent_users": task.get("concurrent_users", 100),
                    "duration_seconds": task.get("duration_seconds", 120),
                    "ramp_up": task.get("ramp_up", 30),
                }
            )
            phases.append(
                {
                    "name": f"stress_{ep_label}",
                    "target": f"{target}{path}",
                    "method": "GET",
                    "concurrent_users": task.get("concurrent_users", 100) * 5,
                    "duration_seconds": 60,
                    "ramp_up": 10,
                }
            )

        thresholds = {
            "response_time_p95_ms": task.get("p95_threshold", 500),
            "error_rate_percent": task.get("error_threshold", 1.0),
            "throughput_rps": task.get("min_throughput", 50),
        }

        return {
            "plan_name": f"benchmark_{task_type}",
            "phases": phases,
            "thresholds": thresholds,
            "tools": ["locust", "k6", "wrk", "vegeta"],
            "reports": [
                "latency_distribution",
                "throughput_over_time",
                "error_analysis",
                "resource_utilization",
            ],
        }


class PerformanceModule:
    NAME = "performance"
    DESCRIPTION = "Performance profiling suggestions, benchmarking plans, load testing configs, bottleneck detection"

    def __init__(self) -> None:
        self.profiler: Profiler = Profiler()

    def execute(self, task: Any) -> dict[str, Any]:
        task_title = getattr(task, "title", "")
        task_tags = getattr(task, "tags", [])

        code = ""
        if hasattr(task, "content"):
            code = task.content

        action = "analyze"
        if "benchmark" in str(task_title).lower() or "benchmark" in task_tags:
            action = "benchmark"
        elif "optimize" in str(task_title).lower() or "optimization" in task_tags:
            action = "optimize"
        elif "load" in str(task_title).lower() or "loadtest" in task_tags:
            action = "benchmark"

        if action == "analyze" and code:
            result: dict[str, Any] = self.profiler.analyze_code_performance(code)
            result["_confidence"] = 0.80
            return result
        elif action == "optimize":
            metrics: dict[str, Any] = {}
            if hasattr(task, "metadata"):
                metrics = getattr(task, "metadata", {})
            optimizations = self.profiler.suggest_optimizations(metrics)
            return {
                "action": "optimize",
                "optimizations": optimizations,
                "_confidence": 0.75,
            }
        elif action == "benchmark":
            task_dict: dict[str, Any] = {}
            if hasattr(task, "metadata"):
                task_dict = getattr(task, "metadata", {})
            plan = self.profiler.generate_benchmark_plan(task_dict)
            return {
                "action": "benchmark",
                "plan": plan,
                "_confidence": 0.80,
            }

        return {
            "action": "analyze",
            "message": "No code or specific task provided for performance analysis",
            "suggestion": "Provide code in task.content or specify action in task title/tags",
            "_confidence": 0.30,
        }
