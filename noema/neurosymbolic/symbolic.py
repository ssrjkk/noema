"""Symbolic engine — Z3-backed constraint parsing and solution verification.

The engine maintains a fixed-size solver pool (:attr:`_solver_pool`) so that
concurrent verification rounds never exhaust process resources. Every
checked-out solver is ``reset()`` before use and returned to the pool in a
``finally`` block; every ``push()`` is paired with ``pop()`` in ``finally`` so
a failed check can never leave a solver polluted.

All public methods are async; the underlying Z3 calls are CPU-bound but
serialized through the bounded pool.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import structlog

logger = structlog.get_logger(__name__)

#: Domain-separation prefix for the task source hash (stable across processes,
#: unlike Python's randomized ``hash()``).
_SOURCE_HASH_DOMAIN: bytes = b"noema.symbolic.v1"


@dataclass
class Constraint:
    name: str
    expression: Any
    description: str
    priority: int = 1
    variable: Any = None


@dataclass
class TaskGraph:
    requirements: list[Constraint] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _sanitize_var_name(name: Any, fallback: str) -> str:
    """Coerce an arbitrary label into a valid Z3 variable identifier."""
    text = str(name).strip()
    if not text:
        return fallback
    safe = re.sub(r"[^0-9a-zA-Z_]", "_", text)
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe or safe[0].isdigit():
        safe = f"v_{safe}"
    return safe or fallback


def _to_num(value: Any) -> float | None:
    """Coerce an untrusted numeric-ish value to ``float`` or return ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _coerce_int(value: Any, default: int) -> int:
    """Coerce an untrusted ``priority``-style value to an int, falling back."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _candidate_name(req: Any, index: int) -> str:
    """Pick a raw variable-name candidate for a requirement, else a fallback."""
    if not isinstance(req, dict):
        if hasattr(req, "model_dump"):
            req = req.model_dump()
        else:
            return f"req_{index}"
    return str(req.get("name") or req.get("category") or "")


def _render_bounds(name: str, lower: float | None, upper: float | None) -> str:
    """Human-readable description of an extracted ``(min, max)`` range."""
    if lower is not None and upper is not None:
        return f"{name} in [{lower:g}, {upper:g}]"
    if lower is not None:
        return f"{name} >= {lower:g}"
    if upper is not None:
        return f"{name} <= {upper:g}"
    return ""


def _unique_names(names: list[str]) -> list[str]:
    """Make every name unique by appending ``_2``, ``_3`` … to collisions."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            out.append(name)
            continue
        seen[name] += 1
        candidate = f"{name}_{seen[name]}"
        while candidate in seen:
            seen[name] += 1
            candidate = f"{name}_{seen[name]}"
        seen[candidate] = 0
        out.append(candidate)
    return out


#: Fields of a requirement dict that may carry natural-language bound text.
_BOUND_TEXT_KEYS: tuple[str, ...] = (
    "constraints",
    "description",
    "requirement",
    "text",
    "condition",
    "rule",
    "statement",
)

#: Natural-language bound patterns. Each maps a phrase to a bound kind
#: (``lower``, ``upper`` or ``exact``); group 1 is the number and group 2 the
#: optional unit suffix (ms/us are scaled to seconds, other units are 1:1).
#: Phrases are ordered so narrower phrasings never shadow generic ones — all
#: matches are applied and the range is narrowed by each.
_BOUND_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"no\s+less\s+than\s+([-+]?\d+(?:\.\d+)?)\s*([a-zµ]*)", "lower"),
    (r"no\s+more\s+than\s+([-+]?\d+(?:\.\d+)?)\s*([a-zµ]*)", "upper"),
    (r"must\s+stay\s+(?:below|under)\s+([-+]?\d+(?:\.\d+)?)\s*([a-zµ]*)", "upper"),
    (r"must\s+stay\s+(?:above|over)\s+([-+]?\d+(?:\.\d+)?)\s*([a-zµ]*)", "lower"),
    (r"must\s+not\s+exceed\s+([-+]?\d+(?:\.\d+)?)\s*([a-zµ]*)", "upper"),
    (r"must\s+not\s+go\s+above\s+([-+]?\d+(?:\.\d+)?)\s*([a-zµ]*)", "upper"),
    (r"greater\s+than\s+or\s+equal\s+to\s+([-+]?\d+(?:\.\d+)?)\s*([a-zµ]*)", "lower"),
    (r"less\s+than\s+or\s+equal\s+to\s+([-+]?\d+(?:\.\d+)?)\s*([a-zµ]*)", "upper"),
    (r"at\s+least\s+([-+]?\d+(?:\.\d+)?)\s*([a-zµ]*)", "lower"),
    (r"at\s+most\s+([-+]?\d+(?:\.\d+)?)\s*([a-zµ]*)", "upper"),
    (r"minimum(?:ly)?\s+([-+]?\d+(?:\.\d+)?)\s*([a-zµ]*)", "lower"),
    (r"maximum(?:ly)?\s+([-+]?\d+(?:\.\d+)?)\s*([a-zµ]*)", "upper"),
    (r"up\s+to\s+([-+]?\d+(?:\.\d+)?)\s*([a-zµ]*)", "upper"),
    (r"exactly\s+([-+]?\d+(?:\.\d+)?)\s*([a-zµ]*)", "exact"),
    (r"\bexceeds?\s+([-+]?\d+(?:\.\d+)?)\s*([a-zµ]*)", "upper"),
    (r"\bbelow\s+([-+]?\d+(?:\.\d+)?)\s*([a-zµ]*)", "upper"),
    (r"\babove\s+([-+]?\d+(?:\.\d+)?)\s*([a-zµ]*)", "lower"),
)


def _unit_scale(suffix: str) -> float:
    """Scale factor for a captured unit suffix; ms/us become seconds."""
    unit = suffix.strip().lower()
    if unit in {"ms", "millisecond", "milliseconds"}:
        return 0.001
    if unit in {"us", "µs", "microsecond", "microseconds"}:
        return 1e-6
    return 1.0


def _bound_text_parts(req: dict[str, Any]) -> list[str]:
    """All natural-language text snippets of a requirement dict."""
    parts: list[str] = []
    for key in _BOUND_TEXT_KEYS:
        value = req.get(key)
        if isinstance(value, str):
            if value:
                parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
    return parts


def _extract_bounds(req: dict[str, Any]) -> tuple[float | None, float | None]:
    """Extract numeric ``(min, max)`` bounds from a requirement.

    Priority: explicit ``min``/``max`` fields, then natural-language-ish
    entries (``constraints``, ``description``, ``requirement``, …) such as
    ``"x >= 0.5"``, ``"x in [1, 10]"``, ``"min=3 max=9"``, ``"between 2 and 5"``,
    ``"at least N"``, ``"at most N"``, ``"must stay below N"``,
    ``"must not exceed N"``, ``"exactly N"``, percentages and time units
    (``ms``/``us`` scale to seconds). Repeated matches narrow the range
    (``lower`` = max of all lower bounds, ``upper`` = min of all upper bounds)
    so a solution can never satisfy *wider* bounds than the task declared.
    """
    lower = _to_num(req.get("min"))
    upper = _to_num(req.get("max"))

    for text in _bound_text_parts(req):
        lowered = text.lower()
        in_match = re.search(r"in\s*[\[\(]\s*([-+]?[\d.]+)\s*,\s*([-+]?[\d.]+)\s*[\]\)]", lowered)
        if in_match:
            lo, hi = float(in_match.group(1)), float(in_match.group(2))
            lower = lo if lower is None else max(lower, lo)
            upper = hi if upper is None else min(upper, hi)
        between_match = re.search(r"between\s+([-+]?[\d.]+)\s+and\s+([-+]?[\d.]+)", lowered)
        if between_match:
            lo, hi = float(between_match.group(1)), float(between_match.group(2))
            lower = lo if lower is None else max(lower, lo)
            upper = hi if upper is None else min(upper, hi)
        ge_match = re.search(r">=\s*([-+]?[\d.]+)", lowered)
        if ge_match:
            lo = float(ge_match.group(1))
            lower = lo if lower is None else max(lower, lo)
        le_match = re.search(r"<=\s*([-+]?[\d.]+)", lowered)
        if le_match:
            hi = float(le_match.group(1))
            upper = hi if upper is None else min(upper, hi)
        min_match = re.search(r"\bmin\s*[:=]?\s*([-+]?[\d.]+)", lowered)
        if min_match:
            lo = float(min_match.group(1))
            lower = lo if lower is None else max(lower, lo)
        max_match = re.search(r"\bmax\s*[:=]?\s*([-+]?[\d.]+)", lowered)
        if max_match:
            hi = float(max_match.group(1))
            upper = hi if upper is None else min(upper, hi)
        for pattern, kind in _BOUND_PATTERNS:
            match = re.search(pattern, lowered)
            if not match:
                continue
            value = float(match.group(1)) * _unit_scale(match.group(2))
            if kind == "lower":
                lower = value if lower is None else max(lower, value)
            elif kind == "upper":
                upper = value if upper is None else min(upper, value)
            else:  # exact
                lower = value if lower is None else max(lower, value)
                upper = value if upper is None else min(upper, value)

    return lower, upper


class SymbolicVerificationError(Exception):
    def __init__(self, message: str, violations: list[str]) -> None:
        super().__init__(message)
        self.violations = violations


def _coerce_value(var: Any, value: Any) -> Any:
    """Coerce an untrusted candidate value to a Z3-compatible literal.

    Returns ``None`` when the value cannot be represented for the variable's
    sort — callers must fail closed in that case instead of passing the raw
    value into ``z3`` (which would raise or mis-compare).
    """
    from z3 import ArithRef, BoolRef

    if isinstance(var, BoolRef):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"true", "1", "yes", "on"}:
                return True
            if low in {"false", "0", "no", "off"}:
                return False
        return None

    if isinstance(var, ArithRef):
        num = _to_num(value)
        if num is None:
            return None
        if var.is_int():
            return int(num) if num.is_integer() else None
        return num

    return None


class SymbolicEngine:
    def __init__(self, verification_timeout: float = 5.0, max_constraints: int = 1000) -> None:
        self.verification_timeout = verification_timeout
        self.max_constraints = max_constraints
        self._solver_pool: asyncio.Queue[Any] = asyncio.Queue(maxsize=10)
        self._initialized = False
        self._degraded = False

        logger.info(
            "symbolic_engine_initialized",
            timeout=verification_timeout,
            max_constraints=max_constraints,
        )

    async def initialize(self) -> None:
        if self._initialized:
            return
        try:
            from z3 import Solver
        except ImportError:
            logger.warning("z3 not installed; symbolic engine running in degraded mode")
            self._degraded = True
            self._initialized = True
            return

        for _ in range(10):
            await self._solver_pool.put(Solver())

        self._degraded = False
        self._initialized = True
        logger.info("symbolic_engine_pool_ready", pool_size=10)

    @asynccontextmanager
    async def _get_solver(self) -> AsyncIterator[Any]:
        solver = None
        try:
            solver = await asyncio.wait_for(self._solver_pool.get(), timeout=1.0)
            solver.reset()
            yield solver
        except TimeoutError:
            logger.error("solver_pool_exhausted")
            raise SymbolicVerificationError(
                "Solver pool exhausted", ["resource_exhaustion"]
            ) from None
        finally:
            if solver is not None:
                await self._solver_pool.put(solver)

    async def parse_task(self, structured_task: dict) -> TaskGraph:
        """Parse a structured task into a symbolic :class:`TaskGraph`.

        Args:
            structured_task: A dict with ``requirements`` (required list),
                ``constraints``, ``goals`` and ``variables`` keys.

        Returns:
            A :class:`TaskGraph` populated with parsed constraints.

        Raises:
            ValueError: If the task structure is invalid.
        """
        try:
            canonical_payload: bytes = json.dumps(
                structured_task, sort_keys=True, default=str
            ).encode("utf-8")
            graph = TaskGraph(
                metadata={
                    "parsed_at": time.monotonic(),
                    "source_hash": hashlib.sha256(
                        _SOURCE_HASH_DOMAIN + canonical_payload
                    ).hexdigest(),
                }
            )

            self._validate_task_structure(structured_task)

            reqs = structured_task.get("requirements", [])
            req_names = _unique_names(
                [_sanitize_var_name(_candidate_name(r, i), f"req_{i}") for i, r in enumerate(reqs)]
            )
            for req, name in zip(reqs, req_names, strict=False):
                if len(graph.requirements) >= self.max_constraints:
                    logger.warning(
                        "max_constraints_reached",
                        constraint_type="requirements",
                        limit=self.max_constraints,
                    )
                    break

                constraint = await self._parse_requirement(req, name)
                if constraint:
                    graph.requirements.append(constraint)
                    graph.variables[constraint.name] = constraint.variable

            consts = structured_task.get("constraints", [])
            const_names = _unique_names(
                [
                    _sanitize_var_name(_candidate_name(c, i), f"constraint_{i}")
                    for i, c in enumerate(consts)
                ]
            )
            for const, name in zip(consts, const_names, strict=False):
                if len(graph.constraints) >= self.max_constraints:
                    break

                constraint = await self._parse_constraint(const, name)
                if constraint:
                    graph.constraints.append(constraint)
                    graph.variables[constraint.name] = constraint.variable

            graph.goals = structured_task.get("goals", [])

            logger.info(
                "task_parsed",
                requirements=len(graph.requirements),
                constraints=len(graph.constraints),
                goals=len(graph.goals),
            )

            return graph

        except Exception as e:
            logger.error(
                "task_parsing_failed", error=str(e), task_keys=list(structured_task.keys())
            )
            raise

    async def verify_solution(
        self, solution: dict, task_graph: TaskGraph
    ) -> tuple[bool, list[str]]:
        """Verify a candidate solution against the symbolic task graph.

        Args:
            solution: Candidate solution mapping variable name to value.
            task_graph: Parsed symbolic task constraints.

        Returns:
            A tuple ``(is_valid, violations)`` where ``violations`` lists
            human-readable reasons (empty when the solution is valid).
        """
        if not task_graph:
            return False, ["task_graph_not_initialized"]

        # Fail closed: without Z3 there is no way to verify anything. Returning
        # True here would silently accept every candidate, so it must not happen.
        if self._degraded:
            return False, ["symbolic_engine_unavailable_z3_not_installed"]

        try:
            async with self._get_solver() as solver:
                for req in task_graph.requirements:
                    solver.add(req.expression)

                if not task_graph.requirements and not task_graph.constraints:
                    logger.warning(
                        "nothing_to_verify",
                        solution_keys=list(solution.keys()),
                    )
                    return True, []

                try:
                    return await asyncio.wait_for(
                        self._check_solution(solver, solution, task_graph),
                        timeout=self.verification_timeout,
                    )

                except TimeoutError:
                    logger.error(
                        "verification_timeout",
                        timeout=self.verification_timeout,
                        solution_keys=list(solution.keys()),
                    )
                    return False, ["verification_timeout"]

        except SymbolicVerificationError:
            raise
        except Exception as e:
            logger.error("verification_failed", error=str(e), error_type=type(e).__name__)
            return False, [f"verification_error: {str(e)}"]

    async def _check_solution(
        self, solver: Any, solution: dict, task_graph: TaskGraph
    ) -> tuple[bool, list[str]]:
        """Check a candidate solution, failing closed on any uncertainty."""
        from z3 import unsat

        # The accumulated requirements must be satisfiable on their own, or no
        # candidate value can ever be valid and we must not pretend otherwise.
        solver.push()
        try:
            if solver.check() == unsat:
                return False, ["constraints_unsatisfiable"]
        finally:
            solver.pop()

        checked = 0
        for name, value in solution.items():
            var = task_graph.variables.get(name)
            if var is None:
                continue

            coerced = _coerce_value(var, value)
            if coerced is None:
                return False, [f"requirement_{name}_uncheckable_value: value={value!r}"]

            solver.push()
            try:
                solver.add(var == coerced)
                if solver.check() == unsat:
                    return False, [f"requirement_{name}_violated: value={value}"]
            finally:
                solver.pop()
            checked += 1

        if checked == 0 and solution:
            logger.warning(
                "no_solution_variable_matched",
                solution_keys=list(solution.keys()),
                known_variables=list(task_graph.variables.keys()),
            )
            # Verification must not pass when nothing was actually checked.
            return False, ["no_solution_variable_matched"]

        return True, []

    def _validate_task_structure(self, task: dict[str, Any]) -> None:
        required_fields = ["requirements"]

        for req_field in required_fields:
            if req_field not in task:
                raise ValueError(f"Missing required field: {req_field}")

        if not isinstance(task["requirements"], list):
            raise ValueError("requirements must be a list")

    async def _parse_requirement(self, req: Any, name: str | None = None) -> Constraint | None:
        try:
            from z3 import And, Int, Real
        except ImportError:
            logger.warning("z3_not_installed; requirement skipped")
            return None

        try:
            if not isinstance(req, dict):
                if hasattr(req, "model_dump"):
                    req = req.model_dump()
                else:
                    raise TypeError(f"requirement must be a dict, got {type(req).__name__}")

            name = name or _sanitize_var_name(req.get("name") or req.get("category"), "req")
            req_type = req.get("type", "numeric")

            if req_type == "numeric":
                lower, upper = _extract_bounds(req)
                if lower is None and upper is None:
                    logger.warning(
                        "requirement_no_numeric_bounds; skipped",
                        requirement_name=name,
                    )
                    return None

                # Fractional bounds need a Real sort; Int would raise on them.
                use_real = any(v is not None and not v.is_integer() for v in (lower, upper))
                var = Real(name) if use_real else Int(name)
                clauses = []
                if lower is not None:
                    clauses.append(var >= lower)
                if upper is not None:
                    clauses.append(var <= upper)

                return Constraint(
                    name=name,
                    expression=And(*clauses),
                    description=str(req.get("description") or _render_bounds(name, lower, upper)),
                    priority=_coerce_int(req.get("priority"), 1),
                    variable=var,
                )

            logger.warning("requirement_type_unsupported", requirement_type=req_type, name=name)
            return None

        except Exception as e:
            logger.warning("requirement_parsing_failed", requirement=req, error=str(e))
            return None

    async def _parse_constraint(self, const: Any, name: str | None = None) -> Constraint | None:
        try:
            from z3 import Bool

            if not isinstance(const, dict):
                if hasattr(const, "model_dump"):
                    const = const.model_dump()
                else:
                    raise TypeError(f"constraint must be a dict, got {type(const).__name__}")

            name = name or _sanitize_var_name(
                const.get("name") or const.get("category"), "constraint"
            )
            var = Bool(name)

            return Constraint(
                name=name,
                expression=var,
                description=str(const.get("condition") or const.get("description") or ""),
                priority=_coerce_int(const.get("priority"), 1),
                variable=var,
            )
        except Exception as e:
            logger.warning("constraint_parsing_failed", constraint=const, error=str(e))
            return None
