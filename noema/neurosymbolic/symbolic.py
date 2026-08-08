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


def _z3_int(name: str) -> Any:
    try:
        from z3 import Int

        return Int(name)
    except ImportError:
        return None


def _z3_bool(name: str) -> Any:
    try:
        from z3 import Bool

        return Bool(name)
    except ImportError:
        return None


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _extract_bounds(req: dict[str, Any]) -> tuple[float | None, float | None]:
    """Extract numeric ``(min, max)`` bounds from a requirement.

    Priority: explicit ``min``/``max`` fields, then natural-language-ish
    ``constraints`` entries such as ``"x >= 0.5"``, ``"x in [1, 10]"``,
    ``"min=3 max=9"`` or ``"between 2 and 5"``.
    """
    raw_min = req.get("min")
    raw_max = req.get("max")
    lower = float(raw_min) if raw_min is not None and isinstance(raw_min, (int, float)) else None
    upper = float(raw_max) if raw_max is not None and isinstance(raw_max, (int, float)) else None

    for raw in req.get("constraints") or []:
        text = str(raw).lower()
        in_match = re.search(r"in\s*[\[\(]\s*([-+]?[\d.]+)\s*,\s*([-+]?[\d.]+)\s*[\]\)]", text)
        if in_match:
            lower = lower if lower is not None else float(in_match.group(1))
            upper = upper if upper is not None else float(in_match.group(2))
        between_match = re.search(r"between\s+([-+]?[\d.]+)\s+and\s+([-+]?[\d.]+)", text)
        if between_match:
            lower = lower if lower is not None else float(between_match.group(1))
            upper = upper if upper is not None else float(between_match.group(2))
        ge_match = re.search(r">=\s*([-+]?[\d.]+)", text)
        if ge_match:
            lower = lower if lower is not None else float(ge_match.group(1))
        le_match = re.search(r"<=\s*([-+]?[\d.]+)", text)
        if le_match:
            upper = upper if upper is not None else float(le_match.group(1))
        min_match = re.search(r"\bmin\s*[:=]?\s*([-+]?[\d.]+)", text)
        if min_match:
            lower = lower if lower is not None else float(min_match.group(1))
        max_match = re.search(r"\bmax\s*[:=]?\s*([-+]?[\d.]+)", text)
        if max_match:
            upper = upper if upper is not None else float(max_match.group(1))

    return lower, upper


class SymbolicVerificationError(Exception):
    def __init__(self, message: str, violations: list[str]) -> None:
        super().__init__(message)
        self.violations = violations


class SymbolicEngine:
    def __init__(self, verification_timeout: float = 5.0, max_constraints: int = 1000) -> None:
        self.verification_timeout = verification_timeout
        self.max_constraints = max_constraints
        self._solver_pool: asyncio.Queue[Any] = asyncio.Queue(maxsize=10)
        self._initialized = False

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
            self._initialized = True
            return

        for _ in range(10):
            await self._solver_pool.put(Solver())

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

            for req in structured_task.get("requirements", []):
                if len(graph.requirements) >= self.max_constraints:
                    logger.warning(
                        "max_constraints_reached",
                        constraint_type="requirements",
                        limit=self.max_constraints,
                    )
                    break

                constraint = await self._parse_requirement(req)
                if constraint:
                    graph.requirements.append(constraint)
                    graph.variables[constraint.name] = _z3_int(constraint.name)

            for const in structured_task.get("constraints", []):
                if len(graph.constraints) >= self.max_constraints:
                    break

                constraint = await self._parse_constraint(const)
                if constraint:
                    graph.constraints.append(constraint)
                    graph.variables[constraint.name] = _z3_bool(constraint.name)

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

        violations = []

        try:
            async with self._get_solver() as solver:
                for req in task_graph.requirements:
                    solver.add(req.expression)

                try:
                    is_valid = await asyncio.wait_for(
                        self._check_solution(solver, solution, task_graph),
                        timeout=self.verification_timeout,
                    )

                    if not is_valid:
                        violations = await self._extract_violations(solver, solution, task_graph)

                    return is_valid, violations

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

    async def _check_solution(self, solver: Any, solution: dict, task_graph: TaskGraph) -> bool:
        from z3 import unsat

        checked = 0
        for name, value in solution.items():
            var = task_graph.variables.get(name)
            if var is None:
                continue

            solver.push()
            try:
                solver.add(var == value)
                if solver.check() == unsat:
                    return False
            finally:
                solver.pop()
            checked += 1

        if checked == 0 and solution:
            logger.warning(
                "no_solution_variable_matched",
                solution_keys=list(solution.keys()),
                known_variables=list(task_graph.variables.keys()),
            )
        return True

    async def _extract_violations(
        self, solver: Any, solution: dict, task_graph: TaskGraph
    ) -> list[str]:
        """Enumerate which candidate values violate the task constraints.

        Each candidate variable is asserted in its own ``push``/``pop`` scope;
        the ``pop`` always runs (``finally``) so a failing check cannot corrupt
        the pooled solver.

        Args:
            solver: Checked-out Z3 solver (owned by the caller).
            solution: Candidate solution mapping variable name to value.
            task_graph: Parsed symbolic task constraints.

        Returns:
            A list of ``"requirement_<name>_violated: value=<value>"`` strings.
        """
        violations = []

        from z3 import unsat

        for name, value in solution.items():
            if name in task_graph.variables:
                var = task_graph.variables[name]

                solver.push()
                try:
                    solver.add(var == value)
                    if solver.check() == unsat:
                        violations.append(f"requirement_{name}_violated: value={value}")
                finally:
                    solver.pop()

        return violations

    def _validate_task_structure(self, task: dict[str, Any]) -> None:
        required_fields = ["requirements"]

        for req_field in required_fields:
            if req_field not in task:
                raise ValueError(f"Missing required field: {req_field}")

        if not isinstance(task["requirements"], list):
            raise ValueError("requirements must be a list")

    async def _parse_requirement(self, req: Any) -> Constraint | None:
        try:
            from z3 import And, Int
        except ImportError:
            logger.warning("z3_not_installed; requirement skipped")
            return None

        try:
            if not isinstance(req, dict):
                if hasattr(req, "model_dump"):
                    req = req.model_dump()
                else:
                    raise TypeError(f"requirement must be a dict, got {type(req).__name__}")

            name = _sanitize_var_name(req.get("name") or req.get("category"), "req")
            req_type = req.get("type", "numeric")

            if req_type == "numeric":
                lower, upper = _extract_bounds(req)
                if lower is None and upper is None:
                    logger.warning(
                        "requirement_no_numeric_bounds; skipped",
                        requirement_name=name,
                    )
                    return None

                var = Int(name)
                clauses = []
                if lower is not None:
                    clauses.append(var >= lower)
                if upper is not None:
                    clauses.append(var <= upper)

                return Constraint(
                    name=name,
                    expression=And(*clauses),
                    description=str(req.get("description") or ""),
                    priority=int(req.get("priority", 1)),
                )

            logger.warning("requirement_type_unsupported", requirement_type=req_type, name=name)
            return None

        except Exception as e:
            logger.warning("requirement_parsing_failed", requirement=req, error=str(e))
            return None

    async def _parse_constraint(self, const: Any) -> Constraint | None:
        try:
            from z3 import Bool

            if not isinstance(const, dict):
                if hasattr(const, "model_dump"):
                    const = const.model_dump()
                else:
                    raise TypeError(f"constraint must be a dict, got {type(const).__name__}")

            name = _sanitize_var_name(const.get("name") or const.get("category"), "constraint")
            var = Bool(name)

            return Constraint(
                name=name,
                expression=var,
                description=str(const.get("condition") or const.get("description") or ""),
                priority=int(const.get("priority", 1)),
            )
        except Exception as e:
            logger.warning("constraint_parsing_failed", constraint=const, error=str(e))
            return None
