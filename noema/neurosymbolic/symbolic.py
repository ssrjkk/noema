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

            for const in structured_task.get("constraints", []):
                if len(graph.constraints) >= self.max_constraints:
                    break

                constraint = await self._parse_constraint(const)
                if constraint:
                    graph.constraints.append(constraint)

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
        from z3 import Int, unsat

        for name, value in solution.items():
            if name in task_graph.variables:
                var = task_graph.variables[name]

                solver.push()
                try:
                    if isinstance(var, type(Int(name))):
                        solver.add(var == value)

                        result = solver.check()
                        if result == unsat:
                            return False
                finally:
                    solver.pop()

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

    async def _parse_requirement(self, req: dict) -> Constraint | None:
        try:
            from z3 import And, Int

            name = req["name"]
            req_type = req.get("type", "numeric")

            if req_type == "numeric":
                var = Int(name)
                min_val = req.get("min", 0)
                max_val = req.get("max", 1000)

                return Constraint(
                    name=name,
                    expression=And(var >= min_val, var <= max_val),
                    description=f"{name} in [{min_val}, {max_val}]",
                    priority=req.get("priority", 1),
                )

            return None

        except Exception as e:
            logger.warning("requirement_parsing_failed", requirement=req, error=str(e))
            return None

    async def _parse_constraint(self, const: dict) -> Constraint | None:
        try:
            from z3 import Bool

            name = const["name"]
            var = Bool(name)

            return Constraint(
                name=name,
                expression=var,
                description=const.get("condition", ""),
                priority=const.get("priority", 1),
            )
        except Exception as e:
            logger.warning("constraint_parsing_failed", constraint=const, error=str(e))
            return None
