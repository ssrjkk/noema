"""NeuroSymbolic engine — application-layer orchestration of symbolic reasoning and neural hypotheses.

Layered architecture (per Domain-Driven Design):
- **Domain**: :class:`TaskGraph`, :class:`ThinkStage`, event ``TypedDict``s — pure models with no I/O.
- **Application**: :class:`NeuroSymbolicEngine` — the use-case orchestrator that owns the refine-verify loop.
- **Infrastructure**: :class:`SymbolicEngine` (Z3 solver pool), :class:`NeuralInterface` (LLM),
  :class:`CausalEngine`, :class:`EvolutionEngine` — all injected via composition.

Concurrency contract:
- Every blocking, CPU-bound step (causal graph construction, counterfactual
  estimation) is executed in a worker thread via :func:`asyncio.to_thread`, so
  the event loop is never blocked by synchronous work.
- The Z3 solver pool inside :class:`SymbolicEngine` bounds concurrent solver
  check-outs, serializing access to the (non-thread-safe) solvers.

Complexity:
- Parsing: ``O(R + C)`` where ``R`` = requirements, ``C`` = constraints.
- Verification: ``O(A · V · S)`` — ``A`` attempts, ``V`` solution variables, ``S`` per-solver cost.
- Causal analysis: ``O(R²)`` worst case for the pairwise dominance scan
  (inherent: the output edge set is quadratic in the requirement count), then
  ``O(N)`` counterfactual passes over the ``N`` graph nodes.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypedDict

import structlog
from pydantic import BaseModel, ConfigDict, Field

from noema.causal import CausalEngine
from noema.errors import NoemaError
from noema.neurosymbolic.evolution import EvolutionEngine
from noema.neurosymbolic.neural import NeuralInterface
from noema.neurosymbolic.symbolic import SymbolicEngine

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping, Sequence

logger = structlog.get_logger(__name__)

#: Hard bound on the number of refinement attempts per task.
_DEFAULT_MAX_REFINEMENT_ATTEMPTS: int = 3
#: Cap on graph nodes considered for counterfactual estimation.
_DEFAULT_MAX_COUNTERFACTUALS: int = 5


class ThinkStage(StrEnum):
    """Ordered stages of a single :meth:`NeuroSymbolicEngine.think` stream."""

    PARSING = "parsing"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    VERIFICATION = "verification"
    REFINEMENT = "refinement"
    CAUSAL_ANALYSIS = "causal_analysis"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"


class CausalAnalysisEntry(TypedDict):
    """One counterfactual estimation result for a causal graph."""

    variable: str
    intervention_1x: dict[str, Any]
    intervention_2x: dict[str, Any]
    backdoor_variables: list[str]
    mediator_variables: list[str]


class ThinkEvent(TypedDict, total=False):
    """Streamed event emitted by :meth:`NeuroSymbolicEngine.think`.

    ``stage`` is always present; the remaining keys vary by stage.
    """

    stage: str
    status: str
    correlation_id: str
    attempt: int
    requirements_count: int
    constraints_count: int
    hypothesis_keys: list[str]
    is_valid: bool
    violations_count: int
    counterfactuals_count: int
    solution: dict[str, Any]
    attempts: int
    causal_analysis: list[dict[str, Any]]
    causal_metrics: dict[str, Any]
    reason: str
    error: str
    error_type: str


class NeuroSymbolicTaskInput(BaseModel):
    """Strict, zero-trust validation of the task dict entering the engine.

    Every inbound field is coerced only in the strictest sense allowed by the
    domain: unknown keys are rejected and field types must match exactly so a
    hostile caller cannot smuggle malformed structure into the symbolic layer.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    requirements: list[dict[str, Any]] = Field(default_factory=list)
    constraints: list[dict[str, Any]] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    variables: dict[str, Any] = Field(default_factory=dict)


class NeuroSymbolicError(NoemaError):
    """Base error for the neurosymbolic engine."""


class MaxRefinementsExceededError(NeuroSymbolicError):
    """Raised when the engine exhausts refinement attempts without a valid solution."""


class VerificationFailedError(NeuroSymbolicError):
    """Raised when symbolic verification fails irrecoverably."""


def _coerce_priority(value: Any, default: int = 5) -> int:
    """Coerce an untrusted ``priority`` value to an int, falling back on junk.

    Args:
        value: Raw value from task input (may be any type).
        default: Priority used when ``value`` is not numeric.

    Returns:
        An integer priority in ``[0, 10]``.
    """
    try:
        return min(max(int(value), 0), 10)
    except (TypeError, ValueError, OverflowError):
        return default


def _dependency_node_name(requirement: Mapping[str, Any]) -> str:
    """Build a deterministic node label for a requirement in the causal graph.

    Args:
        requirement: A requirement mapping (``category`` / ``description``).

    Returns:
        A stable ``"category_description"`` label truncated to 21 characters.
    """
    category = str(requirement.get("category", "unknown"))
    description = str(requirement.get("description", ""))
    return f"{category}_{description[:20]}"


class NeuroSymbolicEngine:
    """Application orchestrator for the symbolic + neural reasoning pipeline.

    Pipeline (bounded by ``max_refinement_attempts``):
        1. Parse the task into a symbolic :class:`TaskGraph`.
        2. Generate a candidate hypothesis with the neural (LLM) interface.
        3. Verify the candidate against the symbolic graph.
        4. On violation, refine the hypothesis and repeat from step 3.
        5. On success, optionally estimate causal counterfactuals.

    Args:
        max_refinement_attempts: Maximum hypothesis/verify cycles per task.
        verification_timeout: Per-verification timeout in seconds.
        enable_evolution: Record successful/failed outcomes for evolution.
        enable_causal: Run causal counterfactual analysis on success.
        max_counterfactuals: Maximum counterfactual results per task.

    Raises:
        ValueError: If any configuration value is out of its allowed domain.
    """

    def __init__(
        self,
        max_refinement_attempts: int = _DEFAULT_MAX_REFINEMENT_ATTEMPTS,
        verification_timeout: float = 5.0,
        enable_evolution: bool = True,
        enable_causal: bool = True,
        max_counterfactuals: int = _DEFAULT_MAX_COUNTERFACTUALS,
    ) -> None:
        # ── Fail-fast configuration validation (directive: strict inputs) ──
        if max_refinement_attempts < 1:
            raise ValueError("max_refinement_attempts must be >= 1")
        if verification_timeout <= 0:
            raise ValueError("verification_timeout must be > 0")
        if max_counterfactuals < 0:
            raise ValueError("max_counterfactuals must be >= 0")

        self.max_refinement_attempts = max_refinement_attempts
        self.enable_evolution = enable_evolution
        self.enable_causal = enable_causal

        self.symbolic = SymbolicEngine(verification_timeout=verification_timeout)
        self.neural = NeuralInterface()
        self.evolution = EvolutionEngine() if enable_evolution else None
        self.causal: CausalEngine | None = (
            CausalEngine(enabled=enable_causal, max_counterfactuals=max_counterfactuals)
            if enable_causal
            else None
        )

        self._started = False
        self._metrics: dict[str, int | float] = {
            "tasks_processed": 0,
            "tasks_successful": 0,
            "tasks_failed": 0,
            "total_refinements": 0,
            "total_llm_calls": 0,
        }

        logger.info(
            "neurosymbolic_engine_created",
            max_refinements=max_refinement_attempts,
            evolution_enabled=enable_evolution,
            causal_enabled=enable_causal,
        )

    async def start(self) -> None:
        """Start the engine: initialize the solver pool and the neural interface.

        Idempotent — repeated calls are no-ops while the engine is running.
        """
        if self._started:
            return
        await self.symbolic.initialize()
        await self.neural.start()
        self._started = True
        logger.info("neurosymbolic_engine_started")

    async def stop(self) -> None:
        """Stop the engine and release the neural interface.

        Idempotent — repeated calls are no-ops while the engine is stopped.
        """
        if not self._started:
            return
        await self.neural.stop()
        self._started = False
        logger.info("neurosymbolic_engine_stopped", metrics=self._metrics)

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[NeuroSymbolicEngine, None]:
        """Context manager that starts the engine on enter and stops it on exit.

        Yields:
            The started engine instance.
        """
        await self.start()
        try:
            yield self
        finally:
            await self.stop()

    async def think(self, task: Mapping[str, Any]) -> AsyncGenerator[ThinkEvent, None]:
        """Run the full refine-verify pipeline for one task, streaming stage events.

        Args:
            task: Strictly validated task description with ``requirements``,
                ``constraints``, ``goals`` and ``variables`` keys.

        Yields:
            A :class:`ThinkEvent` per stage transition; the terminal stage is
            either ``completed`` (with ``solution``) or ``failed``.

        Raises:
            RuntimeError: If the engine is not started.
            MaxRefinementsExceededError: If the engine is configured without
                fallback and all refinement attempts are exhausted.
            NeuroSymbolicError: Re-raised after streaming an ``error`` event
                whenever an unexpected failure occurs.
        """
        if not self._started:
            raise RuntimeError("Engine not started. Call start() first.")

        # Zero-trust boundary: reject malformed/hostile task payloads up front.
        validated_input = NeuroSymbolicTaskInput.model_validate(dict(task))
        task_dict: dict[str, Any] = validated_input.model_dump()
        correlation_id: str = uuid.uuid4().hex

        logger.info(
            "think_started", correlation_id=correlation_id, task_keys=list(task_dict.keys())
        )

        try:
            yield {
                "stage": ThinkStage.PARSING.value,
                "status": "started",
                "correlation_id": correlation_id,
            }
            task_graph = await self.symbolic.parse_task(task_dict)
            yield {
                "stage": ThinkStage.PARSING.value,
                "status": "completed",
                "correlation_id": correlation_id,
                "requirements_count": len(task_graph.requirements),
                "constraints_count": len(task_graph.constraints),
            }

            yield {
                "stage": ThinkStage.HYPOTHESIS_GENERATION.value,
                "status": "started",
                "correlation_id": correlation_id,
            }
            hypothesis: dict[str, Any] = await self.neural.generate_hypothesis(task_dict)
            self._metrics["total_llm_calls"] += 1
            yield {
                "stage": ThinkStage.HYPOTHESIS_GENERATION.value,
                "status": "completed",
                "correlation_id": correlation_id,
                "hypothesis_keys": list(hypothesis.keys()),
            }

            # max_refinement_attempts >= 1 (validated in __init__); pre-bind so
            # the failure path below can reference the last verification round.
            violations: list[str] = []
            for attempt in range(self.max_refinement_attempts):
                yield {
                    "stage": ThinkStage.VERIFICATION.value,
                    "attempt": attempt + 1,
                    "status": "started",
                    "correlation_id": correlation_id,
                }
                is_valid, violations = await self.symbolic.verify_solution(hypothesis, task_graph)
                yield {
                    "stage": ThinkStage.VERIFICATION.value,
                    "attempt": attempt + 1,
                    "status": "completed",
                    "correlation_id": correlation_id,
                    "is_valid": is_valid,
                    "violations_count": len(violations),
                }

                if is_valid:
                    self._metrics["tasks_successful"] += 1
                    self._metrics["tasks_processed"] += 1

                    if self.evolution:
                        await self.evolution.record_outcome(
                            task=task_dict, hypothesis=hypothesis, is_successful=True
                        )

                    causal_analysis: list[dict[str, Any]] | None = None
                    if self.causal:
                        yield {
                            "stage": ThinkStage.CAUSAL_ANALYSIS.value,
                            "status": "started",
                            "correlation_id": correlation_id,
                        }
                        causal_analysis = await self._run_causal_analysis(task_dict)
                        yield {
                            "stage": ThinkStage.CAUSAL_ANALYSIS.value,
                            "status": "completed",
                            "correlation_id": correlation_id,
                            "counterfactuals_count": len(causal_analysis),
                        }

                    result: ThinkEvent = {
                        "stage": ThinkStage.COMPLETED.value,
                        "solution": hypothesis,
                        "correlation_id": correlation_id,
                        "attempts": attempt + 1,
                    }
                    if causal_analysis and self.causal is not None:
                        result["causal_analysis"] = causal_analysis
                        result["causal_metrics"] = dict(self.causal.get_metrics())

                    yield result

                    logger.info(
                        "think_completed",
                        correlation_id=correlation_id,
                        attempts=attempt + 1,
                        causal=bool(causal_analysis),
                    )
                    return

                if attempt < self.max_refinement_attempts - 1:
                    yield {
                        "stage": ThinkStage.REFINEMENT.value,
                        "attempt": attempt + 1,
                        "status": "started",
                        "correlation_id": correlation_id,
                    }
                    hypothesis = await self.neural.refine_hypothesis(
                        hypothesis=hypothesis, violations=violations, task_graph=task_dict
                    )
                    self._metrics["total_llm_calls"] += 1
                    self._metrics["total_refinements"] += 1
                    yield {
                        "stage": ThinkStage.REFINEMENT.value,
                        "attempt": attempt + 1,
                        "status": "completed",
                        "correlation_id": correlation_id,
                    }

            self._metrics["tasks_failed"] += 1
            self._metrics["tasks_processed"] += 1

            if self.evolution:
                await self.evolution.record_outcome(
                    task=task_dict,
                    hypothesis=hypothesis,
                    is_successful=False,
                    violations=violations,
                )

            yield {
                "stage": ThinkStage.FAILED.value,
                "reason": "max_refinements_exceeded",
                "correlation_id": correlation_id,
                "attempts": self.max_refinement_attempts,
            }
            logger.warning(
                "think_failed", correlation_id=correlation_id, reason="max_refinements_exceeded"
            )

        except Exception as exc:
            self._metrics["tasks_failed"] += 1
            self._metrics["tasks_processed"] += 1
            logger.error(
                "think_error",
                correlation_id=correlation_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            yield {
                "stage": ThinkStage.ERROR.value,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "correlation_id": correlation_id,
            }
            raise

    async def _run_causal_analysis(self, task: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Estimate causal counterfactuals for a validated task, off the event loop.

        Args:
            task: Strictly validated task mapping.

        Returns:
            A list of counterfactual estimation results, or ``[]`` when the
            task carries no requirements/constraints or causal analysis is
            disabled.
        """
        if self.causal is None:
            return []
        requirements: list[dict[str, Any]] = [
            {
                "category": str(requirement.get("category", "unknown")),
                "description": str(requirement.get("description", "")),
                "priority": _coerce_priority(requirement.get("priority", 5)),
            }
            for requirement in task.get("requirements", [])
        ]
        constraints: list[dict[str, Any]] = [
            {"name": str(constraint.get("name", f"c_{index}"))}
            for index, constraint in enumerate(task.get("constraints", []))
        ]
        if not requirements and not constraints:
            return []
        description: str = str(task.get("description", ""))
        # CPU-bound graph construction runs in a worker thread (directive: no
        # blocking calls in the asyncio loop).
        return await asyncio.to_thread(
            self._compute_counterfactuals_blocking,
            requirements,
            constraints,
            description,
        )

    def _compute_counterfactuals_blocking(
        self,
        requirements: Sequence[dict[str, Any]],
        constraints: Sequence[dict[str, Any]],
        description: str,
    ) -> list[dict[str, Any]]:
        """Build the causal graph and estimate counterfactuals (blocking worker).

        Complexity: the dependency scan is a pairwise dominance check, O(R²) in
        the worst case (R = number of requirements); this is inherent because
        the produced edge set is quadratic in ``R``. The counterfactual pass is
        O(N) over the ``N`` graph nodes. Both are bounded by the input size.

        Args:
            requirements: Normalized requirement dicts with ``priority`` ints.
            constraints: Normalized constraint dicts.
            description: Task description used as the causal analysis seed.

        Returns:
            A list of counterfactual estimation results.
        """
        causal = self.causal
        if causal is None:
            return []

        dependencies: list[tuple[str, str, float]] = []
        for requirement in requirements:
            requirement_priority = requirement.get("priority", 5)
            for other in requirements:
                if other is requirement:
                    continue
                if requirement_priority > other.get("priority", 5):
                    dependencies.append(
                        (_dependency_node_name(requirement), _dependency_node_name(other), 0.5)
                    )

        graph = causal.build_graph(list(requirements), list(constraints), dependencies)
        return causal.analyze_all_counterfactuals(graph, description)

    def get_metrics(self) -> dict[str, Any]:
        """Return engine metrics plus the derived success rate.

        Returns:
            A dict with the engine counters, ``success_rate`` in ``[0, 1]``,
            and — when causal analysis is enabled — a nested ``causal`` section.
        """
        base: dict[str, Any] = {
            **self._metrics,
            "success_rate": (
                self._metrics["tasks_successful"] / self._metrics["tasks_processed"]
                if self._metrics["tasks_processed"] > 0
                else 0.0
            ),
        }
        if self.causal:
            base["causal"] = dict(self.causal.get_metrics())
        return base
