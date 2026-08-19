from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from noema.llm.providers import BaseLLMProvider
    from noema.ontology import OntologyGraph


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` when it is awaitable (e.g. a coroutine), else return it."""
    if inspect.isawaitable(value):
        return await value
    return value


async def _notify(callback: Callable, hr: Any) -> None:
    """Invoke an ``on_heal`` callback whether it is sync or async."""
    try:
        result = callback(hr)
        if inspect.isawaitable(result):
            await result
    except Exception:  # noqa: BLE001 - a notifier must never mask the result
        pass


def _supports_kwarg(func: Callable, name: str) -> bool:
    """Return True when ``func`` can be called with keyword argument ``name``."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == name:
            return True
    return False


class HealingAction(StrEnum):
    RETRY = "retry"
    RETRY_WITH_CONTEXT = "retry_with_context"
    FALLBACK = "fallback"
    ROLLBACK = "rollback"
    SKIP = "skip"
    ESCALATE = "escalate"


class HealingResult(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = Field(default_factory=time.time)
    original_error: str = ""
    action_taken: str = ""
    succeeded: bool = False
    recovery_time_ms: float = 0.0
    details: str = ""


class HealingStrategy(BaseModel):
    name: str = ""
    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_max: float = 30.0
    timeout: float = 120.0
    fallback_value: Any = None
    actions: list[HealingAction] = Field(
        default_factory=lambda: [
            HealingAction.RETRY,
            HealingAction.RETRY_WITH_CONTEXT,
            HealingAction.FALLBACK,
        ]
    )


class SelfHealer:
    """Self-healing system: detects failures, retries, recovers.

    ``llm`` / ``ontology`` / ``ontology_persist_path`` are optional ORL
    dependencies: when present, :meth:`_propose_ontological_axiom` can
    crystallize a lesson from a failed→fixed code pair into the ontology.
    """

    def __init__(
        self,
        strategy: HealingStrategy | None = None,
        llm: BaseLLMProvider | None = None,
        ontology: OntologyGraph | None = None,
        ontology_persist_path: Path | str | None = None,
    ) -> None:
        self.strategy = strategy or HealingStrategy()
        self.history: list[HealingResult] = []
        self._success_count = 0
        self._failure_count = 0
        self._strategies_applied: dict[str, int] = {}
        self.llm = llm
        self.ontology = ontology
        self.ontology_persist_path = ontology_persist_path

    async def _propose_ontological_axiom(
        self,
        failed_code: str,
        fixed_code: str,
        error_summary: str = "",
        task_id: str = "",
    ) -> bool:
        """Run the ORL pipeline on a failed→fixed code pair.

        The LLM proposes an axiom, the epistemic validator decides, and only
        accepted axioms mutate the ontology (with provenance). Never raises:
        any failure degrades to ``False``.
        """
        if self.llm is None or self.ontology is None:
            return False
        from noema.ontology import crystallize_axiom

        try:
            result = await crystallize_axiom(
                llm=self.llm,
                ontology=self.ontology,
                failed_code=failed_code,
                fixed_code=fixed_code,
                error_summary=error_summary,
                task_id=task_id,
                persist_path=self.ontology_persist_path,
            )
        except Exception:  # noqa: BLE001 - ORL must never crash the healing path
            return False
        return result.accepted

    async def execute_with_healing(
        self,
        func: Callable,
        *args: Any,
        fallback: Any = None,
        on_heal: Callable[[HealingResult], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute a function with automatic healing on failure."""
        last_error: Exception | None = None
        context: dict[str, Any] = {"args": args, "kwargs": dict(kwargs), "attempts": []}
        healing_context: dict[str, Any] = {}
        supports_context = _supports_kwarg(func, "_healing_context")

        for action in self.strategy.actions:
            for attempt in range(self.strategy.max_retries):
                call_kwargs = dict(kwargs)
                if healing_context and supports_context:
                    call_kwargs["_healing_context"] = healing_context
                try:
                    start = time.time()
                    candidate = _maybe_await(func(*args, **call_kwargs))
                    if inspect.isawaitable(candidate):
                        # A hung coroutine must not block the healing cycle
                        # (or its caller) forever.
                        result = await asyncio.wait_for(candidate, timeout=self.strategy.timeout)
                    else:
                        result = candidate
                    elapsed = (time.time() - start) * 1000

                    self._success_count += 1
                    hr = HealingResult(
                        original_error=str(last_error) if last_error else "",
                        action_taken=f"succeeded_on_{action.value}_attempt_{attempt + 1}",
                        succeeded=True,
                        recovery_time_ms=elapsed,
                    )
                    self.history.append(hr)
                    if on_heal:
                        await _notify(on_heal, hr)
                    return result

                except Exception as e:
                    last_error = e
                    context["attempts"].append(
                        {"error": str(e), "action": action.value, "attempt": attempt}
                    )
                    self._strategies_applied[action.value] = (
                        self._strategies_applied.get(action.value, 0) + 1
                    )

                    if action in (HealingAction.RETRY, HealingAction.RETRY_WITH_CONTEXT):
                        if action == HealingAction.RETRY_WITH_CONTEXT:
                            healing_context = {
                                "previous_error": str(e),
                                "attempt": attempt,
                                "retry_count": attempt,
                            }
                        delay = min(
                            self.strategy.backoff_base * (2**attempt),
                            self.strategy.backoff_max,
                        )
                        await asyncio.sleep(delay)
                        continue

                    elif action == HealingAction.FALLBACK:
                        self._failure_count += 1
                        hr = HealingResult(
                            original_error=str(e),
                            action_taken="fallback",
                            succeeded=fallback is not None,
                            details="Used fallback value",
                        )
                        self.history.append(hr)
                        if on_heal:
                            await _notify(on_heal, hr)
                        return fallback

                    elif action == HealingAction.SKIP:
                        self._failure_count += 1
                        hr = HealingResult(
                            original_error=str(e),
                            action_taken="skip",
                            succeeded=False,
                            details="Skipped due to failure",
                        )
                        self.history.append(hr)
                        if on_heal:
                            await _notify(on_heal, hr)
                        return None

                    elif action == HealingAction.ROLLBACK:
                        self._failure_count += 1
                        hr = HealingResult(
                            original_error=str(e),
                            action_taken="rollback",
                            succeeded=False,
                            details="Rollback requested",
                        )
                        self.history.append(hr)
                        if on_heal:
                            await _notify(on_heal, hr)
                        raise

                    elif action == HealingAction.ESCALATE:
                        self._failure_count += 1
                        hr = HealingResult(
                            original_error=str(e),
                            action_taken="escalate",
                            succeeded=False,
                            details="Escalating to caller",
                        )
                        self.history.append(hr)
                        if on_heal:
                            await _notify(on_heal, hr)
                        raise

        # all strategies exhausted
        self._failure_count += 1
        hr = HealingResult(
            original_error=str(last_error),
            action_taken="all_strategies_exhausted",
            succeeded=False,
        )
        self.history.append(hr)
        if on_heal:
            await _notify(on_heal, hr)
        if last_error:
            raise last_error
        return fallback

    def get_stats(self) -> dict[str, Any]:
        total = self._success_count + self._failure_count
        return {
            "total_executions": total,
            "successes": self._success_count,
            "failures": self._failure_count,
            "healing_rate": self._success_count / max(total, 1),
            "strategies_used": dict(self._strategies_applied),
        }
