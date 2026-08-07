"""Система пайплайнов — цепочки ядер для обработки."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from noema.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from noema.core.types import Task

logger = get_logger(__name__)


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStep:
    """Шаг пайплайна."""

    name: str
    kernel_name: str | None = None
    func: Callable[..., Coroutine] | None = None
    phase: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    condition: Callable[[dict], bool] | None = None
    retry_count: int = 0
    timeout_seconds: float = 30.0
    status: StepStatus = StepStatus.PENDING
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class PipelineResult:
    """Результат выполнения пайплайна."""

    pipeline_name: str
    steps: list[PipelineStep]
    final_output: dict[str, Any]
    total_duration_ms: float = 0.0
    success: bool = True

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)

    @property
    def failed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == StepStatus.FAILED)


class Pipeline:
    """
    Пайплайн — цепочка шагов для обработки задач.

    Каждый шаг может быть ядром, функцией или условием.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.steps: list[PipelineStep] = []
        self._context: dict[str, Any] = {}
        self._on_step_complete: list[Callable] = []
        self._on_error: list[Callable] = []

    def add_step(
        self,
        name: str,
        kernel_name: str | None = None,
        func: Callable[..., Coroutine] | None = None,
        phase: str = "",
        config: dict[str, Any] | None = None,
        condition: Callable[[dict], bool] | None = None,
        retry_count: int = 0,
        timeout: float = 30.0,
    ) -> Pipeline:
        """Добавить шаг в пайплайн (fluent API)."""
        self.steps.append(
            PipelineStep(
                name=name,
                kernel_name=kernel_name,
                func=func,
                phase=phase,
                config=config or {},
                condition=condition,
                retry_count=retry_count,
                timeout_seconds=timeout,
            )
        )
        return self

    def on_step_complete(self, callback: Callable[[dict[str, Any]], Any]) -> Pipeline:
        self._on_step_complete.append(callback)
        return self

    def on_error(self, callback: Callable[[dict[str, Any]], Any]) -> Pipeline:
        self._on_error.append(callback)
        return self

    async def execute(
        self,
        task: Task,
        noema: Any = None,
        initial_context: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Выполнить пайплайн."""
        self._context = initial_context or {}
        self._context["task"] = task

        result = PipelineResult(
            pipeline_name=self.name,
            steps=[],
            final_output={},
        )
        t0 = time.monotonic()

        for step in self.steps:
            result.steps.append(step)

            # Проверка условия
            if step.condition and not step.condition(self._context):
                step.status = StepStatus.SKIPPED
                logger.info(f"[Pipeline:{self.name}] Step '{step.name}' skipped (condition)")
                continue

            # Выполнение с retry
            step.status = StepStatus.RUNNING
            step_t0 = time.monotonic()

            for attempt in range(step.retry_count + 1):
                try:
                    if step.kernel_name and noema:
                        kernel = noema.kernels.get(step.kernel_name)
                        if kernel:
                            step.result = await asyncio.wait_for(
                                kernel.execute(task, **step.config, phase=step.phase),
                                timeout=step.timeout_seconds,
                            )
                        else:
                            step.result = {"error": f"Kernel '{step.kernel_name}' not found"}
                    elif step.func:
                        step.result = await asyncio.wait_for(
                            step.func(task, self._context, **step.config),
                            timeout=step.timeout_seconds,
                        )
                    else:
                        step.result = {"status": "noop"}

                    step.status = StepStatus.COMPLETED
                    self._context[step.name] = step.result
                    break

                except TimeoutError:
                    step.error = f"Timeout after {step.timeout_seconds}s"
                    if attempt < step.retry_count:
                        logger.warning(
                            f"[Pipeline:{self.name}] Step '{step.name}' timeout, retrying..."
                        )
                        continue
                    step.status = StepStatus.FAILED
                except Exception as e:
                    step.error = str(e)
                    if attempt < step.retry_count:
                        logger.warning(
                            f"[Pipeline:{self.name}] Step '{step.name}' error: {e}, retrying..."
                        )
                        continue
                    step.status = StepStatus.FAILED

            step.duration_ms = (time.monotonic() - step_t0) * 1000

            # Хуки
            for cb in self._on_step_complete:
                if callable(cb):
                    await cb(step)

            if step.status == StepStatus.FAILED:
                result.success = False
                for cb in self._on_error:
                    if callable(cb):
                        await cb(step, step.error)
                logger.error(f"[Pipeline:{self.name}] Step '{step.name}' failed: {step.error}")

        result.total_duration_ms = max(0.1, (time.monotonic() - t0) * 1000)
        result.final_output = self._context
        return result


# ── Встроенные пайплайны ────────────────────────────────────────────────────


def create_fullstack_pipeline() -> Pipeline:
    """Пайплайн полного цикла: анализ → архитектура → код → оптимизация → безопасность."""
    return (
        Pipeline("fullstack_generation")
        .add_step("analysis", kernel_name="analysis", phase="full")
        .add_step("architecture", kernel_name="architecture", phase="design")
        .add_step("codegen", kernel_name="codegen")
        .add_step("optimization", kernel_name="optimization")
        .add_step("security", kernel_name="security")
    )


def create_quick_prototype_pipeline() -> Pipeline:
    """Быстрый пайплайн для прототипа: только анализ + код."""
    return (
        Pipeline("quick_prototype")
        .add_step("analysis", kernel_name="analysis", phase="full", timeout=10.0)
        .add_step("codegen", kernel_name="codegen", timeout=15.0)
    )


def create_security_audit_pipeline() -> Pipeline:
    """Пайплайн security-аудита."""
    return (
        Pipeline("security_audit")
        .add_step("analysis", kernel_name="analysis", phase="full")
        .add_step("security", kernel_name="security")
        .add_step("optimization", kernel_name="optimization")
    )


def create_architecture_review_pipeline() -> Pipeline:
    """Пайплайн ревью архитектуры."""
    return (
        Pipeline("architecture_review")
        .add_step("analysis", kernel_name="analysis", phase="analyze")
        .add_step("architecture", kernel_name="architecture", phase="design")
        .add_step("security", kernel_name="security")
    )
