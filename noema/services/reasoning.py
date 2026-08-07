"""Reasoning service — wraps ChainOfThought + LLM orchestration."""

from __future__ import annotations

import contextlib
import json
import time
from typing import TYPE_CHECKING, Any

from noema.core.types import (
    ArchitecturePattern,
    CodeBlock,
    Solution,
    SolutionQuality,
    Task,
    TechStack,
    ThoughtProcess,
)
from noema.logging import get_logger

if TYPE_CHECKING:
    from noema.core.chain_of_thought import ChainOfThought
    from noema.core.events import EventBus
    from noema.llm.providers import BaseLLMProvider

log = get_logger(__name__)


class ReasoningService:
    """Orchestrates LLM-based reasoning via ChainOfThought."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        cot: ChainOfThought,
        event_bus: EventBus | None = None,
    ) -> None:
        self.llm = llm
        self.cot = cot
        self.event_bus = event_bus

    async def think(self, task: Task) -> tuple[Solution, ThoughtProcess]:
        """Run the full reasoning pipeline."""
        t0 = time.monotonic()

        log.info("reasoning_start", task_id=task.id, title=task.title)

        requirements = [r.model_dump() for r in task.requirements]
        context = await self.cot.reason(
            task_description=task.title,
            task_tags=task.tags,
            requirements=requirements,
            complexity=task.complexity.value,
        )
        solution = self._build_solution(task, context)
        thought = self._build_thought_process(task, context)

        elapsed = (time.monotonic() - t0) * 1000
        thought.duration_ms = elapsed

        if self.event_bus:
            await self.event_bus.emit(
                "reasoning.completed",
                {
                    "task_id": task.id,
                    "solution_id": solution.id,
                    "quality": solution.quality.value,
                    "confidence": solution.confidence,
                    "duration_ms": elapsed,
                },
                source="reasoning_service",
            )

        log.info(
            "reasoning_done",
            task_id=task.id,
            quality=solution.quality.value,
            confidence=solution.confidence,
            duration_ms=round(elapsed, 1),
        )

        return solution, thought

    def _build_solution(self, task: Task, context: dict[str, Any]) -> Solution:
        """Assemble a Solution from the CoT context dict."""
        code_blocks: list[CodeBlock] = []
        architecture: ArchitecturePattern | None = None
        summary = ""

        raw_codegen = context.get("codegen", "")
        if isinstance(raw_codegen, str):
            with contextlib.suppress(ValueError, TypeError):
                data = json.loads(raw_codegen)
                if isinstance(data, dict) and isinstance(data.get("files"), list):
                    for f in data["files"]:
                        if isinstance(f, dict) and f.get("content"):
                            code_blocks.append(
                                CodeBlock(
                                    filename=f.get("path") or f.get("filename") or "main.py",
                                    language=f.get("language", "python"),
                                    content=f["content"],
                                    description=f.get("description", ""),
                                )
                            )

        raw_arch = context.get("architecture", "")
        if isinstance(raw_arch, str):
            with contextlib.suppress(ValueError, TypeError):
                data = json.loads(raw_arch)
                if isinstance(data, dict) and isinstance(data.get("pattern"), dict):
                    pattern = data["pattern"]
                    architecture = ArchitecturePattern(
                        name=pattern.get("name", "unknown"),
                        description=pattern.get("description", ""),
                    )

        raw_review = context.get("review", "")
        if isinstance(raw_review, str):
            with contextlib.suppress(ValueError, TypeError):
                data = json.loads(raw_review)
                if isinstance(data, dict) and isinstance(data.get("final_summary"), str):
                    summary = data["final_summary"]
        if not summary:
            summary = task.description or f"Solution for: {task.title}"

        stack = task.preferred_stack or TechStack(languages=["Python"])

        quality = SolutionQuality.GOOD if code_blocks else SolutionQuality.DRAFT

        return Solution(
            task_id=task.id,
            title=f"Solution: {task.title}",
            summary=summary,
            architecture=architecture,
            stack=stack,
            code_blocks=code_blocks,
            deployment={},
            performance_notes=[],
            security_notes=[],
            quality=quality,
            confidence=0.7 if code_blocks else 0.4,
            metadata={
                "reasoning_steps": len(self.cot.get_steps()),
                "llm_provider": self.llm.name,
                "llm_model": self.llm.model_name,
            },
        )

    def _build_thought_process(self, task: Task, context: dict[str, Any]) -> ThoughtProcess:
        tp = ThoughtProcess(task_id=task.id)
        for step in self.cot.get_steps():
            tp.add_step(
                kernel=step.name,
                input_summary=step.user_prompt[:200],
                output_summary=step.result[:500],
                confidence=step.confidence,
            )
        tp.final_reasoning = context.get("review", "")
        return tp
