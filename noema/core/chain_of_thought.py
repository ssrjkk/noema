"""DAG-based Chain-of-Thought engine — параллельное рассуждение с Reflexion.

Architecture:
- :class:`StepPlanner` — pure function mapping (tags, complexity, error context) to an
  ordered step list; deterministic, no I/O.
- :class:`ChainOfThought` — DAG executor: builds a step graph, computes topological
  levels, then runs every level's steps concurrently via ``asyncio``.

Concurrency contract:
- Steps within one topological level run concurrently; steps across levels are
  ordered by their ``depends_on`` edges.
- The LLM provider is the only blocking-ish dependency and it is awaited, so the
  event loop is never blocked by this module.

Complexity:
- Planning: ``O(S)`` for the planner's fixed step universe ``S``.
- Topological levels: ``O(V + E)`` per Kahn pass; worst case ``O(V²)`` for dense
  dependency graphs (V = steps, E = edges).
- Execution: ``O(S · L)`` LLM calls, where ``S`` is the number of selected steps
  and ``L`` the depth of the DAG.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from noema.llm.providers import BaseLLMProvider, LLMMessage
from noema.logging import get_logger

if TYPE_CHECKING:
    from noema.budget.token_budget import TokenBudget
    from noema.routing.model_router import ModelRouter

logger = get_logger(__name__)

StreamCallback = Callable[[str, str, int, int], Coroutine[Any, Any, None] | None]


@dataclass
class ReflexionState:
    """Short-term memory of past failures within a task (Reflexion pattern)."""

    attempt: int = 1
    failed_step: str = ""
    error_summary: str = ""
    last_errors: list[str] = field(default_factory=list)
    max_attempts: int = 3


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class CoTStep:
    name: str
    role: str
    depends_on: list[str]
    system_prompt: str
    user_prompt: str
    result: str = ""
    confidence: float = 0.0
    status: StepStatus = StepStatus.PENDING
    duration_ms: float = 0.0
    tokens_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 1),
            "confidence": round(self.confidence, 2),
            "tokens_used": self.tokens_used,
            "result_preview": self.result[:200] if self.result else "",
        }


@dataclass
class ThoughtNode:
    step: int
    role: str
    prompt: str
    response: str
    name: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list[ThoughtNode] = field(default_factory=list)


class StepPlanner:
    ALL_STEPS = [
        "repair",
        "analysis",
        "architecture",
        "stack",
        "components",
        "data_model",
        "api_design",
        "codegen",
        "testing",
        "security",
        "optimization",
        "deployment",
        "review",
    ]

    MINIMAL_STEPS = ["analysis", "architecture", "stack", "codegen", "review"]

    def plan(
        self,
        task_tags: list[str],
        complexity: str,
        error_context: str = "",
        past_attempts: list[str] | None = None,
    ) -> list[str]:
        """Select the ordered step list for a task.

        Complexity: ``O(S)`` in the planner's fixed step universe ``S`` (a
        constant); tag lookups are ``O(1)`` per membership test on a set.

        Reflexion: when ``error_context`` and ``past_attempts`` are supplied, a
        ``repair`` step is inserted right after ``analysis`` so the failed design
        is revisited before architecture decisions are made.
        """
        tags_lower = {t.lower() for t in task_tags}
        complexity = complexity.lower()

        if complexity in ("trivial", "simple"):
            return self.MINIMAL_STEPS

        steps = ["analysis", "architecture", "stack", "components"]

        if "data" in tags_lower or "database" in tags_lower:
            steps.append("data_model")
        if "api" in tags_lower or "web" in tags_lower or "rest" in tags_lower:
            steps.append("api_design")

        steps.append("codegen")

        if "test" in tags_lower or "qa" in tags_lower:
            steps.append("testing")
        if "security" in tags_lower or "auth" in tags_lower:
            steps.append("security")
        if "performance" in tags_lower or "high-load" in tags_lower:
            steps.append("optimization")

        steps.append("deployment")
        steps.append("review")

        if error_context and past_attempts:
            insert_at = 1 if "analysis" in steps else 0
            steps.insert(insert_at, "repair")

        return steps


class ChainOfThought:
    def __init__(
        self,
        llm: BaseLLMProvider,
        max_steps: int = 12,
        on_step_start: StreamCallback | None = None,
        on_step_end: StreamCallback | None = None,
        model_router: ModelRouter | None = None,
        token_budget: TokenBudget | None = None,
    ) -> None:
        """Initialize the DAG executor.

        Args:
            llm: LLM provider used for every step's generation.
            max_steps: Upper bound on steps planned per reasoning run.
            on_step_start/on_step_end: Optional streaming callbacks.
            model_router: Optional router for degraded-mode model selection.
            token_budget: Optional token budget consulted before each step.
        """
        self.llm = llm
        self.max_steps = max_steps
        self.chain: list[ThoughtNode] = []
        self._context: dict[str, Any] = {}
        self._steps: list[CoTStep] = []
        self._planner = StepPlanner()
        self.on_step_start = on_step_start
        self.on_step_end = on_step_end
        self._model_router = model_router
        self._token_budget = token_budget

    async def reason(
        self,
        task_description: str,
        task_tags: list[str],
        requirements: list[dict[str, Any]],
        knowledge_context: str = "",
        graph_context: str = "",
        ontology_context: str = "",
        complexity: str = "moderate",
        reflexion_errors: list[str] | None = None,
        reflexion_attempt: int = 1,
        resume_context: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run the planned DAG and return the accumulated step context.

        When ``resume_context`` maps step name → previous result, matching steps
        are restored from the checkpoint instead of calling the LLM again
        (resumable execution for crashed/interrupted tasks).

        ``ontology_context`` carries the ontological axioms for this task; it
        is injected into the first (analysis) prompt as hard guidance.

        Complexity: ``O(V · L)`` LLM calls in the worst case, where ``V`` is the
        number of selected steps and ``L`` the DAG depth; per-level scheduling is
        ``O(V + E)``.
        """
        self.chain = []
        self._context = {
            "task": task_description,
            "tags": task_tags,
            "requirements": requirements,
            "knowledge": knowledge_context,
            "graph": graph_context,
            "ontology": ontology_context,
        }

        if reflexion_errors:
            self._context["_reflexion_errors"] = reflexion_errors

        planned_steps = self._planner.plan(
            task_tags,
            complexity,
            error_context="\n".join(reflexion_errors[-3:]) if reflexion_errors else "",
            past_attempts=reflexion_errors,
        )
        logger.info(
            f"[CoT] Reflexion attempt {reflexion_attempt} | Planner selected {len(planned_steps)} steps: {planned_steps}"
        )

        if self.on_step_start:
            cb = self.on_step_start
            if asyncio.iscoroutinefunction(cb):
                await cb("planner", "Planner", 0, len(planned_steps))

        self._build_dag(
            planned_steps,
            task_description,
            task_tags,
            requirements,
            knowledge_context,
            graph_context,
            ontology_context,
        )

        resume_context = resume_context or {}
        for step in self._steps:
            if step.name in resume_context:
                step.status = StepStatus.COMPLETED
                step.result = resume_context[step.name]
                self._context[step.name] = step.result
        if resume_context:
            logger.info(
                "resuming_steps",
                restored=[s.name for s in self._steps if s.status == StepStatus.COMPLETED],
            )

        steps_by_name: dict[str, CoTStep] = {step.name: step for step in self._steps}
        levels = self._topological_levels()
        total = len(self._steps)
        completed = 0

        for level in levels:
            if not level:
                continue

            tasks: dict[str, asyncio.Task[None]] = {}
            for step in level:
                if step.status in (StepStatus.SKIPPED, StepStatus.COMPLETED):
                    completed += 1
                    if step.status == StepStatus.COMPLETED and self.on_step_end:
                        cb = self.on_step_end
                        if asyncio.iscoroutinefunction(cb):
                            await cb(step.name, step.result[:200], completed, total)
                    continue
                step.status = StepStatus.RUNNING
                if self.on_step_start:
                    cb = self.on_step_start
                    if asyncio.iscoroutinefunction(cb):
                        await cb(step.name, f"{step.role}:{step.name}", completed, total)
                tasks[step.name] = asyncio.create_task(self._execute_step(step))

            results = await asyncio.gather(*tasks.values(), return_exceptions=True)

            for step_name, result in zip(tasks.keys(), results, strict=False):
                done_step = steps_by_name.get(step_name)
                if done_step is None:
                    logger.error(f"step_not_found: {step_name}")
                    continue
                if isinstance(result, BaseException):
                    done_step.status = StepStatus.FAILED
                    done_step.result = f"Error: {result}"
                    logger.error(f"[CoT] Step {done_step.name} failed: {result}")
                    completed += 1
                    if self.on_step_end:
                        cb = self.on_step_end
                        if asyncio.iscoroutinefunction(cb):
                            await cb(done_step.name, f"FAILED: {result}", completed, total)
                else:
                    if done_step.status == StepStatus.SKIPPED:
                        completed += 1
                        if self.on_step_end:
                            cb = self.on_step_end
                            if asyncio.iscoroutinefunction(cb):
                                await cb(done_step.name, "SKIPPED", completed, total)
                        continue
                    done_step.status = StepStatus.COMPLETED
                    self._context[done_step.name] = done_step.result
                    completed += 1
                    if self.on_step_end:
                        cb = self.on_step_end
                        if asyncio.iscoroutinefunction(cb):
                            await cb(done_step.name, done_step.result[:200], completed, total)

        return self._context

    async def reason_with_reflexion(
        self,
        task_description: str,
        task_tags: list[str],
        requirements: list[dict[str, Any]],
        knowledge_context: str = "",
        graph_context: str = "",
        ontology_context: str = "",
        complexity: str = "moderate",
        judge_feedback: str = "",
    ) -> tuple[dict[str, Any], ReflexionState | None]:
        """Run CoT with Reflexion: retry with replanning on critical failure.

        The number of attempts is bounded by ``ReflexionState.max_attempts``;
        a ``None`` reflexion state is returned when the first attempt succeeded.
        """
        state = ReflexionState(attempt=1)
        last_context: dict[str, Any] = {}

        while state.attempt <= state.max_attempts:
            errors = list(state.last_errors)
            last_context = await self.reason(
                task_description=task_description,
                task_tags=task_tags,
                requirements=requirements,
                knowledge_context=knowledge_context,
                graph_context=graph_context,
                ontology_context=ontology_context,
                complexity=complexity,
                reflexion_errors=errors if state.attempt > 1 else None,
                reflexion_attempt=state.attempt,
            )

            state.attempt += 1

            if not judge_feedback or state.attempt > state.max_attempts:
                return last_context, state if state.attempt > 2 else None

            state.last_errors.append(f"Attempt {state.attempt - 1} judge: {judge_feedback[:200]}")

        return last_context, state

    def _build_dag(
        self,
        planned_steps: list[str],
        task_description: str,
        task_tags: list[str],
        requirements: list[dict[str, Any]],
        knowledge_context: str,
        graph_context: str,
        ontology_context: str = "",
    ) -> None:
        self._steps = []
        step_defs = {
            "analysis": CoTStep(
                name="analysis",
                role="analyst",
                depends_on=[],
                system_prompt=self._analyst_system(),
                user_prompt=self._analyst_prompt(
                    task_description, task_tags, requirements, knowledge_context, ontology_context
                ),
            ),
            "architecture": CoTStep(
                name="architecture",
                role="architect",
                depends_on=["analysis"],
                system_prompt=self._architect_system(),
                user_prompt="",  # filled dynamically
            ),
            "stack": CoTStep(
                name="stack",
                role="architect",
                depends_on=["architecture"],
                system_prompt=self._stack_system(),
                user_prompt="",
            ),
            "components": CoTStep(
                name="components",
                role="architect",
                depends_on=["architecture", "stack"],
                system_prompt=self._component_system(),
                user_prompt="",
            ),
            "data_model": CoTStep(
                name="data_model",
                role="developer",
                depends_on=["components", "stack"],
                system_prompt=self._data_model_system(),
                user_prompt="",
            ),
            "api_design": CoTStep(
                name="api_design",
                role="developer",
                depends_on=["components", "stack"],
                system_prompt=self._api_system(),
                user_prompt="",
            ),
            "codegen": CoTStep(
                name="codegen",
                role="developer",
                depends_on=["stack", "api_design", "data_model", "components"],
                system_prompt=self._codegen_system(),
                user_prompt="",
            ),
            "testing": CoTStep(
                name="testing",
                role="developer",
                depends_on=["codegen", "stack", "api_design", "components"],
                system_prompt=self._testing_system(),
                user_prompt="",
            ),
            "security": CoTStep(
                name="security",
                role="reviewer",
                depends_on=["codegen", "stack", "api_design", "components"],
                system_prompt=self._security_system(),
                user_prompt="",
            ),
            "optimization": CoTStep(
                name="optimization",
                role="optimizer",
                depends_on=["codegen", "stack", "components", "architecture"],
                system_prompt=self._optimization_system(),
                user_prompt="",
            ),
            "deployment": CoTStep(
                name="deployment",
                role="devops",
                depends_on=["stack", "components"],
                system_prompt=self._deployment_system(),
                user_prompt="",
            ),
            "review": CoTStep(
                name="review",
                role="reviewer",
                depends_on=[
                    "analysis",
                    "architecture",
                    "stack",
                    "components",
                    "data_model",
                    "api_design",
                    "codegen",
                    "testing",
                    "security",
                    "optimization",
                    "deployment",
                ],
                system_prompt=self._review_system(),
                user_prompt="",
            ),
            "repair": CoTStep(
                name="repair",
                role="debugger",
                depends_on=["analysis"],
                system_prompt=self._repair_system(),
                user_prompt="",
            ),
        }

        step_set = set(planned_steps)
        for name in self._planner.ALL_STEPS:
            if name not in step_set:
                continue
            step = step_defs[name]
            step.depends_on = [d for d in step.depends_on if d in step_set]
            self._steps.append(step)

    def _topological_levels(self) -> list[list[CoTStep]]:
        """Partition steps into parallel-executable topological levels (Kahn).

        Complexity: each pass is ``O(V + E)``; the worst case for a chain-shaped
        dependency graph is ``O(V)`` passes, i.e. ``O(V²)`` overall.
        """
        levels: list[list[CoTStep]] = []
        added: set[str] = set()
        remaining = list(self._steps)

        while remaining:
            level = []
            for step in remaining[:]:
                if all(dep in added for dep in step.depends_on):
                    level.append(step)
                    remaining.remove(step)
            if not level:
                remaining.clear()
                break
            levels.append(level)
            added.update(s.name for s in level)

        return levels

    async def _execute_step(self, step: CoTStep) -> None:
        step.user_prompt = self._resolve_prompt(step)
        if not step.user_prompt:
            step.status = StepStatus.SKIPPED
            return

        messages = [
            LLMMessage(role="system", content=step.system_prompt),
            LLMMessage(role="user", content=step.user_prompt),
        ]

        # Token budget check
        estimated_tokens = len(step.system_prompt) + len(step.user_prompt or "")
        budget = self._token_budget
        if budget:
            action = budget.check(estimated_cost=estimated_tokens // 2, step_name=step.name)
            if action.name == "SKIP":
                step.status = StepStatus.SKIPPED
                step.result = f"[Budget skipped] Token budget exhausted for: {step.name}"
                return

        # Model routing
        llm: BaseLLMProvider = self.llm
        if self._model_router and budget and budget.should_degrade():
            llm = self._model_router.select(complexity="simple", degraded=True)

        t0 = time.monotonic()
        response = await llm.complete(messages, temperature=0.4, max_tokens=4096)
        duration = (time.monotonic() - t0) * 1000

        step.result = response.content
        step.duration_ms = duration
        step.tokens_used = response.tokens_used or 0
        step.confidence = min(1.0, step.tokens_used / 1000) if step.tokens_used else 0.7

        if budget and response.tokens_used:
            budget.record(response.tokens_used)

        node = ThoughtNode(
            step=len(self.chain) + 1,
            role=step.role,
            prompt=step.user_prompt[:500],
            response=response.content,
            name=step.name,
            confidence=step.confidence,
        )
        self.chain.append(node)

    def _compress(self, text: str, max_chars: int = 1500) -> str:
        """Compress a prior step's output before feeding it as context.

        Structured (JSON) payloads are summarized field-wise; anything else is
        truncated. Complexity: ``O(n)`` in the input size.
        """
        if len(text) <= max_chars:
            return text
        with contextlib.suppress(ValueError, TypeError):
            data = json.loads(text)
            if isinstance(data, dict):
                compressed: dict[str, Any] = {}
                for k, v in data.items():
                    if isinstance(v, str) and len(v) > 200:
                        compressed[k] = v[:200] + "..."
                    elif isinstance(v, list) and len(v) > 5:
                        compressed[k] = v[:5]
                    elif isinstance(v, dict):
                        compressed[k] = {sk: str(sv)[:100] for sk, sv in list(v.items())[:8]}
                    else:
                        compressed[k] = v
                result = json.dumps(compressed, ensure_ascii=False)
                if len(result) <= max_chars:
                    return result
            elif isinstance(data, list) and len(data) > 5:
                return json.dumps(data[:5], ensure_ascii=False)
        return text[:max_chars] + "..."

    def _resolve_prompt(self, step: CoTStep) -> str:
        ctx = self._context
        if step.name == "analysis":
            return step.user_prompt  # populated at DAG build time
        if step.name == "architecture":
            return self._architect_prompt(
                self._compress(ctx.get("analysis", ""), 2000),
                ctx.get("graph", ""),
            )
        elif step.name == "stack":
            return self._stack_prompt(
                self._compress(ctx.get("architecture", ""), 2000),
                ctx.get("tags", []),
            )
        elif step.name == "components":
            return self._component_prompt(
                self._compress(ctx.get("architecture", ""), 1500),
                self._compress(ctx.get("stack", ""), 800),
            )
        elif step.name == "data_model":
            return self._data_model_prompt(
                self._compress(ctx.get("components", ""), 1500),
                self._compress(ctx.get("stack", ""), 400),
                ctx.get("requirements", []),
            )
        elif step.name == "api_design":
            return self._api_prompt(
                self._compress(ctx.get("components", ""), 1500),
                self._compress(ctx.get("stack", ""), 400),
                ctx.get("requirements", []),
            )
        elif step.name == "codegen":
            return self._codegen_prompt(
                self._compress(ctx.get("stack", ""), 800),
                self._compress(ctx.get("api_design", ""), 1500),
                self._compress(ctx.get("data_model", ""), 1500),
                self._compress(ctx.get("components", ""), 800),
            )
        elif step.name == "testing":
            return self._testing_prompt(
                self._compress(ctx.get("stack", ""), 400),
                self._compress(ctx.get("api_design", ""), 1000),
                self._compress(ctx.get("components", ""), 800),
            )
        elif step.name == "security":
            return self._security_prompt(
                self._compress(ctx.get("stack", ""), 400),
                self._compress(ctx.get("api_design", ""), 1000),
                self._compress(ctx.get("components", ""), 800),
            )
        elif step.name == "optimization":
            return self._optimization_prompt(
                self._compress(ctx.get("stack", ""), 400),
                self._compress(ctx.get("components", ""), 800),
                self._compress(ctx.get("architecture", ""), 800),
            )
        elif step.name == "deployment":
            return self._deployment_prompt(
                self._compress(ctx.get("stack", ""), 400),
                self._compress(ctx.get("components", ""), 800),
            )
        elif step.name == "review":
            return self._review_prompt(ctx)
        elif step.name == "repair":
            return self._repair_prompt(ctx)
        return ""

    def _analyst_system(self) -> str:
        return """You are a senior systems analyst. Analyze the given task deeply.
Return a structured JSON analysis with:
{
  "domain": "string",
  "complexity": "trivial|simple|moderate|complex|extreme",
  "scale": "small|medium|large|enterprise",
  "key_challenges": ["list"],
  "non_functional_requirements": ["list"],
  "estimated_team_size": number,
  "estimated_timeline_weeks": number,
  "risk_factors": [{"risk": "...", "probability": "low|medium|high", "impact": "low|medium|high", "mitigation": "..."}],
  "success_criteria": ["list"],
  "similar_projects": ["list"]
}
Return ONLY valid JSON, no markdown blocks."""

    def _architect_system(self) -> str:
        return """You are a principal software architect with 15+ years of experience.
Design the optimal architecture for the given task.
Return structured JSON:
{
  "pattern": {"name": "...", "description": "...", "pros": [...], "cons": [...], "when_to_use": "..."},
  "high_level_design": "...",
  "components": [{"name": "...", "type": "service|database|cache|queue|gateway|worker", "responsibility": "...", "tech": "...", "interfaces": [...]}],
  "communication": {"sync": "REST|gRPC|GraphQL", "async": "Kafka|RabbitMQ|Redis Streams", "patterns": ["CQRS", "Event Sourcing", "Saga"]},
  "data_flow": "...",
  "scalability_strategy": "...",
  "trade_offs": [{"decision": "...", "rationale": "...", "alternatives_rejected": [...]}]
}
Return ONLY valid JSON."""

    def _stack_system(self) -> str:
        return """You are a technology advisor. Select the optimal tech stack.
Consider: maturity, ecosystem, team expertise, performance, cost, long-term maintenance.
Return structured JSON:
{
  "languages": [{"name": "...", "version": "...", "why": "..."}],
  "frameworks": [{"name": "...", "version": "...", "why": "...", "alternatives_considered": [...]}],
  "databases": [{"name": "...", "type": "relational|document|cache|search|time-series", "why": "...", "use_case": "..."}],
  "infrastructure": [{"tool": "...", "purpose": "..."}],
  "monitoring": [{"tool": "...", "purpose": "..."}],
  "development_tools": [{"tool": "...", "purpose": "..."}],
  "total_monthly_cost_estimate": "...",
  "learning_curve": "low|medium|high",
  "ecosystem_maturity": "experimental|growing|mature|enterprise"
}
Return ONLY valid JSON."""

    def _component_system(self) -> str:
        return """You are a software designer. Design detailed component specifications.
Return structured JSON:
{
  "components": [
    {
      "name": "...", "type": "...", "description": "...",
      "interfaces": [{"method": "...", "input": "...", "output": "...", "description": "..."}],
      "dependencies": ["..."],
      "internal_modules": [{"name": "...", "responsibility": "..."}],
      "error_handling": "...", "logging_strategy": "..."
    }
  ],
  "dependency_graph": "...",
  "shared_libraries": ["..."]
}
Return ONLY valid JSON."""

    def _data_model_system(self) -> str:
        return """You are a database architect. Design the data model.
Return structured JSON:
{
  "database": "PostgreSQL|MongoDB|etc",
  "tables": [
    {
      "name": "...", "description": "...",
      "columns": [{"name": "...", "type": "...", "constraints": "...", "description": "..."}],
      "indexes": [{"columns": [...], "type": "btree|hash|gin|gist", "purpose": "..."}],
      "partitioning": {"strategy": "range|hash|list", "key": "...", "reason": "..."}
    }
  ],
  "relationships": [{"from": "...", "to": "...", "type": "one-to-one|one-to-many|many-to-many", "foreign_key": "..."}],
  "migrations_strategy": "...",
  "seed_data_strategy": "..."
}
Return ONLY valid JSON."""

    def _api_system(self) -> str:
        return """You are an API designer. Design RESTful or GraphQL APIs.
Return structured JSON:
{
  "api_style": "REST|GraphQL|gRPC",
  "base_url": "/api/v1",
  "authentication": "JWT|OAuth2|API_KEY",
  "endpoints": [
    {
      "method": "GET|POST|PUT|DELETE|PATCH", "path": "/resource",
      "description": "...", "request_body": {"field": "type"}, "response": {"field": "type"},
      "status_codes": [200, 201, 400, 404, 500],
      "rate_limit": "...", "permissions": ["..."]
    }
  ],
  "error_response_format": {"error": "string", "code": "string", "details": {}},
  "pagination": "cursor|offset|keyset",
  "versioning_strategy": "..."
}
Return ONLY valid JSON."""

    def _codegen_system(self) -> str:
        return """You are a senior developer. Generate PRODUCTION-READY code.
Write complete, working code. No placeholders, no TODOs, no pseudocode.
Use proper error handling, type hints, docstrings, logging.

For each file return:
{
  "files": [
    {
      "path": "src/module/file.py",
      "language": "python",
      "content": "full working code here",
      "description": "what this file does"
    }
  ],
  "entry_points": ["main.py"],
  "setup_commands": ["pip install ..."]
}

Code must be:
- Complete and runnable
- Well-typed (type hints)
- Properly structured (classes, functions, modules)
- With error handling
- With logging
- Following clean architecture principles

Return ONLY valid JSON."""

    def _testing_system(self) -> str:
        return """You are a QA architect. Design comprehensive testing strategy.
Return structured JSON:
{
  "testing_pyramid": {"unit": "%", "integration": "%", "e2e": "%"},
  "test_framework": "...",
  "test_files": [{"path": "...", "description": "...", "test_count": number, "key_scenarios": ["..."]}],
  "ci_integration": "...",
  "coverage_target": "90%",
  "performance_testing": "...",
  "security_testing": "..."
}
Return ONLY valid JSON."""

    def _security_system(self) -> str:
        return """You are a security engineer. Perform security analysis and provide hardening measures.
Return structured JSON:
{
  "threat_model": [{"threat": "...", "likelihood": "low|medium|high", "impact": "low|medium|high", "mitigation": "..."}],
  "owasp_checks": [{"category": "...", "status": "pass|needs_attention", "recommendation": "..."}],
  "authentication": {"method": "...", "implementation": "..."},
  "authorization": {"model": "RBAC|ABAC", "implementation": "..."},
  "data_protection": {"encryption_at_rest": "...", "encryption_in_transit": "...", "pii_handling": "..."},
  "secrets_management": "...",
  "dependency_audit": "...",
  "penetration_testing": "..."
}
Return ONLY valid JSON."""

    def _optimization_system(self) -> str:
        return """You are a performance engineer. Design optimization strategy.
Return structured JSON:
{
  "caching": {"strategy": "...", "layers": [{"level": "L1|L2|L3", "tool": "...", "ttl": "...", "invalidation": "..."}]},
  "database_optimization": ["..."],
  "async_processing": {"tool": "...", "patterns": ["..."]},
  "cdn_strategy": "...",
  "lazy_loading": "...",
  "connection_pooling": {"min": ..., "max": ..., "timeout": ...},
  "monitoring_metrics": ["..."],
  "load_testing": {"tool": "...", "targets": [...]}
}
Return ONLY valid JSON."""

    def _deployment_system(self) -> str:
        return """You are a DevOps engineer. Design deployment and infrastructure.
Return structured JSON:
{
  "containerization": {"dockerfile": "...", "multi_stage": true, "base_image": "..."},
  "orchestration": {"platform": "kubernetes|ecs|cloud-run", "config": "..."},
  "ci_cd": {"provider": "github-actions|gitlab-ci|jenkins", "stages": [...]},
  "environment_strategy": {"dev": "...", "staging": "...", "production": "..."},
  "monitoring": {"metrics": "...", "logging": "...", "tracing": "..."},
  "scaling": {"horizontal": "...", "vertical": "...", "auto_scaling_rules": [...]},
  "backup_strategy": "...",
  "disaster_recovery": "..."
}
Return ONLY valid JSON."""

    def _repair_system(self) -> str:
        return """You are a senior debugger. The previous attempt failed with specific errors.
Analyze the errors and propose a repaired plan.
Return structured JSON:
{
  "root_cause": "what went wrong",
  "fixes": [{"step": "...", "action": "...", "rationale": "..."}],
  "revised_architecture": "changes to architecture if needed",
  "risk_of_regression": "low|medium|high"
}
Return ONLY valid JSON."""

    def _review_system(self) -> str:
        return """You are a technical lead performing final review.
Review all decisions and provide overall assessment.
Return structured JSON:
{
  "overall_quality": "1-10",
  "architecture_score": "1-10",
  "code_quality_score": "1-10",
  "security_score": "1-10",
  "performance_score": "1-10",
  "maintainability_score": "1-10",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "improvements": ["..."],
  "production_readiness": "ready|needs-work|major-issues",
  "confidence": "0.0-1.0",
  "final_summary": "one paragraph summary"
}
Return ONLY valid JSON."""

    def _analyst_prompt(
        self,
        task: str,
        tags: list[str],
        reqs: list[dict[str, Any]],
        knowledge: str,
        ontology: str = "",
    ) -> str:
        parts = [f"Task: {task}", f"Tags: {', '.join(tags)}"]
        if reqs:
            parts.append("Requirements:")
            for r in reqs:
                parts.append(
                    f"  - [{r.get('category', 'general')}] {r.get('description', '')} (priority: {r.get('priority', 5)})"
                )
        if ontology:
            parts.append(f"\n[ONTOLOGICAL AXIOMS - DO NOT VIOLATE]\n{ontology[:2000]}")
        if knowledge:
            parts.append(f"\nRelevant knowledge:\n{knowledge[:2000]}")
        return "\n".join(parts)

    def _architect_prompt(self, analysis: str, graph: str) -> str:
        parts = [f"Analysis from previous step:\n{analysis[:3000]}"]
        if graph:
            parts.append(f"\nKnowledge graph suggestions:\n{graph[:1000]}")
        parts.append("\nDesign the optimal architecture. Return ONLY valid JSON.")
        return "\n".join(parts)

    def _stack_prompt(self, architecture: str, tags: list[str]) -> str:
        return f"Architecture design:\n{architecture[:3000]}\n\nTags: {', '.join(tags)}\n\nSelect the optimal tech stack. Return ONLY valid JSON."

    def _component_prompt(self, architecture: str, stack: str) -> str:
        return f"Architecture:\n{architecture[:2000]}\n\nStack:\n{stack[:1000]}\n\nDesign detailed component specifications. Return ONLY valid JSON."

    def _data_model_prompt(self, components: str, stack: str, reqs: list[dict[str, Any]]) -> str:
        return f"Components:\n{components[:2000]}\n\nStack:\n{stack[:500]}\n\nRequirements: {reqs}\n\nDesign the data model. Return ONLY valid JSON."

    def _api_prompt(self, components: str, stack: str, reqs: list[dict[str, Any]]) -> str:
        return f"Components:\n{components[:2000]}\n\nStack:\n{stack[:500]}\n\nDesign the API layer. Return ONLY valid JSON."

    def _codegen_prompt(self, stack: str, api: str, data: str, components: str) -> str:
        return (
            f"Stack:\n{stack[:1000]}\n\n"
            f"API Design:\n{api[:2000]}\n\n"
            f"Data Model:\n{data[:2000]}\n\n"
            f"Components:\n{components[:1000]}\n\n"
            "Generate complete, production-ready code for ALL files. Return ONLY valid JSON."
        )

    def _testing_prompt(self, stack: str, api: str, components: str) -> str:
        return f"Stack:\n{stack[:500]}\n\nAPI:\n{api[:1500]}\n\nComponents:\n{components[:1000]}\n\nDesign testing strategy. Return ONLY valid JSON."

    def _security_prompt(self, stack: str, api: str, components: str) -> str:
        return f"Stack:\n{stack[:500]}\n\nAPI:\n{api[:1500]}\n\nComponents:\n{components[:1000]}\n\nPerform security analysis. Return ONLY valid JSON."

    def _optimization_prompt(self, stack: str, components: str, arch: str) -> str:
        return f"Stack:\n{stack[:500]}\n\nComponents:\n{components[:1000]}\n\nArchitecture:\n{arch[:1000]}\n\nDesign optimization strategy. Return ONLY valid JSON."

    def _deployment_prompt(self, stack: str, components: str) -> str:
        return f"Stack:\n{stack[:500]}\n\nComponents:\n{components[:1000]}\n\nDesign deployment infrastructure. Return ONLY valid JSON."

    def _repair_prompt(self, context: dict) -> str:
        errors = context.get("_reflexion_errors", [])
        error_block = "\n".join(f"  - {e}" for e in errors[-5:]) if errors else "No prior errors"
        analysis = context.get("analysis", "No analysis available")[:2000]
        return (
            f"Previous attempt errors:\n{error_block}\n\n"
            f"Analysis:\n{analysis}\n\n"
            "Analyze these failures and propose a repaired design. Return ONLY valid JSON."
        )

    def _review_prompt(self, context: dict) -> str:
        summary = json.dumps(
            {k: str(v)[:500] for k, v in context.items()}, indent=2, ensure_ascii=False
        )
        return f"Full project context:\n{summary[:4000]}\n\nPerform final review. Return ONLY valid JSON."

    def get_steps(self) -> list[CoTStep]:
        return self._steps

    def get_chain_summary(self) -> list[dict]:
        return [
            {
                "step": n.step,
                "role": n.role,
                "response_preview": n.response[:200],
                "confidence": n.confidence,
            }
            for n in self.chain
        ]
