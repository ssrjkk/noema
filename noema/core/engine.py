"""NoemaEngine — LLM-first orchestrator for solution generation.

Architecture:
- :class:`NoemaEngine` owns the orchestration graph: knowledge stores, kernels,
  Chain-of-Thought (DAG) executor, neurosymbolic engine, sandbox, memory, and
  module registry. Components are lazily initialized in :meth:`NoemaEngine.initialize`.
- ``think()`` is the single entry point; it may take the neurosymbolic fast path
  or fall back to the DAG-based Chain-of-Thought path with Reflexion retries.

Concurrency contract:
- All I/O is awaited; the only blocking operation is sync ``MemoryStore.save``,
  which is offloaded with :func:`asyncio.to_thread` in :meth:`_persist_memory`.
- ``WorkerPool``/``WorkerHierarchy`` bound parallelism; the event loop is never
  blocked by this module.

Complexity:
- ``think()``: ``O(S · L)`` LLM calls for ``S`` selected CoT steps across ``L``
  DAG levels, plus ``O(K)`` kernel calls in degraded mode (K = kernels).
- ``initialize()``: ``O(M)`` component setup plus ``O(P)`` plugin registrations.
- Assembly/parsing helpers: ``O(N)`` in the number of reasoning artifacts.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from noema.agents.orchestrator import AgentOrchestrator
from noema.budget.token_budget import TokenBudget
from noema.config.settings import SandboxSettings, get_settings
from noema.context import get_tenant_id, reset_tenant_id, set_tenant_id
from noema.core.chain_of_thought import ChainOfThought, ReflexionState
from noema.core.checkpoint import CheckpointStore, DAGCheckpoint
from noema.core.types import (
    ArchitecturePattern,
    CodeBlock,
    JudgeError,
    SandboxValidationError,
    Solution,
    SolutionQuality,
    Task,
    TechStack,
    ThinkTimeoutError,
    ThoughtProcess,
)
from noema.discovery.keys import KeyDiscovery
from noema.evolution.engine import EvolutionEngine
from noema.feedback.store import FeedbackEntry, FeedbackStore
from noema.healer.engine import SelfHealer
from noema.ingestion.loader import KnowledgeLoader
from noema.judge import evaluate_solution
from noema.knowledge.graph import KnowledgeGraph
from noema.knowledge.store import KnowledgeStore
from noema.llm.providers import BaseLLMProvider, create_llm_provider
from noema.logging import get_logger
from noema.memory.store import MemoryStore
from noema.modules.registry import get_registry
from noema.neurosymbolic import (
    MaxRefinementsExceededError,
    NeuroSymbolicEngine,
)
from noema.ontology import OntologyGraph
from noema.plugins.manager import PluginManager
from noema.routing.model_router import ModelRouter
from noema.sandbox.engine import SandboxConfig, SandboxEngine, SandboxResult
from noema.scaffolder.generator import ProjectScaffolder
from noema.tracing.tracer import Tracer, get_tracer
from noema.utils.json_utils import extract_fenced_json
from noema.workers.hierarchy import WorkerHierarchy
from noema.workers.pool import WorkerPool

if TYPE_CHECKING:
    from noema.kernels.base import BaseKernel
    from noema.neurosymbolic.engine import ThinkEvent

log = get_logger(__name__)

StreamCallback = Callable[[str, str, int, int], Coroutine[Any, Any, None] | None]


def _parse_memory_mb(value: str) -> int:
    """Parse a Docker-style memory limit (``256m``, ``1g``, ``512``) into MB."""
    match = re.fullmatch(r"\s*(\d+)\s*([kmg]?)\s*", value.lower())
    if not match:
        return 256
    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "g":
        return max(1, amount * 1024)
    if unit == "k":
        return max(1, amount // 1024)
    return max(1, amount)


def _sandbox_config_from_settings(sb: SandboxSettings) -> SandboxConfig:
    """Map the pydantic ``SandboxSettings`` onto the runtime :class:`SandboxConfig`.

    Removes the historical hard-coded sandbox configuration in
    :class:`NoemaEngine` so every cap (memory, CPUs, image, network isolation,
    timeout) flows from ``NOEMA_SANDBOX_*`` env vars / ``settings.yaml``.
    """
    return SandboxConfig(
        enabled=sb.enabled,
        timeout=sb.timeout,
        max_memory_mb=_parse_memory_mb(sb.max_memory),
        max_cpus=sb.max_cpus,
        network_isolation=sb.network_disabled,
        docker_image=sb.docker_image,
    )


class NoemaEngine:
    """Orchestrator for the full solution-generation pipeline.

    Responsibilities: kernel registry, Chain-of-Thought execution, Reflexion
    retries with judge feedback, neurosymbolic fast path, memory/knowledge
    persistence, sandbox validation, scaffolding, and module dispatch.
    """

    def __init__(
        self,
        worker_count: int = 4,
        knowledge_path: str | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        config: dict[str, Any] | None = None,
        project_root: str = ".",
        tenant_id: str = "",
        token_budget: TokenBudget | None = None,
        model_router: ModelRouter | None = None,
    ) -> None:
        self.config = config or {}
        self.project_root = project_root
        # tenant_id from context var by default; explicit param wins. The
        # context var itself is NOT mutated here: think() scopes it per call.
        self._tenant_id = tenant_id or get_tenant_id()
        self.kernels: dict[str, BaseKernel] = {}
        self.worker_pool = WorkerPool(max_workers=worker_count)
        self.worker_hierarchy = WorkerHierarchy(max_depth=10, max_concurrent=50)
        self.orchestrator = AgentOrchestrator()
        self.knowledge = KnowledgeStore(persist_path=knowledge_path)
        self.knowledge_graph = KnowledgeGraph()
        self.ontology = OntologyGraph()
        self.llm: BaseLLMProvider = create_llm_provider(llm_provider, llm_model)
        self.cot: ChainOfThought | None = None
        self.feedback = FeedbackStore()
        self.plugins = PluginManager()
        self.scaffolder: ProjectScaffolder | None = None
        self.memory = MemoryStore(persist_dir=f"{project_root}/.noema/memory/{self._tenant_id}")
        self.discovery = KeyDiscovery(project_root=project_root)
        self.model_router = model_router or ModelRouter()
        self.token_budget = token_budget or TokenBudget()
        self.evolution = EvolutionEngine(llm_provider=self.llm, project_root=project_root)
        self.healer = SelfHealer()
        self.ingestion = KnowledgeLoader(knowledge_store=self.knowledge)
        self.modules = get_registry()
        self._settings = get_settings()
        self.sandbox = SandboxEngine(_sandbox_config_from_settings(self._settings.sandbox))
        self.healer.llm = self.llm
        self.healer.ontology = self.ontology
        self.healer.ontology_persist_path = self._settings.ontology_persist_path
        self.tracer: Tracer = get_tracer()
        self.checkpointer = CheckpointStore(persist_dir=f"{project_root}/.noema/checkpoints")
        ns_cfg = self._settings.neurosymbolic
        self.neurosymbolic = (
            NeuroSymbolicEngine(
                max_refinement_attempts=ns_cfg.max_refinement_attempts,
                verification_timeout=ns_cfg.verification_timeout,
                enable_evolution=ns_cfg.evolution_enabled,
                enable_causal=ns_cfg.causal_enabled,
                max_counterfactuals=ns_cfg.causal_max_counterfactuals,
            )
            if ns_cfg.enabled
            else None
        )
        self._initialized = False
        self._on_step_start: StreamCallback | None = None
        self._on_step_end: StreamCallback | None = None

    async def initialize(self) -> None:
        """Bring up every component once (idempotent).

        Loads knowledge/feedback, starts the worker pool and orchestrator, registers
        default + plugin kernels, runs resource discovery off-thread, and starts the
        neurosymbolic engine when enabled.

        Complexity: ``O(M + P)`` for M managed components and P plugin artifacts.
        """
        if self._initialized:
            return
        log.info("Инициализация NoemaEngine (LLM-first)...")
        await self.knowledge.load()
        await self.feedback.load()
        ontology_path = self._settings.ontology_persist_path
        try:
            loaded = OntologyGraph.load(ontology_path)
            if loaded.stats()["entities"]:
                self.ontology = loaded
                self.healer.ontology = loaded
                log.info(
                    "ontology_loaded",
                    path=str(ontology_path),
                    entities=loaded.stats()["entities"],
                    relations=loaded.stats()["relations"],
                )
        except Exception as e:  # noqa: BLE001 - a corrupt ontology must not block startup
            log.warning("ontology_load_failed", path=str(ontology_path), error=str(e))
        await self.worker_pool.start()
        await self.orchestrator.initialize()
        self._register_default_kernels()
        plugin_kernels = self.plugins.get_all_kernels()
        for kernel in plugin_kernels:
            self.register_kernel(kernel)
        plugin_agents = self.plugins.get_all_agents()
        for agent in plugin_agents:
            self.orchestrator.register_agent(agent)

        # Discover available keys and resources
        try:
            discovery_result = await asyncio.to_thread(self.discovery.discover_all)
            providers_found = discovery_result.get("providers_available", [])
            resources_found = discovery_result.get("resources", [])
            log.info(
                f"Discovery: {len(providers_found)} providers, "
                f"{len(resources_found)} resources detected"
            )
        except Exception as e:
            log.warning(f"Discovery failed: {e}")

        if self.neurosymbolic:
            await self.neurosymbolic.start()
            log.info("neurosymbolic_engine_started")

        self._initialized = True
        log.info(
            f"NoemaEngine ready | LLM: {self.llm.name} ({self.llm.model_name}) | "
            f"Kernels: {len(self.kernels)} | Agents: {len(self.orchestrator.agents)} | "
            f"Modules: {len(self.modules.modules)} | "
            f"Graph: {self.knowledge_graph.get_stats()['total_nodes']} nodes | "
            f"Memory: {self.memory.stats()['episodic_count']} episodes"
        )

    async def shutdown(self) -> None:
        """Tear down workers, orchestrator, and persist all stores.

        Every step tolerates individual failures so shutdown always completes.
        """
        results = await asyncio.gather(
            self.worker_pool.shutdown(),
            self.orchestrator.shutdown(),
            self.knowledge.persist(),
            self.feedback.persist(),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                log.warning("shutdown_error", error=str(r))
        try:
            await self._persist_memory()
        except Exception as e:
            log.warning("memory_save_error", error=str(e))

        if self.neurosymbolic:
            try:
                await self.neurosymbolic.stop()
            except Exception as e:
                log.warning("neurosymbolic_stop_error", error=str(e))

        self._initialized = False

    async def _persist_memory(self) -> None:
        """Persist memory, supporting both sync (file) and async (Postgres) backends.

        The base :class:`MemoryStore.save` is synchronous; the
        ``PostgresMemoryStore`` subclass overrides it with a coroutine. The
        coroutine-fact is inspected on the bound method so the call itself is
        never made twice and both implementations are handled uniformly.

        Blocking contract: the sync branch is offloaded to a worker thread via
        :func:`asyncio.to_thread` so the event loop is never blocked.
        """
        save = self.memory.save
        if inspect.iscoroutinefunction(save):
            await save()
        else:
            await asyncio.to_thread(save)

    def register_kernel(self, kernel: BaseKernel) -> None:
        self.kernels[kernel.name] = kernel

    def on_step_start(self, callback: StreamCallback) -> None:
        self._on_step_start = callback

    def on_step_end(self, callback: StreamCallback) -> None:
        self._on_step_end = callback

    async def think(
        self,
        task: Task,
        on_step_start: StreamCallback | None = None,
        on_step_end: StreamCallback | None = None,
    ) -> tuple[Solution, ThoughtProcess]:
        """Run the full solution pipeline for a task and return (solution, thought).

        Flow:
        1. Validate the task inputs (zero-trust): reject resource-exhaustion
           attacks (oversized description / excessive tags). Blank titles are
           tolerated and handled by the fallback path.
        2. Try the neurosymbolic fast path when enabled.
        3. Fall back to DAG-based Chain-of-Thought with Reflexion + judge loop.
        4. Assemble a :class:`Solution`, record memory, clear checkpoints.

        Complexity: ``O(S · L)`` LLM calls (S steps across L DAG levels) plus
        ``O(P)`` memory searches per attempt, bounded by 3 Reflexion attempts.

        Tenant semantics: an explicitly set caller tenant (e.g. the API
        request's ``x-tenant-id``) wins inside the call; the engine's own
        tenant is only a fallback for callers on the default context. The
        effective tenant is restored afterwards, so concurrent calls can
        never clobber each other's context. The call is also bounded by
        ``think_timeout_seconds`` (fail-closed, checkpoints survive).
        """
        caller_tenant = get_tenant_id()
        effective_tenant = caller_tenant if caller_tenant != "default" else self._tenant_id
        token = set_tenant_id(effective_tenant)
        try:
            timeout = self._settings.think_timeout_seconds
            try:
                return await asyncio.wait_for(
                    self._think_impl(task, on_step_start, on_step_end), timeout=timeout
                )
            except TimeoutError:
                raise ThinkTimeoutError(
                    f"think() exceeded {timeout}s (fail-closed; checkpoints remain for resume)"
                ) from None
        finally:
            reset_tenant_id(token)

    async def _think_impl(
        self,
        task: Task,
        on_step_start: StreamCallback | None = None,
        on_step_end: StreamCallback | None = None,
    ) -> tuple[Solution, ThoughtProcess]:
        """Implementation of :meth:`think` (tenant already scoped by the caller)."""
        if task.description and len(task.description) > 100_000:
            raise ValueError(f"Task description too large: {len(task.description)} chars")
        if len(task.tags) > 100:
            raise ValueError(f"Too many tags: {len(task.tags)}")

        if not self._initialized:
            await self.initialize()

        # Check for existing checkpoint (resumable execution). The checkpoint
        # file is kept until the run completes: if the process crashes mid-run
        # the recorded progress survives, so it can be resumed on retry.
        ckpt = await self.checkpointer.load(task.id, get_tenant_id())
        resume_context: dict[str, str] | None = None
        if ckpt:
            log.info("resuming_from_checkpoint", task=task.id, steps=len(ckpt.completed_steps))
            resume_context = dict(ckpt.step_results or {})

        thought = ThoughtProcess(task_id=task.id)
        t0 = time.perf_counter()

        log.info(f"[Noema] Thinking about: {task.title} (LLM: {self.llm.name})")

        trace_span = self.tracer.start_span(
            f"think.{task.title[:30]}",
            kind="pipeline",
            attributes={
                "task_id": task.id,
                "complexity": task.complexity.value,
                "tenant_id": self._tenant_id,
            },
        )

        # ── NeuroSymbolic path ────────────────────────────────────────
        if self.neurosymbolic:
            try:
                ns_task = self._prepare_neurosymbolic_task(task)
                final_solution = None
                completed_result = None
                async for result in self.neurosymbolic.think(ns_task):
                    if result["stage"] == "completed":
                        final_solution = result["solution"]
                        completed_result = result
                        break
                    elif result["stage"] == "failed":
                        if self._settings.neurosymbolic.fallback_to_cot:
                            log.warning("neurosymbolic_failed_fallback_to_cot", task_id=task.id)
                            break
                        raise MaxRefinementsExceededError(
                            f"NeuroSymbolic failed after {result.get('attempts', 0)} attempts"
                        )

                if final_solution is not None:
                    solution = await self._convert_neurosymbolic_to_solution(
                        final_solution, task, completed_result
                    )
                    if completed_result and completed_result.get("causal_analysis"):
                        solution.metadata["causal_analysis"] = completed_result["causal_analysis"]
                        solution.metadata["causal_metrics"] = completed_result.get(
                            "causal_metrics", {}
                        )
                    await self._finalize_solution(task, solution, thought, t0, trace_span)
                    return solution, thought
                log.info("neurosymbolic_fallback_to_cot_proceeding")
            except SandboxValidationError:
                raise
            except MaxRefinementsExceededError:
                raise
            except Exception as e:
                log.error("neurosymbolic_think_error", error=str(e))
                if not self._settings.neurosymbolic.fallback_to_cot:
                    raise

        past_episodes = self.memory.search_episodes(f"{task.title} {task.description}", limit=5)

        knowledge_context = await self._gather_knowledge_context(task)
        graph_context = self._gather_graph_context(task)
        ontology_context = self._gather_ontology_context(task)

        if past_episodes:
            memory_context = "\n".join(
                f"[Past] {ep.task_description[:100]} → {ep.outcome}" for ep in past_episodes
            )
            knowledge_context = f"{knowledge_context}\n\n{memory_context}"

        self.cot = ChainOfThought(
            self.llm,
            on_step_start=on_step_start or self._on_step_start,
            on_step_end=on_step_end or self._on_step_end,
            model_router=self.model_router,
            token_budget=self.token_budget,
        )

        reflexion_state: ReflexionState | None = None
        judge_passed_flag = False
        first_failed_code = ""

        if self.llm.name != "fallback":
            for reflexion_attempt in range(1, 4):
                reasoning_result = await self.healer.execute_with_healing(
                    self.cot.reason,
                    task_description=f"{task.title}: {task.description}",
                    task_tags=task.tags,
                    requirements=[r.model_dump() for r in task.requirements],
                    knowledge_context=knowledge_context,
                    graph_context=graph_context,
                    ontology_context=ontology_context,
                    complexity=task.complexity.value,
                    reflexion_errors=(
                        [f"Judge: {reflexion_state.error_summary}"] if reflexion_state else None
                    ),
                    reflexion_attempt=reflexion_attempt,
                    resume_context=resume_context if reflexion_attempt == 1 else None,
                    fallback={"raw": "Chain-of-Thought reasoning failed, using minimal fallback"},
                )

                for node in self.cot.chain:
                    thought.add_step(
                        node.role, node.prompt[:200], node.response[:200], node.confidence
                    )

                solution = await self._assemble_solution_from_reasoning(task, reasoning_result)
                solution = self._evaluate_quality(solution, thought)
                if reflexion_attempt == 1:
                    first_failed_code = "\n\n".join(
                        b.content
                        for b in solution.code_blocks
                        if b.language.lower() in ("python", "py")
                    )

                try:
                    judge_span = self.tracer.start_span(
                        "judge.evaluate",
                        kind="judge",
                        attributes={"task_id": task.id, "reflexion_attempt": reflexion_attempt},
                    )
                    verdict = await evaluate_solution(
                        self.llm,
                        solution,
                        f"{task.title}: {task.description}",
                        task.tags,
                    )
                    self.tracer.end_span(judge_span, status="ok")
                    solution.metadata["judge_passed"] = verdict.passed
                    solution.metadata["judge_reflexion_attempt"] = reflexion_attempt

                    if verdict.passed:
                        log.info(
                            "judge_passed",
                            attempt=reflexion_attempt,
                            overall=verdict.scores.overall,
                        )
                        judge_passed_flag = True
                        break

                    reflexion_state = ReflexionState(
                        attempt=reflexion_attempt + 1,
                        error_summary="; ".join(verdict.weaknesses[:3]),
                        last_errors=verdict.weaknesses,
                    )
                    log.warning(
                        "judge_low_score_reflexion",
                        attempt=reflexion_attempt,
                        overall=verdict.scores.overall,
                        weaknesses=verdict.weaknesses,
                    )
                    if reflexion_attempt < 3:
                        # Save checkpoint before retry
                        await self.checkpointer.save(
                            DAGCheckpoint(
                                task_id=task.id,
                                tenant_id=get_tenant_id(),
                                session_id=self.tracer.current_span_id,
                                attempt=reflexion_attempt + 1,
                                completed_steps=[n.name for n in self.cot.chain if n.name],
                                step_results={n.name: n.response for n in self.cot.chain if n.name},
                                token_budget_used=getattr(self.token_budget, "_used", 0)
                                if self.token_budget
                                else 0,
                                context={"reflexion_state": reflexion_state.error_summary},
                            )
                        )
                        self.cot = ChainOfThought(
                            self.llm,
                            on_step_start=on_step_start or self._on_step_start,
                            on_step_end=on_step_end or self._on_step_end,
                            model_router=self.model_router,
                            token_budget=self.token_budget,
                        )
                        continue
                except Exception as e:
                    # Fail closed: a crashed judge must not silently return
                    # an unverified solution.
                    log.warning(f"Judge evaluation failed: {e}")
                    solution.metadata["judge_passed"] = False
                    solution.metadata["judge_error"] = str(e)
                    break

            if reflexion_attempt > 1 and judge_passed_flag and first_failed_code:
                fixed_code = "\n\n".join(
                    b.content
                    for b in solution.code_blocks
                    if b.language.lower() in ("python", "py")
                )
                await self.healer._propose_ontological_axiom(
                    failed_code=first_failed_code,
                    fixed_code=fixed_code,
                    error_summary=(
                        reflexion_state.error_summary
                        if reflexion_state and reflexion_state.attempt > 1
                        else ""
                    ),
                    task_id=task.id,
                )

        else:
            reasoning_result = await self._kernel_based_reasoning(task, thought)
            for node in self.cot.chain if self.cot else []:
                thought.add_step(node.role, node.prompt[:200], node.response[:200], node.confidence)
            solution = await self._assemble_solution_from_reasoning(task, reasoning_result)
            solution = self._evaluate_quality(solution, thought)

        thought.duration_ms = (time.perf_counter() - t0) * 1000
        thought.final_reasoning = self._build_reasoning(thought, solution)

        await self._finalize_solution(task, solution, thought, t0, trace_span)
        return solution, thought

    async def _verify_think_solution(self, solution: Solution) -> dict[str, Any]:
        """Run the think() verification gate on a generated solution.

        Enabled by ``NOEMA_SANDBOX__VERIFY_THINK``. Every Python code block is
        checked with the pure-AST static pass (syntax + import hygiene +
        undefined-name call graph) — the deterministic, fail-closed half of the
        sandbox contract. The full sandbox run is intentionally *not* executed
        here: it stays where execution is already wired (experiments runner and
        the autonomy fixer), so this gate is fast and works without Docker.

        Returns a verdict dict merged into ``solution.metadata["sandbox"]``.
        Never raises unless ``verify_think_enforce`` is set (fail-closed).
        """
        verdict: dict[str, Any] = {
            "enabled": bool(self._settings.sandbox.verify_think),
            "passed": True,
            "files": [],
            "summary": "",
        }
        if not verdict["enabled"]:
            return verdict

        python_blocks = [b for b in solution.code_blocks if b.language.lower() in ("python", "py")]
        if not python_blocks:
            verdict["summary"] = "no python code blocks to verify"
            return verdict

        for block in python_blocks:
            vr = await self.sandbox.validate_code_block(
                code=block.content,
                language="python",
                filename=block.filename,
            )
            record = {
                "file": block.filename,
                "ast_valid": vr.ast_valid,
                "static_passed": vr.static_passed,
                "ast_errors": vr.ast_errors,
                "static_issues": vr.static_issues,
            }
            verdict["files"].append(record)
            if not (vr.ast_valid and vr.static_passed):
                verdict["passed"] = False

        passed = sum(1 for f in verdict["files"] if f["ast_valid"] and f["static_passed"])
        total = len(verdict["files"])
        verdict["summary"] = f"{passed}/{total} python files passed the static gate"
        log.info(
            "think_sandbox_gate",
            passed=passed,
            total=total,
            enforce=bool(self._settings.sandbox.verify_think_enforce),
        )
        return verdict

    async def _finalize_solution(
        self,
        task: Task,
        solution: Solution,
        thought: ThoughtProcess,
        t0: float,
        trace_span: Any,
    ) -> None:
        """Shared post-processing for every think() exit path.

        Both the neurosymbolic fast path and the Chain-of-Thought path converge
        here so a solution can never leak checkpoints, skip memory recording, or
        bypass the sandbox gate (previously the NS path returned early).

        Order matters: the sandbox gate runs first so an enforced rejection
        leaves the checkpoint in place and the run stays resumable.
        """
        thought.duration_ms = (time.perf_counter() - t0) * 1000
        thought.final_reasoning = self._build_reasoning(thought, solution)

        sandbox_verdict = await self._verify_think_solution(solution)
        if sandbox_verdict.get("enabled"):
            solution.metadata["sandbox"] = sandbox_verdict
            if not sandbox_verdict["passed"] and self._settings.sandbox.verify_think_enforce:
                raise SandboxValidationError(sandbox_verdict["summary"])

        # Judge gate (fail-closed when enforced). Only solutions that went
        # through the judge loop carry the verdict; the neurosymbolic fast
        # path is unaffected by design.
        judge_passed = solution.metadata.get("judge_passed")
        if judge_passed is False and self._settings.judge.enforce:
            raise JudgeError(
                "Solution failed the judge gate (enforce=true); refusing to "
                "return an unverified artifact"
            )

        self.memory.record_episode(
            task_description=f"{task.title}: {task.description}",
            solution_summary=solution.summary[:500],
            tech_stack=", ".join(solution.stack.languages + solution.stack.frameworks)
            if solution.stack
            else "",
            outcome="success"
            if solution.quality
            in (SolutionQuality.MASTERPIECE, SolutionQuality.EXCELLENT, SolutionQuality.GOOD)
            else "partial",
            duration_seconds=thought.duration_ms / 1000,
            tags=task.tags,
            context={"quality": solution.quality.value, "confidence": solution.confidence},
        )

        # Clear checkpoint on success
        await self.checkpointer.delete(task.id, get_tenant_id())

        if solution.quality in (SolutionQuality.MASTERPIECE, SolutionQuality.EXCELLENT):
            self.memory.store_procedure(
                procedure_name=f"solution_{task.title[:30].replace(' ', '_')}",
                steps=[step.kernel for step in thought.steps],
                tags=task.tags,
            )

        log.info(
            f"[Noema] Solution generated in {thought.duration_ms:.0f}ms | "
            f"Quality: {solution.quality.value} | "
            f"Confidence: {thought.avg_confidence:.0%} | "
            f"LLM: {self.llm.name} | "
            f"Thought steps: {len(thought.steps)} | "
            f"Memory: {self.memory.stats()['episodic_count']} episodes"
        )

        trace_span.attributes.update(
            {
                "quality": solution.quality.value,
                "confidence": solution.confidence,
                "duration_ms": round(thought.duration_ms, 1),
                "steps": len(thought.steps),
                "tokens": self.tracer.get_stats().get("total_tokens", 0),
            }
        )
        self.tracer.end_span(trace_span, status="ok")

    def _prepare_neurosymbolic_task(self, task: Task) -> dict[str, Any]:
        """Project a :class:`Task` onto the strict neurosymbolic input contract.

        Domain :class:`Requirement` objects carry ``category``/``description``/
        ``priority``/``constraints``; the symbolic engine consumes numeric
        requirements with ``name`` + optional ``min``/``max``. Category is used
        as the variable name and any constraint text (``>= N``, ``in [a, b]``,
        ``min=N`` …) is forwarded so bounds can be extracted downstream.

        Complexity: ``O(R)`` for R task requirements.
        """
        requirements = []
        for index, requirement in enumerate(task.requirements):
            req = requirement.model_dump() if hasattr(requirement, "model_dump") else {}
            requirements.append(
                {
                    "name": req.get("category") or f"req_{index}",
                    "type": "numeric",
                    "min": req.get("min"),
                    "max": req.get("max"),
                    "constraints": list(req.get("constraints") or []),
                    "description": req.get("description") or "",
                    "priority": int(req.get("priority", 1)),
                }
            )
        return {
            "requirements": requirements,
            "constraints": [],
            "goals": [task.title, task.description],
            "variables": {},
        }

    async def _convert_neurosymbolic_to_solution(
        self, ns_solution: dict, task: Task, ns_result: ThinkEvent | None = None
    ) -> Solution:
        """Map a neurosymbolic engine result onto a :class:`Solution`.

        Only well-formed fields are copied; malformed entries are skipped, so
        untrusted engine output cannot corrupt the solution (zero-trust).

        Complexity: ``O(F)`` for up to 10 code blocks F.
        """
        solution = Solution(
            task_id=task.id,
            title=f"NeuroSymbolic Solution: {task.title}",
            summary=ns_solution.get("summary", ns_solution.get("description", "")),
        )
        if "architecture" in ns_solution:
            arch = ns_solution["architecture"]
            if isinstance(arch, dict):
                solution.architecture = ArchitecturePattern(
                    name=arch.get("name", "NeuroSymbolic Design"),
                    description=arch.get("description", ""),
                    pros=arch.get("pros", []),
                    cons=arch.get("cons", []),
                )
        if "stack" in ns_solution:
            stack = ns_solution["stack"]
            if isinstance(stack, dict):
                solution.stack = TechStack(
                    languages=stack.get("languages", []),
                    frameworks=stack.get("frameworks", []),
                    databases=stack.get("databases", []),
                    infrastructure=stack.get("infrastructure", []),
                )
        code_blocks_data = ns_solution.get("code_blocks", ns_solution.get("files", []))
        if isinstance(code_blocks_data, list):
            for cb in code_blocks_data[:10]:
                if isinstance(cb, dict) and cb.get("content"):
                    path = cb.get("path") or cb.get("filename") or "main.py"
                    ext = path.rsplit(".", 1)[-1] if "." in path else "py"
                    lang_map = {"py": "python", "ts": "typescript", "js": "javascript"}
                    solution.code_blocks.append(
                        CodeBlock(
                            filename=path,
                            language=lang_map.get(ext, ext),
                            content=cb["content"],
                            description=cb.get("description", ""),
                        )
                    )
        return solution

    async def _gather_knowledge_context(self, task: Task) -> str:
        """Gather relevant context from the knowledge store.

        Complexity: ``O(K log K)`` for the store search plus ``O(K)`` formatting
        of the top-K results.
        """
        query = f"{task.title} {task.description} {' '.join(task.tags)}"
        results = await self.knowledge.search(query, top_k=5)
        parts = []
        for r in results:
            title = r.get("title", r.get("name", ""))
            content = r.get("content", r.get("description", ""))
            parts.append(f"[{r.get('type', 'knowledge')}] {title}: {content[:300]}")
        return "\n".join(parts)

    def _gather_graph_context(self, task: Task) -> str:
        """Gather architecture suggestions from the knowledge graph.

        Complexity: ``O(R)`` for up to 15 graph recommendations R.
        """
        recs = self.knowledge_graph.suggest_architecture(task.tags)
        parts = []
        for comp in recs.get("components", [])[:15]:
            parts.append(f"{comp['from']} -> {comp['to']} ({comp['relationship']})")
        return "\n".join(parts) if parts else ""

    def _gather_ontology_context(self, task: Task) -> str:
        """Gather ontological axioms relevant to the task (symbolic RAG).

        Entities are matched deterministically: normalized task tokens
        (title + description + tags) are matched against entity names, then the
        directed subgraph around matched roots is rendered as imperative rules.

        This is prompt-level guidance, not a mathematical guarantee: the axioms
        are only as strong as the authored graph.

        Complexity: ``O(T + V + E)`` for T task tokens and the subgraph BFS.
        """
        if not self.ontology.stats()["entities"]:
            return ""
        text = f"{task.title} {task.description} {' '.join(task.tags)}"
        tokens = set(re.findall(r"[A-Za-z0-9_-]{2,}", text.lower()))
        roots: list[str] = []
        for entity in self.ontology.entities():
            name = entity.name.lower()
            if name in tokens or any(name in t for t in tokens if len(t) > 3):
                roots.append(entity.name)
                if len(roots) >= 10:
                    break
        if not roots:
            return ""
        subgraph = self.ontology.get_subgraph(roots, depth=2)
        if not subgraph.relations():
            return ""
        rules = subgraph.to_rules(limit=30)
        log.info(
            "ontology_axioms_injected",
            roots=roots,
            rules=len(rules.splitlines()),
            task_id=task.id,
        )
        return rules

    async def _kernel_based_reasoning(self, task: Task, thought: ThoughtProcess) -> dict[str, Any]:
        """Degraded-mode reasoning through kernels when the LLM is unavailable.

        Complexity: ``O(K)`` kernel executions for a fixed kernel set K (<= 5).
        """
        analysis = await self._run_kernel("analysis", task, thought, phase="full")
        stack = await self._select_stack(task, thought)
        architecture = await self._run_kernel("architecture", task, thought, phase="design")
        optimizations = await self._run_kernel("optimization", task, thought)
        security = await self._run_kernel("security", task, thought)
        return {
            "analysis": analysis,
            "architecture": architecture,
            "stack": stack.model_dump() if stack else {},
            "optimization": optimizations,
            "security": security,
            "code": {"files": []},
        }

    async def _assemble_solution_from_reasoning(
        self, task: Task, reasoning: dict[str, Any]
    ) -> Solution:
        """Assemble a :class:`Solution` from Chain-of-Thought reasoning dict.

        All nested artifacts are parsed defensively; malformed fields degrade
        gracefully instead of raising (zero-trust).

        Complexity: ``O(N)`` over the number of reasoning artifacts.
        """
        solution = Solution(
            task_id=task.id,
            title=f"Solution: {task.title}",
            summary=self._extract_summary(reasoning),
        )

        # Stack
        stack_data = self._safe_parse(reasoning.get("stack", "{}"))
        if isinstance(stack_data, dict):
            solution.stack = TechStack(
                languages=[
                    lang.get("name", "") if isinstance(lang, dict) else str(lang)
                    for lang in stack_data.get("languages", [])
                ],
                frameworks=[
                    f.get("name", "") if isinstance(f, dict) else str(f)
                    for f in stack_data.get("frameworks", [])
                ],
                databases=[
                    d.get("name", "") if isinstance(d, dict) else str(d)
                    for d in stack_data.get("databases", [])
                ],
                infrastructure=[
                    i.get("tool", "") if isinstance(i, dict) else str(i)
                    for i in stack_data.get("infrastructure", [])
                ],
            )

        # Architecture
        arch_data = self._safe_parse(reasoning.get("architecture", "{}"))
        if isinstance(arch_data, dict) and "pattern" in arch_data:
            pattern = arch_data["pattern"]
            if isinstance(pattern, dict):
                solution.architecture = ArchitecturePattern(
                    name=str(pattern.get("name", "Custom")),
                    description=str(
                        pattern.get("description", arch_data.get("high_level_design", ""))
                    ),
                    pros=[
                        str(p)
                        for p in (
                            pattern.get("pros", [])
                            if isinstance(pattern.get("pros", []), list)
                            else []
                        )
                    ],
                    cons=[
                        str(c)
                        for c in (
                            pattern.get("cons", [])
                            if isinstance(pattern.get("cons", []), list)
                            else []
                        )
                    ],
                )

        # Code blocks
        code_data = self._safe_parse(reasoning.get("code", "{}"))
        if isinstance(code_data, dict) and "files" in code_data:
            for f in code_data["files"]:
                if isinstance(f, dict):
                    path = f.get("path") or f.get("filename") or "main.py"
                    content = f.get("content", "")
                    if content and len(content) > 10:
                        ext = path.rsplit(".", 1)[-1] if "." in path else "py"
                        lang_map = {
                            "py": "python",
                            "ts": "typescript",
                            "js": "javascript",
                            "go": "go",
                            "rs": "rust",
                            "java": "java",
                            "yml": "yaml",
                            "yaml": "yaml",
                        }
                        solution.code_blocks.append(
                            CodeBlock(
                                filename=path,
                                language=lang_map.get(ext, ext),
                                content=content,
                                description=f.get("description", ""),
                            )
                        )

        # Performance notes
        opt_data = self._safe_parse(reasoning.get("optimization", "{}"))
        if isinstance(opt_data, dict):
            for key, val in opt_data.items():
                if isinstance(val, dict):
                    solution.performance_notes.append(
                        f"[{key}] {json.dumps(val, ensure_ascii=False)[:200]}"
                    )
                elif isinstance(val, list):
                    for item in val[:3]:
                        solution.performance_notes.append(f"[{key}] {str(item)[:200]}")

        # Security notes
        sec_data = self._safe_parse(reasoning.get("security", "{}"))
        if isinstance(sec_data, dict):
            for threat in sec_data.get("threat_model", [])[:5]:
                if isinstance(threat, dict):
                    solution.security_notes.append(
                        f"[{threat.get('likelihood', 'medium')}] {threat.get('threat', '')}: {threat.get('mitigation', '')}"
                    )

        solution.metadata["llm_provider"] = self.llm.name
        solution.metadata["llm_model"] = self.llm.model_name
        solution.metadata["review"] = str(reasoning.get("review", ""))[:500]

        return solution

    def _extract_summary(self, reasoning: dict) -> str:
        """Build a short human-readable summary from reasoning artifacts.

        Complexity: ``O(A)`` over the architecture and review artifacts.
        """
        parts = []
        arch = self._safe_parse(reasoning.get("architecture", "{}"))
        if isinstance(arch, dict):
            parts.append(arch.get("high_level_design", "")[:200])
        review = self._safe_parse(reasoning.get("review", "{}"))
        if isinstance(review, dict):
            parts.append(review.get("final_summary", "")[:200])
        return " | ".join(parts) if parts else "Solution generated via Chain-of-Thought reasoning"

    def _safe_parse(self, data: Any) -> Any:
        """Parse arbitrary reasoning artifacts as JSON, degrading gracefully.

        Accepts already-structured containers as-is, extracts fenced JSON from
        strings, and otherwise wraps raw text so callers never see exceptions.

        Complexity: ``O(len(data))``.
        """
        if isinstance(data, (dict, list)):
            return data
        if not isinstance(data, str):
            return {}
        # Try to extract JSON from a markdown code block
        parsed = extract_fenced_json(data, default=None)
        if parsed is not None:
            return parsed
        return {"raw": data.strip()[:500]}

    async def _run_kernel(
        self, kernel_name: str, task: Task, thought: ThoughtProcess, **kwargs
    ) -> dict[str, Any]:
        """Execute one registered kernel, recording its outcome on the thought.

        Complexity: ``O(1)`` dispatch plus whatever the kernel does.
        """
        kernel = self.kernels.get(kernel_name)
        if not kernel:
            thought.add_step(kernel_name, task.title, "Kernel not found", 0.0)
            return {}
        input_summary = f"Task: {task.title} | {kwargs}"
        result = await kernel.execute(task, **kwargs)
        confidence = result.get("_confidence", 0.7)
        thought.add_step(kernel_name, input_summary, str(result)[:200], confidence)
        return result

    async def _select_stack(self, task: Task, thought: ThoughtProcess) -> TechStack:
        """Pick a tech stack: preferred, from knowledge, or a safe default.

        Complexity: ``O(K)`` over candidate stacks from knowledge.
        """
        if task.preferred_stack:
            return task.preferred_stack
        candidates = await self.knowledge.find_relevant_stacks(task)
        return (
            candidates[0]
            if candidates
            else TechStack(
                languages=["Python", "TypeScript"],
                frameworks=["FastAPI", "React"],
                databases=["PostgreSQL", "Redis"],
                infrastructure=["Docker"],
            )
        )

    def _evaluate_quality(self, solution: Solution, thought: ThoughtProcess) -> Solution:
        """Map average step confidence to a quality tier.

        Complexity: ``O(1)``.
        """
        score = thought.avg_confidence
        if score >= 0.9:
            solution.quality = SolutionQuality.MASTERPIECE
        elif score >= 0.75:
            solution.quality = SolutionQuality.EXCELLENT
        elif score >= 0.6:
            solution.quality = SolutionQuality.GOOD
        elif score >= 0.4:
            solution.quality = SolutionQuality.ACCEPTABLE
        else:
            solution.quality = SolutionQuality.DRAFT
        solution.confidence = score
        return solution

    def _build_reasoning(self, thought: ThoughtProcess, solution: Solution) -> str:
        """Render the thought trajectory as a readable reasoning summary.

        Complexity: ``O(S)`` for S thought steps.
        """
        parts = [f"Task: {solution.title}"]
        parts.append(f"Steps: {len(thought.steps)} | LLM: {self.llm.name}")
        for step in thought.steps:
            parts.append(
                f"  [{step.kernel}] -> {step.output_summary[:80]}... ({step.confidence:.0%})"
            )
        parts.append(
            f"Confidence: {thought.avg_confidence:.0%} | Quality: {solution.quality.value}"
        )
        return "\n".join(parts)

    async def record_feedback(
        self,
        solution: Solution,
        task: Task,
        rating: int,
        comments: str = "",
        would_use_again: bool = True,
        improvements: list[str] | None = None,
    ) -> FeedbackEntry:
        """Record user feedback and fold it into episodic memory.

        Complexity: ``O(1)`` store write plus ``O(1)`` memory episode.
        """
        result = await self.feedback.record_feedback(
            solution=solution,
            task=task,
            rating=rating,
            comments=comments,
            would_use_again=would_use_again,
            improvements=improvements,
        )
        # Also record feedback outcome in memory
        outcome = "success" if rating >= 4 else "failure" if rating <= 2 else "partial"
        self.memory.record_episode(
            task_description=f"Feedback on: {task.title}",
            solution_summary=f"Rating: {rating}/5. {comments}",
            outcome=outcome,
            tags=task.tags + ["feedback"],
        )
        return result

    def get_feedback_analysis(self) -> dict[str, Any]:
        """Return aggregated feedback patterns. Complexity: ``O(N)`` in feedback entries."""
        return self.feedback.analyze_patterns()

    async def scaffold_project(
        self, solution: Solution, task: Task, output_dir: str = "."
    ) -> dict[str, Any]:
        """Materialize a scaffolded project on disk. Complexity: ``O(F)`` files written."""
        scaffolder = ProjectScaffolder(output_dir=output_dir)
        return await scaffolder.scaffold(solution, task)

    # ── Self-Evolution ────────────────────────────────────────

    async def evolve(self) -> dict[str, Any]:
        """Run one self-evolution cycle: analyze self, generate patches, apply improvements.

        Complexity: bounded by the evolution engine's patch generation.
        """
        result = await self.evolution.run_evolution_cycle()
        self.memory.record_episode(
            task_description="Self-evolution cycle",
            solution_summary=result.summary,
            outcome="success" if result.patches_applied > 0 else "partial",
            tags=["evolution", "self-improvement"],
        )
        return result.model_dump()

    # ── Knowledge Ingestion ───────────────────────────────────

    async def ingest_file(self, path: str) -> dict[str, Any]:
        result = await self.ingestion.ingest_file(path, tags=["ingested"])
        return result.model_dump()

    async def ingest_directory(
        self, path: str, patterns: list[str] | None = None
    ) -> dict[str, Any]:
        result = await self.ingestion.ingest_directory(path, patterns=patterns, tags=["ingested"])
        return result.model_dump()

    async def ingest_text(self, text: str, source: str = "direct") -> dict[str, Any]:
        result = await self.ingestion.ingest_text(text, source_name=source, tags=["ingested"])
        return result.model_dump()

    # ── Resource Discovery ────────────────────────────────────

    def discover_resources(self) -> dict[str, Any]:
        return self.discovery.discover_all()

    # ── Memory ────────────────────────────────────────────────

    def search_memory(self, query: str, kind: str = "all") -> dict[str, Any]:
        result: dict[str, Any] = {}
        if kind in ("all", "episodes"):
            result["episodes"] = [ep.model_dump() for ep in self.memory.search_episodes(query)]
        if kind in ("all", "knowledge"):
            result["knowledge"] = [k.model_dump() for k in self.memory.search_knowledge(query)]
        if kind in ("all", "procedures"):
            result["procedures"] = [p.model_dump() for p in self.memory.search_procedures(query)]
        return result

    def memory_stats(self) -> dict[str, Any]:
        return self.memory.stats()

    # ── Worker Hierarchy ──────────────────────────────────────

    async def execute_hierarchical(
        self,
        description: str,
        decomposer: Callable[..., Coroutine] | None = None,
        executor: Callable[..., Coroutine] | None = None,
        aggregator: Callable[..., Coroutine] | None = None,
    ) -> dict[str, Any]:
        """Execute a task through the worker hierarchy with optional custom callbacks.

        Complexity: ``O(N)`` for N spawned subtasks, bounded by hierarchy limits.
        """
        task = await self.worker_hierarchy.execute(
            description,
            decomposer=decomposer,
            executor=executor,
            aggregator=aggregator,
        )
        return {
            "task_id": task.id,
            "state": task.state.value,
            "result": task.result,
            "subtasks": len(task.subtasks),
            "depth": task.depth,
        }

    def hierarchy_stats(self) -> dict[str, Any]:
        return self.worker_hierarchy.get_stats()

    # ── Sandbox — Code Validation ────────────────────────────────

    async def validate_solution(self, solution: Solution, run_tests: bool = False) -> SandboxResult:
        """Validate generated code through sandbox (AST, lint, run, tests).

        Complexity: ``O(F)`` files validated; heavy checks are sandbox-bound.
        """
        files = [
            {
                "path": cb.filename,
                "language": cb.language,
                "content": cb.content,
            }
            for cb in solution.code_blocks
        ]
        if not files:
            return SandboxResult(all_valid=True, summary="No code to validate")

        result = await self.sandbox.validate_files(files, run_tests=run_tests)

        solution.metadata["sandbox_valid"] = result.all_valid
        solution.metadata["sandbox_summary"] = result.summary
        solution.metadata["sandbox_files_valid"] = sum(1 for vr in result.files if vr.ast_valid)

        return result

    # ── Module System ──────────────────────────────────────────

    def list_modules(self) -> list[dict[str, str]]:
        """List all registered Noema modules."""
        return self.modules.list_modules()

    def execute_module(self, module_name: str, task: Task) -> dict[str, Any]:
        """Execute a specific module on a task."""
        return self.modules.execute_module(module_name, task)

    def execute_all_modules(
        self, task: Task, filter_tags: list[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Execute all (or filtered) modules on a task — combined intelligence."""
        return self.modules.execute_all(task, filter_tags)

    def get_module(self, name: str) -> Any:
        """Get a module instance by name."""
        return self.modules.get_instance(name)

    def modules_stats(self) -> dict[str, Any]:
        """Get stats on all modules."""
        return self.modules.stats()

    def _register_default_kernels(self) -> None:
        from noema.kernels.ai_ml import AIMLKernel
        from noema.kernels.analysis import AnalysisKernel
        from noema.kernels.architecture import ArchitectureKernel
        from noema.kernels.codegen import CodegenKernel
        from noema.kernels.data import DataKernel
        from noema.kernels.devops import DevOpsKernel
        from noema.kernels.frontend import FrontendKernel
        from noema.kernels.optimization import OptimizationKernel
        from noema.kernels.security import SecurityKernel

        kernel_factories: list[Callable[..., BaseKernel]] = [
            ArchitectureKernel,
            CodegenKernel,
            OptimizationKernel,
            SecurityKernel,
            AnalysisKernel,
            FrontendKernel,
            DevOpsKernel,
            DataKernel,
            AIMLKernel,
        ]
        for factory in kernel_factories:
            self.register_kernel(factory(knowledge=self.knowledge))
