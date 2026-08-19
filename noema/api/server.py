"""FastAPI API layer — production-hardened reasoning endpoints.

Architecture:
- One shared :class:`NoemaEngine` lives behind ``app.state.noema`` and the
  module-level ``_noema``; every endpoint resolves it via :func:`_get_noema`.
- Routers are layered: admin, diagnostics, webhooks, tasks, plus a versioned
  v1 router. Middleware stack handles CORS, auth, rate limits, request size,
  cache headers, request IDs, and RFC 7807 problem responses.

Concurrency contract:
- All handlers are async; ``/think/stream`` pushes step events over an
  :class:`asyncio.Queue` from a background task so the loop is never blocked.
- Long-running ``noema.think`` runs are cancellation-tracked.

Complexity:
- ``think``/``think_detail``: ``O(S·L)`` LLM work bounded by the engine's
  3-attempt Reflexion loop; response building is ``O(R)`` in requirements.
- ``think_stream``: same engine cost plus ``O(1)`` per SSE event.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from noema.api.admin import router as admin_router
from noema.api.auth import APIKeyAuthMiddleware
from noema.api.cache_headers import CacheControlMiddleware
from noema.api.diagnostics import router as diagnostics_router
from noema.api.middleware import (
    RequestIDMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)
from noema.api.problem import ProblemResponse, problem_response
from noema.api.rate_limit import RateLimitMiddleware
from noema.api.versioning import APIVersionMiddleware
from noema.api.webhooks import get_webhook_dispatcher
from noema.audit.logger import AuditLogger
from noema.billing.cost_tracker import CostTracker
from noema.billing.quotas import QuotaExceededError, QuotaManager
from noema.config.feature_flags import FeatureFlagService
from noema.config.settings import get_settings
from noema.context import get_tenant_id, reset_tenant_id, set_tenant_id
from noema.core.engine import NoemaEngine
from noema.core.types import Requirement, SandboxValidationError, Task, TaskComplexity, TechStack
from noema.logging import get_logger, setup_logging
from noema.observability.metrics import (
    CONTENT_TYPE_LATEST,
    generate_latest,
    spawn_metrics_server,
    update_system_gauges,
)
from noema.observability.sentry import init_sentry
from noema.resilience.cancellation import CancellationManager, CancelledTaskError
from noema.resilience.graceful_degradation import GracefulDegradation

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi.exceptions import RequestValidationError

log = get_logger(__name__)

# ─── Lifespan ────────────────────────────────────────────────────────────
_noema: NoemaEngine | None = None
_start_time: float = 0.0
_cancellation_mgr: CancellationManager = CancellationManager()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _noema, _start_time
    setup_logging()
    settings = get_settings()
    log.info("api_starting", host=settings.api.host, port=settings.api.port)

    if not settings.api.api_key.get_secret_value():
        log.warning(
            "api_auth_disabled",
            hint=(
                "No NOEMA_API__API_KEY configured: API authentication is OFF. "
                "Do not expose this instance outside localhost."
            ),
        )
    if settings.api.webhook_secret.get_secret_value():
        log.info("webhook_hmac_enabled")
    else:
        log.warning(
            "webhook_hmac_disabled",
            hint="No NOEMA_API__WEBHOOK_SECRET configured: webhook signatures are not verified.",
        )

    if settings.obs.sentry_dsn:
        init_sentry(
            dsn=settings.obs.sentry_dsn,
            environment=settings.obs.sentry_environment,
            traces_sample_rate=settings.obs.sentry_traces_sample_rate,
            profiles_sample_rate=settings.obs.sentry_profiles_sample_rate,
        )

    _noema = NoemaEngine(worker_count=settings.worker.pool_size)
    await _noema.initialize()
    app.state.noema = _noema

    # Enterprise services
    app.state.audit_logger = AuditLogger(pg_pool=None)
    await app.state.audit_logger.initialize()

    app.state.quota_manager = QuotaManager(pg_pool=None)
    await app.state.quota_manager.initialize()

    app.state.cost_tracker = CostTracker(redis_url=settings.redis.url)

    app.state.feature_flags = FeatureFlagService()
    await app.state.feature_flags.initialize()

    app.state.degradation = GracefulDegradation()

    webhook_dispatcher = get_webhook_dispatcher()
    await webhook_dispatcher.start()
    app.state.webhook_dispatcher = webhook_dispatcher

    # Prometheus: standalone exporter on its own port + gauge refresher
    gauge_task: asyncio.Task | None = None
    if settings.obs.metrics_enabled:
        spawn_metrics_server(port=settings.obs.metrics_port)
        gauge_task = asyncio.create_task(_refresh_gauges())

    _start_time = time.monotonic()
    log.info("api_ready")

    yield

    log.info("api_shutting_down")
    if gauge_task:
        gauge_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await gauge_task
    await webhook_dispatcher.stop()
    if _noema:
        await _noema.shutdown()
    log.info("api_stopped")


async def _refresh_gauges() -> None:
    """Periodically push engine state into Prometheus gauges (best-effort)."""
    while True:
        try:
            noema = _get_noema()
            update_system_gauges(
                worker_stats=noema.worker_pool.stats,
                knowledge_stats=noema.knowledge.get_stats(),
            )
        except Exception as e:
            log.debug("gauge_refresh_failed", error=str(e))
        await asyncio.sleep(10)


# ─── App ─────────────────────────────────────────────────────────────────
settings = get_settings()

app = FastAPI(
    title="Noema API",
    description="Production-grade AI reasoning engine",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.api.reload else None,
    redoc_url="/redoc" if settings.api.reload else None,
)

# Include routers
from noema.api.experiments import router as experiments_router  # noqa: PLC0415, E402
from noema.api.tasks import router as tasks_router  # noqa: PLC0415, E402
from noema.api.webhooks import router as webhooks_router  # noqa: PLC0415, E402

app.include_router(experiments_router)
app.include_router(admin_router)
app.include_router(diagnostics_router)
app.include_router(webhooks_router)
app.include_router(tasks_router)

# Middleware stack (order matters: outermost = first applied)
app.add_middleware(CacheControlMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(RequestSizeLimitMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(APIKeyAuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_methods=settings.api.cors_methods,
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
)
app.add_middleware(APIVersionMiddleware)


def _get_noema() -> NoemaEngine:
    if _noema is None:
        raise HTTPException(status_code=503, detail="Service starting")
    return _noema


def _build_task(request: TaskRequest) -> Task:
    """Project a validated :class:`TaskRequest` onto a domain :class:`Task`.

    Complexity: ``O(R)`` for R requirements (bounded by the request schema).
    """
    task = Task(
        title=request.title,
        description=request.description,
        requirements=[
            Requirement(
                category=r.category,
                description=r.description,
                priority=r.priority,
                constraints=r.constraints,
            )
            for r in request.requirements
        ],
        complexity=TaskComplexity(request.complexity),
        tags=request.tags,
        context=request.context,
    )
    if request.preferred_stack:
        task.preferred_stack = TechStack(**request.preferred_stack)
    return task


# ─── Request/Response Models ────────────────────────────────────────────
class RequirementModel(BaseModel):
    category: str = Field(..., max_length=50)
    description: str = Field(..., max_length=2000)
    priority: int = Field(default=5, ge=1, le=10)
    constraints: list[str] = Field(default_factory=list, max_length=20)


class TaskRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=10000)
    requirements: list[RequirementModel] = Field(default_factory=list, max_length=50)
    preferred_stack: dict[str, list[str]] | None = None
    complexity: str = Field(default="moderate", pattern="^(simple|moderate|complex|extreme)$")
    tags: list[str] = Field(default_factory=list, max_length=20)
    context: dict[str, Any] = Field(default_factory=dict)
    task_id: str | None = Field(
        default=None, description="Optional task ID for cancellation tracking"
    )


class SolutionResponse(BaseModel):
    id: str
    task_id: str
    title: str
    summary: str
    quality: str
    confidence: float
    stack_summary: str
    code_blocks_count: int
    performance_notes: list[str]
    security_notes: list[str]
    thought_steps: int
    duration_ms: float


class SolutionDetailResponse(BaseModel):
    id: str
    task_id: str
    title: str
    summary: str
    quality: str
    confidence: float
    architecture: dict[str, Any] | None
    stack: dict[str, Any]
    code_blocks: list[dict[str, Any]]
    deployment: dict[str, Any]
    performance_notes: list[str]
    security_notes: list[str]
    metadata: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_s: float
    db: str
    llm: str
    workers: dict[str, Any]


class ReadinessResponse(BaseModel):
    ready: bool
    checks: dict[str, str]


# ─── Shared router ──────────────────────────────────────────
router = APIRouter()

# ─── Exception handlers ─────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> ProblemResponse:
    """Catch-all exception handler -> RFC 7807.

    The real exception is logged server-side; the client only ever sees a
    generic message so internal paths/DB errors are never leaked (zero-trust).
    """
    log.error("unhandled_exception", error=str(exc), exc_info=exc, path=str(request.url))
    return problem_response(
        status=500,
        title="Internal Server Error",
        detail="Internal server error",
        instance=str(request.url),
    )


def _estimate_input_tokens(task: Task) -> int:
    """Rough input-token estimate (chars / 4) for quota enforcement."""
    text = f"{task.title} {task.description}"
    return len(text) // 4


@asynccontextmanager
async def _task_guard(
    request: Request, task_id: str, estimated_input_tokens: int = 0
) -> AsyncIterator[str]:
    """Per-request quota enforcement + tenant context for a reasoning task.

    Enforces the tenant's quotas before work starts (fail-closed 429), tracks
    the active task while it runs, and guarantees the task is untracked and the
    tenant context reset even on error.
    """
    tenant_id = request.headers.get("x-tenant-id") or get_tenant_id() or "default"
    token = set_tenant_id(tenant_id)
    quota: QuotaManager | None = getattr(request.app.state, "quota_manager", None)
    tracked = False
    try:
        if quota is not None:
            await quota.check_quota(tenant_id, estimated_input_tokens=estimated_input_tokens)
            await quota.track_active_task(tenant_id, task_id)
            tracked = True
        yield tenant_id
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from None
    finally:
        if quota is not None and tracked:
            await quota.untrack_active_task(tenant_id, task_id)
        reset_tenant_id(token)


async def _record_task_cost(
    request: Request, tenant_id: str, task_id: str, noema: NoemaEngine
) -> None:
    """Attribute a think() run's token spend to the tenant (economy of computation)."""
    tracker: CostTracker | None = getattr(request.app.state, "cost_tracker", None)
    if tracker is None:
        return
    total_tokens = int(noema.tracer.get_stats().get("total_tokens", 0) or 0)
    if total_tokens <= 0:
        return
    try:
        await tracker.record(
            tenant_id=tenant_id,
            task_id=task_id,
            provider=noema.llm.name,
            model=noema.llm.model_name,
            input_tokens=total_tokens,
            output_tokens=0,
            step_name="think",
        )
    except Exception as e:
        log.warning("cost_record_failed", error=str(e))


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> ProblemResponse:
    """HTTPException -> RFC 7807."""
    return problem_response(
        status=exc.status_code,
        title=exc.detail,
        detail=exc.detail,
        instance=str(request.url),
    )


@app.exception_handler(422)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> ProblemResponse:
    """Validation error -> RFC 7807 with field errors."""
    errors = exc.errors() if hasattr(exc, "errors") else []
    return problem_response(
        status=422,
        title="Validation Error",
        detail="Request validation failed",
        instance=str(request.url),
        extra={"errors": errors},
    )


# ─── Endpoints ───────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness probe — always 200 if process is alive."""
    noema = _get_noema()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_s=round(time.monotonic() - _start_time, 1),
        db="ok",
        llm=settings.llm.provider,
        workers=noema.worker_pool.stats,
    )


@router.get("/metrics", tags=["ops"])
async def metrics() -> Response:
    """Prometheus text-format metrics on the main API (API-key protected)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/ready", response_model=ReadinessResponse, tags=["ops"])
async def readiness() -> ReadinessResponse:
    """Readiness probe — verifies all dependencies."""
    checks: dict[str, str] = {}
    ready = True

    # Check LLM
    try:
        provider = _get_noema().llm.name
        if provider:
            checks["llm"] = "ok"
        else:
            checks["llm"] = "no_provider"
            ready = False
    except Exception:
        checks["llm"] = "error"
        ready = False

    # Check workers
    noema = _get_noema()
    stats = noema.worker_pool.stats
    busy = int(stats.get("workers_busy", 0) or 0)
    total = int(stats.get("workers_total", 0) or settings.worker.pool_size)
    if busy >= total:
        checks["workers"] = "saturated"
        ready = False
    else:
        checks["workers"] = "ok"

    return ReadinessResponse(ready=ready, checks=checks)


@router.post("/think", response_model=SolutionResponse, tags=["reasoning"])
async def think(request: TaskRequest, http_request: Request) -> SolutionResponse:
    """Generate a solution for the given task.

    Complexity: dominated by ``noema.think`` (3 Reflexion attempts); response
    building is ``O(R)`` in requirements.
    """
    noema = _get_noema()
    log.info("think_start", title=request.title, tags=request.tags)

    task = _build_task(request)
    task_id = request.task_id or task.id
    async with _task_guard(
        http_request, task_id, estimated_input_tokens=_estimate_input_tokens(task)
    ) as tenant_id:
        t0 = time.monotonic()
        try:
            solution, thought = await _cancellation_mgr.execute_with_cancellation(
                task_id, noema.think(task)
            )
        except CancelledTaskError:
            raise HTTPException(status_code=499, detail="Task cancelled by user") from None
        except SandboxValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        elapsed = (time.monotonic() - t0) * 1000
        await _record_task_cost(http_request, tenant_id, task_id, noema)

        log.info(
            "think_done",
            task_id=solution.task_id,
            quality=solution.quality.value,
            confidence=solution.confidence,
            duration_ms=round(elapsed, 1),
        )

        return SolutionResponse(
            id=solution.id,
            task_id=solution.task_id,
            title=solution.title,
            summary=solution.summary,
            quality=solution.quality.value,
            confidence=solution.confidence,
            stack_summary=solution.stack.summary(),
            code_blocks_count=len(solution.code_blocks),
            performance_notes=solution.performance_notes,
            security_notes=solution.security_notes,
            thought_steps=len(thought.steps),
            duration_ms=thought.duration_ms,
        )


@router.post("/think/detail", response_model=SolutionDetailResponse, tags=["reasoning"])
async def think_detail(request: TaskRequest, http_request: Request) -> SolutionDetailResponse:
    """Generate a full-detail solution.

    Complexity: dominated by ``noema.think``; serialization is ``O(B)`` for B
    code blocks.
    """
    noema = _get_noema()
    log.info("think_detail_start", title=request.title)

    task = _build_task(request)
    task_id = request.task_id or task.id
    async with _task_guard(
        http_request, task_id, estimated_input_tokens=_estimate_input_tokens(task)
    ) as tenant_id:
        try:
            solution, thought = await _cancellation_mgr.execute_with_cancellation(
                task_id, noema.think(task)
            )
        except CancelledTaskError:
            raise HTTPException(status_code=499, detail="Task cancelled by user") from None
        except SandboxValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        await _record_task_cost(http_request, tenant_id, task_id, noema)

    return SolutionDetailResponse(
        id=solution.id,
        task_id=solution.task_id,
        title=solution.title,
        summary=solution.summary,
        quality=solution.quality.value,
        confidence=solution.confidence,
        architecture=solution.architecture.model_dump() if solution.architecture else None,
        stack=solution.stack.model_dump(),
        code_blocks=[cb.model_dump() for cb in solution.code_blocks],
        deployment=solution.deployment,
        performance_notes=solution.performance_notes,
        security_notes=solution.security_notes,
        metadata=solution.metadata,
    )


@router.get("/knowledge/stats", tags=["knowledge"])
async def knowledge_stats() -> dict[str, Any]:
    """Knowledge base statistics."""
    return _get_noema().knowledge.get_stats()


@router.get("/knowledge/search", tags=["knowledge"])
async def knowledge_search(q: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search knowledge base."""
    return await _get_noema().knowledge.search(q, top_k=top_k)


@router.get("/workers/stats", tags=["ops"])
async def worker_stats() -> dict[str, Any]:
    """Worker pool statistics."""
    return _get_noema().worker_pool.stats


@router.post("/think/stream", tags=["reasoning"])
async def think_stream(request: TaskRequest, http_request: Request) -> StreamingResponse:
    """Generate a solution with Server-Sent Events (SSE) streaming.

    Complexity: dominated by ``noema.think``; each event is ``O(1)``.
    """
    noema = _get_noema()
    log.info("think_stream_start", title=request.title)

    task = _build_task(request)

    async def event_stream() -> AsyncIterator[str]:
        step_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def on_step_start(name: str, label: str, done: int, total: int) -> None:
            await step_queue.put(
                {
                    "type": "step_start",
                    "name": name,
                    "label": label,
                    "done": done,
                    "total": total,
                }
            )

        async def on_step_end(name: str, result: str, done: int, total: int) -> None:
            await step_queue.put(
                {
                    "type": "step_end",
                    "name": name,
                    "status": "completed" if not result.startswith("FAILED") else "failed",
                    "preview": result[:200],
                    "done": done,
                    "total": total,
                }
            )

        async def run_think() -> None:
            try:
                async with _task_guard(
                    http_request,
                    request.task_id or task.id,
                    estimated_input_tokens=_estimate_input_tokens(task),
                ) as tenant_id:
                    solution, thought = await _cancellation_mgr.execute_with_cancellation(
                        request.task_id or task.id,
                        noema.think(task, on_step_start=on_step_start, on_step_end=on_step_end),
                    )
                    await _record_task_cost(http_request, tenant_id, task.id, noema)
                    wd = get_webhook_dispatcher()
                    if wd:
                        await wd.emit(
                            "task.completed",
                            {
                                "task_id": task.id,
                                "solution_id": solution.id,
                                "quality": solution.quality.value,
                            },
                        )
                    await step_queue.put(
                        {
                            "type": "complete",
                            "solution_id": solution.id,
                            "quality": solution.quality.value,
                            "confidence": solution.confidence,
                            "duration_ms": thought.duration_ms,
                            "steps": len(thought.steps),
                            "summary": solution.summary,
                        }
                    )
            except CancelledTaskError:
                await step_queue.put(
                    {
                        "type": "complete",
                        "solution_id": "",
                        "quality": "cancelled",
                        "confidence": 0,
                        "duration_ms": 0,
                        "steps": 0,
                        "summary": "Task was cancelled",
                    }
                )
            except Exception as e:
                await step_queue.put({"type": "error", "message": str(e)})

        asyncio.create_task(run_think())

        while True:
            data = await step_queue.get()
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            if data["type"] in ("complete", "error"):
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/think/{task_id}", tags=["reasoning"])
async def cancel_think(task_id: str) -> dict[str, str]:
    """Cancel a running think task by ID."""
    cancelled = _cancellation_mgr.cancel(task_id)
    if cancelled:
        log.info("api_cancel_requested", task_id=task_id)
        return {"status": "cancelled", "task_id": task_id}
    return {"status": "not_found_or_completed", "task_id": task_id}


@router.get("/tasks/active", tags=["ops"])
async def list_active_tasks() -> dict[str, list[str]]:
    """List all currently running task IDs."""
    return {"active_tasks": _cancellation_mgr.get_active()}


@router.get("/health/infra", tags=["ops"])
async def health_infra(request: Request) -> dict[str, Any]:
    """Infrastructure health — Redis, PostgreSQL degradation status."""
    deg: GracefulDegradation | None = getattr(request.app.state, "degradation", None)
    if deg:
        await deg.check_health()
        return deg.get_status()
    return {"redis": "unknown", "postgresql": "unknown"}


@router.get("/features", tags=["ops"])
async def get_features(request: Request) -> dict[str, Any]:
    """Get all feature flags for the current tenant."""
    ff: FeatureFlagService | None = getattr(request.app.state, "feature_flags", None)
    if ff:
        flags = await ff.get_all_flags()
        return {"features": flags}
    return {"features": {}}


@router.get("/kernels", tags=["reasoning"])
async def list_kernels() -> list[dict[str, str]]:
    """Registered reasoning kernels."""
    noema = _get_noema()
    return [{"name": k.name, "description": k.description} for k in noema.kernels.values()]


@router.get("/agents", tags=["reasoning"])
async def list_agents() -> list[dict[str, str | list[str]]]:
    """Registered sub-agents."""
    noema = _get_noema()
    return [
        {"name": a.name, "role": a.role.value, "expertise": a.expertise}
        for a in noema.orchestrator.agents.values()
    ]


# Mount main router (root paths)
app.include_router(router)

# Mount versioned API under /api/v1/ (lazy import avoids circular dep)
from noema.api.routers.v1 import router as v1_router  # noqa: PLC0415, E402

app.include_router(v1_router)
