"""Coverage for noema.api.server pure helpers: _build_task, _task_guard,
_record_task_cost, exception handlers, health/metrics/readiness endpoints."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

import noema.api.server as srv
from noema.billing.quotas import QuotaExceededError
from noema.context import get_tenant_id
from noema.core.types import Task, TaskComplexity


def _request(**headers):
    return SimpleNamespace(
        headers=headers,
        app=SimpleNamespace(state=SimpleNamespace(quota_manager=None, cost_tracker=None)),
        url="http://test/foo",
    )


def _task_request(**overrides):
    base = {
        "title": "Build an API",
        "description": "A REST API for users",
        "requirements": [],
        "tags": ["api"],
    }
    base.update(overrides)
    return srv.TaskRequest(**base)


# ── _build_task ───────────────────────────────────────────────────────────


def test_build_task_maps_basic_fields():
    req = _task_request()
    task = srv._build_task(req)
    assert isinstance(task, Task)
    assert task.title == "Build an API"
    assert task.description == "A REST API for users"
    assert task.tags == ["api"]
    assert task.complexity == TaskComplexity.MODERATE
    assert task.requirements == []
    assert task.preferred_stack is None


def test_build_task_maps_requirements_and_complexity():
    req = _task_request(
        complexity="complex",
        requirements=[
            {
                "category": "functional",
                "description": "Users can log in",
                "priority": 9,
                "constraints": ["jwt"],
            }
        ],
    )
    task = srv._build_task(req)
    assert task.complexity == TaskComplexity.COMPLEX
    assert len(task.requirements) == 1
    requirement = task.requirements[0]
    assert requirement.category == "functional"
    assert requirement.description == "Users can log in"
    assert requirement.priority == 9
    assert requirement.constraints == ["jwt"]


def test_build_task_applies_preferred_stack():
    req = _task_request(preferred_stack={"languages": ["python", "fastapi"]})
    task = srv._build_task(req)
    assert task.preferred_stack is not None
    assert task.preferred_stack.languages == ["python", "fastapi"]


# ── TaskRequest validation ────────────────────────────────────────────────


def test_task_request_rejects_unknown_complexity():
    with pytest.raises(ValidationError):
        _task_request(complexity="trivial")


def test_task_request_rejects_out_of_range_priority():
    with pytest.raises(ValidationError):
        _task_request(requirements=[{"category": "c", "description": "d", "priority": 0}])


def test_task_request_rejects_empty_description():
    with pytest.raises(ValidationError):
        srv.TaskRequest(title="x", description="")


def test_task_request_accepts_task_id():
    req = _task_request(task_id="custom-id")
    assert req.task_id == "custom-id"


# ── _get_noema ────────────────────────────────────────────────────────────


def test_get_noema_raises_503_when_uninitialized(monkeypatch):
    monkeypatch.setattr(srv, "_noema", None)
    with pytest.raises(HTTPException) as exc_info:
        srv._get_noema()
    assert exc_info.value.status_code == 503
    assert "Service starting" in exc_info.value.detail


def test_get_noema_returns_engine(monkeypatch):
    engine = object()
    monkeypatch.setattr(srv, "_noema", engine)
    assert srv._get_noema() is engine


# ── Exception handlers ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_global_exception_handler_returns_500_problem():
    resp = await srv.global_exception_handler(_request(), RuntimeError("boom"))
    assert resp.status_code == 500
    body = json.loads(resp.body)
    assert body["title"] == "Internal Server Error"
    assert body["detail"] == "Internal server error"
    assert body["instance"] == "http://test/foo"


@pytest.mark.asyncio
async def test_http_exception_handler_maps_status_and_detail():
    resp = await srv.http_exception_handler(
        _request(), HTTPException(status_code=404, detail="Missing")
    )
    assert resp.status_code == 404
    body = json.loads(resp.body)
    assert body["title"] == "Missing"
    assert body["detail"] == "Missing"


@pytest.mark.asyncio
async def test_validation_exception_handler_returns_422_with_errors():
    exc = RequestValidationError(
        [{"loc": ("body", "title"), "msg": "field required", "type": "value_error"}]
    )
    resp = await srv.validation_exception_handler(_request(), exc)
    assert resp.status_code == 422
    body = json.loads(resp.body)
    assert body["title"] == "Validation Error"
    assert body["errors"] == [
        {"loc": ["body", "title"], "msg": "field required", "type": "value_error"}
    ]


# ── health / metrics / readiness ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(
        srv,
        "_noema",
        SimpleNamespace(worker_pool=SimpleNamespace(stats={"workers_busy": 1, "workers_total": 4})),
    )
    result = await srv.health()
    assert result.status == "ok"
    assert result.version == "1.0.0"
    assert result.db == "ok"
    assert result.workers["workers_busy"] == 1


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_prometheus_text():
    resp = await srv.metrics()
    assert resp.media_type.startswith("text/plain")
    assert len(resp.body) > 0


@pytest.mark.asyncio
async def test_readiness_ok(monkeypatch):
    monkeypatch.setattr(
        srv,
        "_get_noema",
        lambda: SimpleNamespace(
            llm=SimpleNamespace(name="fallback"),
            worker_pool=SimpleNamespace(stats={"workers_busy": 1, "workers_total": 4}),
        ),
    )
    result = await srv.readiness()
    assert result.ready is True
    assert result.checks["llm"] == "ok"
    assert result.checks["workers"] == "ok"


@pytest.mark.asyncio
async def test_readiness_saturated_workers(monkeypatch):
    monkeypatch.setattr(
        srv,
        "_get_noema",
        lambda: SimpleNamespace(
            llm=SimpleNamespace(name="fallback"),
            worker_pool=SimpleNamespace(stats={"workers_busy": 4, "workers_total": 4}),
        ),
    )
    result = await srv.readiness()
    assert result.ready is False
    assert result.checks["workers"] == "saturated"


@pytest.mark.asyncio
async def test_readiness_no_provider(monkeypatch):
    monkeypatch.setattr(
        srv,
        "_get_noema",
        lambda: SimpleNamespace(
            llm=SimpleNamespace(name=""),
            worker_pool=SimpleNamespace(stats={"workers_busy": 0, "workers_total": 4}),
        ),
    )
    result = await srv.readiness()
    assert result.ready is False
    assert result.checks["llm"] == "no_provider"


@pytest.mark.asyncio
async def test_readiness_without_engine_raises_503(monkeypatch):
    def _raise():
        raise HTTPException(status_code=503, detail="Service starting")

    monkeypatch.setattr(srv, "_get_noema", _raise)
    with pytest.raises(HTTPException) as exc_info:
        await srv.readiness()
    assert exc_info.value.status_code == 503


# ── _task_guard ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_task_guard_default_tenant_no_quota():
    before = get_tenant_id()
    async with srv._task_guard(_request(), "task-1") as tenant:
        assert tenant == "default"
        assert get_tenant_id() == "default"
    assert get_tenant_id() == before


@pytest.mark.asyncio
async def test_task_guard_header_tenant():
    before = get_tenant_id()
    async with srv._task_guard(_request(**{"x-tenant-id": "tenA"}), "task-1") as tenant:
        assert tenant == "tenA"
    assert get_tenant_id() == before


@pytest.mark.asyncio
async def test_task_guard_quota_exceeded_raises_429():
    class _OverQuota:
        async def check_quota(self, tenant_id):
            raise QuotaExceededError("quota exhausted")

    req = _request()
    req.app.state.quota_manager = _OverQuota()
    before = get_tenant_id()
    with pytest.raises(HTTPException) as exc_info:
        async with srv._task_guard(req, "task-1"):
            pass
    assert exc_info.value.status_code == 429
    assert "quota exhausted" in exc_info.value.detail
    assert get_tenant_id() == before


@pytest.mark.asyncio
async def test_task_guard_tracks_and_untracks():
    class _Quota:
        def __init__(self) -> None:
            self.tracked: list[tuple[str, str]] = []
            self.untracked: list[tuple[str, str]] = []

        async def check_quota(self, tenant_id):
            pass

        async def track_active_task(self, tenant_id, task_id):
            self.tracked.append((tenant_id, task_id))

        async def untrack_active_task(self, tenant_id, task_id):
            self.untracked.append((tenant_id, task_id))

    quota = _Quota()
    req = _request(**{"x-tenant-id": "tenA"})
    req.app.state.quota_manager = quota
    async with srv._task_guard(req, "task-9") as tenant:
        assert tenant == "tenA"
    assert quota.tracked == [("tenA", "task-9")]
    assert quota.untracked == [("tenA", "task-9")]


# ── _record_task_cost ─────────────────────────────────────────────────────


def _noema_with_tokens(total_tokens: int):
    return SimpleNamespace(
        llm=SimpleNamespace(name="fallback", model_name="model-x"),
        tracer=SimpleNamespace(get_stats=lambda: {"total_tokens": total_tokens}),
    )


@pytest.mark.asyncio
async def test_record_task_cost_no_tracker_is_noop():
    await srv._record_task_cost(_request(), "tenA", "task-1", _noema_with_tokens(100))


@pytest.mark.asyncio
async def test_record_task_cost_skips_zero_tokens():
    class _Tracker:
        def __init__(self) -> None:
            self.records: list[dict[str, Any]] = []

        async def record(self, **kwargs):
            self.records.append(kwargs)

    tracker = _Tracker()
    req = _request()
    req.app.state.cost_tracker = tracker
    await srv._record_task_cost(req, "tenA", "task-1", _noema_with_tokens(0))
    assert tracker.records == []


@pytest.mark.asyncio
async def test_record_task_cost_records_think_spend():
    class _Tracker:
        def __init__(self) -> None:
            self.records: list[dict[str, Any]] = []

        async def record(self, **kwargs):
            self.records.append(kwargs)

    tracker = _Tracker()
    req = _request()
    req.app.state.cost_tracker = tracker
    await srv._record_task_cost(req, "tenA", "task-1", _noema_with_tokens(500))
    assert tracker.records == [
        {
            "tenant_id": "tenA",
            "task_id": "task-1",
            "provider": "fallback",
            "model": "model-x",
            "input_tokens": 500,
            "output_tokens": 0,
            "step_name": "think",
        }
    ]


@pytest.mark.asyncio
async def test_record_task_cost_tolerates_tracker_failure():
    class _BrokenTracker:
        async def record(self, **kwargs):
            raise RuntimeError("redis down")

    req = _request()
    req.app.state.cost_tracker = _BrokenTracker()
    await srv._record_task_cost(req, "tenA", "task-1", _noema_with_tokens(100))
