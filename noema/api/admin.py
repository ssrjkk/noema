"""Admin API — metrics, task history, tenant monitoring."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Query, Request

from noema.config.settings import get_settings
from noema.logging import get_logger

if TYPE_CHECKING:
    from noema.audit.logger import AuditLogger
    from noema.core.engine import NoemaEngine

log = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

_metrics_cache: dict[str, Any] = {}
_metrics_cache_at: float = 0.0
_METRICS_CACHE_TTL = 5.0


def _get_noema(request: Request) -> NoemaEngine:
    noema = getattr(request.app.state, "noema", None)
    if noema is None:
        raise HTTPException(status_code=503, detail="Service starting")
    return cast("NoemaEngine", noema)


@router.get("/metrics")
async def admin_metrics(request: Request) -> dict[str, Any]:
    """Prometheus-style metrics as JSON."""
    global _metrics_cache, _metrics_cache_at
    now = time.monotonic()
    if _metrics_cache and (now - _metrics_cache_at) < _METRICS_CACHE_TTL:
        return _metrics_cache

    noema = _get_noema(request)
    settings = get_settings()

    ns = noema.neurosymbolic
    ns_metrics = ns.get_metrics() if ns else {}

    tasks_processed = ns_metrics.get("tasks_processed", 0)
    tasks_successful = ns_metrics.get("tasks_successful", 0)
    tasks_failed = ns_metrics.get("tasks_failed", 0)
    success_rate = ns_metrics.get("success_rate", 0.0)
    total_llm_calls = ns_metrics.get("total_llm_calls", 0)
    total_refinements = ns_metrics.get("total_refinements", 0)

    worker_stats = noema.worker_pool.stats if hasattr(noema, "worker_pool") else {}
    memory_stats = noema.memory_stats() if hasattr(noema, "memory_stats") else {}
    from noema.api.server import _start_time as server_start_time

    uptime = time.monotonic() - server_start_time if server_start_time else 0

    result = {
        "tasks": {
            "total": tasks_processed,
            "successful": tasks_successful,
            "failed": tasks_failed,
            "success_rate": round(success_rate, 4),
        },
        "llm": {
            "total_calls": total_llm_calls,
            "total_refinements": total_refinements,
            "provider": settings.llm.provider,
        },
        "workers": worker_stats,
        "memory": {
            "episodic_count": memory_stats.get("episodic_count", 0),
            "procedural_count": memory_stats.get("procedural_count", 0),
            "knowledge_count": memory_stats.get("knowledge_count", 0),
        },
        "uptime_s": round(uptime, 1),
    }

    _metrics_cache = result
    _metrics_cache_at = now
    return result


@router.get("/tasks/history")
async def admin_task_history(
    request: Request,
    tenant_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[dict[str, Any]]:
    """Return task history from the audit logger."""
    audit = getattr(request.app.state, "audit_logger", None)
    if audit is None:
        raise HTTPException(status_code=503, detail="Audit logger not available")

    events: list[dict[str, Any]]
    if tenant_id:
        events = await audit.query(tenant_id=tenant_id, limit=limit + offset)
    else:
        events = await _query_all_tenants(audit, limit + offset)

    return events[offset:][:limit]


async def _query_all_tenants(audit: AuditLogger, max_results: int) -> list[dict[str, Any]]:
    """Query audit events across all tenants when tenant_id is not specified."""
    from pathlib import Path

    fallback_dir = getattr(audit, "_fallback_dir", ".noema/audit")
    base = Path(fallback_dir)
    if not base.exists():
        return []

    all_events: list[dict[str, Any]] = []
    for fpath in base.glob("*.jsonl"):
        tenant = fpath.stem
        try:
            events = await audit.query(tenant_id=tenant, limit=max_results)
            all_events.extend(events)
        except Exception:
            continue
        if len(all_events) >= max_results:
            break
    all_events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return all_events[:max_results]


@router.get("/tenants/{tenant_id}/metrics")
async def admin_tenant_metrics(tenant_id: str, request: Request) -> dict[str, Any]:
    """Tenant-specific metrics."""
    ff = getattr(request.app.state, "feature_flags", None)
    qm = getattr(request.app.state, "quota_manager", None)
    audit = getattr(request.app.state, "audit_logger", None)

    features = {}
    if ff:
        try:
            features = await ff.get_all_flags(tenant_id)
        except Exception as e:
            log.warning("feature_flags_error", tenant_id=tenant_id, error=str(e))

    quota_usage: dict[str, Any] | None = None
    if qm:
        try:
            quota = await qm.get_quota(tenant_id)
            quota_usage = {
                "monthly_budget_usd": quota.monthly_budget_usd,
                "max_concurrent_tasks": quota.max_concurrent_tasks,
                "max_tasks_per_hour": quota.max_tasks_per_hour,
            }
        except Exception as e:
            log.warning("quota_error", tenant_id=tenant_id, error=str(e))

    recent_task_count = 0
    if audit:
        try:
            recent = await audit.query(tenant_id=tenant_id, limit=1000)
            recent_task_count = len(recent)
        except Exception as e:
            log.warning("audit_query_failed", tenant_id=tenant_id, error=str(e))

    return {
        "tenant_id": tenant_id,
        "features": features,
        "quota": quota_usage,
        "recent_task_count": recent_task_count,
    }


@router.get("/audit/proof/{task_id}")
async def admin_audit_proof(
    task_id: str, request: Request, tenant_id: str | None = Query(None)
) -> dict[str, Any]:
    """Cryptographic inclusion proof for a completed task (Merkle)."""
    audit = getattr(request.app.state, "audit_logger", None)
    if audit is None:
        raise HTTPException(status_code=503, detail="Audit logger not available")
    if not tenant_id:
        from noema.context import get_tenant_id

        tenant_id = get_tenant_id() or "default"
    try:
        proof = await audit.get_proof_for_task(tenant_id=tenant_id, task_id=task_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        log.warning("audit_proof_error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate proof") from e
    return {"tenant_id": tenant_id, "task_id": task_id, "proof": proof}


@router.post("/audit/verify")
async def admin_audit_verify(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    """Verify an inclusion proof without trusting the server."""
    from noema.audit.merkle_proof import InclusionProof, verify_inclusion_proof

    proof_data = payload.get("proof")
    if not proof_data:
        raise HTTPException(status_code=422, detail="'proof' is required")
    if not isinstance(proof_data, dict):
        raise HTTPException(status_code=422, detail="'proof' must be an object")
    leaf_data = payload.get("leaf_data", proof_data.get("leaf_data"))
    if leaf_data is None:
        raise HTTPException(
            status_code=422, detail="'leaf_data' is required (or embed it in the proof)"
        )
    try:
        proof = InclusionProof.from_dict(proof_data)
        valid = verify_inclusion_proof(proof, leaf_data)
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid proof structure: {e}") from e
    return {
        "valid": valid,
        "block_index": proof.block_index,
        "root_hash": proof.root_hash.hex(),
    }


@router.get("/neurosymbolic/stats")
async def admin_neurosymbolic_stats(request: Request) -> dict[str, Any]:
    """NeuroSymbolic engine statistics."""
    settings = get_settings()
    ns_enabled = settings.neurosymbolic.enabled
    noema = _get_noema(request)

    result: dict[str, Any] = {
        "enabled": ns_enabled,
        "settings": {
            "max_refinement_attempts": settings.neurosymbolic.max_refinement_attempts,
            "verification_timeout": settings.neurosymbolic.verification_timeout,
            "evolution_enabled": settings.neurosymbolic.evolution_enabled,
            "fallback_to_cot": settings.neurosymbolic.fallback_to_cot,
        },
    }

    if ns_enabled and noema.neurosymbolic:
        ns = noema.neurosymbolic
        metrics = ns.get_metrics()
        result["metrics"] = {
            "tasks_processed": metrics.get("tasks_processed", 0),
            "tasks_successful": metrics.get("tasks_successful", 0),
            "tasks_failed": metrics.get("tasks_failed", 0),
            "success_rate": round(metrics.get("success_rate", 0.0), 4),
            "total_refinements": metrics.get("total_refinements", 0),
            "total_llm_calls": metrics.get("total_llm_calls", 0),
        }
        if ns.evolution:
            try:
                result["evolution"] = ns.evolution.get_stats()
            except Exception:
                result["evolution"] = {"error": "unavailable"}
    return result


def _getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)
