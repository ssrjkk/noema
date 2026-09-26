"""Tests for admin API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from noema.api.admin import router


@pytest.fixture
def app():
    """Create a test FastAPI app with admin router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_noema():
    """Mock NoemaEngine with required attributes."""
    noema = MagicMock()
    noema.neurosymbolic = MagicMock()
    noema.neurosymbolic.get_metrics.return_value = {
        "tasks_processed": 100,
        "tasks_successful": 95,
        "tasks_failed": 5,
        "success_rate": 0.95,
        "total_llm_calls": 200,
        "total_refinements": 50,
    }
    noema.worker_pool.stats = {"active": 4, "idle": 0}
    noema.memory_stats.return_value = {
        "episodic_count": 10,
        "procedural_count": 5,
        "knowledge_count": 20,
    }
    return noema


def test_admin_metrics_returns_stats(client, app, mock_noema):
    """GET /admin/metrics should return aggregated metrics."""
    app.state.noema = mock_noema

    with (
        patch("noema.api.server._start_time", 1000.0),
        patch("noema.api.admin.get_settings") as mock_settings,
    ):
        mock_settings.return_value.llm.provider = "openai"
        with patch("time.monotonic", return_value=1100.0):
            response = client.get("/admin/metrics")

    assert response.status_code == 200
    data = response.json()
    assert data["tasks"]["total"] == 100
    assert data["tasks"]["successful"] == 95
    assert data["tasks"]["failed"] == 5
    assert data["tasks"]["success_rate"] == 0.95
    assert data["llm"]["total_calls"] == 200
    assert data["llm"]["provider"] == "openai"
    assert data["workers"]["active"] == 4
    assert data["memory"]["episodic_count"] == 10
    assert data["uptime_s"] == 100.0


def test_admin_metrics_caches_result(client, app, mock_noema):
    """Metrics endpoint should cache results for 5 seconds."""

    import noema.api.admin as admin_module

    admin_module._metrics_cache = {}
    admin_module._metrics_cache_at = 0.0

    app.state.noema = mock_noema

    with (
        patch("noema.api.server._start_time", 1000.0),
        patch("noema.api.admin.get_settings") as mock_settings,
    ):
        mock_settings.return_value.llm.provider = "openai"
        with patch("time.monotonic", return_value=1100.0):
            response1 = client.get("/admin/metrics")
            response2 = client.get("/admin/metrics")

    assert response1.json() == response2.json()
    assert mock_noema.neurosymbolic.get_metrics.call_count == 1


def test_admin_metrics_no_noema_returns_503(client, app):
    """GET /admin/metrics should return 503 if noema not initialized."""
    import noema.api.admin as admin_module

    admin_module._metrics_cache = {}
    admin_module._metrics_cache_at = 0.0

    app.state.noema = None
    response = client.get("/admin/metrics")
    assert response.status_code == 503
    assert "Service starting" in response.json()["detail"]


def test_admin_task_history_with_tenant(client, app):
    """GET /admin/tasks/history with tenant_id should query that tenant."""
    mock_audit = AsyncMock()
    mock_audit.query = AsyncMock(
        return_value=[{"task_id": "t1", "timestamp": "2026-01-01T00:00:00"}]
    )
    app.state.audit_logger = mock_audit

    response = client.get("/admin/tasks/history?tenant_id=tenant1&limit=10")

    assert response.status_code == 200
    mock_audit.query.assert_called_once_with(tenant_id="tenant1", limit=10)


def test_admin_task_history_all_tenants(client, app, tmp_path):
    """GET /admin/tasks/history without tenant_id should query all tenants."""
    mock_audit = AsyncMock()
    mock_audit._fallback_dir = str(tmp_path)
    mock_audit.query = AsyncMock(return_value=[])

    (tmp_path / "tenant1.jsonl").write_text("")
    (tmp_path / "tenant2.jsonl").write_text("")

    app.state.audit_logger = mock_audit

    response = client.get("/admin/tasks/history?limit=10")

    assert response.status_code == 200
    assert response.json() == []


def test_admin_task_history_all_tenants_with_error(client, app, tmp_path):
    """_query_all_tenants should handle query errors gracefully."""
    mock_audit = AsyncMock()
    mock_audit._fallback_dir = str(tmp_path)

    def query_side_effect(tenant_id, limit):
        if tenant_id == "tenant1":
            raise Exception("Query failed")
        return [{"task_id": "t1", "timestamp": "2026-01-01T00:00:00"}]

    mock_audit.query = AsyncMock(side_effect=query_side_effect)

    (tmp_path / "tenant1.jsonl").write_text("")
    (tmp_path / "tenant2.jsonl").write_text("")

    app.state.audit_logger = mock_audit

    response = client.get("/admin/tasks/history?limit=10")

    assert response.status_code == 200
    assert len(response.json()) == 1


def test_admin_task_history_all_tenants_early_break(client, app, tmp_path):
    """_query_all_tenants should stop when max_results reached."""
    mock_audit = AsyncMock()
    mock_audit._fallback_dir = str(tmp_path)
    mock_audit.query = AsyncMock(
        return_value=[
            {"task_id": f"t{i}", "timestamp": f"2026-01-0{i}T00:00:00"} for i in range(1, 6)
        ]
    )

    (tmp_path / "tenant1.jsonl").write_text("")
    (tmp_path / "tenant2.jsonl").write_text("")
    (tmp_path / "tenant3.jsonl").write_text("")

    app.state.audit_logger = mock_audit

    response = client.get("/admin/tasks/history?limit=5")

    assert response.status_code == 200
    assert len(response.json()) == 5


def test_admin_task_history_no_audit_returns_503(client, app):
    """GET /admin/tasks/history should return 503 if audit logger not available."""
    app.state.audit_logger = None
    response = client.get("/admin/tasks/history")
    assert response.status_code == 503
    assert "Audit logger not available" in response.json()["detail"]


def test_admin_tenant_metrics_with_all_services(client, app):
    """GET /admin/tenants/{id}/metrics should return tenant-specific data."""
    mock_ff = AsyncMock()
    mock_ff.get_all_flags = AsyncMock(return_value={"feature_a": True})

    mock_quota = MagicMock()
    mock_quota.monthly_budget_usd = 100.0
    mock_quota.max_concurrent_tasks = 10
    mock_quota.max_tasks_per_hour = 50

    mock_qm = AsyncMock()
    mock_qm.get_quota = AsyncMock(return_value=mock_quota)

    mock_audit = AsyncMock()
    mock_audit.query = AsyncMock(return_value=[{"task_id": f"t{i}"} for i in range(5)])

    app.state.feature_flags = mock_ff
    app.state.quota_manager = mock_qm
    app.state.audit_logger = mock_audit

    response = client.get("/admin/tenants/tenant1/metrics")

    assert response.status_code == 200
    data = response.json()
    assert data["tenant_id"] == "tenant1"
    assert data["features"]["feature_a"] is True
    assert data["quota"]["monthly_budget_usd"] == 100.0
    assert data["recent_task_count"] == 5


def test_admin_tenant_metrics_feature_flags_error(client, app):
    """Feature flags errors should be logged but not fail the request."""
    mock_ff = AsyncMock()
    mock_ff.get_all_flags = AsyncMock(side_effect=Exception("DB error"))

    app.state.feature_flags = mock_ff
    app.state.quota_manager = None
    app.state.audit_logger = None

    response = client.get("/admin/tenants/tenant1/metrics")

    assert response.status_code == 200
    data = response.json()
    assert data["features"] == {}
    assert data["quota"] is None


def test_admin_tenant_metrics_quota_error(client, app):
    """Quota manager errors should be logged but not fail the request."""
    mock_qm = AsyncMock()
    mock_qm.get_quota = AsyncMock(side_effect=Exception("Quota DB error"))

    app.state.feature_flags = None
    app.state.quota_manager = mock_qm
    app.state.audit_logger = None

    response = client.get("/admin/tenants/tenant1/metrics")

    assert response.status_code == 200
    data = response.json()
    assert data["quota"] is None


def test_admin_tenant_metrics_audit_error(client, app):
    """Audit query errors should be logged but not fail the request."""
    mock_audit = AsyncMock()
    mock_audit.query = AsyncMock(side_effect=Exception("Audit error"))

    app.state.feature_flags = None
    app.state.quota_manager = None
    app.state.audit_logger = mock_audit

    response = client.get("/admin/tenants/tenant1/metrics")

    assert response.status_code == 200
    data = response.json()
    assert data["recent_task_count"] == 0


def test_admin_audit_proof_with_tenant(client, app):
    """GET /admin/audit/proof/{task_id} with tenant_id should use it."""
    mock_audit = AsyncMock()
    mock_audit.get_proof_for_task = AsyncMock(
        return_value={"root_hash": "abc123", "proof_path": []}
    )
    app.state.audit_logger = mock_audit

    response = client.get("/admin/audit/proof/task1?tenant_id=tenant1")

    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "task1"
    assert data["tenant_id"] == "tenant1"
    assert "proof" in data


def test_admin_audit_proof_without_tenant_uses_context(client, app):
    """GET /admin/audit/proof/{task_id} without tenant_id should use context var."""
    mock_audit = AsyncMock()
    mock_audit.get_proof_for_task = AsyncMock(return_value={"root_hash": "abc123"})
    app.state.audit_logger = mock_audit

    with patch("noema.context.get_tenant_id", return_value="context-tenant"):
        response = client.get("/admin/audit/proof/task1")

    assert response.status_code == 200
    data = response.json()
    assert data["tenant_id"] == "context-tenant"


def test_admin_audit_proof_not_found(client, app):
    """GET /admin/audit/proof/{task_id} should return 404 if proof not found."""
    mock_audit = AsyncMock()
    mock_audit.get_proof_for_task = AsyncMock(side_effect=ValueError("No proof found"))
    app.state.audit_logger = mock_audit

    response = client.get("/admin/audit/proof/task1?tenant_id=tenant1")

    assert response.status_code == 404
    assert "No proof found" in response.json()["detail"]


def test_admin_audit_proof_error(client, app):
    """GET /admin/audit/proof/{task_id} should return 500 on unexpected error."""
    mock_audit = AsyncMock()
    mock_audit.get_proof_for_task = AsyncMock(side_effect=Exception("Unexpected"))
    app.state.audit_logger = mock_audit

    response = client.get("/admin/audit/proof/task1?tenant_id=tenant1")

    assert response.status_code == 500
    assert "Failed to generate proof" in response.json()["detail"]


def test_admin_audit_verify_valid(client, app):
    """POST /admin/audit/verify should verify a valid inclusion proof."""
    mock_proof = MagicMock()
    mock_proof.block_index = 5
    mock_proof.root_hash = b"\x00\x01\x02\x03"

    with (
        patch("noema.audit.merkle_proof.InclusionProof") as mock_inclusion,
        patch("noema.audit.merkle_proof.verify_inclusion_proof", return_value=True),
    ):
        mock_inclusion.from_dict.return_value = mock_proof

        payload = {
            "proof": {"block_index": 5, "root_hash": "00010203", "proof_path": []},
            "leaf_data": {"task_id": "task1"},
        }
        response = client.post("/admin/audit/verify", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["block_index"] == 5


def test_admin_audit_verify_missing_proof(client, app):
    """POST /admin/audit/verify should return 422 if proof is missing."""
    response = client.post("/admin/audit/verify", json={"leaf_data": {}})
    assert response.status_code == 422
    assert "'proof' is required" in response.json()["detail"]


def test_admin_audit_verify_proof_not_object(client, app):
    """POST /admin/audit/verify should return 422 if proof is not an object."""
    response = client.post("/admin/audit/verify", json={"proof": "not-an-object"})
    assert response.status_code == 422
    assert "'proof' must be an object" in response.json()["detail"]


def test_admin_audit_verify_missing_leaf_data(client, app):
    """POST /admin/audit/verify should return 422 if leaf_data is missing."""
    response = client.post(
        "/admin/audit/verify", json={"proof": {"block_index": 5, "root_hash": "00010203"}}
    )
    assert response.status_code == 422
    assert "'leaf_data' is required" in response.json()["detail"]


def test_admin_audit_verify_invalid_proof_structure(client, app):
    """POST /admin/audit/verify should return 422 for invalid proof structure."""
    with patch("noema.audit.merkle_proof.InclusionProof") as mock_inclusion:
        mock_inclusion.from_dict.side_effect = KeyError("missing field")

        payload = {
            "proof": {"block_index": 5},
            "leaf_data": {"task_id": "task1"},
        }
        response = client.post("/admin/audit/verify", json=payload)

    assert response.status_code == 422
    assert "Invalid proof structure" in response.json()["detail"]


def test_admin_neurosymbolic_stats_enabled(client, app, mock_noema):
    """GET /admin/neurosymbolic/stats should return stats when enabled."""
    mock_noema.neurosymbolic.evolution = MagicMock()
    mock_noema.neurosymbolic.evolution.get_stats.return_value = {"generations": 10}

    app.state.noema = mock_noema

    with patch("noema.api.admin.get_settings") as mock_settings:
        mock_settings.return_value.neurosymbolic.enabled = True
        mock_settings.return_value.neurosymbolic.max_refinement_attempts = 3
        mock_settings.return_value.neurosymbolic.verification_timeout = 30
        mock_settings.return_value.neurosymbolic.evolution_enabled = True
        mock_settings.return_value.neurosymbolic.fallback_to_cot = True

        response = client.get("/admin/neurosymbolic/stats")

    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is True
    assert data["settings"]["max_refinement_attempts"] == 3
    assert data["metrics"]["tasks_processed"] == 100
    assert data["evolution"]["generations"] == 10


def test_admin_neurosymbolic_stats_disabled(client, app):
    """GET /admin/neurosymbolic/stats should return settings when disabled."""
    mock_noema = MagicMock()
    mock_noema.neurosymbolic = None
    app.state.noema = mock_noema

    with patch("noema.api.admin.get_settings") as mock_settings:
        mock_settings.return_value.neurosymbolic.enabled = False
        mock_settings.return_value.neurosymbolic.max_refinement_attempts = 3
        mock_settings.return_value.neurosymbolic.verification_timeout = 30
        mock_settings.return_value.neurosymbolic.evolution_enabled = False
        mock_settings.return_value.neurosymbolic.fallback_to_cot = False

        response = client.get("/admin/neurosymbolic/stats")

    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is False
    assert "metrics" not in data


def test_admin_neurosymbolic_stats_evolution_error(client, app, mock_noema):
    """Evolution stats errors should be caught and reported."""
    mock_noema.neurosymbolic.evolution = MagicMock()
    mock_noema.neurosymbolic.evolution.get_stats.side_effect = Exception("Evolution error")

    app.state.noema = mock_noema

    with patch("noema.api.admin.get_settings") as mock_settings:
        mock_settings.return_value.neurosymbolic.enabled = True
        mock_settings.return_value.neurosymbolic.max_refinement_attempts = 3
        mock_settings.return_value.neurosymbolic.verification_timeout = 30
        mock_settings.return_value.neurosymbolic.evolution_enabled = True
        mock_settings.return_value.neurosymbolic.fallback_to_cot = True

        response = client.get("/admin/neurosymbolic/stats")

    assert response.status_code == 200
    data = response.json()
    assert data["evolution"]["error"] == "unavailable"
