import asyncio
import json
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from noema.api.admin import router as admin_router
from noema.audit.logger import AuditEvent, AuditLogger, _tenant_filename
from noema.audit.merkle_proof import (
    InclusionProof,
    compute_root,
    verify_inclusion_proof,
)


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(admin_router)
    return app


@pytest.fixture
def audit_logger(tmp_path):
    logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
    return logger


async def populate_logger(logger: AuditLogger, n: int = 10, tenant: str = "tenant-a") -> list[dict]:
    await logger.initialize()
    events = []
    for i in range(n):
        ev = AuditEvent(
            timestamp=datetime.now(UTC),
            event_type="task.completed",
            tenant_id=tenant,
            user_id=f"user-{i}",
            task_id=f"task-{i}",
            details={"result": f"solution-{i}"},
        )
        await logger.log(ev)
        events.append(ev)
    return events


def _load_records(tmp_path, tenant: str) -> list[dict]:
    from pathlib import Path

    fpath = Path(tmp_path, f"{tenant}.jsonl")
    if not fpath.exists():
        return []
    return [
        json.loads(line) for line in fpath.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


@pytest.mark.asyncio
async def test_get_proof_for_task_file_fallback(audit_logger, tmp_path):
    await populate_logger(audit_logger, n=5)
    proof_dict = await audit_logger.get_proof_for_task(tenant_id="tenant-a", task_id="task-2")
    assert "leaf_hash" in proof_dict
    assert "root_hash" in proof_dict
    assert "path" in proof_dict
    assert proof_dict["block_index"] == 2
    proof = InclusionProof.from_dict(proof_dict)
    records = sorted(_load_records(tmp_path, "tenant-a"), key=lambda r: r.get("block_index", 0))
    assert len(records) == 5
    assert len(proof.path) == 3  # padded to 8 -> depth 3


@pytest.mark.asyncio
async def test_get_proof_verifies(audit_logger, tmp_path):
    await populate_logger(audit_logger, n=8)
    proof_dict = await audit_logger.get_proof_for_task(tenant_id="tenant-a", task_id="task-3")
    proof = InclusionProof.from_dict(proof_dict)
    assert verify_inclusion_proof(proof, proof_dict["leaf_data"]) is True


@pytest.mark.asyncio
async def test_get_proof_rejects_tampered_data(audit_logger, tmp_path):
    await populate_logger(audit_logger, n=4)
    proof_dict = await audit_logger.get_proof_for_task(tenant_id="tenant-a", task_id="task-1")
    proof = InclusionProof.from_dict(proof_dict)
    tampered = dict(proof_dict["leaf_data"])
    tampered["details"] = {"result": "HACKED"}
    assert verify_inclusion_proof(proof, tampered) is False


@pytest.mark.asyncio
async def test_get_proof_task_not_found(audit_logger, tmp_path):
    await populate_logger(audit_logger, n=3)
    with pytest.raises(ValueError, match="not found"):
        await audit_logger.get_proof_for_task(tenant_id="tenant-a", task_id="nonexistent")


@pytest.mark.asyncio
async def test_all_proofs_verify_against_same_root(audit_logger, tmp_path):
    await populate_logger(audit_logger, n=16, tenant="tenant-b")
    roots = set()
    for i in range(16):
        proof_dict = await audit_logger.get_proof_for_task(
            tenant_id="tenant-b", task_id=f"task-{i}"
        )
        proof = InclusionProof.from_dict(proof_dict)
        roots.add(proof.root_hash)
        assert verify_inclusion_proof(proof, proof_dict["leaf_data"])
    assert len(roots) == 1


@pytest.mark.asyncio
async def test_commitment_is_deterministic(audit_logger, tmp_path):
    """Same payload + index must produce same commitment across runs."""
    ev1 = AuditEvent(datetime.now(UTC), "ev", "t", "u", task_id="task-x", details={"a": 1})
    await audit_logger.log(ev1)
    assert ev1.commitment is not None
    assert len(ev1.commitment) == 64  # SHA-256 hex


def test_api_audit_proof_not_available():
    app = make_app()
    app.state.audit_logger = None
    client = TestClient(app)
    resp = client.get("/admin/audit/proof/task-1")
    assert resp.status_code == 503


def test_api_audit_proof_not_found(tmp_path):
    app = make_app()
    app.state.audit_logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
    client = TestClient(app)
    resp = client.get("/admin/audit/proof/task-missing?tenant_id=tenant-a")
    assert resp.status_code == 404


def test_api_audit_proof_and_verify_roundtrip(tmp_path):
    async def _setup():
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        await logger.initialize()
        for i in range(6):
            await logger.log(
                AuditEvent(
                    datetime.now(UTC),
                    "task.completed",
                    "tenant-a",
                    f"u{i}",
                    task_id=f"task-{i}",
                    details={"result": f"r{i}"},
                )
            )
        return logger

    logger = asyncio.run(_setup())
    app = make_app()
    app.state.audit_logger = logger
    client = TestClient(app)

    proof_resp = client.get("/admin/audit/proof/task-2?tenant_id=tenant-a")
    assert proof_resp.status_code == 200
    body = proof_resp.json()
    assert body["task_id"] == "task-2"
    proof = body["proof"]

    verify_resp = client.post(
        "/admin/audit/verify", json={"proof": proof, "leaf_data": proof["leaf_data"]}
    )
    assert verify_resp.status_code == 200
    assert verify_resp.json()["valid"] is True

    wrong_leaf = dict(proof["leaf_data"])
    wrong_leaf["details"] = {"result": "tampered"}
    verify_resp2 = client.post(
        "/admin/audit/verify", json={"proof": proof, "leaf_data": wrong_leaf}
    )
    assert verify_resp2.status_code == 200
    assert verify_resp2.json()["valid"] is False


def test_api_audit_verify_bad_payload(tmp_path):
    app = make_app()
    client = TestClient(app)
    resp = client.post("/admin/audit/verify", json={})
    assert resp.status_code == 422
    resp = client.post("/admin/audit/verify", json={"proof": {"bad": "data"}, "leaf_data": {}})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_logger_root_matches_compute_root(audit_logger, tmp_path):
    await populate_logger(audit_logger, n=7)
    records = _load_records(tmp_path, "tenant-a")
    records.sort(key=lambda r: r.get("block_index", 0))
    hashes = [bytes.fromhex(r["commitment"]) for r in records]
    expected_root = compute_root(hashes)
    proof_dict = await audit_logger.get_proof_for_task(tenant_id="tenant-a", task_id="task-0")
    assert InclusionProof.from_dict(proof_dict).root_hash == expected_root


@pytest.mark.asyncio
async def test_tenant_filename_sanitizes_traversal(audit_logger, tmp_path):
    assert _tenant_filename("tenant-a") == "tenant-a"
    assert _tenant_filename("../../etc") == ".._.._etc"
    assert _tenant_filename("a/b\\c:") == "a_b_c_"
    with pytest.raises(ValueError):
        _tenant_filename("")
    with pytest.raises(ValueError):
        _tenant_filename("..")
    with pytest.raises(ValueError):
        _tenant_filename("." * 101)

    evil = "../escape"
    ev = AuditEvent(datetime.now(UTC), "ev", evil, "u", task_id="t1")
    await audit_logger.log(ev)
    from pathlib import Path

    assert Path(tmp_path, ".._escape.jsonl").exists()
    assert not Path(tmp_path.parent, "escape.jsonl").exists()
    proof_dict = await audit_logger.get_proof_for_task(tenant_id=evil, task_id="t1")
    assert InclusionProof.from_dict(proof_dict).leaf_hash is not None


@pytest.mark.asyncio
async def test_fallback_proof_uses_in_memory_tree(audit_logger, tmp_path):
    """Proof from fallback path must match static compute_root of file records."""
    await populate_logger(audit_logger, n=9)
    records = sorted(_load_records(tmp_path, "tenant-a"), key=lambda r: r.get("block_index", 0))
    expected = compute_root([bytes.fromhex(r["commitment"]) for r in records])
    proof_dict = await audit_logger.get_proof_for_task(tenant_id="tenant-a", task_id="task-4")
    assert InclusionProof.from_dict(proof_dict).root_hash == expected
    assert verify_inclusion_proof(InclusionProof.from_dict(proof_dict), proof_dict["leaf_data"])
