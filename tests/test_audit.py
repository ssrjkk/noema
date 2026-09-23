"""Comprehensive tests for noema.audit.logger and noema.audit.merkle.

Covers AuditLogger (init, log, query, proof, DB integration, error handling,
file fallback) and MerkleChainAudit / AuditBlock / MerkleProof (building,
verification, edge cases, export/import).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from noema.audit.logger import (
    CREATE_AUDIT_TABLE_SQL,
    AuditEvent,
    AuditLogger,
    _tenant_filename,
)
from noema.audit.merkle import (
    AuditBlock,
    MerkleChainAudit,
    MerkleProof,
    _compute_block_hash,
    _compute_commitment,
    _hash,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_event(**overrides: Any) -> AuditEvent:
    defaults = {
        "timestamp": datetime.now(UTC),
        "event_type": "test.event",
        "tenant_id": "tenant-1",
        "user_id": "user-1",
        "task_id": "task-1",
        "details": {"key": "value"},
    }
    defaults.update(overrides)
    return AuditEvent(**defaults)


def _make_mock_pg() -> AsyncMock:
    """Create a mock PostgreSQL pool that succeeds by default."""
    pg = AsyncMock()
    pg.execute = AsyncMock()
    pg.fetch = AsyncMock(return_value=[])
    pg.fetchrow = AsyncMock(return_value=None)
    return pg


# ══════════════════════════════════════════════════════════════════════════════
# AuditLogger — Initialization
# ══════════════════════════════════════════════════════════════════════════════


class TestAuditLoggerInit:
    """AuditLogger initialization paths."""

    def test_init_defaults(self):
        logger = AuditLogger()
        assert logger.pg is None
        assert logger._fallback_dir == ".noema/audit"
        assert logger._initialized is False
        assert logger._file_fallback is False
        assert logger._leaf_hashes == []
        assert logger._tree is None

    def test_init_custom_params(self, tmp_path):
        pg = MagicMock()
        logger = AuditLogger(pg_pool=pg, fallback_dir=str(tmp_path))
        assert logger.pg is pg
        assert logger._fallback_dir == str(tmp_path)

    @pytest.mark.asyncio
    async def test_initialize_no_pg_falls_to_file(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        await logger.initialize()
        assert logger._initialized is True
        assert logger._file_fallback is True
        assert Path(tmp_path).exists()

    @pytest.mark.asyncio
    async def test_initialize_with_pg_success(self, tmp_path):
        pg = _make_mock_pg()
        logger = AuditLogger(pg_pool=pg, fallback_dir=str(tmp_path))
        await logger.initialize()
        assert logger._initialized is True
        assert logger._file_fallback is False
        # execute called for each SQL statement
        assert pg.execute.called

    @pytest.mark.asyncio
    async def test_initialize_pg_fails_falls_to_file(self, tmp_path):
        pg = _make_mock_pg()
        pg.execute.side_effect = Exception("connection refused")
        logger = AuditLogger(pg_pool=pg, fallback_dir=str(tmp_path))
        await logger.initialize()
        assert logger._initialized is True
        assert logger._file_fallback is True

    @pytest.mark.asyncio
    async def test_initialize_creates_fallback_dir(self, tmp_path):
        nested = str(tmp_path / "deep" / "nested" / "dir")
        logger = AuditLogger(pg_pool=None, fallback_dir=nested)
        await logger.initialize()
        assert Path(nested).exists()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        await logger.initialize()
        assert logger._initialized is True
        # Second call should not raise
        await logger.initialize()
        assert logger._initialized is True


# ══════════════════════════════════════════════════════════════════════════════
# AuditLogger — Logging events
# ══════════════════════════════════════════════════════════════════════════════


class TestAuditLoggerLog:
    """AuditLogger.log() writes and commitment chain."""

    @pytest.mark.asyncio
    async def test_log_auto_initializes(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        assert logger._initialized is False
        await logger.log(_make_event())
        assert logger._initialized is True

    @pytest.mark.asyncio
    async def test_log_sets_commitment_and_chain_link(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        ev = _make_event()
        await logger.log(ev)
        assert ev.commitment is not None
        assert len(ev.commitment) == 64  # hex of 32 bytes
        assert ev.chain_link is not None
        assert len(ev.chain_link) == 64
        assert ev.block_index == 0

    @pytest.mark.asyncio
    async def test_log_increments_block_index(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        for i in range(5):
            ev = _make_event(task_id=f"task-{i}")
            await logger.log(ev)
            assert ev.block_index == i

    @pytest.mark.asyncio
    async def test_log_writes_jsonl_file(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        await logger.log(_make_event(tenant_id="acme"))
        fpath = Path(tmp_path) / "acme.jsonl"
        assert fpath.exists()
        content = fpath.read_text(encoding="utf-8").strip()
        record = json.loads(content)
        assert record["tenant_id"] == "acme"
        assert record["event_type"] == "test.event"
        assert "commitment" in record

    @pytest.mark.asyncio
    async def test_log_multiple_events_same_tenant(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        for i in range(3):
            await logger.log(_make_event(task_id=f"t-{i}"))
        fpath = Path(tmp_path) / "tenant-1.jsonl"
        lines = [line for line in fpath.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 3

    @pytest.mark.asyncio
    async def test_log_different_tenants_different_files(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        await logger.log(_make_event(tenant_id="alpha"))
        await logger.log(_make_event(tenant_id="beta"))
        assert (Path(tmp_path) / "alpha.jsonl").exists()
        assert (Path(tmp_path) / "beta.jsonl").exists()

    @pytest.mark.asyncio
    async def test_log_with_pg_success(self, tmp_path):
        pg = _make_mock_pg()
        logger = AuditLogger(pg_pool=pg, fallback_dir=str(tmp_path))
        await logger.initialize()
        ev = _make_event()
        await logger.log(ev)
        assert pg.execute.called  # INSERT called
        assert ev.commitment is not None
        assert logger._file_fallback is False

    @pytest.mark.asyncio
    async def test_log_pg_write_fails_falls_to_file(self, tmp_path):
        pg = _make_mock_pg()
        logger = AuditLogger(pg_pool=pg, fallback_dir=str(tmp_path))
        await logger.initialize()
        # Make the INSERT fail
        pg.execute.side_effect = Exception("write failed")
        ev = _make_event()
        await logger.log(ev)
        assert logger._file_fallback is True
        # Should have written to fallback file
        fpath = Path(tmp_path) / "tenant-1.jsonl"
        assert fpath.exists()

    @pytest.mark.asyncio
    async def test_log_leaf_hashes_grow(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        for i in range(4):
            await logger.log(_make_event(task_id=f"t-{i}"))
        assert len(logger._leaf_hashes) == 4

    @pytest.mark.asyncio
    async def test_log_with_none_details(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        ev = AuditEvent(
            timestamp=datetime.now(UTC),
            event_type="ev",
            tenant_id="t",
            user_id="u",
            details=None,
        )
        await logger.log(ev)
        assert ev.commitment is not None

    @pytest.mark.asyncio
    async def test_log_with_ip_and_user_agent(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        ev = _make_event(ip_address="10.0.0.1", user_agent="NoemaTest/1.0")
        await logger.log(ev)
        fpath = Path(tmp_path) / "tenant-1.jsonl"
        record = json.loads(fpath.read_text(encoding="utf-8").strip())
        assert record["ip_address"] == "10.0.0.1"
        assert record["user_agent"] == "NoemaTest/1.0"


# ══════════════════════════════════════════════════════════════════════════════
# AuditLogger — Querying
# ══════════════════════════════════════════════════════════════════════════════


class TestAuditLoggerQuery:
    """AuditLogger.query() paths."""

    @pytest.mark.asyncio
    async def test_query_empty_returns_empty(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        await logger.initialize()
        results = await logger.query("nonexistent")
        assert results == []

    @pytest.mark.asyncio
    async def test_query_by_tenant(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        await logger.log(_make_event(tenant_id="a"))
        await logger.log(_make_event(tenant_id="b"))
        await logger.log(_make_event(tenant_id="a"))
        results = await logger.query("a")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_by_event_type(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        await logger.log(_make_event(event_type="login"))
        await logger.log(_make_event(event_type="logout"))
        await logger.log(_make_event(event_type="login"))
        results = await logger.query("tenant-1", event_type="login")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_by_start_time(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        now = datetime.now(UTC)
        await logger.log(_make_event(timestamp=now - timedelta(hours=3)))
        await logger.log(_make_event(timestamp=now))
        results = await logger.query("tenant-1", start_time=now - timedelta(hours=1))
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_by_end_time(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        now = datetime.now(UTC)
        await logger.log(_make_event(timestamp=now - timedelta(hours=3)))
        await logger.log(_make_event(timestamp=now))
        results = await logger.query("tenant-1", end_time=now - timedelta(hours=1))
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_by_time_range(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        now = datetime.now(UTC)
        await logger.log(_make_event(timestamp=now - timedelta(hours=5)))
        await logger.log(_make_event(timestamp=now - timedelta(hours=2)))
        await logger.log(_make_event(timestamp=now))
        results = await logger.query(
            "tenant-1",
            start_time=now - timedelta(hours=3),
            end_time=now - timedelta(hours=1),
        )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_limit(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        for i in range(10):
            await logger.log(_make_event(task_id=f"t-{i}"))
        results = await logger.query("tenant-1", limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_query_results_sorted_desc_by_time(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        now = datetime.now(UTC)
        await logger.log(_make_event(timestamp=now - timedelta(hours=2), task_id="old"))
        await logger.log(_make_event(timestamp=now, task_id="new"))
        results = await logger.query("tenant-1")
        assert results[0]["task_id"] == "new"

    @pytest.mark.asyncio
    async def test_query_with_pg(self, tmp_path):
        pg = _make_mock_pg()
        mock_row = {"id": 1, "event_type": "test", "tenant_id": "t1"}
        pg.fetch = AsyncMock(return_value=[mock_row])
        logger = AuditLogger(pg_pool=pg, fallback_dir=str(tmp_path))
        await logger.initialize()
        results = await logger.query("t1")
        assert len(results) == 1
        assert results[0]["event_type"] == "test"

    @pytest.mark.asyncio
    async def test_query_pg_fails_falls_to_file(self, tmp_path):
        pg = _make_mock_pg()
        logger = AuditLogger(pg_pool=pg, fallback_dir=str(tmp_path))
        await logger.initialize()
        # Force file fallback for logging so data is in the JSONL file
        logger._file_fallback = True
        await logger.log(_make_event())
        # Now make pg query fail — should fall to file-based query
        results = await logger.query("tenant-1")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_query_with_all_filters_combined(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        now = datetime.now(UTC)
        await logger.log(
            _make_event(timestamp=now - timedelta(hours=4), event_type="a", task_id="1")
        )
        await logger.log(
            _make_event(timestamp=now - timedelta(hours=2), event_type="b", task_id="2")
        )
        await logger.log(_make_event(timestamp=now, event_type="a", task_id="3"))
        results = await logger.query(
            "tenant-1",
            event_type="a",
            start_time=now - timedelta(hours=3),
            end_time=now + timedelta(hours=1),
            limit=10,
        )
        assert len(results) == 1
        assert results[0]["task_id"] == "3"


# ══════════════════════════════════════════════════════════════════════════════
# AuditLogger — _tenant_filename
# ══════════════════════════════════════════════════════════════════════════════


class TestTenantFilename:
    """_tenant_filename sanitization."""

    def test_normal_tenant(self):
        assert _tenant_filename("my-tenant") == "my-tenant"

    def test_dots_and_underscores_preserved(self):
        assert _tenant_filename("a.b_c") == "a.b_c"

    def test_slashes_replaced(self):
        assert _tenant_filename("a/b\\c") == "a_b_c"

    def test_special_chars_replaced(self):
        result = _tenant_filename("ten@nt#id!")
        assert "/" not in result
        assert "\\" not in result
        assert "@" not in result

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            _tenant_filename("")

    def test_dot_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            _tenant_filename(".")

    def test_double_dot_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            _tenant_filename("..")

    def test_too_long_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            _tenant_filename("x" * 101)

    def test_exactly_100_ok(self):
        result = _tenant_filename("a" * 100)
        assert len(result) == 100

    def test_path_traversal_sanitized(self):
        result = _tenant_filename("../../etc/passwd")
        # Slashes are replaced with underscores, dots are preserved
        assert "/" not in result
        assert "\\" not in result
        assert result == ".._.._etc_passwd"


# ══════════════════════════════════════════════════════════════════════════════
# AuditLogger — _build_leaf_data
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildLeafData:
    """AuditLogger._build_leaf_data for AuditEvent and dict inputs."""

    def test_build_from_audit_event(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        ev = _make_event(task_id="t1", details={"x": 1})
        data = logger._build_leaf_data(ev)
        assert data["event_type"] == "test.event"
        assert data["tenant_id"] == "tenant-1"
        assert data["user_id"] == "user-1"
        assert data["task_id"] == "t1"
        assert data["details"] == {"x": 1}
        assert "timestamp" in data

    def test_build_from_dict(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        now = datetime.now(UTC)
        d = {
            "timestamp": now,
            "event_type": "ev",
            "tenant_id": "t",
            "user_id": "u",
            "task_id": "t1",
            "details": {"a": 1},
        }
        data = logger._build_leaf_data(d)
        assert data["event_type"] == "ev"
        assert data["task_id"] == "t1"

    def test_build_from_dict_string_timestamp(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        d = {
            "timestamp": "2024-01-01T00:00:00+00:00",
            "event_type": "ev",
            "tenant_id": "t",
            "user_id": "u",
        }
        data = logger._build_leaf_data(d)
        assert "2024-01-01" in data["timestamp"]

    def test_build_from_event_none_details(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        ev = AuditEvent(
            timestamp=datetime.now(UTC),
            event_type="ev",
            tenant_id="t",
            user_id="u",
            details=None,
        )
        data = logger._build_leaf_data(ev)
        assert data["details"] == {}

    def test_build_from_dict_none_details(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        d = {
            "timestamp": datetime.now(UTC),
            "event_type": "ev",
            "tenant_id": "t",
            "user_id": "u",
            "details": None,
        }
        data = logger._build_leaf_data(d)
        assert data["details"] == {}


# ══════════════════════════════════════════════════════════════════════════════
# AuditLogger — _load_leaf_hashes / _load_leaf_hashes_fallback
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadLeafHashes:
    """Leaf hash loading from DB and file fallback."""

    @pytest.mark.asyncio
    async def test_load_from_pg_success(self, tmp_path):
        pg = _make_mock_pg()
        pg.fetch = AsyncMock(
            return_value=[
                {"commitment": "aa" * 32, "block_index": 0},
                {"commitment": "bb" * 32, "block_index": 1},
            ]
        )
        logger = AuditLogger(pg_pool=pg, fallback_dir=str(tmp_path))
        await logger.initialize()
        assert len(logger._leaf_hashes) == 2
        assert logger._leaf_hashes[0] == bytes.fromhex("aa" * 32)

    @pytest.mark.asyncio
    async def test_load_from_pg_failure_empty(self, tmp_path):
        pg = _make_mock_pg()
        pg.fetch = AsyncMock(side_effect=Exception("fail"))
        logger = AuditLogger(pg_pool=pg, fallback_dir=str(tmp_path))
        await logger.initialize()
        # Falls back to file mode, leaf_hashes empty
        assert logger._leaf_hashes == []

    @pytest.mark.asyncio
    async def test_load_fallback_from_jsonl(self, tmp_path):
        # Pre-populate a JSONL file
        fpath = Path(tmp_path) / "t1.jsonl"
        record = {
            "block_index": 0,
            "commitment": "cc" * 32,
            "event_type": "ev",
            "tenant_id": "t1",
            "user_id": "u",
            "timestamp": "2024-01-01T00:00:00+00:00",
        }
        fpath.write_text(json.dumps(record) + "\n", encoding="utf-8")
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        await logger.initialize()
        assert len(logger._leaf_hashes) == 1
        assert logger._leaf_hashes[0] == bytes.fromhex("cc" * 32)

    @pytest.mark.asyncio
    async def test_load_fallback_skips_bad_json(self, tmp_path):
        fpath = Path(tmp_path) / "t1.jsonl"
        fpath.write_text(
            "not json\n" + json.dumps({"commitment": "dd" * 32, "block_index": 0}) + "\n",
            encoding="utf-8",
        )
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        await logger.initialize()
        assert len(logger._leaf_hashes) == 1

    @pytest.mark.asyncio
    async def test_load_fallback_skips_bad_hex(self, tmp_path):
        fpath = Path(tmp_path) / "t1.jsonl"
        fpath.write_text(
            json.dumps({"commitment": "not-hex!", "block_index": 0}) + "\n",
            encoding="utf-8",
        )
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        await logger.initialize()
        assert len(logger._leaf_hashes) == 0

    @pytest.mark.asyncio
    async def test_load_fallback_no_dir(self, tmp_path):
        nonexistent = str(tmp_path / "nope")
        logger = AuditLogger(pg_pool=None, fallback_dir=nonexistent)
        await logger.initialize()
        assert logger._leaf_hashes == []

    @pytest.mark.asyncio
    async def test_load_fallback_multiple_files(self, tmp_path):
        for tenant in ["a", "b"]:
            fpath = Path(tmp_path) / f"{tenant}.jsonl"
            fpath.write_text(
                json.dumps({"commitment": "ee" * 32, "block_index": 0}) + "\n",
                encoding="utf-8",
            )
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        await logger.initialize()
        # Both files loaded, sorted by block_index
        assert len(logger._leaf_hashes) == 2


# ══════════════════════════════════════════════════════════════════════════════
# AuditLogger — get_proof_for_task (DB path)
# ══════════════════════════════════════════════════════════════════════════════


class TestGetProofForTaskDB:
    """get_proof_for_task with mocked PostgreSQL."""

    @pytest.mark.asyncio
    async def test_proof_pg_path_not_found_raises(self, tmp_path):
        pg = _make_mock_pg()
        pg.fetchrow = AsyncMock(return_value=None)
        logger = AuditLogger(pg_pool=pg, fallback_dir=str(tmp_path))
        await logger.initialize()
        with pytest.raises(ValueError, match="not found"):
            await logger.get_proof_for_task("t1", "nonexistent")

    @pytest.mark.asyncio
    async def test_proof_pg_exception_falls_to_file(self, tmp_path):
        pg = _make_mock_pg()
        logger = AuditLogger(pg_pool=pg, fallback_dir=str(tmp_path))
        await logger.initialize()
        # Force file fallback for logging so data is in the JSONL file
        logger._file_fallback = True
        await logger.log(_make_event(task_id="task-x"))
        # Reset to PG mode; make fetchrow raise a non-ValueError exception
        logger._file_fallback = False
        pg.fetchrow = AsyncMock(side_effect=RuntimeError("db connection lost"))
        result = await logger.get_proof_for_task("tenant-1", "task-x")
        assert "leaf_data" in result


# ══════════════════════════════════════════════════════════════════════════════
# AuditLogger — _write_fallback error handling
# ══════════════════════════════════════════════════════════════════════════════


class TestWriteFallbackErrors:
    """_write_fallback OSError handling."""

    @pytest.mark.asyncio
    async def test_write_to_readonly_dir(self, tmp_path):
        readonly_dir = str(tmp_path / "readonly")
        Path(readonly_dir).mkdir()
        logger = AuditLogger(pg_pool=None, fallback_dir=readonly_dir)
        await logger.initialize()
        # Make directory read-only
        import os

        os.chmod(readonly_dir, 0o444)
        try:
            ev = _make_event()
            # Should not raise (error is logged)
            await logger.log(ev)
        finally:
            os.chmod(readonly_dir, 0o755)


# ══════════════════════════════════════════════════════════════════════════════
# AuditLogger — _get_prev_root and _sync_tree
# ══════════════════════════════════════════════════════════════════════════════


class TestSyncTreeAndPrevRoot:
    """_sync_tree and _get_prev_root internals."""

    def test_get_prev_root_empty(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        root = logger._get_prev_root()
        assert root == b"\x00" * 32

    @pytest.mark.asyncio
    async def test_sync_tree_builds_from_hashes(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        await logger.initialize()
        logger._leaf_hashes = [b"\x01" * 32, b"\x02" * 32]
        logger._tree = None
        logger._sync_tree()
        assert logger._tree is not None
        assert logger._tree.count == 2

    @pytest.mark.asyncio
    async def test_sync_tree_rebuilds_on_count_mismatch(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        await logger.initialize()
        logger._leaf_hashes = [b"\x01" * 32]
        logger._sync_tree()
        assert logger._tree.count == 1
        logger._leaf_hashes.append(b"\x02" * 32)
        logger._sync_tree()
        assert logger._tree.count == 2


# ══════════════════════════════════════════════════════════════════════════════
# AuditEvent dataclass
# ══════════════════════════════════════════════════════════════════════════════


class TestAuditEvent:
    """AuditEvent dataclass behavior."""

    def test_defaults(self):
        ev = AuditEvent(
            timestamp=datetime.now(UTC),
            event_type="test",
            tenant_id="t",
            user_id="u",
        )
        assert ev.details == {}
        assert ev.task_id is None
        assert ev.ip_address is None
        assert ev.user_agent is None
        assert ev.commitment is None
        assert ev.chain_link is None
        assert ev.block_index is None

    def test_custom_values(self):
        now = datetime.now(UTC)
        ev = AuditEvent(
            timestamp=now,
            event_type="login",
            tenant_id="t1",
            user_id="u1",
            task_id="task-1",
            details={"a": 1},
            ip_address="1.2.3.4",
            user_agent="TestAgent",
        )
        assert ev.event_type == "login"
        assert ev.details == {"a": 1}
        assert ev.ip_address == "1.2.3.4"

    def test_post_init_sets_none_details_to_empty(self):
        ev = AuditEvent(
            timestamp=datetime.now(UTC),
            event_type="e",
            tenant_id="t",
            user_id="u",
            details=None,
        )
        assert ev.details == {}


# ══════════════════════════════════════════════════════════════════════════════
# CREATE_AUDIT_TABLE_SQL
# ══════════════════════════════════════════════════════════════════════════════


class TestCreateAuditTableSQL:
    """SQL constant structure."""

    def test_contains_create_table(self):
        assert "CREATE TABLE IF NOT EXISTS audit_log" in CREATE_AUDIT_TABLE_SQL

    def test_contains_indexes(self):
        assert "idx_audit_tenant_time" in CREATE_AUDIT_TABLE_SQL
        assert "idx_audit_task" in CREATE_AUDIT_TABLE_SQL
        assert "idx_audit_event_type" in CREATE_AUDIT_TABLE_SQL
        assert "idx_audit_block_index" in CREATE_AUDIT_TABLE_SQL

    def test_contains_alter_columns(self):
        assert "ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS commitment" in CREATE_AUDIT_TABLE_SQL
        assert "ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS chain_link" in CREATE_AUDIT_TABLE_SQL
        assert (
            "ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS block_index" in CREATE_AUDIT_TABLE_SQL
        )


# ══════════════════════════════════════════════════════════════════════════════
# merkle.py — _hash, _compute_commitment, _compute_block_hash
# ══════════════════════════════════════════════════════════════════════════════


class TestMerkleHashFunctions:
    """Low-level hash primitives in merkle.py."""

    def test_hash_deterministic(self):
        assert _hash(b"hello") == _hash(b"hello")

    def test_hash_different_inputs(self):
        assert _hash(b"a") != _hash(b"b")

    def test_hash_length(self):
        assert len(_hash(b"test")) == 32

    def test_hash_empty_input(self):
        h = _hash(b"")
        assert len(h) == 32

    def test_compute_commitment_deterministic(self):
        c1 = _compute_commitment({"key": "val"})
        c2 = _compute_commitment({"key": "val"})
        assert c1 == c2

    def test_compute_commitment_different_payloads(self):
        c1 = _compute_commitment({"a": 1})
        c2 = _compute_commitment({"a": 2})
        assert c1 != c2

    def test_compute_commitment_with_nonce(self):
        nonce = b"\x01" * 16
        c1 = _compute_commitment({"x": 1}, nonce=nonce)
        c2 = _compute_commitment({"x": 1}, nonce=nonce)
        assert c1 == c2

    def test_compute_commitment_different_nonces(self):
        c1 = _compute_commitment({"x": 1}, nonce=b"\x01" * 16)
        c2 = _compute_commitment({"x": 1}, nonce=b"\x02" * 16)
        assert c1 != c2

    def test_compute_commitment_none_nonce(self):
        c = _compute_commitment({"x": 1}, nonce=None)
        assert len(c) == 32

    def test_compute_block_hash_deterministic(self):
        h1 = _compute_block_hash(0, b"\x00" * 32, b"\x01" * 32, 1000.0, b"meta")
        h2 = _compute_block_hash(0, b"\x00" * 32, b"\x01" * 32, 1000.0, b"meta")
        assert h1 == h2

    def test_compute_block_hash_different_index(self):
        h1 = _compute_block_hash(0, b"\x00" * 32, b"\x01" * 32, 1000.0, b"")
        h2 = _compute_block_hash(1, b"\x00" * 32, b"\x01" * 32, 1000.0, b"")
        assert h1 != h2

    def test_compute_block_hash_different_prev(self):
        h1 = _compute_block_hash(0, b"\x00" * 32, b"\x01" * 32, 1000.0, b"")
        h2 = _compute_block_hash(0, b"\xff" * 32, b"\x01" * 32, 1000.0, b"")
        assert h1 != h2


# ══════════════════════════════════════════════════════════════════════════════
# merkle.py — AuditBlock
# ══════════════════════════════════════════════════════════════════════════════


class TestAuditBlock:
    """AuditBlock dataclass and verify()."""

    def test_verify_valid_block(self):
        commitment = _compute_commitment({"event": "test"})
        block_hash = _compute_block_hash(1, b"\x00" * 32, commitment, 100.0, b"")
        block = AuditBlock(
            index=1,
            prev_hash=b"\x00" * 32,
            timestamp=100.0,
            commitment=commitment,
            metadata=b"",
            block_hash=block_hash,
        )
        assert block.verify() is True

    def test_verify_tampered_commitment(self):
        commitment = _compute_commitment({"event": "test"})
        block_hash = _compute_block_hash(1, b"\x00" * 32, commitment, 100.0, b"")
        block = AuditBlock(
            index=1,
            prev_hash=b"\x00" * 32,
            timestamp=100.0,
            commitment=b"\xff" * 32,  # tampered
            metadata=b"",
            block_hash=block_hash,
        )
        assert block.verify() is False

    def test_verify_tampered_timestamp(self):
        commitment = _compute_commitment({"event": "test"})
        block_hash = _compute_block_hash(1, b"\x00" * 32, commitment, 100.0, b"")
        block = AuditBlock(
            index=1,
            prev_hash=b"\x00" * 32,
            timestamp=999.0,  # tampered
            commitment=commitment,
            metadata=b"",
            block_hash=block_hash,
        )
        assert block.verify() is False

    def test_verify_tampered_prev_hash(self):
        commitment = _compute_commitment({"event": "test"})
        block_hash = _compute_block_hash(1, b"\x00" * 32, commitment, 100.0, b"")
        block = AuditBlock(
            index=1,
            prev_hash=b"\xab" * 32,  # tampered
            timestamp=100.0,
            commitment=commitment,
            metadata=b"",
            block_hash=block_hash,
        )
        assert block.verify() is False

    def test_frozen_dataclass(self):
        commitment = _compute_commitment({"event": "test"})
        block_hash = _compute_block_hash(0, b"\x00" * 32, commitment, 0.0, b"genesis")
        block = AuditBlock(
            index=0,
            prev_hash=b"\x00" * 32,
            timestamp=0.0,
            commitment=commitment,
            metadata=b"genesis",
            block_hash=block_hash,
        )
        with pytest.raises(AttributeError):
            block.index = 99  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
# merkle.py — MerkleProof (standalone)
# ══════════════════════════════════════════════════════════════════════════════


class TestMerkleProofDataclass:
    """MerkleProof.verify() standalone."""

    def test_verify_single_leaf_no_siblings(self):
        leaf = _hash(b"leaf")
        proof = MerkleProof(block_index=0, block_hash=leaf, siblings=[])
        assert proof.verify(root=leaf, leaf=leaf) is True

    def test_verify_wrong_root_fails(self):
        leaf = _hash(b"leaf")
        proof = MerkleProof(block_index=0, block_hash=leaf, siblings=[])
        assert proof.verify(root=b"\xff" * 32, leaf=leaf) is False

    def test_verify_with_sibling(self):
        left = _hash(b"left")
        right = _hash(b"right")
        root = _hash(left + right)
        # Proof for left leaf: sibling is right, is_left=False (sibling on right)
        proof = MerkleProof(block_index=0, block_hash=left, siblings=[(right, False)])
        assert proof.verify(root=root, leaf=left) is True

    def test_verify_with_sibling_wrong_order(self):
        left = _hash(b"left")
        right = _hash(b"right")
        root = _hash(left + right)
        # Wrong: says sibling is on left but it's actually on right
        proof = MerkleProof(block_index=0, block_hash=left, siblings=[(right, True)])
        assert proof.verify(root=root, leaf=left) is False


# ══════════════════════════════════════════════════════════════════════════════
# merkle.py — MerkleChainAudit
# ══════════════════════════════════════════════════════════════════════════════


class TestMerkleChainAuditInit:
    """MerkleChainAudit initialization."""

    def test_default_chain_id(self):
        chain = MerkleChainAudit()
        assert len(chain.chain_id) == 8

    def test_custom_chain_id(self):
        chain = MerkleChainAudit(chain_id="my-chain")
        assert chain.chain_id == "my-chain"

    def test_genesis_block_exists(self):
        chain = MerkleChainAudit()
        assert chain.height == 1
        assert chain.tip.index == 0

    def test_genesis_prev_hash_is_zero(self):
        chain = MerkleChainAudit()
        genesis = chain.get_block(0)
        assert genesis is not None
        assert genesis.prev_hash == b"\x00" * 32

    def test_genesis_verifies(self):
        chain = MerkleChainAudit()
        assert chain.verify_chain() is True

    def test_genesis_metadata(self):
        chain = MerkleChainAudit()
        genesis = chain.get_block(0)
        assert genesis is not None
        assert genesis.metadata == b"genesis"

    def test_empty_chain_id_string_generates_default(self):
        chain = MerkleChainAudit(chain_id="")
        # empty string is falsy, so default is used
        assert len(chain.chain_id) == 8


class TestMerkleChainAuditAppend:
    """MerkleChainAudit.append() behavior."""

    def test_append_increases_height(self):
        chain = MerkleChainAudit()
        chain.append({"event": "test"})
        assert chain.height == 2

    def test_append_returns_block(self):
        chain = MerkleChainAudit()
        block = chain.append({"event": "test"})
        assert isinstance(block, AuditBlock)
        assert block.index == 1

    def test_append_multiple(self):
        chain = MerkleChainAudit()
        for i in range(10):
            block = chain.append({"event": f"e{i}"})
            assert block.index == i + 1
        assert chain.height == 11

    def test_append_chain_still_verifies(self):
        chain = MerkleChainAudit()
        for i in range(5):
            chain.append({"event": f"e{i}"})
        assert chain.verify_chain() is True

    def test_append_with_metadata(self):
        chain = MerkleChainAudit()
        block = chain.append({"event": "test"}, metadata={"key": "val"})
        assert block.metadata != b""

    def test_append_without_metadata(self):
        chain = MerkleChainAudit()
        block = chain.append({"event": "test"}, metadata=None)
        assert isinstance(block.metadata, bytes)

    def test_tip_updates_after_append(self):
        chain = MerkleChainAudit()
        block = chain.append({"event": "test"})
        assert chain.tip == block
        assert chain.tip_hash == block.block_hash

    def test_append_invalidates_tree_cache(self):
        chain = MerkleChainAudit()
        chain.append({"event": "e1"})
        root1 = chain.root
        chain.append({"event": "e2"})
        root2 = chain.root
        assert root1 != root2


class TestMerkleChainAuditVerify:
    """MerkleChainAudit.verify_chain() behavior."""

    def test_valid_chain(self):
        chain = MerkleChainAudit()
        for i in range(5):
            chain.append({"event": f"e{i}"})
        assert chain.verify_chain() is True

    def test_tampered_commitment_detected(self):
        chain = MerkleChainAudit()
        chain.append({"event": "e1"})
        chain.append({"event": "e2"})
        # Tamper with middle block's commitment
        original = chain._blocks[1]
        tampered = AuditBlock(
            index=original.index,
            prev_hash=original.prev_hash,
            timestamp=original.timestamp,
            commitment=b"\xff" * 32,
            metadata=original.metadata,
            block_hash=original.block_hash,
        )
        chain._blocks[1] = tampered
        assert chain.verify_chain() is False

    def test_broken_link_detected(self):
        chain = MerkleChainAudit()
        chain.append({"event": "e1"})
        chain.append({"event": "e2"})
        # Break prev_hash link
        original = chain._blocks[2]
        tampered = AuditBlock(
            index=original.index,
            prev_hash=b"\xab" * 32,  # wrong
            timestamp=original.timestamp,
            commitment=original.commitment,
            metadata=original.metadata,
            block_hash=original.block_hash,
        )
        chain._blocks[2] = tampered
        assert chain.verify_chain() is False


class TestMerkleChainAuditProofs:
    """MerkleChainAudit.prove_inclusion() and verify_proof()."""

    def test_prove_genesis(self):
        chain = MerkleChainAudit()
        proof = chain.prove_inclusion(0)
        assert proof is not None
        assert proof.block_index == 0
        assert chain.verify_proof(proof) is True

    def test_prove_single_block_chain(self):
        chain = MerkleChainAudit()
        # Only genesis
        proof = chain.prove_inclusion(0)
        assert proof is not None
        assert proof.siblings == []

    def test_prove_each_block(self):
        chain = MerkleChainAudit()
        for i in range(7):
            chain.append({"event": f"e{i}"})
        for idx in range(chain.height):
            proof = chain.prove_inclusion(idx)
            assert proof is not None
            assert chain.verify_proof(proof) is True

    def test_prove_nonexistent_negative(self):
        chain = MerkleChainAudit()
        assert chain.prove_inclusion(-1) is None

    def test_prove_nonexistent_too_large(self):
        chain = MerkleChainAudit()
        assert chain.prove_inclusion(100) is None

    def test_verify_proof_wrong_block_hash(self):
        chain = MerkleChainAudit()
        chain.append({"event": "e1"})
        proof = chain.prove_inclusion(0)
        assert proof is not None
        # Tamper with the proof's block_hash
        wrong_proof = MerkleProof(
            block_index=proof.block_index,
            block_hash=b"\xff" * 32,
            siblings=proof.siblings,
        )
        assert chain.verify_proof(wrong_proof) is False

    def test_verify_proof_index_out_of_range(self):
        chain = MerkleChainAudit()
        chain.append({"event": "e1"})
        proof = chain.prove_inclusion(0)
        assert proof is not None
        # Create proof with out-of-range index
        bad_proof = MerkleProof(
            block_index=999,
            block_hash=proof.block_hash,
            siblings=proof.siblings,
        )
        assert chain.verify_proof(bad_proof) is False

    def test_proof_for_various_chain_sizes(self):
        for size in [1, 2, 3, 4, 5, 8, 15, 16, 17, 32]:
            chain = MerkleChainAudit()
            for i in range(size):
                chain.append({"event": f"e{i}"})
            for idx in [0, size // 2, size]:
                proof = chain.prove_inclusion(idx)
                assert proof is not None, f"Failed for size={size}, idx={idx}"
                assert chain.verify_proof(proof), f"Verify failed for size={size}, idx={idx}"


class TestMerkleChainAuditGetBlock:
    """MerkleChainAudit.get_block() behavior."""

    def test_get_existing_block(self):
        chain = MerkleChainAudit()
        chain.append({"event": "e1"})
        block = chain.get_block(0)
        assert block is not None
        assert block.index == 0

    def test_get_genesis(self):
        chain = MerkleChainAudit()
        block = chain.get_block(0)
        assert block is not None
        assert block.metadata == b"genesis"

    def test_get_nonexistent_negative(self):
        chain = MerkleChainAudit()
        assert chain.get_block(-1) is None

    def test_get_nonexistent_too_large(self):
        chain = MerkleChainAudit()
        assert chain.get_block(100) is None


class TestMerkleChainAuditRoot:
    """MerkleChainAudit.root property."""

    def test_root_single_block(self):
        chain = MerkleChainAudit()
        root = chain.root
        assert len(root) == 32

    def test_root_changes_on_append(self):
        chain = MerkleChainAudit()
        root1 = chain.root
        chain.append({"event": "e1"})
        root2 = chain.root
        assert root1 != root2

    def test_root_deterministic(self):
        c1 = MerkleChainAudit(chain_id="det")
        c2 = MerkleChainAudit(chain_id="det")
        # Same genesis -> same root
        assert c1.root == c2.root


class TestMerkleChainAuditExportImport:
    """MerkleChainAudit export/import."""

    def test_export_structure(self):
        chain = MerkleChainAudit()
        chain.append({"event": "e1"})
        exported = chain.export_blocks()
        assert len(exported) == 2
        assert "index" in exported[0]
        assert "prev_hash" in exported[0]
        assert "block_hash" in exported[0]
        assert "commitment" in exported[0]

    def test_export_import_roundtrip(self):
        chain = MerkleChainAudit(chain_id="rt")
        for i in range(5):
            chain.append({"event": f"e{i}"})
        exported = chain.export_blocks()
        imported = MerkleChainAudit.import_blocks("rt", exported)
        assert imported.chain_id == "rt"
        assert imported.height == chain.height
        assert imported.root == chain.root
        assert imported.verify_chain() is True

    def test_import_skips_genesis_in_data(self):
        chain = MerkleChainAudit(chain_id="sk")
        chain.append({"event": "e1"})
        exported = chain.export_blocks()
        # exported[0] is genesis (index=0), should be skipped
        imported = MerkleChainAudit.import_blocks("sk", exported)
        assert imported.height == 2

    def test_import_tampered_raises(self):
        chain = MerkleChainAudit(chain_id="tp")
        chain.append({"event": "e1"})
        exported = chain.export_blocks()
        # Tamper with block_hash
        exported[1]["block_hash"] = "ff" * 32
        with pytest.raises(ValueError, match="failed chain verification"):
            MerkleChainAudit.import_blocks("tp", exported)

    def test_import_broken_link_raises(self):
        chain = MerkleChainAudit(chain_id="bl")
        chain.append({"event": "e1"})
        chain.append({"event": "e2"})
        exported = chain.export_blocks()
        # Break prev_hash of block 2
        exported[2]["prev_hash"] = "ab" * 32
        with pytest.raises(ValueError, match="failed chain verification"):
            MerkleChainAudit.import_blocks("bl", exported)

    def test_import_empty_blocks(self):
        chain = MerkleChainAudit(chain_id="em")
        exported = chain.export_blocks()
        imported = MerkleChainAudit.import_blocks("em", exported)
        assert imported.height == 1  # just genesis

    def test_export_hex_strings(self):
        chain = MerkleChainAudit()
        chain.append({"event": "e1"})
        exported = chain.export_blocks()
        for block_data in exported:
            # All hash fields should be valid hex
            bytes.fromhex(block_data["prev_hash"])
            bytes.fromhex(block_data["commitment"])
            bytes.fromhex(block_data["block_hash"])


class TestMerkleChainAuditToDict:
    """MerkleChainAudit.to_dict() summary."""

    def test_structure(self):
        chain = MerkleChainAudit()
        chain.append({"event": "e1"})
        info = chain.to_dict()
        assert "chain_id" in info
        assert "height" in info
        assert "tip_hash" in info
        assert "root" in info
        assert "verified" in info
        assert info["height"] == 2
        assert info["verified"] is True

    def test_hex_strings(self):
        chain = MerkleChainAudit()
        info = chain.to_dict()
        assert len(info["tip_hash"]) == 64
        assert len(info["root"]) == 64


class TestMerkleChainAuditCreateAuditProof:
    """MerkleChainAudit.create_audit_proof() high-level API."""

    def test_structure(self):
        chain = MerkleChainAudit()
        result = chain.create_audit_proof("task-1", "tenant-1")
        assert result["chain_id"] == chain.chain_id
        assert result["block_index"] > 0
        assert len(result["block_hash"]) == 64
        assert len(result["root"]) == 64
        assert result["verified"] is True
        assert result["proof"] is not None

    def test_proof_has_siblings_and_directions(self):
        chain = MerkleChainAudit()
        chain.append({"event": "prior"})
        result = chain.create_audit_proof("task-1", "tenant-1")
        proof = result["proof"]
        assert "siblings" in proof
        assert "directions" in proof

    def test_creates_new_block(self):
        chain = MerkleChainAudit()
        height_before = chain.height
        chain.create_audit_proof("task-1", "tenant-1")
        assert chain.height == height_before + 1


class TestMerkleChainAuditBuildMerkleTree:
    """MerkleChainAudit._build_merkle_tree static method."""

    def test_empty_leaves(self):
        result = MerkleChainAudit._build_merkle_tree([])
        assert result == []

    def test_single_leaf(self):
        leaf = _hash(b"leaf")
        result = MerkleChainAudit._build_merkle_tree([leaf])
        assert result == [leaf]

    def test_two_leaves(self):
        l1 = _hash(b"l1")
        l2 = _hash(b"l2")
        result = MerkleChainAudit._build_merkle_tree([l1, l2])
        assert len(result) == 1
        assert result[0] == _hash(l1 + l2)

    def test_three_leaves(self):
        leaves = [_hash(f"l{i}".encode()) for i in range(3)]
        result = MerkleChainAudit._build_merkle_tree(leaves)
        assert len(result) == 1  # single root

    def test_four_leaves(self):
        leaves = [_hash(f"l{i}".encode()) for i in range(4)]
        result = MerkleChainAudit._build_merkle_tree(leaves)
        assert len(result) == 1

    def test_deterministic(self):
        leaves = [_hash(f"l{i}".encode()) for i in range(5)]
        r1 = MerkleChainAudit._build_merkle_tree(leaves)
        r2 = MerkleChainAudit._build_merkle_tree(leaves)
        assert r1 == r2


class TestMerkleChainAuditBuildMerkleRoot:
    """MerkleChainAudit._build_merkle_root."""

    def test_with_blocks(self):
        chain = MerkleChainAudit()
        chain.append({"event": "e1"})
        root = chain._build_merkle_root()
        assert len(root) == 32
        assert root == chain.root


# ══════════════════════════════════════════════════════════════════════════════
# Integration: AuditLogger full lifecycle
# ══════════════════════════════════════════════════════════════════════════════


class TestAuditLoggerLifecycle:
    """End-to-end AuditLogger lifecycle tests."""

    @pytest.mark.asyncio
    async def test_log_then_query_all(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        events = []
        for i in range(5):
            ev = _make_event(task_id=f"task-{i}")
            await logger.log(ev)
            events.append(ev)
        results = await logger.query("tenant-1")
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_log_then_proof(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        for i in range(4):
            await logger.log(_make_event(task_id=f"task-{i}"))
        proof = await logger.get_proof_for_task("tenant-1", "task-2")
        assert "leaf_data" in proof
        assert proof["leaf_data"]["task_id"] == "task-2"

    @pytest.mark.asyncio
    async def test_commitment_chain_integrity(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        commitments = []
        for i in range(3):
            ev = _make_event(task_id=f"t-{i}")
            await logger.log(ev)
            commitments.append(ev.commitment)
        # All commitments should be unique
        assert len(set(commitments)) == 3

    @pytest.mark.asyncio
    async def test_fallback_proof_task_not_found(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        await logger.initialize()
        await logger.log(_make_event(task_id="exists"))
        with pytest.raises(ValueError, match="not found"):
            await logger.get_proof_for_task("tenant-1", "missing")

    @pytest.mark.asyncio
    async def test_fallback_proof_no_file(self, tmp_path):
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        await logger.initialize()
        with pytest.raises(ValueError, match="not found"):
            await logger.get_proof_for_task("nonexistent-tenant", "task-1")
