"""Immutable Audit Log — SOC2/GDPR compliance, append-only."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, cast

from noema.logging import get_logger

log = get_logger(__name__)

_TENANT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _tenant_filename(tenant_id: str) -> str:
    """Normalize a tenant id into a safe single path segment.

    Replaces any character that could act as a path separator or escape with
    ``_``, so a tenant id can never traverse out of the fallback directory.
    """
    name = _TENANT_RE.sub("_", str(tenant_id))
    if not name or name in (".", "..") or len(name) > 100:
        raise ValueError("Invalid tenant_id")
    return name


@dataclass
class AuditEvent:
    timestamp: datetime
    event_type: str
    tenant_id: str
    user_id: str
    task_id: str | None = None
    details: dict[str, Any] | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    commitment: str | None = None
    chain_link: str | None = None
    block_index: int | None = None

    def __post_init__(self) -> None:
        if self.details is None:
            self.details = {}


# SQL for table creation (executed on init)
CREATE_AUDIT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(100) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    task_id VARCHAR(100),
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    ip_address VARCHAR(45),
    user_agent TEXT,
    commitment VARCHAR(64),
    chain_link VARCHAR(64),
    block_index INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_tenant_time ON audit_log (tenant_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_task ON audit_log (task_id);
CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log (event_type);
CREATE INDEX IF NOT EXISTS idx_audit_block_index ON audit_log (block_index);
ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS commitment VARCHAR(64);
ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS chain_link VARCHAR(64);
ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS block_index INTEGER;
"""


class AuditLogger:
    """Immutable audit log. All writes are append-only.

    Each entry includes a cryptographic commitment forming a verifiable
    hash chain. Supports Merkle Inclusion Proofs for O(log N) verification
    without revealing the full log.

    Falls back to file-based logging when PostgreSQL is unavailable.
    """

    def __init__(self, pg_pool: Any = None, fallback_dir: str = ".noema/audit") -> None:
        self.pg: Any = pg_pool
        self._fallback_dir = fallback_dir
        self._initialized = False
        self._file_fallback = False
        self._leaf_hashes: list[bytes] = []
        self._tree: Any = None

    def _sync_tree(self) -> None:
        from noema.audit.merkle_proof import IncrementalMerkleTree

        if self._tree is None or self._tree.count != len(self._leaf_hashes):
            self._tree = IncrementalMerkleTree(self._leaf_hashes)

    async def initialize(self) -> None:
        if self.pg and not self._file_fallback:
            try:
                for stmt in CREATE_AUDIT_TABLE_SQL.strip().split(";"):
                    s = stmt.strip()
                    if s:
                        await self.pg.execute(s)
                await self._load_leaf_hashes()
                self._sync_tree()
                log.info("audit_table_ready")
                self._initialized = True
                return
            except Exception as e:
                log.warning("audit_pg_fallback_to_file", error=str(e))
        self._file_fallback = True
        self._initialized = True
        from pathlib import Path

        Path(self._fallback_dir).mkdir(parents=True, exist_ok=True)
        self._load_leaf_hashes_fallback()
        self._sync_tree()
        log.info("audit_file_fallback_ready", dir=self._fallback_dir)

    async def _load_leaf_hashes(self) -> None:
        try:
            rows = await self.pg.fetch(
                "SELECT commitment, block_index FROM audit_log WHERE commitment IS NOT NULL ORDER BY block_index ASC"
            )
            self._leaf_hashes = [bytes.fromhex(r["commitment"]) for r in rows]
        except Exception as e:
            log.warning("audit_load_hashes_failed", error=str(e))
            self._leaf_hashes = []

    def _load_leaf_hashes_fallback(self) -> None:
        from pathlib import Path

        self._leaf_hashes = []
        fallback_dir = Path(self._fallback_dir)
        if not fallback_dir.exists():
            return
        records: list[dict] = []
        for fpath in fallback_dir.glob("*.jsonl"):
            try:
                with open(fpath, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            except OSError:
                continue
        records.sort(
            key=lambda r: r.get("block_index", 0) if isinstance(r.get("block_index"), int) else 0
        )
        for r in records:
            cmt = r.get("commitment")
            if cmt and isinstance(cmt, str):
                try:
                    self._leaf_hashes.append(bytes.fromhex(cmt))
                except (ValueError, TypeError):
                    continue

    def _get_prev_root(self) -> bytes:
        self._sync_tree()
        return self._tree.root if self._tree is not None else b"\x00" * 32

    def _build_leaf_data(self, event: AuditEvent | dict[str, Any]) -> dict[str, Any]:
        if isinstance(event, AuditEvent):
            return {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "tenant_id": event.tenant_id,
                "user_id": event.user_id,
                "task_id": event.task_id,
                "details": dict(event.details or {}),
            }
        return {
            "timestamp": event["timestamp"].isoformat()
            if hasattr(event["timestamp"], "isoformat")
            else str(event["timestamp"]),
            "event_type": event["event_type"],
            "tenant_id": event["tenant_id"],
            "user_id": event["user_id"],
            "task_id": event.get("task_id"),
            "details": event.get("details") or {},
        }

    async def log(self, event: AuditEvent) -> None:
        if not self._initialized:
            await self.initialize()
        from noema.audit.merkle_proof import _safe_hash
        from noema.utils.json_utils import serialize_to_bytes

        payload_bytes = serialize_to_bytes(self._build_leaf_data(event))
        block_index = len(self._leaf_hashes)
        commitment = _safe_hash(payload_bytes)
        prev_root = self._get_prev_root()
        chain_link = _safe_hash(prev_root + payload_bytes + str(block_index).encode("utf-8"))
        commitment_hex = commitment.hex()
        chain_link_hex = chain_link.hex()
        event.commitment = commitment_hex
        event.chain_link = chain_link_hex
        event.block_index = block_index
        self._leaf_hashes.append(commitment)
        if self._tree is not None:
            self._tree.append(commitment)
        else:
            self._sync_tree()
        if not self._file_fallback and self.pg:
            try:
                await self.pg.execute(
                    """INSERT INTO audit_log
                       (timestamp, event_type, tenant_id, user_id, task_id, details, ip_address, user_agent, commitment, chain_link, block_index)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)""",
                    event.timestamp,
                    event.event_type,
                    event.tenant_id,
                    event.user_id,
                    event.task_id,
                    json.dumps(event.details),
                    event.ip_address,
                    event.user_agent,
                    commitment_hex,
                    chain_link_hex,
                    block_index,
                )
                return
            except Exception as e:
                log.warning("audit_pg_write_failed", error=str(e))
                self._file_fallback = True
        self._write_fallback(event)

    def _write_fallback(self, event: AuditEvent) -> None:
        from pathlib import Path

        record = asdict(event)
        record["details"] = dict(event.details or {})
        record["timestamp"] = event.timestamp.isoformat()
        path = Path(self._fallback_dir) / f"{_tenant_filename(event.tenant_id)}.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            log.error("audit_fallback_write_failed", error=str(e))

    async def get_proof_for_task(self, tenant_id: str, task_id: str) -> dict[str, Any]:
        if self._file_fallback or not self.pg:
            return self._get_proof_fallback(tenant_id, task_id)
        try:
            row = await self.pg.fetchrow(
                """SELECT block_index, details, event_type, timestamp, user_id, tenant_id, commitment
                   FROM audit_log WHERE tenant_id = $1 AND task_id = $2 AND commitment IS NOT NULL
                   ORDER BY block_index ASC LIMIT 1""",
                tenant_id,
                task_id,
            )
            if not row:
                raise ValueError(f"Task {task_id} not found in audit log")
            leaf_data = self._build_leaf_data(dict(row))
            if not self._leaf_hashes:
                await self._load_leaf_hashes()
            self._sync_tree()
            if self._tree is None:
                raise RuntimeError("Merkle tree not initialized")
            proof = self._tree.inclusion_proof(leaf_data, row["block_index"])
            result = dict(proof.to_dict())
            result["leaf_data"] = leaf_data
            return result
        except ValueError:
            raise
        except Exception as e:
            log.warning("audit_proof_query_failed", error=str(e))
            return self._get_proof_fallback(tenant_id, task_id)

    def _get_proof_fallback(self, tenant_id: str, task_id: str) -> dict[str, Any]:
        from pathlib import Path

        from noema.audit.merkle_proof import generate_inclusion_proof

        safe_tenant = _tenant_filename(tenant_id)
        path = Path(self._fallback_dir) / f"{safe_tenant}.jsonl"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found in audit log")
        records: list[dict] = []
        target_record: dict | None = None
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("task_id") == task_id and row.get("tenant_id") == tenant_id:
                    target_record = row
                records.append(row)
        if not target_record:
            raise ValueError(f"Task {task_id} not found in audit log")
        records.sort(
            key=lambda r: r.get("block_index", 0) if isinstance(r.get("block_index"), int) else 0
        )
        leaf_hashes = []
        for r in records:
            cmt = r.get("commitment")
            if cmt and isinstance(cmt, str):
                try:
                    leaf_hashes.append(bytes.fromhex(cmt))
                except (ValueError, TypeError):
                    continue
        leaf_data = self._build_leaf_data(target_record)
        target_index = target_record.get("block_index", 0)
        if (
            self._leaf_hashes
            and isinstance(target_index, int)
            and 0 <= target_index < len(self._leaf_hashes)
        ):
            self._sync_tree()
            if self._tree is not None and self._tree.count == len(self._leaf_hashes):
                proof = self._tree.inclusion_proof(leaf_data, target_index)
            else:
                proof = generate_inclusion_proof(leaf_data, target_index, self._leaf_hashes)
        else:
            proof = generate_inclusion_proof(leaf_data, target_index, leaf_hashes)
        result = cast("dict[str, Any]", proof.to_dict())
        result["leaf_data"] = leaf_data
        return result

    async def query(
        self,
        tenant_id: str,
        event_type: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if self._file_fallback or not self.pg:
            return self._query_fallback(tenant_id, event_type, start_time, end_time, limit)
        try:
            params: list[Any] = [tenant_id]
            query = "SELECT * FROM audit_log WHERE tenant_id = $1"
            idx = 2
            if event_type:
                query += f" AND event_type = ${idx}"
                params.append(event_type)
                idx += 1
            if start_time:
                query += f" AND timestamp >= ${idx}"
                params.append(start_time)
                idx += 1
            if end_time:
                query += f" AND timestamp <= ${idx}"
                params.append(end_time)
                idx += 1
            query += f" ORDER BY timestamp DESC LIMIT ${idx}"
            params.append(limit)
            rows = await self.pg.fetch(query, *params)
            return [dict(r) for r in rows]
        except Exception as e:
            log.warning("audit_query_fallback", error=str(e))
            return self._query_fallback(tenant_id, event_type, start_time, end_time, limit)

    def _query_fallback(
        self,
        tenant_id: str,
        event_type: str | None,
        start_time: datetime | None,
        end_time: datetime | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        from pathlib import Path

        path = Path(self._fallback_dir) / f"{_tenant_filename(tenant_id)}.jsonl"
        if not path.exists():
            return []
        results = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("tenant_id") != tenant_id:
                    continue
                if event_type and row.get("event_type") != event_type:
                    continue
                if start_time:
                    ts = row.get("timestamp", "")
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts)
                    if ts < start_time:
                        continue
                if end_time:
                    ts = row.get("timestamp", "")
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts)
                    if ts > end_time:
                        continue
                results.append(row)
        results.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return results[:limit]
