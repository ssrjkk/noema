"""DAG Checkpointing — сохранение прогресса CoT для resumable execution."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from noema.logging import get_logger

log = get_logger(__name__)


@dataclass
class DAGCheckpoint:
    task_id: str = ""
    tenant_id: str = ""
    session_id: str = ""
    attempt: int = 1
    completed_steps: list[str] = field(default_factory=list)
    step_results: dict[str, str] = field(default_factory=dict)
    token_budget_used: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


class CheckpointStore:
    """Saves/loads DAG checkpoints for resumable execution.

    Uses file-based storage by default; Redis backend for distributed mode.
    """

    def __init__(self, persist_dir: str = ".noema/checkpoints", redis_url: str = "") -> None:
        self._dir = Path(persist_dir)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.warning("checkpoint_dir_failed", path=persist_dir, error=str(e))
        self._redis_url = redis_url
        self._redis: Any = None
        self._try_connect()

    def _try_connect(self) -> None:
        if not self._redis_url:
            return
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        except Exception as e:
            log.warning("checkpoint_redis_failed", error=str(e))

    def _checkpoint_key(self, task_id: str, tenant_id: str) -> str:
        return f"noema:ckpt:{tenant_id}:{task_id}"

    @staticmethod
    def _safe_name(value: str) -> str:
        """Keep only filesystem-safe characters; separators could escape the dir."""
        value = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
        value = value.strip("._")
        return value or "unknown"

    def _file_path(self, task_id: str, tenant_id: str) -> Path:
        safe_tenant = self._safe_name(tenant_id)
        safe_task = self._safe_name(task_id)
        return self._dir / f"{safe_tenant}__{safe_task}.json"

    async def save(self, cp: DAGCheckpoint) -> None:
        data = {
            "task_id": cp.task_id,
            "tenant_id": cp.tenant_id,
            "session_id": cp.session_id,
            "attempt": cp.attempt,
            "completed_steps": cp.completed_steps,
            "step_results": cp.step_results,
            "token_budget_used": cp.token_budget_used,
            "context": {k: _safe_str(v, 500) for k, v in cp.context.items()},
            "timestamp": time.time(),
        }
        payload = json.dumps(data, ensure_ascii=False)
        # Atomic write: write to a temp file in the same directory, then rename,
        # so a crash mid-save can never leave a truncated checkpoint behind.
        path = self._file_path(cp.task_id, cp.tenant_id)
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        def _write() -> None:
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, path)

        await asyncio.to_thread(_write)
        log.debug("checkpoint_saved", task=cp.task_id, steps=len(cp.completed_steps))

    async def load(self, task_id: str, tenant_id: str) -> DAGCheckpoint | None:
        path = self._file_path(task_id, tenant_id)
        try:
            exists = await asyncio.to_thread(path.exists)
            if not exists:
                return None
            text = await asyncio.to_thread(path.read_text, encoding="utf-8")
            data = json.loads(text)
            return DAGCheckpoint(
                task_id=data["task_id"],
                tenant_id=data.get("tenant_id", tenant_id),
                session_id=data.get("session_id", ""),
                attempt=data.get("attempt", 1),
                completed_steps=data.get("completed_steps", []),
                step_results=data.get("step_results", {}),
                token_budget_used=data.get("token_budget_used", 0),
                context=data.get("context", {}),
                timestamp=data.get("timestamp", 0.0),
            )
        except (KeyError, json.JSONDecodeError, OSError) as e:
            log.warning("checkpoint_load_failed", task=task_id, error=str(e))
            return None

    async def delete(self, task_id: str, tenant_id: str) -> None:
        path = self._file_path(task_id, tenant_id)
        try:
            exists = await asyncio.to_thread(path.exists)
            if exists:
                await asyncio.to_thread(path.unlink)
        except OSError as e:
            log.warning("checkpoint_delete_failed", task=task_id, error=str(e))

    async def has_checkpoint(self, task_id: str, tenant_id: str) -> bool:
        return await asyncio.to_thread(self._file_path(task_id, tenant_id).exists)


def _safe_str(v: Any, max_len: int = 500) -> str:
    try:
        s = str(v)
        return s[:max_len]
    except Exception:
        return f"<error: {type(v).__name__}>"
