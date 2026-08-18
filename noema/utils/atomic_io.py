"""Atomic file writes with rotating backups."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from noema.logging import get_logger

log = get_logger(__name__)


def atomic_write_json(path: Path, data: Any, backup: bool = True, backup_count: int = 5) -> None:
    """Write JSON atomically: tmp → rename.

    If *backup* is True, rotates previous file to ``<name>.bak.<N>``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Backup current
    if backup and path.is_file():
        _rotate_backups(path, backup_count)

    # Atomic write via tmp + rename
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=path.stem + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        # On Windows, dst must not exist for os.replace
        if path.is_file():
            os.replace(tmp_path, path)
        else:
            os.rename(tmp_path, path)
        # Durability: fsync the directory so the rename survives a power loss
        # (best-effort; not supported on all platforms/filesystems).
        with contextlib.suppress(OSError):
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        log.debug("atomic_write_done", path=str(path), bytes=path.stat().st_size)
    except BaseException:
        # Clean up temp file on failure
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def atomic_read_json(path: Path, default: Any = None) -> Any:
    """Read JSON file. Returns *default* if file missing or corrupt."""
    path = Path(path)
    if not path.is_file():
        return default if default is not None else {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("atomic_read_failed", path=str(path), error=str(exc))
        # Try backup
        backup = path.with_suffix(".bak.1")
        if backup.is_file():
            log.info("atomic_read_fallback", backup=str(backup))
            with (
                contextlib.suppress(json.JSONDecodeError, OSError),
                open(backup, encoding="utf-8") as f,
            ):
                return json.load(f)
        return default if default is not None else {}


def _rotate_backups(path: Path, max_backups: int) -> None:
    """Rotate: .bak.5 → delete, .bak.4 → .bak.5, ... current → .bak.1"""
    if max_backups <= 0:
        return

    # Delete oldest
    oldest = path.with_suffix(f".bak.{max_backups}")
    if oldest.is_file():
        oldest.unlink()

    # Shift existing backups
    for i in range(max_backups - 1, 0, -1):
        src = path.with_suffix(f".bak.{i}")
        dst = path.with_suffix(f".bak.{i + 1}")
        if src.is_file():
            src.rename(dst)

    # Current → .bak.1
    dest = path.with_suffix(".bak.1")
    shutil.copy2(path, dest)
