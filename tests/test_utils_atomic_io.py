"""Tests for noema/utils/atomic_io.py — atomic JSON writes, rotation and recovery."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from noema.utils.atomic_io import _rotate_backups, atomic_read_json, atomic_write_json

if TYPE_CHECKING:
    from pathlib import Path


def _tmp_files_in(directory: Path) -> list[Path]:
    return [p for p in directory.iterdir() if p.suffix == ".tmp"]


class TestAtomicWriteJson:
    def test_writes_expected_shape(self, tmp_path: Path):
        target = tmp_path / "state.json"
        atomic_write_json(target, {"b": 2, "a": 1})

        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": 2}

    def test_accepts_str_path(self, tmp_path: Path):
        target = tmp_path / "state.json"
        atomic_write_json(str(target), [1, 2, 3])

        assert json.loads(target.read_text(encoding="utf-8")) == [1, 2, 3]

    def test_creates_missing_parent_directories(self, tmp_path: Path):
        target = tmp_path / "deep" / "nested" / "state.json"
        atomic_write_json(target, {"ok": True})

        assert target.is_file()

    def test_non_serializable_values_are_stringified(self, tmp_path: Path):
        class Custom:
            def __str__(self) -> str:
                return "custom"

        target = tmp_path / "state.json"
        atomic_write_json(target, {"obj": Custom()})

        assert json.loads(target.read_text(encoding="utf-8")) == {"obj": "custom"}

    def test_non_ascii_written_verbatim(self, tmp_path: Path):
        target = tmp_path / "state.json"
        atomic_write_json(target, {"text": "привет"})

        assert "привет" in target.read_text(encoding="utf-8")

    def test_leaves_no_temporary_files(self, tmp_path: Path):
        atomic_write_json(tmp_path / "state.json", {"a": 1})

        assert _tmp_files_in(tmp_path) == []

    def test_new_file_takeover_and_existing_file_replace_share_result(self, tmp_path: Path):
        first = tmp_path / "a.json"
        atomic_write_json(first, {"v": 1})
        second = tmp_path / "b.json"
        atomic_write_json(second, {"v": 2})
        atomic_write_json(second, {"v": 3}, backup=False)

        assert json.loads(first.read_text(encoding="utf-8")) == {"v": 1}
        assert json.loads(second.read_text(encoding="utf-8")) == {"v": 3}

    def test_backup_rotates_previous_file(self, tmp_path: Path):
        target = tmp_path / "state.json"
        atomic_write_json(target, {"v": 1})
        atomic_write_json(target, {"v": 2})

        backup = target.with_suffix(".bak.1")
        assert json.loads(backup.read_text(encoding="utf-8")) == {"v": 1}
        assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2}

    def test_backup_disabled_writes_no_backup(self, tmp_path: Path):
        target = tmp_path / "state.json"
        atomic_write_json(target, {"v": 1})
        atomic_write_json(target, {"v": 2}, backup=False)

        assert list(tmp_path.glob("*.bak.*")) == []

    def test_rotation_shifts_backups_and_drops_oldest(self, tmp_path: Path):
        target = tmp_path / "state.json"
        for v in range(1, 6):
            atomic_write_json(target, {"v": v}, backup_count=2)

        # Each write pushes the previous contents one slot further back and
        # the slot past ``backup_count`` is discarded.
        assert json.loads(target.read_text(encoding="utf-8")) == {"v": 5}
        assert json.loads(target.with_suffix(".bak.1").read_text(encoding="utf-8")) == {"v": 4}
        assert json.loads(target.with_suffix(".bak.2").read_text(encoding="utf-8")) == {"v": 3}
        assert not target.with_suffix(".bak.3").exists()

    def test_parent_directory_is_fsynced_when_supported(self, tmp_path: Path):
        """Where ``os.open`` accepts a directory (POSIX) the rename itself is
        fsynced. Windows refuses that handle, so a readable probe fd stands in
        for the directory while the real ``fsync``/``close`` still run.
        """
        target = tmp_path / "state.json"
        probe = tmp_path / "probe"
        probe.write_bytes(b"")
        real_os_open = os.open

        def _open(path, flags, *args, **kwargs):
            if str(path) == str(target.parent):
                return real_os_open(str(probe), os.O_RDONLY)
            return real_os_open(path, flags, *args, **kwargs)

        with patch("os.open", side_effect=_open):
            atomic_write_json(target, {"a": 1})

        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}

    def test_backup_count_zero_skips_rotation(self, tmp_path: Path):
        target = tmp_path / "state.json"
        atomic_write_json(target, {"v": 1})
        atomic_write_json(target, {"v": 2}, backup_count=0)

        assert list(target.parent.glob("*.bak.*")) == []
        assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2}


class TestRotateBackups:
    def test_non_positive_count_is_a_noop(self, tmp_path: Path):
        target = tmp_path / "state.json"
        target.write_text("data", encoding="utf-8")

        _rotate_backups(target, 0)

        assert list(tmp_path.glob("*")) == [target]

    def test_shifts_only_existing_backups(self, tmp_path: Path):
        target = tmp_path / "state.json"
        target.write_text("current", encoding="utf-8")
        target.with_suffix(".bak.1").write_text("one", encoding="utf-8")

        _rotate_backups(target, 3)

        assert target.with_suffix(".bak.1").read_text(encoding="utf-8") == "current"
        assert target.with_suffix(".bak.2").read_text(encoding="utf-8") == "one"
        assert not target.with_suffix(".bak.3").exists()

    def test_oldest_backup_is_dropped_to_make_room(self, tmp_path: Path):
        target = tmp_path / "state.json"
        target.write_text("current", encoding="utf-8")
        target.with_suffix(".bak.1").write_text("one", encoding="utf-8")
        target.with_suffix(".bak.2").write_text("two", encoding="utf-8")

        _rotate_backups(target, 2)

        assert not target.with_suffix(".bak.3").exists()
        assert target.with_suffix(".bak.1").read_text(encoding="utf-8") == "current"
        assert target.with_suffix(".bak.2").read_text(encoding="utf-8") == "one"

    def test_copies_metadata_not_just_content(self, tmp_path: Path):
        target = tmp_path / "state.json"
        target.write_text("first", encoding="utf-8")
        stat_before = target.stat()

        _rotate_backups(target, 5)

        backup = target.with_suffix(".bak.1")
        assert backup.read_text(encoding="utf-8") == "first"
        assert backup.stat().st_size == stat_before.st_size


class TestAtomicWriteFailureHandling:
    def test_serialization_error_removes_temp_file(self, tmp_path: Path):
        target = tmp_path / "state.json"

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        with patch("noema.utils.atomic_io.json.dump", side_effect=_boom), pytest.raises(OSError):
            atomic_write_json(target, {"a": 1})

        assert _tmp_files_in(tmp_path) == []
        assert not target.exists()

    def test_base_exception_also_cleans_temp_file(self, tmp_path: Path):
        target = tmp_path / "state.json"

        with (
            patch("noema.utils.atomic_io.os.fsync", side_effect=KeyboardInterrupt),
            pytest.raises(KeyboardInterrupt),
        ):
            atomic_write_json(target, {"a": 1})

        assert _tmp_files_in(tmp_path) == []

    def test_existing_content_survives_a_failed_write(self, tmp_path: Path):
        target = tmp_path / "state.json"
        atomic_write_json(target, {"v": "original"})

        with (
            patch("noema.utils.atomic_io.os.fsync", side_effect=OSError("no fs")),
            pytest.raises(OSError),
        ):
            atomic_write_json(target, {"v": "replacement"})

        assert json.loads(target.read_text(encoding="utf-8")) == {"v": "original"}

    def test_cleanup_failure_does_not_mask_original_error(self, tmp_path: Path):
        target = tmp_path / "state.json"
        real_unlink = os.unlink

        def _unlink(path):
            if str(path).endswith(".tmp"):
                raise OSError("cannot unlink")
            return real_unlink(path)

        with (
            patch("noema.utils.atomic_io.json.dump", side_effect=OSError("disk full")),
            patch("noema.utils.atomic_io.os.unlink", side_effect=_unlink),
            pytest.raises(OSError, match="disk full"),
        ):
            atomic_write_json(target, {"a": 1})


class TestAtomicReadJson:
    def test_reads_written_file(self, tmp_path: Path):
        target = tmp_path / "state.json"
        atomic_write_json(target, {"a": 1})

        assert atomic_read_json(target) == {"a": 1}

    def test_missing_file_returns_empty_dict(self, tmp_path: Path):
        assert atomic_read_json(tmp_path / "absent.json") == {}

    def test_missing_file_honours_default(self, tmp_path: Path):
        assert atomic_read_json(tmp_path / "absent.json", default={"seed": True}) == {"seed": True}

    def test_missing_file_with_none_default_falls_back_to_empty_dict(self, tmp_path: Path):
        assert atomic_read_json(tmp_path / "absent.json", default=None) == {}

    def test_corrupt_file_without_backup_returns_default(self, tmp_path: Path):
        target = tmp_path / "state.json"
        target.write_text("{not json", encoding="utf-8")

        assert atomic_read_json(target, default={"fallback": 1}) == {"fallback": 1}

    def test_corrupt_file_recovers_from_backup(self, tmp_path: Path):
        target = tmp_path / "state.json"
        atomic_write_json(target, {"v": 1})
        atomic_write_json(target, {"v": 2})
        target.write_text("{broken", encoding="utf-8")

        assert atomic_read_json(target) == {"v": 1}

    def test_corrupt_file_and_corrupt_backup_return_default(self, tmp_path: Path):
        target = tmp_path / "state.json"
        atomic_write_json(target, {"v": 1})
        atomic_write_json(target, {"v": 2})
        target.write_text("{broken", encoding="utf-8")
        target.with_suffix(".bak.1").write_text("{also broken", encoding="utf-8")

        assert atomic_read_json(target) == {}

    def test_unreadable_file_falls_back_to_backup(self, tmp_path: Path):
        target = tmp_path / "state.json"
        atomic_write_json(target, {"v": 1})
        atomic_write_json(target, {"v": 2})

        real_open = open

        def _deny(path, *args, **kwargs):
            if str(path) == str(target):
                raise OSError("permission denied")
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=_deny):
            assert atomic_read_json(target) == {"v": 1}

    def test_backup_read_oserror_returns_default(self, tmp_path: Path):
        target = tmp_path / "state.json"
        atomic_write_json(target, {"v": 1})
        atomic_write_json(target, {"v": 2})
        target.write_text("{broken", encoding="utf-8")

        backup = target.with_suffix(".bak.1")
        real_open = open

        def _deny(path, *args, **kwargs):
            if str(path) == str(backup):
                raise OSError("backup unreadable")
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=_deny):
            assert atomic_read_json(target, default={"recovered": False}) == {"recovered": False}

    def test_accepts_str_path(self, tmp_path: Path):
        target = tmp_path / "state.json"
        atomic_write_json(target, {"a": 1})

        assert atomic_read_json(str(target)) == {"a": 1}
