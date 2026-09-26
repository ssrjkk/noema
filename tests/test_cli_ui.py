"""Tests for CLI UI design system."""

from __future__ import annotations

import pytest

from noema.cli.ui import (
    BULLET,
    STATUS_ERR,
    STATUS_OK,
    STATUS_WARN,
    data_table,
    err,
    fmt_duration,
    fmt_path,
    human_rows,
    info,
    kv_panel,
    ok,
    panel,
    print_banner,
    run_async,
    section,
    spinner,
    task_progress,
    warn,
)


def test_print_banner(capsys):
    print_banner()
    captured = capsys.readouterr()
    assert "noema" in captured.out.lower() or "n" in captured.out


def test_ok_prints_status(capsys):
    ok("test message")
    captured = capsys.readouterr()
    assert STATUS_OK in captured.out or "test message" in captured.out


def test_err_prints_status(capsys):
    err("error message")
    captured = capsys.readouterr()
    assert STATUS_ERR in captured.out or "error message" in captured.out


def test_warn_prints_status(capsys):
    warn("warning message")
    captured = capsys.readouterr()
    assert STATUS_WARN in captured.out or "warning message" in captured.out


def test_info_prints_bullet(capsys):
    info("info message")
    captured = capsys.readouterr()
    assert BULLET in captured.out or "info message" in captured.out


def test_panel_renders(capsys):
    panel("body text", title="Test Panel")
    captured = capsys.readouterr()
    assert "body text" in captured.out
    assert "Test Panel" in captured.out


def test_kv_panel_renders(capsys):
    kv_panel("Config", [("key1", "value1"), ("key2", "value2")])
    captured = capsys.readouterr()
    assert "key1" in captured.out
    assert "value1" in captured.out


def test_data_table_renders(capsys):
    data_table("Test Table", ["Col1", "Col2"], [["a", "b"], ["c", "d"]])
    captured = capsys.readouterr()
    assert "Test Table" in captured.out
    assert "Col1" in captured.out
    assert "a" in captured.out


def test_section_renders(capsys):
    section("Section Title")
    captured = capsys.readouterr()
    assert "Section Title" in captured.out


def test_fmt_duration_milliseconds():
    assert fmt_duration(500) == "500 ms"
    assert fmt_duration(1) == "1 ms"
    assert fmt_duration(999) == "999 ms"


def test_fmt_duration_seconds():
    assert fmt_duration(1000) == "1.0 s"
    assert fmt_duration(5500) == "5.5 s"
    assert fmt_duration(59999) == "60.0 s"


def test_fmt_duration_minutes():
    assert fmt_duration(60000) == "1.0 min"
    assert fmt_duration(120000) == "2.0 min"
    assert fmt_duration(3600000) == "60.0 min"


def test_fmt_path():
    result = fmt_path("/some/path")
    assert result == "[path]/some/path[/path]"


@pytest.mark.asyncio
async def test_spinner_context_manager():
    async with spinner("Loading"):
        pass


def test_task_progress_returns_progress():
    progress = task_progress("Processing")
    assert progress is not None
    assert hasattr(progress, "add_task")


def test_human_rows_filters_and_formats():
    data = {
        "user_name": "Alice",
        "item_count": 42,
        "nested": {"key": "value"},
        "list_field": [1, 2, 3],
    }
    rows = human_rows(data)
    assert len(rows) == 2
    assert ("User Name", "Alice") in rows
    assert ("Item Count", "42") in rows


def test_human_rows_excludes_nested_structures():
    data = {
        "simple": "value",
        "dict_field": {"nested": "data"},
        "list_field": [1, 2, 3],
    }
    rows = human_rows(data)
    assert len(rows) == 1
    assert rows[0] == ("Simple", "value")


def test_run_async_executes_coroutine():
    async def factory():
        return "result"

    run_async(factory)


def test_run_async_with_async_function():
    executed = []

    async def factory():
        executed.append(True)

    run_async(factory)
    assert executed == [True]
