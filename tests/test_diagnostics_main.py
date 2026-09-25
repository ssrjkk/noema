"""Tests for noema.api.diagnostics health checks and the python -m noema entrypoint."""

from __future__ import annotations

import asyncio
import runpy
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from noema.api.diagnostics import _check_component, diagnostics


class TestCheckComponent:
    @pytest.mark.asyncio
    async def test_ok_status_passthrough(self):
        async def check():
            return {"status": "ok", "extra": 1}

        result = await _check_component("llm", check)
        assert result["component"] == "llm"
        assert result["status"] == "ok"
        assert result["detail"] == {"status": "ok", "extra": 1}
        assert result["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_error_status_inside_result_is_error(self):
        async def check():
            return {"status": "degraded"}

        assert (await _check_component("db", check))["status"] == "error"

    @pytest.mark.asyncio
    async def test_timeout_reports_fixed_duration(self):
        async def slow():
            await asyncio.sleep(5)

        started = time.monotonic()
        result = await _check_component("redis", slow, timeout=0.05)
        assert result["status"] == "error"
        assert result["detail"] == "timeout"
        assert result["duration_ms"] == pytest.approx(50, rel=0.2)
        assert time.monotonic() - started < 1

    @pytest.mark.asyncio
    async def test_exception_becomes_error_detail(self):
        async def boom():
            raise RuntimeError("connection refused")

        result = await _check_component("sandbox", boom)
        assert result["status"] == "error"
        assert result["detail"] == "connection refused"


def _fake_noema():
    return SimpleNamespace(
        llm=SimpleNamespace(name="openai", model_name="gpt-4"),
        sandbox=object(),
        memory_stats=lambda: {"total_entries": 5},
    )


@pytest.fixture
def patched_env(monkeypatch):
    monkeypatch.setattr("noema.api.server._get_noema", _fake_noema)
    monkeypatch.setattr("noema.api.server._start_time", time.monotonic() - 10)

    db = SimpleNamespace(health_check=AsyncMock(return_value=True))
    monkeypatch.setattr("noema.api.diagnostics.get_db", lambda: db)

    settings = SimpleNamespace(redis=SimpleNamespace(url="redis://localhost:6379"))
    monkeypatch.setattr("noema.api.diagnostics.get_settings", lambda: settings)

    class FakeRedis:
        def __init__(self):
            self.pinged = False
            self.closed = False

        async def ping(self):
            self.pinged = True

        async def aclose(self):
            self.closed = True

    fake = FakeRedis()
    monkeypatch.setattr("redis.asyncio.from_url", MagicMock(return_value=fake))
    return SimpleNamespace(db=db, redis=fake)


class TestDiagnosticsEndpoint:
    @pytest.mark.asyncio
    async def test_all_components_healthy(self, patched_env):
        report = await diagnostics(MagicMock())

        assert report["healthy"] is True
        assert report["uptime_s"] >= 10
        assert report["version"] == "1.0.0"
        components = {c["component"]: c for c in report["checks"]}
        assert set(components) == {"llm", "database", "redis", "sandbox", "memory"}
        assert all(c["status"] == "ok" for c in components.values())
        assert components["llm"]["detail"]["provider"] == "openai"
        assert components["memory"]["detail"]["total_entries"] == 5
        assert patched_env.redis.pinged and patched_env.redis.closed

    @pytest.mark.asyncio
    async def test_unhealthy_database_marks_report_unhealthy(self, patched_env):
        patched_env.db.health_check = MagicMock(return_value=False)
        report = await diagnostics(MagicMock())
        assert report["healthy"] is False
        db_check = next(c for c in report["checks"] if c["component"] == "database")
        assert db_check["status"] == "error"

    @pytest.mark.asyncio
    async def test_redis_failure_is_isolated(self, patched_env, monkeypatch):
        class BrokenRedis:
            async def ping(self):
                raise ConnectionError("down")

            async def aclose(self):
                pass

        monkeypatch.setattr("redis.asyncio.from_url", MagicMock(return_value=BrokenRedis()))
        report = await diagnostics(MagicMock())
        assert report["healthy"] is False
        redis_check = next(c for c in report["checks"] if c["component"] == "redis")
        assert redis_check["status"] == "error"

    @pytest.mark.asyncio
    async def test_no_sandbox_is_still_ok(self, patched_env, monkeypatch):
        no_sandbox = SimpleNamespace(
            llm=SimpleNamespace(name="x", model_name="y"),
            sandbox=None,
            memory_stats=lambda: {},
        )
        monkeypatch.setattr("noema.api.server._get_noema", lambda: no_sandbox)
        report = await diagnostics(MagicMock())
        sandbox_check = next(c for c in report["checks"] if c["component"] == "sandbox")
        assert sandbox_check["status"] == "ok"
        assert sandbox_check["detail"]["available"] is False


class TestMainModule:
    def test_python_dash_m_noema_invokes_cli(self, capsys):
        argv = sys.argv
        sys.argv = ["noema", "--help"]
        try:
            with pytest.raises(SystemExit) as exc:
                runpy.run_module("noema", run_name="__main__")
            assert exc.value.code == 0
        finally:
            sys.argv = argv
        out = capsys.readouterr().out
        assert "noema" in out.lower()
