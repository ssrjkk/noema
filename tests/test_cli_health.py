"""Tests for CLI health commands."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from noema.cli.health import health_app

runner = CliRunner()


def test_health_check_all_ok():
    """check() with all services healthy should exit 0."""
    with (
        patch("noema.cli.health._check_llm") as mock_llm,
        patch("noema.cli.health._check_db") as mock_db,
        patch("noema.cli.health._check_redis") as mock_redis,
        patch("noema.cli.health._check_sandbox") as mock_sandbox,
    ):
        mock_llm.return_value = {"status": "ok", "provider": "ollama"}
        mock_db.return_value = {"status": "ok", "message": "Database reachable"}
        mock_redis.return_value = {"status": "ok", "message": "Redis reachable"}
        mock_sandbox.return_value = {"status": "ok", "message": "Sandbox available"}

        result = runner.invoke(health_app, ["check"])

    assert result.exit_code == 0
    assert "All systems operational" in result.stdout


def test_health_check_with_errors():
    """check() with service errors should exit 1."""
    with (
        patch("noema.cli.health._check_llm") as mock_llm,
        patch("noema.cli.health._check_db") as mock_db,
        patch("noema.cli.health._check_redis") as mock_redis,
        patch("noema.cli.health._check_sandbox") as mock_sandbox,
    ):
        mock_llm.return_value = {"status": "ok", "provider": "ollama"}
        mock_db.return_value = {"status": "error", "error": "Connection refused"}
        mock_redis.return_value = {"status": "ok", "message": "Redis reachable"}
        mock_sandbox.return_value = {"status": "ok", "message": "Sandbox available"}

        result = runner.invoke(health_app, ["check"])

    assert result.exit_code == 1
    assert "Issues detected" in result.stdout


def test_health_check_json_output():
    """check() with json output should print JSON."""
    with (
        patch("noema.cli.health._check_llm") as mock_llm,
        patch("noema.cli.health._check_db") as mock_db,
        patch("noema.cli.health._check_redis") as mock_redis,
        patch("noema.cli.health._check_sandbox") as mock_sandbox,
    ):
        mock_llm.return_value = {"status": "ok", "provider": "ollama"}
        mock_db.return_value = {"status": "ok", "message": "Database reachable"}
        mock_redis.return_value = {"status": "ok", "message": "Redis reachable"}
        mock_sandbox.return_value = {"status": "ok", "message": "Sandbox available"}

        result = runner.invoke(health_app, ["check", "--output", "json"])

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["healthy"] is True
    assert "checks" in data
    assert data["checks"]["llm"]["status"] == "ok"


def test_health_check_llm_only():
    """check_llm() should check only LLM provider."""
    with patch("noema.cli.health._check_llm") as mock_llm:
        mock_llm.return_value = {"status": "ok", "provider": "ollama", "model": "llama2"}

        result = runner.invoke(health_app, ["check-llm"])

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["status"] == "ok"
    assert data["provider"] == "ollama"


def test_check_llm_error():
    """_check_llm() should handle errors gracefully."""
    with patch("noema.config.settings.get_settings") as mock_settings:
        mock_settings.return_value.llm.provider = "invalid"
        mock_settings.return_value.llm.ollama_model = "llama2"

        with patch("noema.llm.providers.create_llm_provider", side_effect=Exception("Bad provider")):
            from noema.cli.health import _check_llm
            import asyncio

            result = asyncio.run(_check_llm())

    assert result["status"] == "error"
    assert "Bad provider" in result["error"]


def test_check_db_error():
    """_check_db() should handle errors gracefully."""
    with patch("noema.db.engine.get_db", side_effect=Exception("DB not configured")):
        from noema.cli.health import _check_db
        import asyncio

        result = asyncio.run(_check_db())

    assert result["status"] == "error"
    assert "DB not configured" in result["error"]


def test_check_redis_skipped():
    """_check_redis() should skip if redis not installed."""
    with patch("noema.config.settings.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6379/0"

        with patch.dict("sys.modules", {"redis.asyncio": None}):
            from noema.cli.health import _check_redis
            import asyncio

            result = asyncio.run(_check_redis())

    assert result["status"] in ["skipped", "error"]


def test_check_sandbox_error():
    """_check_sandbox() should handle errors gracefully."""
    with patch(
        "noema.sandbox.engine.SandboxEngine", side_effect=Exception("Sandbox init failed")
    ):
        from noema.cli.health import _check_sandbox
        import asyncio

        result = asyncio.run(_check_sandbox())

    assert result["status"] == "error"
    assert "Sandbox init failed" in result["error"]
