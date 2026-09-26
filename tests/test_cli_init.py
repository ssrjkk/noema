"""Tests for CLI init commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from noema.cli.init_cmd import init_app

runner = CliRunner()


def test_init_creates_settings_file(tmp_path, monkeypatch):
    """init() should create settings.yaml with provided config."""
    monkeypatch.chdir(tmp_path)

    with patch("noema.cli.init_cmd.NoemaSettings") as mock_settings_cls:
        mock_settings = MagicMock()
        mock_settings.db = MagicMock()
        mock_settings.redis = MagicMock()
        mock_settings.llm = MagicMock()
        mock_settings.obs = MagicMock()
        mock_settings_cls.return_value = mock_settings

        result = runner.invoke(
            init_app,
            [
                "init",
                "--db-url",
                "postgresql://test:test@localhost/test",
                "--redis-url",
                "redis://localhost:6379/1",
                "--llm-provider",
                "openai",
                "--env",
                "production",
            ],
        )

    assert result.exit_code == 0
    assert "Created" in result.stdout
    mock_settings.dump_yaml.assert_called_once()
    assert mock_settings.db.url == "postgresql://test:test@localhost/test"
    assert mock_settings.redis.url == "redis://localhost:6379/1"
    assert mock_settings.llm.provider == "openai"
    assert mock_settings.obs.sentry_environment == "production"


def test_init_refuses_overwrite_without_force(tmp_path, monkeypatch):
    """init() without --force should refuse to overwrite existing settings."""
    monkeypatch.chdir(tmp_path)
    Path("settings.yaml").write_text("existing: true")

    result = runner.invoke(init_app, ["init"])

    assert result.exit_code == 1
    assert "Use --force" in result.stdout


def test_init_allows_overwrite_with_force(tmp_path, monkeypatch):
    """init() with --force should overwrite existing settings."""
    monkeypatch.chdir(tmp_path)
    Path("settings.yaml").write_text("existing: true")

    with patch("noema.cli.init_cmd.NoemaSettings") as mock_settings_cls:
        mock_settings = MagicMock()
        mock_settings.db = MagicMock()
        mock_settings.redis = MagicMock()
        mock_settings.llm = MagicMock()
        mock_settings.obs = MagicMock()
        mock_settings_cls.return_value = mock_settings

        result = runner.invoke(init_app, ["init", "--force"])

    assert result.exit_code == 0
    assert "Created" in result.stdout


def test_init_handles_migration_failure(tmp_path, monkeypatch):
    """init() should warn but not fail if migrations fail."""
    monkeypatch.chdir(tmp_path)

    with (
        patch("noema.cli.init_cmd.NoemaSettings") as mock_settings_cls,
        patch("alembic.command.upgrade", side_effect=Exception("Migration failed")),
    ):
        mock_settings = MagicMock()
        mock_settings.db = MagicMock()
        mock_settings.redis = MagicMock()
        mock_settings.llm = MagicMock()
        mock_settings.obs = MagicMock()
        mock_settings_cls.return_value = mock_settings

        result = runner.invoke(init_app, ["init"])

    assert result.exit_code == 0
    assert "Migration failed" in result.stdout
    assert "alembic upgrade head" in result.stdout


def test_seed_inserts_feature_flags():
    """seed() should insert feature flags into database."""
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()

    mock_db = MagicMock()
    mock_db.session = MagicMock()
    mock_db.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_db.session.return_value.__aexit__ = AsyncMock(return_value=None)

    with patch("noema.db.engine.init_db", return_value=mock_db):
        result = runner.invoke(init_app, ["seed"])

    assert result.exit_code == 0
    assert "seeded" in result.stdout.lower()
    mock_session.execute.assert_called_once()
    mock_session.commit.assert_called_once()


def test_show_config_prints_json():
    """show_config() should print current configuration as JSON."""
    with patch("noema.cli.init_cmd.get_settings") as mock_get_settings:
        mock_settings = MagicMock()
        mock_settings.model_dump.return_value = {
            "db": {"url": "postgresql://localhost/test"},
            "redis": {"url": "redis://localhost:6379/0"},
            "llm": {"provider": "ollama"},
        }
        mock_get_settings.return_value = mock_settings

        result = runner.invoke(init_app, ["show-config"])

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["db"]["url"] == "postgresql://localhost/test"
    assert data["llm"]["provider"] == "ollama"
