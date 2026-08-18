"""Tests for the env-var override layer (``from_yaml`` / ``_env_overrides``).

Env vars flow exclusively through ``from_yaml``; direct construction must
not be surprised by stray ``NOEMA_*`` variables, container fields expect
JSON, and empty/malformed values fail loudly but clearly.
"""

from __future__ import annotations

import json

import pytest

from noema.config.settings import NoemaSettings


def _missing_path(tmp_path):
    return tmp_path / "missing-settings.yaml"


def test_env_double_underscore_override(tmp_path, monkeypatch):
    monkeypatch.setenv("NOEMA_LLM__PROVIDER", "openai")
    s = NoemaSettings.from_yaml(_missing_path(tmp_path))
    assert s.llm.provider == "openai"


def test_env_single_underscore_submodel_override(tmp_path, monkeypatch):
    monkeypatch.setenv("NOEMA_LLM_PROVIDER", "anthropic")
    s = NoemaSettings.from_yaml(_missing_path(tmp_path))
    assert s.llm.provider == "anthropic"


def test_env_container_field_accepts_json(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "NOEMA_AUTONOMY__LEAN_VERIFIER_REQUIRED_PATHS", json.dumps(["crypto/", "auth/"])
    )
    s = NoemaSettings.from_yaml(_missing_path(tmp_path))
    assert s.autonomy.lean_verifier_required_paths == ["crypto/", "auth/"]


def test_env_container_field_malformed_json_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("NOEMA_AUTONOMY__LEAN_VERIFIER_REQUIRED_PATHS", "not-json")
    with pytest.raises(ValueError, match="LEAN_VERIFIER_REQUIRED_PATHS"):
        NoemaSettings.from_yaml(_missing_path(tmp_path))


def test_env_empty_value_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("NOEMA_LLM__PROVIDER", "")
    monkeypatch.setenv("NOEMA_COT_MAX_STEPS", "")
    s = NoemaSettings.from_yaml(_missing_path(tmp_path))
    assert s.llm.provider == "ollama"
    assert s.cot_max_steps == 12


def test_env_stray_var_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("NOEMA_BOGUS_FIELD", "123")
    monkeypatch.setenv("NOEMA_LLM__PROVIDER", "openai")
    s = NoemaSettings.from_yaml(_missing_path(tmp_path))
    assert s.llm.provider == "openai"


def test_direct_construction_ignores_env(monkeypatch):
    monkeypatch.setenv("NOEMA_LLM__PROVIDER", "openai")
    s = NoemaSettings()
    assert s.llm.provider == "ollama"


def test_secret_fields_stay_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("NOEMA_LLM__OPENAI_API_KEY", "sk-super-secret")
    s = NoemaSettings.from_yaml(_missing_path(tmp_path))
    assert s.llm.openai_api_key.get_secret_value() == "sk-super-secret"
    assert "sk-super-secret" not in repr(s.llm.openai_api_key)
