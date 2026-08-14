"""Coverage for noema.discovery.keys — KeyDiscovery deep paths."""

import shutil
import subprocess
from pathlib import Path

from noema.discovery.keys import KeyDiscovery, ResourceInfo


def _discovery(tmp_path: Path) -> KeyDiscovery:
    return KeyDiscovery(project_root=str(tmp_path))


def test_scan_env_detects_known_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-1234567890")
    keys = KeyDiscovery()._scan_env()
    key = next(k for k in keys if k.name == "OPENAI_API_KEY")
    assert key.provider_hint == "openai"
    assert key.source == "env"
    assert key.confidence == 1.0
    assert key.value == "sk-12345..."


def test_scan_env_short_value_not_truncated(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "abc")
    keys = KeyDiscovery()._scan_env()
    key = next(k for k in keys if k.name == "OPENAI_API_KEY")
    assert key.value == "abc"


def test_scan_env_ignores_unknown_vars(monkeypatch):
    monkeypatch.setenv("MY_CUSTOM_VAR", "x")
    assert KeyDiscovery()._scan_env() == []


def test_parse_env_file_extracts_known_keys():
    content = "\n".join(
        [
            "# comment line",
            "",
            "ANTHROPIC_API_KEY=sk-ant-12345",
            'OPENAI_API_KEY="sk-ow-9876543210"',
            "CUSTOM_THING=abc",
            "EMPTY_KEY=",
        ]
    )
    keys = KeyDiscovery()._parse_env_file(content, source="file:.env")
    names = {k.name: k for k in keys}
    assert set(names) == {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}
    assert names["ANTHROPIC_API_KEY"].provider_hint == "anthropic"
    assert names["ANTHROPIC_API_KEY"].value == "sk-ant-1..."
    assert names["ANTHROPIC_API_KEY"].confidence == 0.7
    assert names["OPENAI_API_KEY"].value == "sk-ow-98..."


def test_parse_env_file_ignores_unknown_lines():
    keys = KeyDiscovery()._parse_env_file("FOO=bar\n# hi\n")
    assert keys == []


def test_scan_files_reads_env(tmp_path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-file-key\n", encoding="utf-8")
    keys = _discovery(tmp_path)._scan_files()
    assert any(k.name == "OPENAI_API_KEY" and k.source == "file:.env" for k in keys)


def test_discover_keys_dedup_prefers_env_confidence(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-123456")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-file\n", encoding="utf-8")
    keys = _discovery(tmp_path).discover_keys()
    matching = [k for k in keys if k.name == "ANTHROPIC_API_KEY"]
    assert len(matching) == 1
    assert matching[0].confidence == 1.0
    assert matching[0].source == "env"


def test_discover_all_groups_providers(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-1234567890")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_1234567890")
    result = _discovery(tmp_path).discover_all()
    names = {entry["name"] for entry in result["keys"]}
    assert "OPENAI_API_KEY" in names
    assert "GITHUB_TOKEN" in names
    assert "openai" in result["providers_available"]
    assert "github" in result["providers_available"]
    assert len(result["resources"]) == 5


def test_scan_keychain_without_security(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert KeyDiscovery()._scan_keychain() == []


def test_scan_keychain_with_security(monkeypatch):
    class _Result:
        returncode = 0
        stdout = "super-secret-password\n"

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/security")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())
    keys = KeyDiscovery()._scan_keychain()
    assert len(keys) == 1
    assert keys[0].name == "keychain_noema"
    assert keys[0].source == "keychain"
    assert keys[0].value == "super-se..."

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result() if False else type(
        "R", (), {"returncode": 1, "stdout": ""}
    )())
    assert KeyDiscovery()._scan_keychain() == []


def test_check_cpu():
    resource = KeyDiscovery()._check_cpu()
    assert isinstance(resource, ResourceInfo)
    assert resource.kind == "cpu"
    assert resource.available is True
    assert resource.details["cores"] >= 1


def test_check_ram():
    resource = KeyDiscovery()._check_ram()
    assert resource.kind == "ram"
    assert "total_gb" in resource.details


def test_check_disk():
    resource = KeyDiscovery()._check_disk()
    assert resource.kind == "disk"
    assert "total_gb" in resource.details
    assert "free_gb" in resource.details


def test_check_gpu_without_nvidia_smi(monkeypatch):
    def _no_smi(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess, "run", _no_smi)
    resource = KeyDiscovery()._check_gpu()
    assert resource.kind == "gpu"
    assert resource.available is False


def test_check_network_without_internet(monkeypatch):
    def _no_net(*args, **kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", _no_net)
    resource = KeyDiscovery()._check_network()
    assert resource.kind == "network"
    assert resource.details["internet"] is False


def test_discover_resources_returns_all_kinds():
    resources = KeyDiscovery().discover_resources()
    assert len(resources) == 5
    assert {r.kind for r in resources} == {"cpu", "ram", "disk", "gpu", "network"}
