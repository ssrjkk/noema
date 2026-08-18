"""Chaos tests: fail-closed behavior under hostile environment mutation.

Run with ``pytest --chaos`` to additionally mutate the process environment
(see ``tests/conftest.py``). Every test here is deterministic and asserts that
Noema degrades loudly or safely — never silently wrong.
"""

import os
import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from noema.config.settings import NoemaSettings
from noema.ontology import Entity, OntologyGraph
from noema.sandbox.engine import _build_isolated_env
from noema.utils.atomic_io import atomic_read_json, atomic_write_json


@pytest.mark.chaos
def test_settings_load_fails_closed_on_byte_corruption(tmp_path: Path, chaotic_env: None) -> None:
    """A single flipped byte in the config must raise, not silently misconfigure."""
    src = Path(__file__).resolve().parents[2] / "settings.yaml"
    if not src.is_file():
        pytest.skip("settings.yaml not present")
    copy = tmp_path / "settings.yaml"
    data = src.read_bytes()
    data = (
        data[: len(data) // 2] + bytes([data[len(data) // 2] ^ 0xFF]) + data[len(data) // 2 + 1 :]
    )
    copy.write_bytes(data)
    with pytest.raises((yaml.YAMLError, ValidationError, UnicodeDecodeError)):
        NoemaSettings.from_yaml(copy)


@pytest.mark.chaos
def test_settings_load_with_hostile_env_still_parses(chaotic_env: None) -> None:
    """Hostile NOEMA_* vars must not break parsing (values stay typed)."""
    settings = NoemaSettings.from_yaml()
    assert isinstance(settings, NoemaSettings)


@pytest.mark.chaos
def test_settings_env_overrides_typed_fail_closed(tmp_path: Path, chaotic_env: None) -> None:
    """An invalid typed override must raise, never silently coerce to a wrong value."""
    bad = tmp_path / "settings.yaml"
    bad.write_text("api:\n  host: 127.0.0.1\n", encoding="utf-8")
    saved = os.environ.get("NOEMA_API__PORT")
    os.environ["NOEMA_API__PORT"] = "not-a-number"
    try:
        with pytest.raises(ValidationError):
            NoemaSettings.from_yaml(bad)
    finally:
        if saved is None:
            os.environ.pop("NOEMA_API__PORT", None)
        else:
            os.environ["NOEMA_API__PORT"] = saved


@pytest.mark.chaos
def test_sandbox_isolated_env_strips_inherited_proxies(chaotic_env: None) -> None:
    """Even with proxy vars inherited from a hostile parent, sandbox env is clean."""
    saved = {k: os.environ.get(k) for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy")}
    os.environ["HTTP_PROXY"] = "http://evil-proxy:8080"
    os.environ["HTTPS_PROXY"] = "http://evil-proxy:8080"
    os.environ["http_proxy"] = "http://evil-proxy:8080"
    try:
        env = _build_isolated_env()
        assert env["HTTP_PROXY"] == ""
        assert env["HTTPS_PROXY"] == ""
        assert env["http_proxy"] == ""
        assert env["NO_PROXY"] == "*"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.mark.chaos
def test_corrupt_primary_falls_back_to_rotated_backup(tmp_path: Path, chaotic_env: None) -> None:
    """A killed process mid-write must not lose the last good state (backup restore)."""
    path = tmp_path / "state.json"
    atomic_write_json(path, {"epoch": 1})
    atomic_write_json(path, {"epoch": 2})
    path.write_text('{"epoch": 2,', encoding="utf-8")  # simulate torn write
    recovered = atomic_read_json(path, default=None)
    assert recovered == {"epoch": 1}


@pytest.mark.chaos
def test_ontology_survives_truncated_file(tmp_path: Path, chaotic_env: None) -> None:
    """Truncated ontology (crash mid-persist) degrades to an empty graph."""
    path = tmp_path / "ontology.json"
    graph = OntologyGraph()
    graph.add_entity(Entity("alice", "user"))
    graph.add_entity(Entity("server-eu", "server"))
    graph.save(path)
    path.write_text('{"entities": [{"name": "ali', encoding="utf-8")
    loaded = OntologyGraph.load(path)
    assert loaded.stats()["entities"] == 0
    assert loaded.stats()["relations"] == 0
    assert loaded.has_cycle() is False


@pytest.mark.chaos
def test_ontology_import_after_torn_write(tmp_path: Path, chaotic_env: None) -> None:
    """Backup recovery path: garbage primary, intact backup, graph survives."""
    path = tmp_path / "ontology.json"
    graph = OntologyGraph()
    graph.add_entity(Entity("alice", "user"))
    graph.add_entity(Entity("server-eu", "server"))
    graph.save(path)
    backup = path.with_suffix(".bak.1")
    shutil.copy2(path, backup)
    path.write_text("}{garbage", encoding="utf-8")
    loaded = OntologyGraph.load(path)
    assert loaded.stats()["entities"] == 2


@pytest.mark.chaos
def test_cache_eviction_under_overflow(chaotic_env: None) -> None:
    """Overflowing the semantic cache evicts oldest entries instead of crashing."""
    from noema.cache import SemanticCache

    cache = SemanticCache(max_size=5, similarity_threshold=0.99)
    for i in range(20):
        cache.set(
            [{"role": "user", "content": f"prompt number {i}"}],
            f"response {i}",
            model="test-model",
            tenant_id="tenant-chaos",
        )
    assert len(cache._entries) <= 5  # noqa: SLF001
    assert (
        cache.get(
            [{"role": "user", "content": "prompt number 0"}],
            model="test-model",
            tenant_id="tenant-chaos",
        )
        is None
    )  # oldest evicted


@pytest.mark.chaos
def test_yaml_dump_roundtrip_after_chaos_load(tmp_path: Path, chaotic_env: None) -> None:
    """Loading under hostile env, dumping, reloading must not raise or leak secrets."""
    settings = NoemaSettings.from_yaml()
    base = NoemaSettings(**settings.model_dump())
    with_secret = NoemaSettings(
        **{**base.model_dump(), "api": {**base.api.model_dump(), "api_key": "super-secret-key"}}
    )
    out = tmp_path / "out.yaml"
    with_secret.dump_yaml(out)
    text = out.read_text(encoding="utf-8")
    assert "super-secret-key" not in text  # no secret leakage
    reloaded = NoemaSettings.from_yaml(out)
    assert isinstance(reloaded, NoemaSettings)
    assert reloaded.api.api_key.get_secret_value() != "super-secret-key"
