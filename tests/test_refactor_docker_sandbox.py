"""Coverage for noema.sandbox.docker — PluginSandbox logic without requiring Docker."""

import asyncio
from pathlib import Path

import pytest

from noema.sandbox.docker import PluginSandbox, SandboxConfig, SandboxResult


def test_sandbox_config_defaults():
    cfg = SandboxConfig()
    assert cfg.image == "python:3.12-slim"
    assert cfg.timeout == 60.0
    assert cfg.max_memory == "256m"
    assert cfg.max_cpus == 0.5
    assert cfg.network_disabled is True
    assert cfg.read_only_root is True
    assert cfg.tmpfs_size == "128m"
    assert cfg.allowed_paths == []


def test_plugin_sandbox_init_from_settings():
    sandbox = PluginSandbox()
    assert isinstance(sandbox.config, SandboxConfig)
    assert sandbox.config.read_only_root is True
    assert sandbox.config.network_disabled is True


def test_write_plugin_creates_files(tmp_path):
    sandbox = PluginSandbox()
    sandbox._write_plugin(tmp_path, "def main():\n    return 1\n", "main", {"a": 1})
    assert (tmp_path / "plugin.py").read_text(encoding="utf-8") == "def main():\n    return 1\n"
    runner = (tmp_path / "runner.py").read_text(encoding="utf-8")
    assert "from plugin import main" in runner
    assert "output.json" in runner
    assert (tmp_path / "input.json").read_text(encoding="utf-8") == '{"a": 1}'


def test_write_plugin_rejects_invalid_entry_point(tmp_path):
    sandbox = PluginSandbox()
    with pytest.raises(ValueError, match="Invalid entry_point"):
        sandbox._write_plugin(tmp_path, "def main(): pass", "not-valid", None)


def test_validate_volumes_rejects_unconfigured():
    sandbox = PluginSandbox()
    with pytest.raises(ValueError, match="extra_volumes are disabled"):
        sandbox._validate_volumes({"C:\\tmp\\host": "/mount"}, None)


def test_validate_volumes_allows_configured_base(tmp_path):
    host = tmp_path / "host"
    host.mkdir()
    sub = host / "sub"
    sub.mkdir()
    sandbox = PluginSandbox(config=SandboxConfig(allowed_paths=[str(host)]))
    sandbox._validate_volumes({str(sub): "/mount"}, None)


def test_validate_volumes_rejects_unrelated_path(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    sandbox = PluginSandbox(config=SandboxConfig(allowed_paths=[str(allowed)]))
    with pytest.raises(ValueError, match="Volume host path not allowed"):
        sandbox._validate_volumes({str(other): "/mount"}, None)


def test_validate_volumes_allowed_host_param(tmp_path):
    host = tmp_path / "host"
    host.mkdir()
    sub = host / "sub"
    sub.mkdir()
    sandbox = PluginSandbox()
    sandbox._validate_volumes({str(sub): "/mount"}, allowed_host=host)


def test_execute_file_missing_returns_error(tmp_path):
    sandbox = PluginSandbox()
    result = asyncio.run(sandbox.execute_file(tmp_path / "nope.py"))
    assert isinstance(result, SandboxResult)
    assert result.success is False
    assert result.exit_code == 1
    assert "File not found" in result.error


@pytest.mark.asyncio
async def test_run_container_docker_missing(tmp_path, monkeypatch):
    async def _no_docker(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _no_docker)
    sandbox = PluginSandbox()
    result = await sandbox._run_container(tmp_path, {})
    assert result.success is False
    assert result.exit_code == -1
    assert "Docker not installed" in result.error


@pytest.mark.asyncio
async def test_run_container_execution_error(tmp_path, monkeypatch):
    async def _boom(*args, **kwargs):
        raise RuntimeError("container exploded")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _boom)
    sandbox = PluginSandbox()
    result = await sandbox._run_container(tmp_path, {})
    assert result.success is False
    assert result.exit_code == -1
    assert result.error == "container exploded"


@pytest.mark.asyncio
async def test_run_container_timeout(tmp_path, monkeypatch):
    class _Proc:
        def __init__(self) -> None:
            self.returncode = -1

        async def communicate(self):
            return b"", b""

        def kill(self) -> None:
            pass

        async def wait(self) -> int:
            return -1

    async def _raise_timeout(awaitable, timeout):
        awaitable.close()
        raise TimeoutError()

    async def _fake_exec(*args, **kwargs):
        return _Proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(asyncio, "wait_for", _raise_timeout)
    sandbox = PluginSandbox()
    result = await sandbox._run_container(tmp_path, {})
    assert result.timed_out is True
    assert result.exit_code == -1
    assert "timed out" in result.error


@pytest.mark.asyncio
async def test_run_container_parses_output_json(tmp_path, monkeypatch):
    class _Proc:
        def __init__(self, plugin_dir: Path) -> None:
            self._dir = plugin_dir
            self.returncode = 0

        async def communicate(self):
            (self._dir / "output.json").write_text(
                '{"success": true, "output": "hello world"}', encoding="utf-8"
            )
            return b"", b""

        def kill(self) -> None:
            pass

        async def wait(self) -> int:
            return 0

    async def _fake_exec(*args, **kwargs):
        return _Proc(tmp_path)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    sandbox = PluginSandbox()
    result = await sandbox._run_container(tmp_path, {})
    assert result.success is True
    assert result.output == "hello world"
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_execute_plugin_end_to_end_without_docker(tmp_path, monkeypatch):
    async def _no_docker(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _no_docker)
    sandbox = PluginSandbox()
    result = await sandbox.execute_plugin(
        "def main():\n    return 42\n",
        entry_point="main",
        input_data={"x": 1},
    )
    assert result.success is False
    assert "Docker not installed" in result.error
    assert result.duration_ms >= 0
