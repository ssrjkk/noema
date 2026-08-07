"""Docker-based plugin sandbox for safe code execution."""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from noema.config.settings import get_settings
from noema.logging import get_logger

log = get_logger(__name__)


@dataclass
class SandboxResult:
    """Result of sandboxed execution."""

    success: bool = False
    output: str = ""
    error: str = ""
    exit_code: int = -1
    duration_ms: float = 0.0
    timed_out: bool = False
    memory_used_mb: float = 0.0


@dataclass
class SandboxConfig:
    """Sandbox execution constraints."""

    image: str = "python:3.12-slim"
    timeout: float = 60.0
    max_memory: str = "256m"
    max_cpus: float = 0.5
    network_disabled: bool = True
    read_only_root: bool = True
    tmpfs_size: str = "128m"
    allowed_paths: list[str] = field(default_factory=list)


class PluginSandbox:
    """Executes plugin code in a Docker container with strict resource limits.

    Security model:
    - No network access (unless explicitly allowed)
    - Read-only root filesystem
    - Memory + CPU limits
    - Execution timeout
    - No access to host filesystem (except mounted plugin dir)
    - Dropped all capabilities except SYS_NICE
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        settings = get_settings()
        self.config = config or SandboxConfig(
            image=settings.sandbox.docker_image,
            timeout=settings.sandbox.timeout,
            max_memory=settings.sandbox.max_memory,
            max_cpus=settings.sandbox.max_cpus,
            network_disabled=settings.sandbox.network_disabled,
            read_only_root=settings.sandbox.read_only_root,
        )

    async def execute_plugin(
        self,
        plugin_code: str,
        entry_point: str = "main",
        input_data: dict[str, Any] | None = None,
        extra_volumes: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Execute plugin code in a sandboxed Docker container.

        Args:
            plugin_code: Python source code to execute
            entry_point: Function name to call (default: main)
            input_data: JSON-serializable data passed to the entry point
            extra_volumes: Additional host:container volume mounts

        Returns:
            SandboxResult with output/error/status
        """
        t0 = time.monotonic()

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            self._write_plugin(plugin_dir, plugin_code, entry_point, input_data)
            result = await self._run_container(plugin_dir, extra_volumes or {})
            result.duration_ms = (time.monotonic() - t0) * 1000

            log.info(
                "sandbox_execution",
                success=result.success,
                exit_code=result.exit_code,
                duration_ms=round(result.duration_ms, 1),
                timed_out=result.timed_out,
            )
            return result

    async def execute_file(
        self,
        file_path: str | Path,
        entry_point: str = "main",
        input_data: dict[str, Any] | None = None,
    ) -> SandboxResult:
        """Execute a plugin file in the sandbox."""
        path = Path(file_path).resolve()
        if not path.is_file():
            return SandboxResult(
                success=False,
                error=f"File not found: {path}",
                exit_code=1,
            )

        code = path.read_text(encoding="utf-8")
        extra_volumes = {str(path.parent): "/workspace/mount:ro"}
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            self._write_plugin(plugin_dir, code, entry_point, input_data)
            result = await self._run_container(plugin_dir, extra_volumes, allowed_host=path.parent)
        result.duration_ms = (time.monotonic() - t0) * 1000
        return result

    def _write_plugin(
        self,
        plugin_dir: Path,
        code: str,
        entry_point: str,
        input_data: dict[str, Any] | None,
    ) -> None:
        """Write plugin code and runner to temp directory."""
        # Plugin code
        (plugin_dir / "plugin.py").write_text(code, encoding="utf-8")

        if not entry_point.isidentifier():
            raise ValueError(f"Invalid entry_point: {entry_point}")
        runner = f"""#!/usr/bin/env python3
import json
import sys
import traceback
from plugin import {entry_point}

input_data = None
try:
    with open("/workspace/input.json", "r") as f:
        input_data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    pass

try:
    if input_data:
        result = {entry_point}(**input_data) if isinstance(input_data, dict) else {entry_point}(input_data)
    else:
        result = {entry_point}()

    output = {{"success": True, "output": str(result) if result is not None else ""}}
except Exception as e:
    output = {{"success": False, "error": f"{{type(e).__name__}}: {{e}}", "traceback": traceback.format_exc()}}

with open("/workspace/output.json", "w") as f:
    json.dump(output, f)
"""
        (plugin_dir / "runner.py").write_text(runner, encoding="utf-8")

        # Input data
        input_file = plugin_dir / "input.json"
        input_file.write_text(json.dumps(input_data or {}), encoding="utf-8")

    def _validate_volumes(
        self, extra_volumes: dict[str, str], allowed_host: Path | None = None
    ) -> None:
        """Reject any volume whose host path is not explicitly allowed."""
        if not extra_volumes:
            return
        if allowed_host is not None:
            allowed_host = allowed_host.resolve()
        allowed = [Path(p).resolve() for p in self.config.allowed_paths]
        if allowed_host is None and not allowed:
            raise ValueError("extra_volumes are disabled: configure allowed_paths on SandboxConfig")
        for host in extra_volumes:
            resolved = Path(host).resolve()
            if allowed_host is not None and (
                resolved == allowed_host or allowed_host in resolved.parents
            ):
                continue
            if not any(resolved == base or base in resolved.parents for base in allowed):
                raise ValueError(f"Volume host path not allowed: {host}")

    async def _run_container(
        self,
        plugin_dir: Path,
        extra_volumes: dict[str, str],
        allowed_host: Path | None = None,
    ) -> SandboxResult:
        """Run the Docker container."""
        get_settings()

        self._validate_volumes(extra_volumes, allowed_host)

        cmd = [
            "docker",
            "run",
            "--rm",
            "--read-only" if self.config.read_only_root else "",
            f"--memory={self.config.max_memory}",
            f"--cpus={self.config.max_cpus}",
            f"--network={'' if not self.config.network_disabled else 'none'}",
            f"--tmpfs=/tmp:size={self.config.tmpfs_size}",
            # Drop capabilities
            "--cap-drop=ALL",
            "--cap-add=SYS_NICE",
            # No new privileges
            "--security-opt=no-new-privileges",
            # Volume mounts
            f"-v={plugin_dir}:/workspace:ro",
        ]

        # Add extra volumes (always read-only)
        for host, container in extra_volumes.items():
            mount = f"-v={host}:{container}"
            if not mount.endswith(":ro"):
                mount += ":ro"
            cmd.append(mount)

        # Entry point
        cmd.extend(
            [
                self.config.image,
                "python",
                "/workspace/runner.py",
            ]
        )

        # Filter empty strings
        cmd = [c for c in cmd if c]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.config.timeout,
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return SandboxResult(
                    success=False,
                    error=f"Execution timed out after {self.config.timeout}s",
                    exit_code=-1,
                    timed_out=True,
                )

            # Parse output
            output_file = plugin_dir / "output.json"
            if output_file.is_file():
                with contextlib.suppress(json.JSONDecodeError):
                    output_data = json.loads(output_file.read_text(encoding="utf-8"))
                    return SandboxResult(
                        success=output_data.get("success", False),
                        output=output_data.get("output", ""),
                        error=output_data.get("error", ""),
                        exit_code=proc.returncode or 0,
                    )

            return SandboxResult(
                success=(proc.returncode == 0),
                output=stdout.decode(errors="replace"),
                error=stderr.decode(errors="replace"),
                exit_code=proc.returncode or 0,
            )

        except FileNotFoundError:
            return SandboxResult(
                success=False,
                error="Docker not installed or not in PATH",
                exit_code=-1,
            )
        except Exception as exc:
            return SandboxResult(
                success=False,
                error=str(exc),
                exit_code=-1,
            )
