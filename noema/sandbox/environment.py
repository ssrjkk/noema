"""Execution environments for the sandbox.

Architecture:
- :class:`Environment` is the abstraction seam between validation and the
  medium a candidate solution is executed in. Today there are two
  implementations over Python code (local subprocess with rlimits and Docker
  with network isolation), but the interface is deliberately agnostic to the
  artifact type: a physics engine, a hardware simulator (Verilator), or a
  molecular dynamics backend (OpenMM) would implement the same three stages
  (``lint`` / ``run_code`` / ``run_tests``) over their own artifact model.
- :class:`SandboxEngine` (in ``engine.py``) owns capability probing and stage
  orchestration and dispatches each stage to the best available environment,
  so no environment hardcodes the decision of "who runs first".
- Every environment is fail-closed: timeouts, resource limits and structured
  results instead of exceptions (zero-trust input handling).

Shared helpers (:func:`_safe_join`, :func:`_build_isolated_env`,
:func:`_parse_pytest_counts`, ...) live here so environments stay
self-contained; ``engine.py`` re-exports them for backwards compatibility.
"""

from __future__ import annotations

import asyncio
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from noema.logging import get_logger

if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import Callable

log = get_logger(__name__)

_IS_UNIX = platform.system() != "Windows"


def _set_resource_limits(cpu_sec: float = 10, mem_mb: int = 256) -> None:
    """Set RLIMIT_*, NPROC, NOFILE, FSIZE on Unix systems."""
    if not _IS_UNIX:
        return
    try:
        import resource

        def rlimit(name: str) -> int:
            value = getattr(resource, name)
            if not isinstance(value, int):
                raise ValueError(f"resource limit {name!r} unavailable")
            return value

        setrlimit = getattr(resource, "setrlimit")  # noqa: B009 (unavailable on win32 typeshed)
        setrlimit(rlimit("RLIMIT_CPU"), (int(cpu_sec), int(cpu_sec) + 5))
        mem_bytes = mem_mb * 1024 * 1024
        setrlimit(rlimit("RLIMIT_AS"), (mem_bytes, mem_bytes))
        setrlimit(rlimit("RLIMIT_NPROC"), (10, 10))
        setrlimit(rlimit("RLIMIT_NOFILE"), (64, 64))
        setrlimit(rlimit("RLIMIT_FSIZE"), (10 * 1024 * 1024, 10 * 1024 * 1024))
    except (ImportError, OSError, ResourceWarning, ValueError):
        log.warning("resource_limits_unavailable")


def _check_bubblewrap() -> bool:
    """Check if bubblewrap (bwrap) is available on the system."""
    if not _IS_UNIX:
        return False
    try:
        result = subprocess.run(
            ["bwrap", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _check_docker() -> bool:
    """Check if the docker CLI is available on the system."""
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _safe_join(tmp_dir: Path, rel_path: str) -> Path:
    """Join a user-supplied path under tmp_dir, rejecting traversal escapes."""
    base = tmp_dir.resolve()
    candidate = (base / (rel_path or "main.py").lstrip("/\\")).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise ValueError(f"Unsafe path in sandbox input: {rel_path!r}") from None
    return candidate


def _build_isolated_env() -> dict[str, str]:
    """Build environment without network access."""
    env = os.environ.copy()
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    env["HTTP_PROXY"] = ""
    env["HTTPS_PROXY"] = ""
    env["http_proxy"] = ""
    env["https_proxy"] = ""
    env["PYTHONIOENCODING"] = "utf-8"
    return env


class ValidationLevel(Enum):
    AST = "ast"
    SYNTAX = "syntax"
    LINT = "lint"
    TYPE_CHECK = "type_check"
    RUN = "run"
    TEST = "test"


def _parse_pytest_counts(output: str, returncode: int) -> tuple[int, int]:
    """Extract ``(passed, failed)`` test counts from pytest's summary line.

    ``pytest -q`` does not emit the uppercase ``PASSED``/``FAILED`` markers, so
    counting those substrings misreports green runs as zero passed. The summary
    line ``"N passed, M failed"`` is parsed instead; when it is missing the
    return code is the source of truth (0 ⇒ all green, else ≥1 failure).
    """
    passed = 0
    failed = 0
    tail = output[-2000:]
    match_passed = re.search(r"(\d+)\s+passed", tail)
    match_failed = re.search(r"(\d+)\s+failed", tail)
    match_errors = re.search(r"(\d+)\s+error", tail)
    if match_passed:
        passed = int(match_passed.group(1))
    if match_failed:
        failed += int(match_failed.group(1))
    if match_errors:
        failed += int(match_errors.group(1))
    if not match_passed and not match_failed and not match_errors:
        passed = 1 if returncode == 0 else 0
        failed = 0 if returncode == 0 else 1
    return passed, failed


@dataclass
class SandboxConfig:
    enabled: bool = True
    timeout: float = 60.0
    max_memory_mb: int = 256
    max_cpu_seconds: float = 10.0
    max_cpus: float = 0.5
    network_isolation: bool = True
    docker_image: str = "noema-sandbox:3.12"
    static_check_enabled: bool = True
    static_allow_imports: tuple[str, ...] = ()
    lint_enabled: bool = True
    type_check_enabled: bool = True
    run_enabled: bool = True
    test_enabled: bool = True
    max_parallel: int = 4
    temp_dir: str = ""


@dataclass
class CodeValidationResult:
    file_path: str
    language: str
    ast_valid: bool = True
    ast_errors: list[str] = field(default_factory=list)
    static_passed: bool = True
    static_issues: list[str] = field(default_factory=list)
    lint_passed: bool = True
    lint_errors: list[str] = field(default_factory=list)
    type_check_passed: bool = True
    type_errors: list[str] = field(default_factory=list)
    run_passed: bool = True
    run_output: str = ""
    run_errors: str = ""
    duration_ms: float = 0.0


@dataclass
class SandboxResult:
    all_valid: bool = False
    files: list[CodeValidationResult] = field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0
    total_duration_ms: float = 0.0
    summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class Environment(ABC):
    """Abstract execution medium for validating candidate artifacts.

    Implementations translate the generic three-stage pipeline (lint, run,
    tests) into their own medium: subprocesses, containers, simulators. The
    engine picks the best available environment per stage; environments never
    decide routing themselves.
    """

    id: str = "abstract"
    description: str = ""

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this environment can actually execute right now."""

    @abstractmethod
    async def lint(self, result: SandboxResult, files: list[dict[str, str]]) -> SandboxResult:
        """Static lint stage; failures land in ``result`` (never raises)."""

    @abstractmethod
    async def run_code(self, result: SandboxResult, files: list[dict[str, str]]) -> SandboxResult:
        """Execution stage; failures land in ``result`` (never raises)."""

    @abstractmethod
    async def run_tests(self, result: SandboxResult, files: list[dict[str, str]]) -> SandboxResult:
        """Test stage; counts land in ``result`` (never raises)."""


class LocalEnvironment(Environment):
    """Direct subprocess execution with rlimits (Unix) / timeouts (Windows).

    ``bwrap_enabled`` is set by the engine after capability probing; when set,
    commands are wrapped with bubblewrap for namespace isolation.
    """

    id = "local"
    description = "Direct subprocess with resource limits"

    def __init__(self, config: SandboxConfig) -> None:
        self.config = config
        self.bwrap_enabled = False

    def bwrap_cmd(self, cmd: list[str], tmp_dir: Path) -> list[str]:
        """Wrap a command with bubblewrap for sandbox isolation."""
        if not self.bwrap_enabled:
            return cmd
        return [
            "bwrap",
            "--unshare-all",
            "--ro-bind",
            "/",
            "/",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--bind",
            str(tmp_dir),
            "/sandbox",
            "--chdir",
            "/sandbox",
            "--",
            *cmd,
        ]

    def is_available(self) -> bool:
        return True

    def resource_limits_preexec(self) -> Callable[[], None] | None:
        """Build the ``preexec_fn`` applying this environment's configured limits."""
        if not _IS_UNIX:
            return None
        cpu = self.config.max_cpu_seconds
        mem = self.config.max_memory_mb

        def _apply() -> None:
            _set_resource_limits(cpu_sec=cpu, mem_mb=mem)

        return _apply

    def _write_files(self, files: list[dict[str, str]], tmp_dir: Path) -> None:
        for f in files:
            content = f.get("content", "")
            path = f.get("path", f.get("filename", "main.py"))
            file_path = _safe_join(tmp_dir, path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

    async def lint(self, result: SandboxResult, files: list[dict[str, str]]) -> SandboxResult:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            self._write_files(files, tmp_dir)

            def _lint(i: int, f: dict[str, str]) -> None:
                path = f.get("path", f.get("filename", "main.py"))
                file_path = _safe_join(tmp_dir, path)
                lang = f.get("language", "python")

                if lang == "python" and file_path.suffix == ".py":
                    proc = subprocess.run(
                        [sys.executable, "-m", "ruff", "check", "--select=E,F,W", str(file_path)],
                        capture_output=True,
                        text=True,
                        timeout=self.config.max_cpu_seconds,
                        env=_build_isolated_env(),
                    )
                    if proc.returncode != 0:
                        result.files[i].lint_passed = False
                        combined = (proc.stdout + proc.stderr).strip()
                        result.files[i].lint_errors = (
                            combined.split("\n") if combined else ["lint produced no output"]
                        )

            with ThreadPoolExecutor(
                max_workers=max(1, min(self.config.max_parallel, len(files)))
            ) as pool:
                list(pool.map(_lint, range(len(files)), files))
        except subprocess.TimeoutExpired:
            log.warning("lint_timeout")
        except Exception as e:
            log.warning("lint_direct_error", error=str(e))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return result

    async def run_code(self, result: SandboxResult, files: list[dict[str, str]]) -> SandboxResult:
        tmp_dir = Path(tempfile.mkdtemp())
        semaphore = asyncio.Semaphore(max(1, self.config.max_parallel))
        try:
            self._write_files(files, tmp_dir)

            async def _run_one(i: int, f: dict[str, str]) -> None:
                async with semaphore:
                    path = f.get("path", f.get("filename", "main.py"))
                    lang = f.get("language", "python")
                    if lang != "python":
                        return
                    if not result.files[i].static_passed:
                        return
                    file_path = _safe_join(tmp_dir, path)
                    t0 = time.monotonic()
                    proc: Process | None = None
                    try:
                        run_cmd = [sys.executable, str(file_path)]
                        if self.bwrap_enabled:
                            run_cmd = self.bwrap_cmd(run_cmd, tmp_dir)
                        proc = await asyncio.create_subprocess_exec(
                            *run_cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            env=_build_isolated_env(),
                            preexec_fn=self.resource_limits_preexec()
                            if not self.bwrap_enabled
                            else None,
                        )
                        stdout, stderr = await asyncio.wait_for(
                            proc.communicate(), timeout=self.config.max_cpu_seconds
                        )
                        result.files[i].duration_ms = (time.monotonic() - t0) * 1000
                        combined = (stdout + stderr).decode("utf-8", errors="replace")
                        if proc.returncode == 0:
                            result.files[i].run_passed = True
                            result.files[i].run_output = combined[:500]
                        else:
                            result.files[i].run_passed = False
                            result.files[i].run_errors = combined[:1000]
                    except TimeoutError:
                        result.files[i].run_passed = False
                        result.files[
                            i
                        ].run_errors = f"Execution timed out after {self.config.max_cpu_seconds}s"
                    except Exception as e:
                        result.files[i].run_passed = False
                        result.files[i].run_errors = str(e)[:500]
                    finally:
                        if proc is not None and proc.returncode is None:
                            proc.kill()
                            await proc.wait()

            await asyncio.gather(*(_run_one(i, f) for i, f in enumerate(files)))

        except Exception as e:
            log.warning("run_code_direct_error", error=str(e))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return result

    async def run_tests(self, result: SandboxResult, files: list[dict[str, str]]) -> SandboxResult:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            test_files = [f for f in files if "test" in f.get("path", f.get("filename", ""))]
            if not test_files:
                return result

            self._write_files(files, tmp_dir)

            test_cmd = [
                sys.executable,
                "-m",
                "pytest",
                str(tmp_dir),
                "-x",
                "--tb=short",
                "--timeout=30",
            ]
            if self.bwrap_enabled:
                test_cmd = self.bwrap_cmd(test_cmd, tmp_dir)
            proc: Process | None = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    *test_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=_build_isolated_env(),
                    preexec_fn=self.resource_limits_preexec() if not self.bwrap_enabled else None,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=min(60.0, self.config.timeout)
                )
                combined = (stdout + stderr).decode("utf-8", errors="replace")
                result.tests_passed, result.tests_failed = _parse_pytest_counts(
                    combined, proc.returncode or 0
                )
            except TimeoutError:
                log.warning("tests_timeout")
            except Exception as e:
                log.warning("test_direct_error", error=str(e))
            finally:
                if proc is not None and proc.returncode is None:
                    proc.kill()
                    await proc.wait()

        except Exception as e:
            log.warning("test_direct_error", error=str(e))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return result


class DockerEnvironment(Environment):
    """Container-based execution with network isolation and quota flags."""

    id = "docker"
    description = "Docker container with network isolation"

    def __init__(self, config: SandboxConfig) -> None:
        self.config = config

    def is_available(self) -> bool:
        return _check_docker()

    def docker_run_flags(self) -> list[str]:
        flags = ["--rm"]
        if self.config.network_isolation:
            flags.append("--network=none")
        flags.extend(
            [
                "--memory",
                f"{self.config.max_memory_mb}m",
                "--cpus",
                f"{self.config.max_cpus}",
            ]
        )
        return flags

    def _write_files(self, files: list[dict[str, str]], tmp_dir: Path) -> None:
        for f in files:
            content = f.get("content", "")
            path = f.get("path", f.get("filename", "main.py"))
            file_path = _safe_join(tmp_dir, path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

    async def lint(self, result: SandboxResult, files: list[dict[str, str]]) -> SandboxResult:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            self._write_files(files, tmp_dir)

            for i, f in enumerate(files):
                path = f.get("path", f.get("filename", "main.py"))
                file_path = _safe_join(tmp_dir, path)
                lang = f.get("language", "python")

                if lang == "python" and file_path.suffix == ".py":
                    proc: Process | None = None
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            "docker",
                            "run",
                            *self.docker_run_flags(),
                            "-v",
                            f"{tmp_dir}:/sandbox:ro",
                            "-w",
                            "/sandbox",
                            self.config.docker_image,
                            "ruff",
                            "check",
                            "--no-cache",
                            "--select=E,F,W",
                            f"/sandbox/{file_path.name}",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                        if proc.returncode != 0:
                            out = (stdout + stderr).decode("utf-8", errors="replace")
                            result.files[i].lint_passed = False
                            result.files[i].lint_errors = (
                                out.strip().split("\n")
                                if out.strip()
                                else [stderr.decode("utf-8", errors="replace").strip()]
                            )
                    except TimeoutError:
                        result.files[i].lint_passed = False
                        result.files[i].lint_errors = ["Lint timed out"]
                    finally:
                        if proc is not None and proc.returncode is None:
                            proc.kill()
                            await proc.wait()

        except Exception as e:
            log.warning("lint_docker_error", error=str(e))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return result

    async def run_code(self, result: SandboxResult, files: list[dict[str, str]]) -> SandboxResult:
        tmp_dir = Path(tempfile.mkdtemp())
        semaphore = asyncio.Semaphore(max(1, self.config.max_parallel))
        try:
            self._write_files(files, tmp_dir)

            async def _run_one(i: int, f: dict[str, str]) -> None:
                async with semaphore:
                    path = f.get("path", f.get("filename", "main.py"))
                    lang = f.get("language", "python")
                    if lang != "python":
                        return
                    if not result.files[i].static_passed:
                        return

                    file_path = _safe_join(tmp_dir, path)
                    t0 = time.monotonic()
                    proc: Process | None = None
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            "docker",
                            "run",
                            *self.docker_run_flags(),
                            "-v",
                            f"{tmp_dir}:/sandbox:ro",
                            "-w",
                            "/sandbox",
                            self.config.docker_image,
                            "python",
                            f"/sandbox/{file_path.name}",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        stdout, stderr = await asyncio.wait_for(
                            proc.communicate(), timeout=self.config.max_cpu_seconds
                        )
                        result.files[i].duration_ms = (time.monotonic() - t0) * 1000
                        combined = (stdout + stderr).decode("utf-8", errors="replace")

                        if proc.returncode == 0:
                            result.files[i].run_passed = True
                            result.files[i].run_output = combined[:500]
                        else:
                            result.files[i].run_passed = False
                            result.files[i].run_errors = combined[:1000]
                    except TimeoutError:
                        result.files[i].run_passed = False
                        result.files[
                            i
                        ].run_errors = f"Execution timed out after {self.config.max_cpu_seconds}s"
                    except Exception as e:
                        result.files[i].run_passed = False
                        result.files[i].run_errors = str(e)[:500]
                    finally:
                        if proc is not None and proc.returncode is None:
                            proc.kill()
                            await proc.wait()

            await asyncio.gather(*(_run_one(i, f) for i, f in enumerate(files)))

        except Exception as e:
            log.warning("run_code_docker_error", error=str(e))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return result

    async def run_tests(self, result: SandboxResult, files: list[dict[str, str]]) -> SandboxResult:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            test_files = [f for f in files if "test" in f.get("path", f.get("filename", ""))]
            if not test_files:
                return result

            self._write_files(files, tmp_dir)

            proc: Process | None = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    "docker",
                    "run",
                    *self.docker_run_flags(),
                    "-v",
                    f"{tmp_dir}:/sandbox:ro",
                    "-w",
                    "/sandbox",
                    self.config.docker_image,
                    "python",
                    "-m",
                    "pytest",
                    "/sandbox",
                    "-x",
                    "--tb=short",
                    "--timeout=30",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                combined = (stdout + stderr).decode("utf-8", errors="replace")
                result.tests_passed, result.tests_failed = _parse_pytest_counts(
                    combined, proc.returncode or 0
                )
            except TimeoutError:
                log.warning("tests_docker_timeout")
            except Exception as e:
                log.warning("test_docker_error", error=str(e))
            finally:
                if proc is not None and proc.returncode is None:
                    proc.kill()
                    await proc.wait()

        except Exception as e:
            log.warning("test_docker_error", error=str(e))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return result
