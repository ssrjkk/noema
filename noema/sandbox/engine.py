"""Sandbox engine — isolated linting, running, and testing of generated code.

Architecture:
- :class:`SandboxEngine` validates code in escalating stages: AST parse (pure),
  lint, execution, and pytest. Each stage has a Docker-backed path and a direct
  path with rlimits (Unix) / subprocess timeouts (Windows).
- Path handling is zero-trust: user-supplied filenames are joined under a fresh
  temp dir via :func:`_safe_join`, which rejects traversal escapes.

Concurrency contract:
- Capability probing (docker / bubblewrap) is deferred and offloaded with
  :func:`asyncio.to_thread`, so constructing the engine never blocks the loop.
- Direct subprocesses run through ``asyncio.create_subprocess_exec`` with
  hard timeouts; sync lint runs are offloaded with :func:`asyncio.to_thread`.

Complexity:
- ``validate_files``: ``O(F)`` AST parses plus ``O(F)`` lint/runs for F files;
  per-file runs are parallel (bounded by ``max_parallel``), so wall time is
  ``O(F / max_parallel)`` per stage, each subprocess bounded by ``max_cpu_seconds``.
"""

from __future__ import annotations

import ast
import asyncio
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from noema.logging import get_logger
from noema.sandbox.static_check import _STDLIB_MODULES, analyze_code

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

        resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_sec), int(cpu_sec) + 5))  # type: ignore[attr-defined]
        mem_bytes = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))  # type: ignore[attr-defined]
        resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))  # type: ignore[attr-defined]
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))  # type: ignore[attr-defined]
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))  # type: ignore[attr-defined]
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
    docker_image: str = "python:3.12-slim"
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


class SandboxEngine:
    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()
        # Capability probes are deferred: constructing the engine must not run
        # blocking subprocesses (docker --version / bwrap --version).
        self._has_docker: bool | None = None
        self._has_bwrap: bool | None = None

    def _detect_capabilities(self) -> None:
        """Probe docker + bubblewrap availability synchronously.

        Called from a worker thread (:func:`asyncio.to_thread`) in async paths
        or on demand from sync contexts. Complexity: ``O(1)`` subprocess calls.
        """
        self._has_docker = self._check_docker()
        self._has_bwrap = _check_bubblewrap()

    async def _ensure_capabilities(self) -> None:
        """Resolve capability flags without blocking the event loop."""
        if self._has_docker is None:
            await asyncio.to_thread(self._detect_capabilities)

    def _ensure_capabilities_sync(self) -> None:
        """Resolve capability flags synchronously (blocking; sync callers only)."""
        if self._has_docker is None:
            self._detect_capabilities()

    def _bwrap_cmd(self, cmd: list[str], tmp_dir: Path) -> list[str]:
        """Wrap a command with bubblewrap for sandbox isolation."""
        if not self._has_bwrap:
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

    def _check_docker(self) -> bool:
        try:
            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    async def validate_code_block(
        self,
        code: str,
        language: str = "python",
        filename: str = "main.py",
        allowed_imports: set[str] | None = None,
    ) -> CodeValidationResult:
        """Validate a single code block (AST parse + static pass for Python).

        Never raises: syntax problems are captured in the result's error list.
        Complexity: ``O(len(code))``.
        """
        result = CodeValidationResult(file_path=filename, language=language)
        t0 = time.monotonic()

        if language == "python":
            try:
                ast.parse(code)
                result.ast_valid = True
            except SyntaxError as e:
                result.ast_valid = False
                result.ast_errors.append(str(e))

            if result.ast_valid and self.config.static_check_enabled:
                allowed = (
                    allowed_imports if allowed_imports is not None else self._allowed_imports()
                )
                issues = analyze_code(code, allowed_imports=allowed)
                if issues:
                    result.static_passed = False
                    result.static_issues = [issue.render() for issue in issues]

        result.duration_ms = (time.monotonic() - t0) * 1000
        return result

    def _allowed_imports(self) -> set[str]:
        """Import roots permitted by the sandbox: stdlib + configured extras."""
        return set(_STDLIB_MODULES) | set(self.config.static_allow_imports)

    async def validate_files(
        self, files: list[dict[str, str]], run_tests: bool = False
    ) -> SandboxResult:
        """Validate all files: AST, lint, run, and optionally tests.

        Lint runs in a worker thread; code/test runs are subprocess-bounded
        with timeouts. Every per-file failure is captured in the result instead
        of raising, so callers always get a structured verdict (zero-trust).

        Complexity: ``O(F)`` per stage for F files.
        """
        await self._ensure_capabilities()
        t0 = time.monotonic()
        result = SandboxResult()

        sibling_roots = set()
        for raw in files:
            name = raw.get("path", raw.get("filename", "main.py"))
            parts = name.replace("\\", "/").split("/")
            root = parts[0] if len(parts) > 1 else Path(parts[-1]).stem
            if root:
                sibling_roots.add(root)
        allowed = self._allowed_imports() | sibling_roots

        for f in files:
            vr = await self.validate_code_block(
                code=f.get("content", ""),
                language=f.get("language", "python"),
                filename=f.get("path", f.get("filename", "main.py")),
                allowed_imports=allowed,
            )
            result.files.append(vr)

        if self.config.lint_enabled:
            if self._has_docker:
                result = await self._run_lint_in_docker(result, files)
            else:
                result = await asyncio.to_thread(self._run_lint_direct, result, files)

        if self.config.run_enabled:
            if self._has_docker:
                result = await self._run_code_in_docker(result, files)
            else:
                result = await self._run_code_direct(result, files)

        if run_tests and self.config.test_enabled:
            if self._has_docker:
                result = await self._run_tests_in_docker(result, files)
            else:
                result = await self._run_tests_direct(result, files)

        result.all_valid = all(
            vr.ast_valid and vr.static_passed and vr.lint_passed and vr.run_passed
            for vr in result.files
        )
        result.total_duration_ms = (time.monotonic() - t0) * 1000

        passed = sum(1 for vr in result.files if vr.ast_valid and vr.static_passed)
        total = len(result.files)
        result.summary = (
            f"{passed}/{total} files valid"
            f"{f', tests: {result.tests_passed}/{result.tests_passed + result.tests_failed}' if run_tests else ''}"
        )

        log.info(
            "sandbox_result",
            all_valid=result.all_valid,
            files=total,
            duration_ms=round(result.total_duration_ms, 1),
            tests=f"{result.tests_passed}/{result.tests_passed + result.tests_failed}"
            if run_tests
            else "none",
        )

        return result

    # ── Direct execution with resource limits ──────────────────────────

    def _run_lint_direct(self, result: SandboxResult, files: list[dict[str, str]]) -> SandboxResult:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            for f in files:
                content = f.get("content", "")
                path = f.get("path", f.get("filename", "main.py"))
                file_path = _safe_join(tmp_dir, path)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

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
                        preexec_fn=self._resource_limits_preexec(),
                    )
                    if proc.returncode != 0:
                        result.files[i].lint_passed = False
                        result.files[i].lint_errors = (
                            proc.stdout.strip().split("\n")
                            if proc.stdout.strip()
                            else [proc.stderr.strip()]
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

    async def _run_code_direct(
        self, result: SandboxResult, files: list[dict[str, str]]
    ) -> SandboxResult:
        tmp_dir = Path(tempfile.mkdtemp())
        semaphore = asyncio.Semaphore(max(1, self.config.max_parallel))
        try:
            for f in files:
                content = f.get("content", "")
                path = f.get("path", f.get("filename", "main.py"))
                file_path = _safe_join(tmp_dir, path)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

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
                            if self._has_bwrap:
                                run_cmd = self._bwrap_cmd(run_cmd, tmp_dir)
                            proc = await asyncio.create_subprocess_exec(
                                *run_cmd,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                                env=_build_isolated_env(),
                                preexec_fn=self._resource_limits_preexec()
                                if not self._has_bwrap
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
                            ].run_errors = (
                                f"Execution timed out after {self.config.max_cpu_seconds}s"
                            )
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

    async def _run_tests_direct(
        self, result: SandboxResult, files: list[dict[str, str]]
    ) -> SandboxResult:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            test_files = [f for f in files if "test" in f.get("path", f.get("filename", ""))]
            if not test_files:
                return result

            for f in files:
                content = f.get("content", "")
                path = f.get("path", f.get("filename", "main.py"))
                file_path = _safe_join(tmp_dir, path)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

            test_cmd = [
                sys.executable,
                "-m",
                "pytest",
                str(tmp_dir),
                "-x",
                "--tb=short",
                "--timeout=30",
            ]
            if self._has_bwrap:
                test_cmd = self._bwrap_cmd(test_cmd, tmp_dir)
            proc: Process | None = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    *test_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=_build_isolated_env(),
                    preexec_fn=self._resource_limits_preexec() if not self._has_bwrap else None,
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

        except TimeoutError:
            log.warning("tests_timeout")
        except Exception as e:
            log.warning("test_direct_error", error=str(e))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return result

    # ── Docker-based execution (with network isolation) ────────────────

    def _resource_limits_preexec(self) -> Callable[[], None] | None:
        """Build the ``preexec_fn`` applying this engine's configured limits."""
        if not _IS_UNIX:
            return None
        cpu = self.config.max_cpu_seconds
        mem = self.config.max_memory_mb

        def _apply() -> None:
            _set_resource_limits(cpu_sec=cpu, mem_mb=mem)

        return _apply

    def _docker_run_flags(self) -> list[str]:
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

    async def _run_lint_in_docker(
        self, result: SandboxResult, files: list[dict[str, str]]
    ) -> SandboxResult:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            for f in files:
                content = f.get("content", "")
                path = f.get("path", f.get("filename", "main.py"))
                file_path = _safe_join(tmp_dir, path)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

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
                            *self._docker_run_flags(),
                            "-v",
                            f"{tmp_dir}:/sandbox:ro",
                            "-w",
                            "/sandbox",
                            self.config.docker_image,
                            "ruff",
                            "check",
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

    async def _run_code_in_docker(
        self, result: SandboxResult, files: list[dict[str, str]]
    ) -> SandboxResult:
        tmp_dir = Path(tempfile.mkdtemp())
        semaphore = asyncio.Semaphore(max(1, self.config.max_parallel))
        try:
            for f in files:
                content = f.get("content", "")
                path = f.get("path", f.get("filename", "main.py"))
                file_path = _safe_join(tmp_dir, path)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

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
                                *self._docker_run_flags(),
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
                            ].run_errors = (
                                f"Execution timed out after {self.config.max_cpu_seconds}s"
                            )
                        except Exception as e:
                            result.files[i].run_passed = False
                            result.files[i].run_errors = str(e)[:500]
                        finally:
                            if proc is not None and proc.returncode is None:
                                proc.kill()
                                await proc.wait()

            await asyncio.gather(*(_run_one(i, f) for i, f in enumerate(files)))

        except TimeoutError:
            log.warning("run_code_docker_timeout")
        except Exception as e:
            log.warning("run_code_docker_error", error=str(e))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return result

    async def _run_tests_in_docker(
        self, result: SandboxResult, files: list[dict[str, str]]
    ) -> SandboxResult:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            test_files = [f for f in files if "test" in f.get("path", f.get("filename", ""))]
            if not test_files:
                return result

            for f in files:
                content = f.get("content", "")
                path = f.get("path", f.get("filename", "main.py"))
                file_path = _safe_join(tmp_dir, path)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

            proc: Process | None = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    "docker",
                    "run",
                    *self._docker_run_flags(),
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

    def is_available(self) -> bool:
        self._ensure_capabilities_sync()
        return bool(self._has_docker) and self.config.enabled
