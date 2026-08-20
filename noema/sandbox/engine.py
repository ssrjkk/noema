"""Sandbox engine — isolated linting, running, and testing of generated code.

Architecture:
- :class:`SandboxEngine` validates code in escalating stages: AST parse (pure),
  lint, execution, and pytest. Each stage is dispatched to the best available
  :class:`~noema.sandbox.environment.Environment` (Docker when present, direct
  subprocess with rlimits otherwise); the environment never decides routing.
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
import time
from pathlib import Path
from typing import TYPE_CHECKING

from noema.logging import get_logger
from noema.sandbox.environment import (
    CodeValidationResult,
    DockerEnvironment,
    Environment,
    LocalEnvironment,
    SandboxConfig,
    SandboxResult,
    _build_isolated_env,
    _check_bubblewrap,
    _parse_pytest_counts,
    _safe_join,
    _set_resource_limits,
)
from noema.sandbox.static_check import _STDLIB_MODULES, analyze_code

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "CodeValidationResult",
    "Environment",
    "SandboxConfig",
    "SandboxEngine",
    "SandboxResult",
    "_build_isolated_env",
    "_check_bubblewrap",
    "_parse_pytest_counts",
    "_safe_join",
    "_set_resource_limits",
]

log = get_logger(__name__)


class SandboxEngine:
    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()
        self._local = LocalEnvironment(self.config)
        self._docker = DockerEnvironment(self.config)
        self._envs: list[Environment] = [self._docker, self._local]
        # Capability probes are deferred: constructing the engine must not run
        # blocking subprocesses (docker --version / bwrap --version).
        self._has_docker: bool | None = None
        self._has_bwrap: bool | None = None

    def _detect_capabilities(self) -> None:
        """Probe docker + bubblewrap availability synchronously.

        Called from a worker thread (:func:`asyncio.to_thread`) in async paths
        or on demand from sync contexts. Complexity: ``O(1)`` subprocess calls.
        """
        self._has_docker = self._docker.is_available()
        self._has_bwrap = _check_bubblewrap()

    async def _ensure_capabilities(self) -> None:
        """Resolve capability flags without blocking the event loop."""
        if self._has_docker is None:
            await asyncio.to_thread(self._detect_capabilities)

    def _ensure_capabilities_sync(self) -> None:
        """Resolve capability flags synchronously (blocking; sync callers only)."""
        if self._has_docker is None:
            self._detect_capabilities()

    def _pick_env(self) -> Environment:
        """Best available environment for the next stage."""
        return self._docker if self._has_docker else self._local

    def _bwrap_cmd(self, cmd: list[str], tmp_dir: Path) -> list[str]:
        """Wrap a command with bubblewrap (delegates to the local environment)."""
        return self._local.bwrap_cmd(cmd, tmp_dir)

    def _docker_run_flags(self) -> list[str]:
        """Docker quota/network flags (delegates to the docker environment)."""
        return self._docker.docker_run_flags()

    def _resource_limits_preexec(self) -> Callable[[], None] | None:
        """``preexec_fn`` for the direct path (delegates to the local environment)."""
        return self._local.resource_limits_preexec()

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
            allowed = allowed_imports if allowed_imports is not None else self._allowed_imports()

            def _python_check() -> tuple[bool, list[str], bool, list[str]]:
                ast_valid = True
                ast_errors: list[str] = []
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    ast_valid = False
                    ast_errors.append(str(e))

                static_passed = True
                static_issues: list[str] = []
                if ast_valid and self.config.static_check_enabled:
                    issues = analyze_code(code, allowed_imports=allowed)
                    if issues:
                        static_passed = False
                        static_issues = [issue.render() for issue in issues]
                return ast_valid, ast_errors, static_passed, static_issues

            # AST parse + static analysis are CPU-bound: offload large blocks
            # so they never block the event loop, but keep tiny blocks inline
            # (a thread hop costs more than parsing a few hundred bytes).
            if len(code) > 10_000:
                (
                    result.ast_valid,
                    result.ast_errors,
                    result.static_passed,
                    result.static_issues,
                ) = await asyncio.to_thread(_python_check)
            else:
                (
                    result.ast_valid,
                    result.ast_errors,
                    result.static_passed,
                    result.static_issues,
                ) = _python_check()

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

        Each stage is dispatched to the best available environment.

        Complexity: ``O(F)`` per stage for F files.
        """
        await self._ensure_capabilities()
        self._local.bwrap_enabled = bool(self._has_bwrap)
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

        env = self._pick_env()
        if self.config.lint_enabled:
            result = await env.lint(result, files)

        if self.config.run_enabled:
            result = await env.run_code(result, files)

        if run_tests and self.config.test_enabled:
            result = await env.run_tests(result, files)

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

    def is_available(self) -> bool:
        self._ensure_capabilities_sync()
        return bool(self._has_docker) and self.config.enabled
