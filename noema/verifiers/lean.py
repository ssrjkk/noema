"""Lean 4 formal verification bridge — theorem prover as a merge-gate stage.

Honest positioning: Noema cannot prove arbitrary imperative code bug-free;
that is a research-level problem. What this module provides is the *seam*:

1. A formal specification lives as a ``.lean`` file next to the code it
   describes. It declares the interface of the implementation as axioms and
   carries the theorem obligations (e.g. ``theorem total_never_div_by_zero``).
2. :class:`LeanVerifier` compiles those obligations with the real Lean 4
   toolchain when it is installed. A failed proof is a failed verification
   (fail-closed: the merge gate blocks).
3. When ``lean`` is absent the verifier degrades gracefully
   (``error="lean_not_installed"``) and never crashes the pipeline. It only
   blocks when the gate explicitly configures a verifier.

The runner is injectable so tests exercise the full flow without the binary.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_TIMEOUT = 30.0
MAX_PARALLEL_COMPILES = 4

LeanRunner = Callable[[str, str, float], Awaitable[tuple[int, str, str]]]


@dataclass
class VerifierResult:
    """Verdict of one formal verification run."""

    verified: bool = False
    files_checked: int = 0
    output: str = ""
    error: str = ""
    duration_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "files_checked": self.files_checked,
            "output": self.output[:2000],
            "error": self.error[:500],
            "duration_ms": round(self.duration_ms, 1),
        }


def make_lean_module(theorem: str, imports: tuple[str, ...] = ()) -> str:
    """Wrap a theorem/proof text into a runnable Lean 4 module.

    ``imports`` default to core Lean (no Mathlib), so a bare ``lean`` binary
    can check the obligation. Example::

        make_lean_module(
            "theorem total_never_negative (a b : Nat) : a + b >= a := by simp",
            imports=("Init",),
        )
    """
    parts = [f"import {m}" for m in imports]
    if imports:
        parts.append("")
    parts.append(theorem.rstrip())
    parts.append("")
    return "\n".join(parts)


async def _default_runner(binary: str, file_path: str, timeout: float) -> tuple[int, str, str]:
    """Run the ``lean`` binary over a module file (async, bounded)."""
    proc = await asyncio.create_subprocess_exec(
        binary,
        file_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        with contextlib.suppress(Exception):  # noqa: BLE001 - reap the child
            await proc.communicate()
        return 1, "", "timeout"
    return (
        proc.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


class LeanVerifier:
    """Compile ``.lean`` proof obligations with the Lean 4 theorem prover.

    Args:
        lean_binary: Path/name of the ``lean`` executable (default ``"lean"``).
        timeout: Per-module compile timeout in seconds.
        runner: Injectable ``(binary, path, timeout) -> (code, out, err)``
            coroutine (tests use a fake; default shells out to ``lean``).
    """

    def __init__(
        self,
        lean_binary: str = "lean",
        timeout: float = DEFAULT_TIMEOUT,
        runner: LeanRunner | None = None,
    ) -> None:
        self.lean_binary = lean_binary
        self.timeout = timeout
        self._runner = runner or _default_runner
        # An injected runner simulates the toolchain; the binary probe then
        # applies to the default runner only.
        self._available: bool | None = True if runner is not None else None

    def available(self) -> bool:
        """Probe the ``lean`` binary once and cache the result."""
        if self._available is None:
            self._available = bool(shutil.which(self.lean_binary))
        return self._available

    async def verify_lean_source(
        self, source: str, file_name: str = "obligation.lean"
    ) -> VerifierResult:
        """Compile one Lean module and report the verdict.

        Returns ``verified=False`` with ``error="lean_not_installed"`` when
        the binary is missing; never raises on toolchain failure.
        """
        t0 = time.monotonic()
        if not self.available():
            return VerifierResult(
                verified=False,
                error=f"lean_not_installed:{self.lean_binary}",
                duration_ms=(time.monotonic() - t0) * 1000,
            )
        tmp_dir = Path(tempfile.mkdtemp(prefix="noema-lean-"))
        try:
            module_path = tmp_dir / file_name
            module_path.write_text(source, encoding="utf-8")
            code, stdout, stderr = await self._runner(
                self.lean_binary, str(module_path), self.timeout
            )
        except Exception as e:  # noqa: BLE001 - toolchain failures are verdicts
            return VerifierResult(
                verified=False,
                error=f"lean_run_failed:{e}",
                duration_ms=(time.monotonic() - t0) * 1000,
            )
        finally:
            _rmtree(tmp_dir)

        output = (stdout + "\n" + stderr).strip()
        return VerifierResult(
            verified=code == 0,
            files_checked=1,
            output=output,
            error="" if code == 0 else output[:500] or f"lean_exit_{code}",
            duration_ms=(time.monotonic() - t0) * 1000,
        )

    async def verify_files(self, files: list[tuple[str, str]]) -> VerifierResult:
        """Compile every ``.lean`` file in ``files``; all must prove.

        Files with other suffixes are ignored (the runtime sandbox covers
        them). Returns ``verified=True`` vacuously when there are no ``.lean``
        files — the gate decides whether a verifier is required at all.
        """
        t0 = time.monotonic()
        lean_files = [(p, c) for p, c in files if p.endswith(".lean")]
        if not lean_files:
            return VerifierResult(verified=True, files_checked=0)

        sem = asyncio.Semaphore(MAX_PARALLEL_COMPILES)

        async def _check(path: str, content: str) -> VerifierResult:
            async with sem:
                return await self.verify_lean_source(
                    content, file_name=Path(path).name or "obligation.lean"
                )

        results = await asyncio.gather(*(_check(path, content) for path, content in lean_files))
        verified = all(r.verified for r in results)
        errors = [
            f"{path}: {r.error}"
            for (path, _), r in zip(lean_files, results, strict=True)
            if not r.verified and r.error
        ]
        outputs = [r.output for r in results if r.output]
        return VerifierResult(
            verified=verified,
            files_checked=len(lean_files),
            output="\n".join(outputs)[:2000],
            error="; ".join(errors)[:500],
            duration_ms=(time.monotonic() - t0) * 1000,
        )


def _rmtree(path: Path) -> None:
    import shutil as _shutil

    _shutil.rmtree(path, ignore_errors=True)
