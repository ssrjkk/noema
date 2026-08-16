"""Formal verification bridges — theorem provers as an optional pipeline stage.

The only implemented backend is Lean 4 (:mod:`noema.verifiers.lean`): the
verifier compiles proof obligations (``.lean`` files) with the real theorem
prover and reports pass/fail. It is a *seam*: when the ``lean`` binary is not
installed the verifier degrades gracefully (``lean_not_installed``) and never
crashes the pipeline.

Future backends slot in here with the same :class:`VerifierProtocol` shape
(e.g. Coq, or a bounded model checker for hardware specs).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VerifierProtocol(Protocol):
    """A formal verifier that can be wired into the merge gate."""

    async def verify_files(self, files: list[tuple[str, str]]) -> Any:
        """Verify ``(path, content)`` pairs; return a ``VerifierResult``."""
        ...


__all__ = ["VerifierProtocol", "LeanVerifier", "VerifierResult"]

from noema.verifiers.lean import LeanVerifier, VerifierResult  # noqa: E402
