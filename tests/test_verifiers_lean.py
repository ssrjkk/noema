"""Lean 4 formal verification bridge — proof obligations in the merge gate.

Covers ``noema/verifiers/lean``:
- module generation from a theorem,
- graceful degradation when the ``lean`` binary is missing,
- fake-runner success/failure verdicts,
- temp-file hygiene,
- merge-gate integration: a configured verifier blocks failed proofs
  (``formal_verification``) and passes proven ones.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from noema.experiments.gate import GateConfig, run_merge_gate
from noema.verifiers.lean import LeanVerifier, make_lean_module

GOOD_THEOREM = "theorem add_comm_obligation (a b : Nat) : a + b = b + a := by simp [Nat.add_comm]"


def _ok_runner(binary: str, path: str, timeout: float):
    async def _run(binary, path, timeout):
        assert (await _read_file(path)).strip() == make_lean_module(GOOD_THEOREM).strip()
        return 0, "no errors", ""

    return _run


async def _read_file(path: str) -> str:
    from pathlib import Path

    return Path(path).read_text(encoding="utf-8")


class TestMakeLeanModule:
    def test_wraps_theorem_with_imports(self):
        module = make_lean_module(GOOD_THEOREM, imports=("Init",))
        assert module.startswith("import Init\n")
        assert "theorem add_comm_obligation" in module
        assert module.endswith("\n")

    def test_no_imports_by_default(self):
        module = make_lean_module(GOOD_THEOREM)
        assert not module.startswith("import")
        assert module == GOOD_THEOREM + "\n"


@pytest.mark.asyncio
class TestLeanVerifier:
    async def test_degrades_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr("noema.verifiers.lean.shutil.which", lambda *a, **k: None)
        verifier = LeanVerifier()
        result = await verifier.verify_lean_source(make_lean_module(GOOD_THEOREM))
        assert result.verified is False
        assert result.error.startswith("lean_not_installed")
        assert result.files_checked == 0

    async def test_success_with_fake_runner(self):
        verifier = LeanVerifier(runner=_ok_runner("lean", "", 0))
        result = await verifier.verify_lean_source(make_lean_module(GOOD_THEOREM))
        assert result.verified is True
        assert result.files_checked == 1
        assert result.error == ""

    async def test_failure_with_fake_runner(self):
        async def _failing(binary, path, timeout):
            return 1, "", "error: unknown identifier 'add_comm_obligation'"

        verifier = LeanVerifier(runner=_failing)
        result = await verifier.verify_lean_source(make_lean_module(GOOD_THEOREM))
        assert result.verified is False
        assert "unknown identifier" in result.error

    async def test_runner_exception_is_a_verdict(self):
        async def _crash(binary, path, timeout):
            raise RuntimeError("toolchain exploded")

        verifier = LeanVerifier(runner=_crash)
        result = await verifier.verify_lean_source(make_lean_module(GOOD_THEOREM))
        assert result.verified is False
        assert "lean_run_failed" in result.error

    async def test_verify_files_skips_non_lean(self):
        verifier = LeanVerifier()
        result = await verifier.verify_files([("app/billing.py", "x = 1\n")])
        assert result.verified is True
        assert result.files_checked == 0

    async def test_verify_files_all_must_prove(self):
        async def _ok(binary, path, timeout):
            return 0, "", ""

        async def _fail(binary, path, timeout):
            return 1, "", "error: proof failed"

        calls = {"n": 0}

        async def _alternating(binary, path, timeout):
            calls["n"] += 1
            return await (_fail if calls["n"] == 1 else _ok)(binary, path, timeout)

        verifier = LeanVerifier(runner=_alternating)
        result = await verifier.verify_files(
            [
                ("specs/one.lean", make_lean_module("theorem one : True := by trivial")),
                ("specs/two.lean", make_lean_module("theorem two : True := by trivial")),
            ]
        )
        assert result.verified is False
        assert result.files_checked == 2
        assert "proof failed" in result.error


@pytest.mark.asyncio
class TestGateFormalVerification:
    @staticmethod
    async def _validate(files, run_tests: bool = False):
        return SimpleNamespace(all_valid=True, summary=f"{len(files)} files valid")

    def _gate(self, verifier, files):
        async def _judge(solution):
            return SimpleNamespace(passed=True, summary="ok", scores=SimpleNamespace(overall=0.9))

        return GateConfig(
            explicit_files=files,
            judge=_judge,
            sandbox=SimpleNamespace(all_valid=True, summary="ok", validate_files=self._validate),
            verifier=verifier,
        )

    async def test_gate_blocks_failed_proof(self):
        async def _failing(binary, path, timeout):
            return 1, "", "error: proof failed"

        cfg = self._gate(
            LeanVerifier(runner=_failing),
            [
                {"path": "app/billing.py", "content": "def total(a, b):\n    return a / b\n"},
                {"path": "specs/billing.lean", "content": make_lean_module(GOOD_THEOREM)},
            ],
        )
        report = await run_merge_gate(cfg)
        assert report.passed is False
        assert "formal_verification" in report.blocked_by
        assert report.formal_verified is False
        assert "proof failed" in report.formal_error

    async def test_gate_passes_proven_spec(self):
        cfg = self._gate(
            LeanVerifier(runner=_ok_runner("lean", "", 0)),
            [
                {"path": "app/billing.py", "content": "def total(a, b):\n    return a / b\n"},
                {"path": "specs/billing.lean", "content": make_lean_module(GOOD_THEOREM)},
            ],
        )
        report = await run_merge_gate(cfg)
        assert report.passed is True
        assert report.formal_verified is True
        assert report.blocked_by == []

    async def test_gate_without_verifier_is_unaffected(self):
        cfg = self._gate(
            None,
            [{"path": "app/billing.py", "content": "def total(a, b):\n    return a / b\n"}],
        )
        report = await run_merge_gate(cfg)
        assert report.passed is True
        assert report.formal_verified is None
