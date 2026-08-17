"""Lean 4 formal verification bridge — proof obligations in the merge gate.

Covers ``noema/verifiers/lean``:
- module generation from a theorem,
- graceful degradation when the ``lean`` binary is missing,
- fake-runner success/failure verdicts,
- bounded-parallel multi-file verification,
- temp-file hygiene,
- merge-gate integration: a configured verifier blocks failed proofs
  (``formal_verification``), verifies spec-only PRs, and enforces
  ``require_spec_patterns`` (anti-vacuum),
- fail-closed fixer wiring: ``autonomy.lean_verifier`` blocks the gate when
  the toolchain is absent.
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

    async def test_verify_files_parallel_many(self):
        async def _ok(binary, path, timeout):
            return 0, "", ""

        verifier = LeanVerifier(runner=_ok)
        result = await verifier.verify_files(
            [
                (
                    f"specs/obligation_{i}.lean",
                    make_lean_module(f"theorem t{i} : True := by trivial"),
                )
                for i in range(6)
            ]
        )
        assert result.verified is True
        assert result.files_checked == 6
        assert result.error == ""


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

    async def test_gate_verifies_lean_only_pr(self):
        """Spec-only PRs must still be theorem-checked (no early skip)."""
        cfg = self._gate(
            LeanVerifier(runner=_ok_runner("lean", "", 0)),
            [{"path": "specs/billing.lean", "content": make_lean_module(GOOD_THEOREM)}],
        )
        report = await run_merge_gate(cfg)
        assert report.passed is True
        assert report.formal_verified is True
        assert report.changed_files == 0
        assert report.note == "no_changed_files"

    async def test_gate_blocks_missing_required_spec(self):
        cfg = self._gate(
            LeanVerifier(runner=_ok_runner("lean", "", 0)),
            [{"path": "app/billing.py", "content": "def total(a, b):\n    return a / b\n"}],
        )
        cfg.require_spec_patterns = ("app/",)
        report = await run_merge_gate(cfg)
        assert report.passed is False
        assert "missing_formal_spec:app/billing.py" in report.blocked_by
        assert "formal_verification" not in report.blocked_by

    async def test_gate_passes_when_required_spec_present(self):
        cfg = self._gate(
            LeanVerifier(runner=_ok_runner("lean", "", 0)),
            [
                {"path": "app/billing.py", "content": "def total(a, b):\n    return a / b\n"},
                {"path": "specs/app/billing.lean", "content": make_lean_module(GOOD_THEOREM)},
            ],
        )
        cfg.require_spec_patterns = ("app/",)
        report = await run_merge_gate(cfg)
        assert report.passed is True
        assert report.blocked_by == []
        assert report.formal_verified is True

    async def test_require_specs_not_enforced_without_verifier(self):
        cfg = self._gate(
            None,
            [{"path": "app/billing.py", "content": "def total(a, b):\n    return a / b\n"}],
        )
        cfg.require_spec_patterns = ("app/",)
        report = await run_merge_gate(cfg)
        assert report.passed is True
        assert report.blocked_by == []

    async def test_gate_blocks_lean_only_pr_when_toolchain_missing(self, monkeypatch):
        monkeypatch.setattr("noema.verifiers.lean.shutil.which", lambda *a, **k: None)
        cfg = self._gate(
            LeanVerifier(),
            [{"path": "specs/billing.lean", "content": make_lean_module(GOOD_THEOREM)}],
        )
        report = await run_merge_gate(cfg)
        assert report.passed is False
        assert "formal_verification" in report.blocked_by
        assert "lean_not_installed" in report.formal_error


@pytest.mark.asyncio
class TestFixerGateIntegration:
    async def test_fixer_gate_verifier_construction_failure_is_fail_closed(self, monkeypatch):
        """A verifier construction failure must propagate (never skip the stage)."""
        from noema.autonomy.fixer import _default_gate_runner
        from noema.config.settings import reset_settings

        def _boom(*args, **kwargs):
            raise RuntimeError("verifier module broken")

        monkeypatch.setenv("NOEMA_AUTONOMY__LEAN_VERIFIER", "true")
        monkeypatch.setattr("noema.verifiers.lean.LeanVerifier", _boom)
        reset_settings()
        try:
            with pytest.raises(RuntimeError, match="verifier module broken"):
                await _default_gate_runner([("specs/billing.lean", make_lean_module(GOOD_THEOREM))])
        finally:
            reset_settings()

    async def test_fixer_gate_fail_closed_without_toolchain(self, monkeypatch):
        """lean_verifier=true + missing binary must BLOCK, not skip."""
        from noema.autonomy.fixer import _default_gate_runner
        from noema.config.settings import reset_settings

        monkeypatch.setenv("NOEMA_AUTONOMY__LEAN_VERIFIER", "true")
        monkeypatch.setattr("noema.verifiers.lean.shutil.which", lambda *a, **k: None)
        reset_settings()
        try:
            report = await _default_gate_runner(
                [("specs/billing.lean", make_lean_module(GOOD_THEOREM))]
            )
        finally:
            reset_settings()
        assert report.passed is False
        assert "formal_verification" in report.blocked_by
        assert report.formal_verified is False

    async def test_fixer_gate_fail_closed_on_required_path_without_spec(self, monkeypatch):
        """lean_verifier=true + protected path without spec must BLOCK."""
        from noema.autonomy.fixer import _default_gate_runner
        from noema.config.settings import reset_settings

        async def _judge(solution):
            return SimpleNamespace(passed=True, summary="ok", scores=SimpleNamespace(overall=0.9))

        class _FakeSandbox:
            def __init__(self, *args, **kwargs):
                pass

            async def validate_files(self, files, run_tests: bool = False):
                return SimpleNamespace(all_valid=True, summary="ok")

        monkeypatch.setenv("NOEMA_AUTONOMY__LEAN_VERIFIER", "true")
        monkeypatch.setenv("NOEMA_AUTONOMY__LEAN_VERIFIER_REQUIRED_PATHS", '["crypto/"]')
        monkeypatch.setattr("noema.verifiers.lean.shutil.which", lambda *a, **k: None)
        monkeypatch.setattr("noema.experiments.gate._default_judge", lambda: _judge)
        monkeypatch.setattr("noema.experiments.gate.SandboxEngine", _FakeSandbox)
        reset_settings()
        try:
            report = await _default_gate_runner(
                [("crypto/sign.py", "def sign(data):\n    return data\n")]
            )
        finally:
            reset_settings()
        assert report.passed is False
        assert "missing_formal_spec:crypto/sign.py" in report.blocked_by
