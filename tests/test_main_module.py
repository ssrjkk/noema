"""Coverage for the ``python -m noema`` module entry point (noema/__main__.py).

The CLI emits deterministic UTF-8 regardless of the local console code page,
so subprocesses must be decoded explicitly as UTF-8 — on a cp1251 Windows
console the default locale codec would fail to decode the output.
"""

from __future__ import annotations

import subprocess
import sys


def _run_help() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "noema", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )


def test_python_m_noema_help() -> None:
    """``python -m noema --help`` must surface the CLI without an environment."""
    proc = _run_help()
    assert proc.returncode == 0
    assert "Usage: python -m noema" in proc.stdout
    assert "COMMAND" in proc.stdout


def test_python_m_noema_commands_listed() -> None:
    """The CLI help must advertise the core command groups."""
    proc = _run_help()
    assert proc.returncode == 0
    lowered = proc.stdout.lower()
    for command in ("think", "serve", "pipeline", "knowledge", "grid", "arq"):
        assert command in lowered
