"""Pytest plugin: chaos mode for fail-closed verification.

``--chaos`` mutates the process environment for tests marked ``chaos``: it
wipes every ``NOEMA_*`` and proxy variable and injects a poison override
(``NOEMA_API__HOST=chaos-broken-host``), then restores the original
environment afterwards. Tests marked ``chaos`` must be written so their
assertions hold in both modes: with ``--chaos`` they verify the framework
fails closed under hostile configuration, without it they are ordinary
deterministic tests.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--chaos",
        action="store_true",
        default=False,
        help="mutate the environment for chaos-marked tests (fail-closed verification)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "chaos: fail-closed verification under hostile environment mutation (--chaos)",
    )


@pytest.fixture
def chaotic_env(request: pytest.FixtureRequest) -> Iterator[None]:
    """Wipe NOEMA_*/proxy vars and poison one override; restore afterwards."""
    saved = dict(os.environ)
    try:
        if request.config.getoption("--chaos"):
            for key in [k for k in os.environ if k.startswith("NOEMA_") or "PROXY" in k.upper()]:
                os.environ.pop(key, None)
            os.environ["NOEMA_API__HOST"] = "chaos-broken-host"
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
