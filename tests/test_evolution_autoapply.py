"""T2.3 tests: evolution auto-apply honors the landing policy.

The guarantee under test: **no mutation is auto-applied without a green test run.**
The tests use a real temporary project, a real ``pytest`` subprocess (the engine's
own default test runner) and a deterministic patch generator — no mocks or stubs.
"""

import pytest

from noema.evolution.engine import EvolutionEngine, EvolutionPatch
from noema.llm.providers import FallbackProvider

APP_ORIGINAL = """\
def add(a, b):
    \"\"\"Add two numbers.\"\"\"
    try:
        return a + b
    except Exception:
        return 0
"""

APP_PATCHED_GREEN = """\
def add(a, b):
    \"\"\"Add two numbers (improved).\"\"\"
    try:
        return a + b
    except Exception:
        return 0
"""

APP_PATCHED_RED = """\
def add(a, b):
    \"\"\"Add two numbers.\"\"\"
    try:
        return a + b + 1
    except Exception:
        return 0
"""

TEST_APP = """\
from app import add


def test_add():
    assert add(1, 2) == 3
"""


def _make_project(tmp_path):
    (tmp_path / "app.py").write_text(APP_ORIGINAL, encoding="utf-8")
    (tmp_path / "test_app.py").write_text(TEST_APP, encoding="utf-8")
    return tmp_path


def _patch_for(patched_code: str, description: str):
    async def _generate(file: str, issue: dict[str, str]) -> EvolutionPatch:
        return EvolutionPatch(
            target="app.py",
            description=description,
            original_code=APP_ORIGINAL,
            patched_code=patched_code,
            rationale="deterministic test patch",
            confidence=1.0,
        )

    return _generate


@pytest.mark.asyncio
async def test_evolution_disabled_applies_nothing(tmp_path):
    project = _make_project(tmp_path)
    engine = EvolutionEngine(
        llm_provider=FallbackProvider(),
        project_root=str(project),
        enabled=False,
    )
    result = await engine.run_evolution_cycle()
    assert result.patches_generated == 0
    assert result.patches_applied == 0
    assert "disabled" in result.summary
    assert (project / "app.py").read_text(encoding="utf-8") == APP_ORIGINAL


@pytest.mark.asyncio
async def test_mutation_does_not_land_without_green_run(tmp_path):
    """A red test run on the patched code reverts the file and rejects the patch."""
    project = _make_project(tmp_path)
    engine = EvolutionEngine(
        llm_provider=FallbackProvider(),
        project_root=str(project),
        enabled=True,
        test_before_apply=True,
        auto_apply=True,
    )
    result = await engine.run_evolution_cycle(
        patch_generator=_patch_for(APP_PATCHED_RED, "break add")
    )

    assert result.patches_applied == 0
    assert result.patches_rejected == 1
    assert len(engine.patches) == 1
    patch = engine.patches[0]
    assert patch.applied is False
    assert patch.tests_passed is False
    assert "failed" in patch.tests_output or "1 failed" in patch.tests_output
    assert (project / "app.py").read_text(encoding="utf-8") == APP_ORIGINAL


@pytest.mark.asyncio
async def test_mutation_lands_on_green_run(tmp_path):
    """A green test run on the patched code lands the mutation."""
    project = _make_project(tmp_path)
    engine = EvolutionEngine(
        llm_provider=FallbackProvider(),
        project_root=str(project),
        enabled=True,
        test_before_apply=True,
        auto_apply=True,
    )
    result = await engine.run_evolution_cycle(
        patch_generator=_patch_for(APP_PATCHED_GREEN, "improve add")
    )

    assert result.patches_applied == 1
    assert result.patches_rejected == 0
    assert engine.patches[0].applied is True
    assert engine.patches[0].tests_passed is True
    assert (project / "app.py").read_text(encoding="utf-8") == APP_PATCHED_GREEN


@pytest.mark.asyncio
async def test_without_auto_apply_patch_is_proposed_not_landed(tmp_path):
    """Without auto_apply the worktree is untouched and the patch is only proposed."""
    project = _make_project(tmp_path)
    engine = EvolutionEngine(
        llm_provider=FallbackProvider(),
        project_root=str(project),
        enabled=True,
        test_before_apply=True,
        auto_apply=False,
    )
    result = await engine.run_evolution_cycle(
        patch_generator=_patch_for(APP_PATCHED_GREEN, "improve add")
    )

    assert result.patches_applied == 0
    assert result.patches_proposed == 1
    assert engine.patches[0].applied is False
    assert engine.patches[0].tests_passed is True
    assert (project / "app.py").read_text(encoding="utf-8") == APP_ORIGINAL


@pytest.mark.asyncio
async def test_test_before_apply_false_lands_without_verification(tmp_path):
    """Explicit opt-out: with test_before_apply=False the patch lands unverified."""
    project = _make_project(tmp_path)
    engine = EvolutionEngine(
        llm_provider=FallbackProvider(),
        project_root=str(project),
        enabled=True,
        test_before_apply=False,
        auto_apply=True,
    )
    result = await engine.run_evolution_cycle(
        patch_generator=_patch_for(APP_PATCHED_GREEN, "improve add")
    )

    assert result.patches_applied == 1
    assert engine.patches[0].applied is True
    assert engine.patches[0].tests_passed is False
    assert (project / "app.py").read_text(encoding="utf-8") == APP_PATCHED_GREEN
