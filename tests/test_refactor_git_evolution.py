"""Deep-path coverage for GitEvolution using a real temporary git repo."""

import pytest

from noema.evolution.git_evolution import EvolutionProposal, GitEvolution


def _fake_tests(ok: bool, output: str = "ok"):
    async def _run() -> tuple[bool, str]:
        return ok, output

    return _run


@pytest.fixture
def git_identity(monkeypatch):
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Evolution Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "evo@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Evolution Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "evo@example.com")


@pytest.mark.asyncio
async def test_is_git_repo_false_for_plain_dir(tmp_path):
    ge = GitEvolution(project_root=str(tmp_path))
    assert await ge.is_git_repo() is False


@pytest.mark.asyncio
async def test_init_repo_on_fresh_dir(tmp_path):
    ge = GitEvolution(project_root=str(tmp_path))
    assert await ge.is_git_repo() is False
    assert await ge.init_repo() is True
    assert await ge.is_git_repo() is True


@pytest.mark.asyncio
async def test_apply_and_commit_green(tmp_path, monkeypatch, git_identity):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    ge = GitEvolution(project_root=str(tmp_path))
    monkeypatch.setattr(ge, "_run_tests", _fake_tests(True))

    proposal = await ge.apply_and_commit("app.py", "x = 2\n", "improve", branch="feature/improve")

    assert isinstance(proposal, EvolutionProposal)
    assert proposal.status == "passed"
    assert proposal.tests_passed is True
    assert proposal.branch == "feature/improve"
    assert proposal.original_code == "x = 1\n"
    assert proposal.proposed_code == "x = 2\n"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "x = 2\n"
    assert len(ge.proposals) == 1
    assert ge.stats()["passed"] == 1


@pytest.mark.asyncio
async def test_apply_and_commit_auto_branch(tmp_path, monkeypatch, git_identity):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    ge = GitEvolution(project_root=str(tmp_path))
    monkeypatch.setattr(ge, "_run_tests", _fake_tests(True))

    proposal = await ge.apply_and_commit("app.py", "x = 2\n", "improve")

    assert proposal.status == "passed"
    assert proposal.branch.startswith("evolution/")


@pytest.mark.asyncio
async def test_apply_and_commit_red_test_fails_proposal(tmp_path, monkeypatch, git_identity):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    ge = GitEvolution(project_root=str(tmp_path))
    monkeypatch.setattr(ge, "_run_tests", _fake_tests(False, "FAILED"))

    proposal = await ge.apply_and_commit("app.py", "x = 2\n", "improve")

    assert proposal.status == "failed"
    assert proposal.tests_passed is False
    assert proposal.tests_output == "FAILED"
    assert ge.stats()["failed"] == 1


@pytest.mark.asyncio
async def test_apply_and_commit_new_file(tmp_path, monkeypatch, git_identity):
    ge = GitEvolution(project_root=str(tmp_path))
    monkeypatch.setattr(ge, "_run_tests", _fake_tests(True))

    proposal = await ge.apply_and_commit("sub/app.py", "y = 9\n", "add file")

    assert proposal.status == "passed"
    assert proposal.original_code == ""
    assert (tmp_path / "sub" / "app.py").read_text(encoding="utf-8") == "y = 9\n"


@pytest.mark.asyncio
async def test_get_log_empty_without_evolution_commits(tmp_path):
    ge = GitEvolution(project_root=str(tmp_path))
    assert await ge.get_log() == []


@pytest.mark.asyncio
async def test_get_log_lists_evolution_commits(tmp_path, monkeypatch, git_identity):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    ge = GitEvolution(project_root=str(tmp_path))
    monkeypatch.setattr(ge, "_run_tests", _fake_tests(True))

    await ge.apply_and_commit("app.py", "x = 2\n", "first change")
    await ge.apply_and_commit("app.py", "x = 3\n", "second change")

    commits = await ge.get_log()
    assert len(commits) == 2
    assert all(commit["hash"] for commit in commits)
    assert commits[0]["message"].startswith("[noema-evolution] second change")


@pytest.mark.asyncio
async def test_revert_proposal_restores_original(tmp_path, monkeypatch, git_identity):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    ge = GitEvolution(project_root=str(tmp_path))
    monkeypatch.setattr(ge, "_run_tests", _fake_tests(True))

    proposal = await ge.apply_and_commit("app.py", "x = 2\n", "improve")
    assert proposal.status == "passed"

    assert await ge.revert_proposal(proposal) is True
    assert proposal.status == "rejected"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "x = 1\n"
    assert len(await ge.get_log()) == 2
    assert ge.stats()["total_proposals"] == 1


@pytest.mark.asyncio
async def test_revert_proposal_without_original_returns_false(tmp_path, git_identity):
    ge = GitEvolution(project_root=str(tmp_path))
    proposal = EvolutionProposal(file_path="app.py", original_code="")
    assert await ge.revert_proposal(proposal) is False
    assert proposal.status == "pending"


@pytest.mark.asyncio
async def test_stats_counts_statuses(tmp_path, monkeypatch, git_identity):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    ge = GitEvolution(project_root=str(tmp_path))
    monkeypatch.setattr(ge, "_run_tests", _fake_tests(True))

    await ge.apply_and_commit("app.py", "x = 2\n", "good change")
    monkeypatch.setattr(ge, "_run_tests", _fake_tests(False, "boom"))
    await ge.apply_and_commit("app.py", "x = 3\n", "bad change")

    stats = ge.stats()
    assert stats["total_proposals"] == 2
    assert stats["passed"] == 1
    assert stats["failed"] == 1
    assert stats["pending"] == 0
    assert stats["auto_apply"] is False
    assert stats["test_before_apply"] is True
