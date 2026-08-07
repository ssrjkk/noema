"""Git-based self-evolution — PR workflow with test verification."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from noema.config.settings import get_settings
from noema.logging import get_logger

log = get_logger(__name__)


@dataclass
class EvolutionProposal:
    """A proposed code change that goes through review."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    branch: str = ""
    description: str = ""
    file_path: str = ""
    original_code: str = ""
    proposed_code: str = ""
    rationale: str = ""
    confidence: float = 0.5
    status: str = "pending"  # pending, testing, passed, failed, applied, rejected
    tests_output: str = ""
    tests_passed: bool = False
    created_at: float = field(default_factory=time.time)


class GitEvolution:
    """Self-evolution through git branches and PRs.

    Workflow:
    1. LLM analyzes code and proposes changes
    2. Changes are applied to a feature branch
    3. Tests are run automatically
    4. If tests pass → creates a summary for human review
    5. Human reviews → merge or reject
    """

    def __init__(
        self,
        project_root: str = ".",
        auto_apply: bool = False,
        test_before_apply: bool = True,
    ) -> None:
        settings = get_settings()
        self.project_root = Path(project_root)
        self.auto_apply = auto_apply or settings.evolution_auto_apply
        self.test_before_apply = test_before_apply or settings.evolution_test_before_apply
        self._proposals: list[EvolutionProposal] = []

    async def _run_git(self, *args: str) -> tuple[int, str, str]:
        """Run a git command and return (returncode, stdout, stderr)."""
        cmd = ["git", "-C", str(self.project_root), *args]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return (
            proc.returncode or 0,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )

    async def is_git_repo(self) -> bool:
        """Check if project root is a git repository."""
        code, _, _ = await self._run_git("rev-parse", "--is-inside-work-tree")
        return code == 0

    async def init_repo(self) -> bool:
        """Initialize a git repo if not already one."""
        if await self.is_git_repo():
            return True
        code, _, err = await self._run_git("init")
        if code == 0:
            log.info("git_repo_initialized", path=str(self.project_root))
        else:
            log.error("git_init_failed", error=err)
        return code == 0

    async def create_branch(self, branch_name: str | None = None) -> str:
        """Create and checkout a new branch."""
        branch = branch_name or f"evolution/{uuid.uuid4().hex[:8]}"
        code, _, err = await self._run_git("checkout", "-b", branch)
        if code != 0:
            log.error("git_branch_failed", branch=branch, error=err)
        else:
            log.info("git_branch_created", branch=branch)
        return branch

    async def apply_and_commit(
        self,
        file_path: str,
        new_content: str,
        message: str,
        branch: str | None = None,
    ) -> EvolutionProposal:
        """Apply a code change, commit, and optionally run tests.

        Returns an EvolutionProposal with status.
        """
        proposal = EvolutionProposal(
            branch=branch or "",
            description=message,
            file_path=file_path,
            proposed_code=new_content,
        )

        # Ensure git repo
        if not await self.is_git_repo():
            await self.init_repo()

        # Create branch
        if not proposal.branch:
            proposal.branch = await self.create_branch()

        # Read original
        target = self.project_root / file_path
        if target.is_file():
            proposal.original_code = target.read_text(encoding="utf-8")

        # Write new content
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_content, encoding="utf-8")

        # Stage
        await self._run_git("add", file_path)

        # Commit
        code, _, err = await self._run_git(
            "commit", "-m", f"[noema-evolution] {message}\n\nProposal ID: {proposal.id}"
        )
        if code != 0:
            proposal.status = "failed"
            proposal.tests_output = err
            self._proposals.append(proposal)
            return proposal

        # Run tests if configured
        if self.test_before_apply:
            proposal.status = "testing"
            tests_ok, output = await self._run_tests()
            proposal.tests_output = output
            if tests_ok:
                proposal.status = "passed"
                proposal.tests_passed = True
                log.info("evolution_tests_passed", proposal_id=proposal.id, file=file_path)
            else:
                proposal.status = "failed"
                proposal.tests_passed = False
                log.warning("evolution_tests_failed", proposal_id=proposal.id, file=file_path)
                # Revert on test failure
                await self._run_git("checkout", "--", file_path)
        else:
            proposal.status = "passed"

        self._proposals.append(proposal)
        return proposal

    async def revert_proposal(self, proposal: EvolutionProposal) -> bool:
        """Revert a proposal's changes."""
        if proposal.original_code:
            target = self.project_root / proposal.file_path
            target.write_text(proposal.original_code, encoding="utf-8")
            await self._run_git("add", proposal.file_path)
            await self._run_git("commit", "-m", f"[noema-evolution] Revert proposal {proposal.id}")
            proposal.status = "rejected"
            log.info("evolution_reverted", proposal_id=proposal.id)
            return True
        return False

    async def _run_tests(self) -> tuple[bool, str]:
        """Run the project's test suite."""
        # Try pytest first
        proc = await asyncio.create_subprocess_exec(
            "python",
            "-m",
            "pytest",
            "tests/",
            "-x",
            "-q",
            "--tb=short",
            "--timeout=60",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.project_root),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
            return proc.returncode == 0, output
        except TimeoutError:
            proc.kill()
            return False, "Tests timed out after 120s"

    async def get_log(self, limit: int = 20) -> list[dict[str, str]]:
        """Get recent evolution commits."""
        code, output, _ = await self._run_git(
            "log", "--oneline", f"-{limit}", "--grep=noema-evolution"
        )
        if code != 0:
            return []
        commits = []
        for line in output.strip().split("\n"):
            if line.strip():
                parts = line.split(" ", 1)
                commits.append(
                    {
                        "hash": parts[0] if len(parts) > 0 else "",
                        "message": parts[1] if len(parts) > 1 else "",
                    }
                )
        return commits

    @property
    def proposals(self) -> list[EvolutionProposal]:
        return list(self._proposals)

    def stats(self) -> dict[str, Any]:
        passed = sum(1 for p in self._proposals if p.status == "passed")
        failed = sum(1 for p in self._proposals if p.status == "failed")
        pending = sum(1 for p in self._proposals if p.status in ("pending", "testing"))
        return {
            "total_proposals": len(self._proposals),
            "passed": passed,
            "failed": failed,
            "pending": pending,
            "auto_apply": self.auto_apply,
            "test_before_apply": self.test_before_apply,
        }
