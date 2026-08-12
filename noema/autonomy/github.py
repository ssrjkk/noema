"""GitHub client — create fix branches, commit files, and open pull requests.

Built on ``httpx`` so the HTTP transport is injectable (tests pass a
``httpx.MockTransport``); no ``PyGithub`` dependency is required.
"""

from __future__ import annotations

import base64
import uuid
from typing import Any

import httpx
import structlog

from noema.errors import NoemaError

logger = structlog.get_logger(__name__)

_API_ROOT = "https://api.github.com"


class GitHubError(NoemaError):
    """Raised when a GitHub API call fails."""


class GitHubClient:
    """Minimal GitHub REST client for the incident→PR flow.

    Args:
        token: GitHub personal access token.
        repo: ``"owner/name"`` repository identifier.
        base_branch: Branch the fix branch is cut from and the PR targets.
        api_root: Override for the API base URL (tests).
        transport: Optional ``httpx`` transport (e.g. ``httpx.MockTransport``).
    """

    def __init__(
        self,
        token: str,
        repo: str,
        base_branch: str = "main",
        api_root: str = _API_ROOT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not token:
            raise ValueError("token is required")
        if "/" not in repo:
            raise ValueError("repo must be 'owner/name'")
        self.repo = repo
        self.base_branch = base_branch
        self._client = httpx.AsyncClient(
            base_url=f"{api_root}/repos/{repo}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "noema-autonomy",
            },
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── Branches ──────────────────────────────────────────────────────

    async def _base_branch_sha(self) -> str:
        resp = await self._client.get(f"/git/ref/heads/{self.base_branch}")
        _raise_for_status(resp, f"resolve base branch {self.base_branch!r}")
        return str(resp.json()["object"]["sha"])

    async def create_branch(self, branch: str) -> str:
        """Create ``branch`` off the configured base branch.

        Returns:
            The base commit SHA the branch points at.
        """
        base_sha = await self._base_branch_sha()
        resp = await self._client.post(
            "/git/refs",
            json={"ref": f"refs/heads/{branch}", "sha": base_sha},
        )
        if resp.status_code == 422:
            logger.info("branch_already_exists", branch=branch)
            return base_sha
        _raise_for_status(resp, f"create branch {branch!r}")
        return base_sha

    # ── Files ─────────────────────────────────────────────────────────

    async def _file_sha(self, branch: str, path: str) -> str | None:
        """Return the blob SHA of ``path`` on ``branch``, or ``None`` if absent."""
        resp = await self._client.get(f"/contents/{path}", params={"ref": branch})
        if resp.status_code == 404:
            return None
        _raise_for_status(resp, f"stat {path!r}")
        data = resp.json()
        if isinstance(data, list):
            return None
        return str(data.get("sha") or "")

    async def write_file(
        self,
        branch: str,
        path: str,
        content: str,
        message: str,
        create_branch_if_missing: bool = True,
    ) -> None:
        """Create or update a single file on ``branch`` via the contents API."""
        if create_branch_if_missing:
            await self.create_branch(branch)
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        payload: dict[str, Any] = {"branch": branch, "message": message, "content": encoded}
        sha = await self._file_sha(branch, path)
        if sha:
            payload["sha"] = sha
        resp = await self._client.put(f"/contents/{path}", json=payload)
        _raise_for_status(resp, f"write {path!r}")
        logger.info("github_file_written", branch=branch, path=path)

    # ── Pull requests ────────────────────────────────────────────────

    async def open_pr(self, branch: str, title: str, body: str) -> dict[str, Any]:
        """Open a pull request from ``branch`` into the base branch.

        Returns:
            A dict with ``number`` and ``url``.
        """
        resp = await self._client.post(
            "/pulls",
            json={"title": title, "head": branch, "base": self.base_branch, "body": body},
        )
        _raise_for_status(resp, f"open PR for {branch!r}")
        data = resp.json()
        return {"number": int(data["number"]), "url": str(data["html_url"])}

    # ── High-level flow ──────────────────────────────────────────────

    async def submit_fix_pr(
        self,
        *,
        files: list[tuple[str, str]],
        title: str,
        body: str,
        branch: str | None = None,
        base_commit_message: str = "fix: noema automated incident fix",
    ) -> dict[str, Any]:
        """Create a fix branch, commit the given files, and open a PR.

        Args:
            files: ``(path, content)`` pairs to write onto the branch.
            title: PR title.
            body: PR body (e.g. reproducing the incident + fix summary).
            branch: Branch name; defaults to a fresh ``noema-fix/<id>`` name.
            base_commit_message: Commit message used for every file write.

        Returns:
            ``{"branch", "pr_number", "pr_url", "files": n}``
        """
        branch = branch or f"noema-fix/{uuid.uuid4().hex[:10]}"
        await self.create_branch(branch)
        for path, content in files:
            await self.write_file(
                branch,
                path,
                content,
                message=base_commit_message,
                create_branch_if_missing=False,
            )
        pr = await self.open_pr(branch, title, body)
        logger.info(
            "github_fix_pr_submitted",
            repo=self.repo,
            branch=branch,
            pr_number=pr["number"],
            files=len(files),
        )
        return {
            "branch": branch,
            "pr_number": pr["number"],
            "pr_url": pr["url"],
            "files": len(files),
        }


def _raise_for_status(resp: httpx.Response, what: str) -> None:
    if resp.status_code < 300:
        return
    detail = ""
    try:
        detail = str(resp.json().get("message", ""))
    except Exception:
        detail = resp.text[:200]
    raise GitHubError(f"GitHub {what} failed: {resp.status_code} {detail}".strip())
