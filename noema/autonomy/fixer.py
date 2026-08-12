"""IncidentFixer — consume an incident, produce a validated fix, open a PR.

The end-to-end flow (T2.1):
1. Normalize a Sentry/webhook payload into an :class:`Incident`.
2. Build a fix ``Task`` and run it through :class:`NoemaEngine`.
3. Gate the fix on a *passing run* (sandbox + tests via ``validate_solution``).
4. Write the fix's files onto a branch and open a pull request.

Both the engine factory and the GitHub client are injectable so tests exercise
the flow hermetically (mock engine + ``httpx.MockTransport``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from noema.autonomy.github import GitHubClient
from noema.autonomy.incidents import (
    Incident,
    incident_branch_name,
    incident_to_task,
    parse_incident,
)
from noema.core.types import CodeBlock, Solution, Task

logger = structlog.get_logger(__name__)

EngineFactory = Callable[[], Awaitable[Any]]


async def _default_engine_factory() -> Any:
    from noema.core.engine import NoemaEngine

    engine = NoemaEngine(llm_provider="fallback")
    await engine.initialize()
    return engine


class IncidentFixer:
    """Orchestrates incident → validated fix → pull request."""

    def __init__(
        self,
        github: GitHubClient,
        engine_factory: EngineFactory = _default_engine_factory,
    ) -> None:
        self.github = github
        self._engine_factory = engine_factory
        self._engine: Any | None = None

    async def _get_engine(self) -> Any:
        if self._engine is None:
            self._engine = await self._engine_factory()
        return self._engine

    async def close(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None:
            shutdown = getattr(engine, "shutdown", None)
            if shutdown is not None:
                try:
                    await shutdown()
                except Exception as e:  # noqa: BLE001 - teardown must not mask
                    logger.warning("incident_fixer_engine_shutdown_failed", error=str(e))
        await self.github.aclose()

    async def handle_incident(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run the full incident→PR loop for one payload.

        Returns:
            ``{"status": "pr_created", "incident_id", "branch", "pr_number",
            "pr_url", "files", "judge_passed"}`` or a failure summary.
        """
        incident = parse_incident(payload)
        logger.info(
            "incident_received",
            incident_id=incident.id,
            title=incident.title,
            source=incident.source,
        )

        engine = await self._get_engine()
        task = Task(
            title=incident_to_task(incident)["title"],
            description=incident_to_task(incident)["description"],
            tags=incident_to_task(incident)["tags"],
        )

        try:
            solution, thought = await engine.think(task)
        except Exception as e:  # noqa: BLE001 - report failure as a result
            logger.error("incident_fix_generation_failed", incident_id=incident.id, error=str(e))
            return {"status": "fix_generation_failed", "incident_id": incident.id, "error": str(e)}

        # Passing-run gate: the fix must validate in the sandbox (with tests)
        # before any branch/PR is created.
        try:
            run = await engine.validate_solution(solution, run_tests=True)
        except Exception as e:  # noqa: BLE001
            logger.error("incident_fix_validation_failed", incident_id=incident.id, error=str(e))
            return {
                "status": "validation_failed",
                "incident_id": incident.id,
                "error": str(e),
                "solution_id": solution.id,
            }
        if not run.all_valid:
            logger.warning(
                "incident_fix_validation_rejected",
                incident_id=incident.id,
                summary=str(run.summary),
            )
            return {
                "status": "validation_failed",
                "incident_id": incident.id,
                "summary": str(run.summary),
                "solution_id": solution.id,
            }

        if not solution.code_blocks:
            return {
                "status": "no_changes",
                "incident_id": incident.id,
                "solution_id": solution.id,
            }

        files = _solution_files(solution)
        branch = incident_branch_name(incident)
        pr_title = f"fix: {incident.title}"
        pr_body = _pr_body(incident, solution)
        try:
            pr = await self.github.submit_fix_pr(
                files=files, title=pr_title, body=pr_body, branch=branch
            )
        except Exception as e:  # noqa: BLE001 - GitHub failures must not crash the loop
            logger.error("incident_fix_pr_failed", incident_id=incident.id, error=str(e))
            return {
                "status": "pr_failed",
                "incident_id": incident.id,
                "error": str(e),
                "solution_id": solution.id,
            }

        logger.info(
            "incident_fix_pr_created",
            incident_id=incident.id,
            pr_number=pr["pr_number"],
            branch=pr["branch"],
        )
        return {
            "status": "pr_created",
            "incident_id": incident.id,
            "branch": pr["branch"],
            "pr_number": pr["pr_number"],
            "pr_url": pr["pr_url"],
            "files": pr["files"],
            "judge_passed": solution.metadata.get("judge_passed"),
        }


def _solution_files(solution: Solution) -> list[tuple[str, str]]:
    """Map the solution's code blocks onto ``(path, content)`` file pairs."""
    files: list[tuple[str, str]] = []
    seen: set[str] = set()
    for block in solution.code_blocks:
        path = _block_path(block)
        if path in seen:
            continue
        seen.add(path)
        files.append((path, block.content))
    return files


def _block_path(block: CodeBlock) -> str:
    filename = block.filename.strip().lstrip("/")
    if filename:
        return filename
    language = (block.language or "py").lower()
    ext = "py" if language in ("python", "py") else "txt"
    return f"fix_{block.description or 'file'}_".replace(" ", "_") + f".{ext}"


def _pr_body(incident: Incident, solution: Solution) -> str:
    parts = [
        f"Automated fix for incident **{incident.id}**: {incident.title}",
        "",
        f"- Incident source: {incident.source}",
        f"- Solution summary: {solution.summary[:500]}",
        f"- Quality: {solution.quality.value} | Confidence: {solution.confidence:.0%}",
    ]
    if incident.stack_trace:
        parts.append("")
        parts.append("### Reproducing stack trace")
        parts.append("```")
        parts.append(incident.stack_trace[:1500])
        parts.append("```")
    parts.append("")
    parts.append("_Generated by Noema autonomy loop. Validated in the sandbox before opening._")
    return "\n".join(parts)


def build_github_client_from_settings() -> GitHubClient:
    """Build a :class:`GitHubClient` from the ``autonomy`` settings block.

    Raises:
        GitHubError: When the token or repository is not configured.
    """
    from noema.autonomy.github import GitHubError
    from noema.config.settings import get_settings

    settings = get_settings().autonomy
    token = settings.github_token.get_secret_value()
    if not token:
        raise GitHubError(
            "No GitHub token configured (NOEMA_AUTONOMY__GITHUB_TOKEN / "
            "settings.yaml autonomy.github_token)"
        )
    if not settings.github_repo:
        raise GitHubError("No GitHub repository configured (autonomy.github_repo)")
    return GitHubClient(
        token=token,
        repo=settings.github_repo,
        base_branch=settings.github_base_branch,
    )


def get_incident_fixer() -> IncidentFixer:
    """Build a default :class:`IncidentFixer` from settings (for the webhook)."""
    return IncidentFixer(github=build_github_client_from_settings())
