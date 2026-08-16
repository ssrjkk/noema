"""IncidentFixer — consume an incident, produce a validated fix, open a PR.

The end-to-end flow (T2.1):
1. Normalize a Sentry/webhook payload into an :class:`Incident`.
2. Build a fix ``Task`` and run it through :class:`NoemaEngine`.
3. Gate the fix on a *passing run* (sandbox + tests via ``validate_solution``).
4. Write the fix's files onto a branch and open a pull request.
5. Attribute the generation cost onto the changed modules
   (``noema_pr_cost_usd`` / ``noema_code_cost_per_module`` metrics).
6. When ``autonomy.auto_approve`` is set, run the merge gate over the fix
   files and — if it passes — approve the PR; with ``auto_merge`` the PR is
   then merged (Evolution Auto-Apply). Gate failures are fail-closed: no
   auto-approve, no merge.

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
GateRunner = Callable[[list[tuple[str, str]]], Awaitable[Any]]


async def _default_engine_factory() -> Any:
    from noema.core.engine import NoemaEngine

    engine = NoemaEngine(llm_provider="fallback")
    await engine.initialize()
    return engine


async def _default_gate_runner(files: list[tuple[str, str]]) -> Any:
    """Run the merge gate over the fix's files (sandbox static pass + judge).

    Uses ``explicit_files`` so no local git diff is needed; the judge threshold
    is 0.0 (the sandbox is the hard gate), and any judge/LLM failure surfaces
    as a raised error which the fixer treats as *blocked* (fail-closed).

    When ``autonomy.lean_verifier`` is enabled the gate additionally compiles
    every ``.lean`` proof obligation with the Lean 4 theorem prover (if the
    binary is present) and blocks on a failed proof.
    """
    from noema.experiments.gate import GateConfig, run_merge_gate

    cfg = GateConfig(
        judge_threshold=0.0,
        sandbox_enabled=True,
        sandbox_run=False,
        run_tests=False,
        explicit_files=[{"path": path, "content": content} for path, content in files],
    )
    from noema.config.settings import get_settings

    if get_settings().autonomy.lean_verifier:
        try:
            from noema.verifiers.lean import LeanVerifier

            verifier = LeanVerifier()
            if verifier.available():
                cfg.verifier = verifier
        except Exception as e:  # noqa: BLE001 - never break the gate over the prover
            logger.warning("lean_verifier_unavailable", error=str(e))
    return await run_merge_gate(cfg)


def _default_cost_tracker() -> Any:
    from noema.billing.cost_tracker import CostTracker

    return CostTracker()


class IncidentFixer:
    """Orchestrates incident → validated fix → pull request → merge gate."""

    def __init__(
        self,
        github: GitHubClient,
        engine_factory: EngineFactory | None = None,
        gate_runner: GateRunner | None = None,
        cost_tracker: Any | None = None,
        auto_approve: bool | None = None,
        auto_merge: bool | None = None,
    ) -> None:
        self.github = github
        self._engine_factory = engine_factory or _default_engine_factory
        self._gate_runner = gate_runner or _default_gate_runner
        self._cost_tracker = cost_tracker
        from noema.config.settings import get_settings

        settings = get_settings().autonomy
        self._auto_approve = settings.auto_approve if auto_approve is None else auto_approve
        self._auto_merge = settings.auto_merge if auto_merge is None else auto_merge
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

        result: dict[str, Any] = {
            "status": "pr_created",
            "incident_id": incident.id,
            "branch": pr["branch"],
            "pr_number": pr["pr_number"],
            "pr_url": pr["pr_url"],
            "files": pr["files"],
            "judge_passed": solution.metadata.get("judge_passed"),
        }

        # Cost-per-line: attribute the task's token spend onto the changed modules.
        await self._attribute_pr_cost(solution.task_id, pr["pr_number"], files)

        # Merge gate → auto-approve / auto-merge (fail-closed).
        if self._auto_approve:
            gate_report = await self._run_merge_gate(incident, solution, files, pr, result)
            if gate_report is not None:
                result["merge_gate_passed"] = bool(gate_report.passed)
                result["merge_gate_blocked_by"] = list(gate_report.blocked_by)

        logger.info(
            "incident_fix_pr_created",
            incident_id=incident.id,
            pr_number=pr["pr_number"],
            branch=pr["branch"],
            merge_gate_passed=result.get("merge_gate_passed"),
        )
        return result

    async def _attribute_pr_cost(
        self, task_id: str, pr_number: int, files: list[tuple[str, str]]
    ) -> None:
        """Record per-module cost of the PR into the tracker + Prometheus."""
        try:
            tracker = self._cost_tracker or _default_cost_tracker()
            self._cost_tracker = tracker
            attributed = tracker.attribute_pr_cost(
                repo=self.github.repo,
                pr_number=pr_number,
                task_id=task_id,
                files=files,
                model=getattr(getattr(self._engine, "llm", None), "model_name", ""),
            )
            if attributed:
                logger.info(
                    "incident_pr_cost_attributed",
                    pr_number=pr_number,
                    task_id=task_id,
                    total_usd=round(sum(f.cost_usd for f in attributed), 6),
                )
        except Exception as e:  # noqa: BLE001 - billing must never break the loop
            logger.warning("incident_pr_cost_attribution_failed", error=str(e))

    async def _run_merge_gate(
        self,
        incident: Incident,
        solution: Solution,
        files: list[tuple[str, str]],
        pr: dict[str, Any],
        result: dict[str, Any],
    ) -> Any | None:
        """Gate the PR files; on pass approve (and merge when enabled).

        Fail-closed: any gate error or a failing report prevents approval
        and merge; the PR itself stays open for human review.
        """
        try:
            report = await self._gate_runner(files)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "incident_merge_gate_failed",
                incident_id=incident.id,
                pr_number=pr["pr_number"],
                error=str(e),
            )
            result["merge_gate_error"] = str(e)
            return None
        if not getattr(report, "passed", False):
            logger.warning(
                "incident_merge_gate_blocked",
                incident_id=incident.id,
                pr_number=pr["pr_number"],
                blocked_by=list(getattr(report, "blocked_by", [])),
            )
            return report
        try:
            await self.github.approve_pr(
                pr["pr_number"],
                body=(
                    "Approved by Noema merge gate — sandbox validation passed "
                    f"for incident {incident.id}."
                ),
            )
            result["merge_gate_approved"] = True
            if self._auto_merge:
                merged = await self.github.merge_pr(pr["pr_number"])
                result["merged"] = bool(merged.get("merged", True))
                result["merge_sha"] = merged.get("sha", "")
        except Exception as e:  # noqa: BLE001
            logger.error(
                "incident_auto_apply_failed",
                incident_id=incident.id,
                pr_number=pr["pr_number"],
                error=str(e),
            )
            result["merge_gate_error"] = str(e)
        return report


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
