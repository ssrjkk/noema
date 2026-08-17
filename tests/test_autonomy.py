"""T2.1 tests: incident → validated fix → pull request loop.

Covers ``noema/autonomy``:
- incident normalization (Sentry envelope, generic webhook, own schema),
- GitHub client REST calls (branch refs, contents, pulls) via a mock transport,
- the IncidentFixer end-to-end flow with a mocked engine + mocked GitHub,
- the passing-run gate (no PR when the sandbox run is not all-valid).
"""

from unittest.mock import AsyncMock

import httpx
import pytest

from noema.autonomy.fixer import IncidentFixer, _block_path, _solution_files
from noema.autonomy.github import GitHubClient, GitHubError
from noema.autonomy.incidents import incident_branch_name, incident_to_task, parse_incident
from noema.core.types import CodeBlock, Solution

# ── Incident parsing ───────────────────────────────────────────────────

SENTRY_PAYLOAD = {
    "event": {
        "id": "abc123",
        "title": "ZeroDivisionError: division by zero",
        "message": "div by zero in checkout.total()",
        "culprit": "app/billing.py in total",
        "environment": "production",
        "exception": {
            "values": [
                {
                    "type": "ZeroDivisionError",
                    "value": "division by zero",
                    "stacktrace": {
                        "frames": [
                            {"filename": "app/billing.py", "function": "total", "lineno": 42}
                        ]
                    },
                }
            ]
        },
    },
    "source": "sentry",
}


def test_parse_sentry_incident():
    incident = parse_incident(SENTRY_PAYLOAD)
    assert incident.id == "abc123"
    assert incident.source == "sentry"
    assert "ZeroDivisionError" in incident.title
    assert "app/billing.py" in incident.stack_trace
    assert "production" in incident.description


def test_parse_generic_webhook():
    incident = parse_incident(
        {"incident": {"id": "i1", "title": "Timeout in API", "description": "p99 > 2s"}}
    )
    assert incident.id == "i1"
    assert incident.title == "Timeout in API"
    assert incident.source == "webhook"


def test_parse_own_schema_payload():
    incident = parse_incident(
        {"payload": {"title": "boom", "description": "desc", "stack_trace": "trace"}}
    )
    assert incident.title == "boom"
    assert incident.stack_trace == "trace"


def test_incident_to_task_and_branch():
    incident = parse_incident(SENTRY_PAYLOAD)
    task = incident_to_task(incident)
    assert task["title"].startswith("Fix:")
    assert "bug" in task["tags"]
    assert "ZeroDivisionError" in task["title"]
    branch = incident_branch_name(incident)
    assert branch.startswith("noema-fix/abc123-")
    assert all(c.isalnum() or c in "-/" for c in branch)


# ── GitHub client (mock transport) ─────────────────────────────────────


def _json_response(status: int, payload):
    return httpx.Response(status, json=payload)


def _mock_transport():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(
            {"method": request.method, "url": str(request.url), "content": request.content}
        )
        path = request.url.path
        if request.method == "GET" and path.endswith("/git/ref/heads/main"):
            return _json_response(200, {"object": {"sha": "base-sha"}})
        if request.method == "POST" and path.endswith("/git/refs"):
            return _json_response(201, {"ref": "refs/heads/branch"})
        if request.method == "GET" and "/contents/" in path:
            if "new" in path:
                return _json_response(404, {"message": "Not Found"})
            return _json_response(200, {"sha": "file-sha", "type": "file"})
        if request.method == "PUT" and "/contents/" in path:
            return _json_response(201, {"content": {"path": path.rsplit("/", 1)[-1]}})
        if request.method == "POST" and path.endswith("/pulls"):
            return _json_response(201, {"number": 7, "html_url": "https://github.com/o/r/pull/7"})
        return _json_response(404, {"message": "no route"})

    return httpx.MockTransport(handler), calls


@pytest.mark.asyncio
async def test_github_create_branch_and_pr():
    transport, calls = _mock_transport()
    client = GitHubClient(token="t", repo="o/r", transport=transport)
    base_sha = await client.create_branch("noema-fix/1-fix")
    assert base_sha == "base-sha"
    pr = await client.open_pr("noema-fix/1-fix", "fix: x", "body")
    assert pr == {"number": 7, "url": "https://github.com/o/r/pull/7"}
    await client.aclose()
    assert any(c["method"] == "POST" and "/git/refs" in c["url"] for c in calls)
    assert any(c["method"] == "POST" and "/pulls" in c["url"] for c in calls)


@pytest.mark.asyncio
async def test_github_submit_fix_pr_full_flow():
    transport, calls = _mock_transport()
    client = GitHubClient(token="t", repo="o/r", base_branch="main", transport=transport)
    result = await client.submit_fix_pr(
        files=[("app/fix.py", "print('ok')\n"), ("README.md", "# fix\n")],
        title="fix: boom",
        body="body",
        branch="noema-fix/1-fix",
    )
    assert result["pr_number"] == 7
    assert result["pr_url"] == "https://github.com/o/r/pull/7"
    assert result["branch"] == "noema-fix/1-fix"
    assert result["files"] == 2
    await client.aclose()

    contents_puts = [c for c in calls if c["method"] == "PUT" and "/contents/" in c["url"]]
    assert len(contents_puts) == 2
    import json

    for c in contents_puts:
        payload = json.loads(c["content"])
        assert payload["branch"] == "noema-fix/1-fix"
        assert payload["content"]  # base64 encoded
        assert "sha" in payload  # existing file gets a blob sha


@pytest.mark.asyncio
async def test_github_error_mapping():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(401, {"message": "Bad credentials"})

    client = GitHubClient(token="t", repo="o/r", transport=httpx.MockTransport(handler))
    with pytest.raises(GitHubError) as exc:
        await client.create_branch("x")
    assert "401" in str(exc.value)
    assert "Bad credentials" in str(exc.value)


def test_github_requires_token_and_repo():
    with pytest.raises(ValueError):
        GitHubClient(token="", repo="o/r")
    with pytest.raises(ValueError):
        GitHubClient(token="t", repo="badrepo")


# ── IncidentFixer flow ─────────────────────────────────────────────────


def _fake_solution(**metadata) -> Solution:
    return Solution(
        id="sol1",
        task_id="t1",
        title="fix",
        summary="fix summary",
        code_blocks=[
            CodeBlock(filename="app/fix.py", language="python", content="print('fixed')\n"),
            CodeBlock(filename="app/fix.py", language="python", content="print('fixed')\n"),
        ],
        quality="good",
        confidence=0.9,
        metadata=metadata,
    )


async def _fixer_with_mocks(engine, github) -> IncidentFixer:
    async def factory():
        return engine

    return IncidentFixer(github=github, engine_factory=factory)


@pytest.mark.asyncio
async def test_fixer_creates_pr_with_validated_fix():
    engine = AsyncMock()
    engine.think = AsyncMock(return_value=(_fake_solution(judge_passed=True), AsyncMock()))
    run = AsyncMock()
    run.all_valid = True
    run.summary = "all ok"
    engine.validate_solution = AsyncMock(return_value=run)
    github = AsyncMock()
    github.submit_fix_pr = AsyncMock(
        return_value={"branch": "b", "pr_number": 1, "pr_url": "url", "files": 1}
    )

    fixer = await _fixer_with_mocks(engine, github)
    result = await fixer.handle_incident(SENTRY_PAYLOAD)

    assert result["status"] == "pr_created"
    assert result["pr_number"] == 1
    assert result["judge_passed"] is True
    engine.think.assert_awaited_once()
    engine.validate_solution.assert_awaited_once()
    assert engine.validate_solution.await_args.kwargs["run_tests"] is True
    github.submit_fix_pr.assert_awaited_once()
    pr_kwargs = github.submit_fix_pr.await_args.kwargs
    assert pr_kwargs["files"] == [("app/fix.py", "print('fixed')\n")]  # deduplicated
    assert "ZeroDivisionError" in pr_kwargs["body"]


@pytest.mark.asyncio
async def test_fixer_blocks_pr_when_run_fails():
    engine = AsyncMock()
    engine.think = AsyncMock(return_value=(_fake_solution(), AsyncMock()))
    run = AsyncMock()
    run.all_valid = False
    run.summary = "tests failed"
    engine.validate_solution = AsyncMock(return_value=run)
    github = AsyncMock()

    fixer = await _fixer_with_mocks(engine, github)
    result = await fixer.handle_incident(SENTRY_PAYLOAD)

    assert result["status"] == "validation_failed"
    assert result["summary"] == "tests failed"
    github.submit_fix_pr.assert_not_awaited()


@pytest.mark.asyncio
async def test_fixer_no_changes_without_code_blocks():
    engine = AsyncMock()
    solution = _fake_solution()
    solution.code_blocks = []
    engine.think = AsyncMock(return_value=(solution, AsyncMock()))
    run = AsyncMock()
    run.all_valid = True
    engine.validate_solution = AsyncMock(return_value=run)
    github = AsyncMock()

    fixer = await _fixer_with_mocks(engine, github)
    result = await fixer.handle_incident(SENTRY_PAYLOAD)

    assert result["status"] == "no_changes"
    github.submit_fix_pr.assert_not_awaited()


@pytest.mark.asyncio
async def test_fixer_merge_gate_error_is_fail_closed():
    """A raised gate error must prevent approval, never crash the loop."""
    github = AsyncMock()

    async def _crashing_gate(files):
        raise RuntimeError("lean toolchain broken")

    fixer = IncidentFixer(
        github=github,
        gate_runner=_crashing_gate,
        auto_approve=True,
        auto_merge=True,
    )
    result: dict = {}
    report = await fixer._run_merge_gate(
        parse_incident(SENTRY_PAYLOAD),
        _fake_solution(),
        [("app/fix.py", "print('fixed')\n")],
        {"pr_number": 1},
        result,
    )
    assert report is None
    assert "lean toolchain broken" in result["merge_gate_error"]
    assert "merge_gate_approved" not in result
    github.approve_pr.assert_not_awaited()
    github.merge_pr.assert_not_awaited()


def test_solution_files_dedupes_and_falls_back():
    solution = _fake_solution()
    files = _solution_files(solution)
    assert files == [("app/fix.py", "print('fixed')\n")]
    empty = CodeBlock(filename="", language="python", content="x")
    assert _block_path(empty).endswith(".py")


# ── Worker wiring ──────────────────────────────────────────────────────


def test_worker_registers_fix_incident_task():
    from noema.workers.arq_worker import NoemaWorkerSettings

    assert any(f.__name__ == "fix_incident_task" for f in NoemaWorkerSettings.functions)
