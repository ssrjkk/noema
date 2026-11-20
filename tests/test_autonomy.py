"""T2.1 tests: incident → validated fix → pull request loop.

Covers ``noema/autonomy``:
- incident normalization (Sentry envelope, generic webhook, own schema),
- the failure signature that keys one bug's repairs together across alerts,
- GitHub client REST calls (branch refs, contents, pulls) via a mock transport,
- the IncidentFixer end-to-end flow with a mocked engine + mocked GitHub,
- the passing-run gate (no PR when the sandbox run is not all-valid),
- the memory loop: verified patches indexed by signature, recalled as priors,
  blocked or errored gate runs kept out of the index, a dead store survivable.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from noema.autonomy.fixer import IncidentFixer, _block_path, _solution_files
from noema.autonomy.github import GitHubClient, GitHubError
from noema.autonomy.incidents import (
    error_signature,
    incident_branch_name,
    incident_to_task,
    parse_incident,
)
from noema.core.types import CodeBlock, Solution
from noema.experiments.gate import GateReport
from noema.memory.hybrid_retriever import FailurePattern
from noema.memory.store import MemoryStore

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

    # These flow tests are about the fix loop, not the failure index: without an
    # explicit store the AsyncMock engine answers every attribute with a mock, and
    # recall would run against a coroutine instead of a memory tier.
    engine.memory = None

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


# ── Incident ↔ memory integration ──────────────────────────────────────


def _alert(
    event_id: str,
    message: str,
    environment: str = "production",
    filename: str = "app/billing.py",
    lineno: int = 42,
) -> dict:
    """A Sentry-shaped alert, so two alerts for one bug can be compared."""
    return {
        "event": {
            "id": event_id,
            "title": "ZeroDivisionError: division by zero",
            "message": message,
            "environment": environment,
            "exception": {
                "values": [
                    {
                        "type": "ZeroDivisionError",
                        "value": "division by zero",
                        "stacktrace": {
                            "frames": [
                                {"filename": filename, "function": "total", "lineno": lineno}
                            ]
                        },
                    }
                ]
            },
        },
        "source": "sentry",
    }


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(persist_dir=str(tmp_path / "memory"), auto_save=False)


class _RecordingEngine:
    """Engine double that keeps the tasks it was asked to solve."""

    def __init__(self, memory, run_ok: bool = True) -> None:
        self.memory = memory
        self.tasks: list = []
        self._solution = _fake_solution(judge_passed=True)
        self._run = SimpleNamespace(all_valid=run_ok, summary="all ok")

    async def think(self, task):
        self.tasks.append(task)
        return self._solution, SimpleNamespace()

    async def validate_solution(self, solution, run_tests=False):
        return self._run


def _github_with_mock():
    github = AsyncMock()
    github.submit_fix_pr = AsyncMock(
        return_value={"branch": "b", "pr_number": 1, "pr_url": "url", "files": 1}
    )
    return github


def _memory_fixer(engine, github, gate_report=None) -> IncidentFixer:
    async def factory():
        return engine

    gate_runner = None
    if gate_report is not None:
        gate_runner = AsyncMock(return_value=gate_report)

    return IncidentFixer(
        github=github,
        engine_factory=factory,
        gate_runner=gate_runner,
        auto_approve=gate_report is not None,
    )


def test_error_signature_keys_the_repair_on_what_and_where_broke():
    first = parse_incident(_alert("abc123", "div by zero at 10:00"))
    second = parse_incident(_alert("zzz999", "checkout blew up", environment="staging"))

    assert first.id != second.id
    assert first.description != second.description
    assert error_signature(first) == error_signature(second)
    assert "division by zero" in error_signature(first)
    assert "app/billing.py" in error_signature(first)

    # A different raise site is a different repair, even for the same exception.
    elsewhere = parse_incident(_alert("q1", "div by zero", filename="app/cart.py"))
    assert error_signature(elsewhere) != error_signature(first)

    # And an alert with no trace at all still keys on something other than nothing.
    bare = parse_incident({"incident": {"id": "b1", "title": "Timeout in API"}})
    assert error_signature(bare) == "Timeout in API | id:b1"

    # With a deep stack the signature must cite the frame that raised, not the
    # outermost one — otherwise two different bugs in one handler collide.
    deep = parse_incident(
        {
            "incident": {
                "id": "d1",
                "title": "KeyError",
                "stack_trace": (
                    "KeyError: 'card'\n"
                    "  File 'app/outer.py', line 1, in handle\n"
                    "  File 'app/cart.py', line 88, in add_item"
                ),
            }
        }
    )
    assert error_signature(deep) == "KeyError: 'card' | File 'app/cart.py', line 88, in add_item"


@pytest.mark.asyncio
async def test_fixer_recalls_a_previous_patch_into_the_fix_task(tmp_path):
    store = _store(tmp_path)
    known = parse_incident(_alert("abc123", "div by zero at 10:00"))
    store.index_failure_fix(
        FailurePattern(
            pattern_id="older-alert",
            error_signature=error_signature(known),
            unsat_core=(),
            verified_patch_diff="def total(amount, rate):\n    if rate == 0: raise ValueError()\n",
            z3_contract="",
        )
    )

    engine = _RecordingEngine(store)
    fixer = _memory_fixer(engine, _github_with_mock())
    result = await fixer.handle_incident(_alert("zzz999", "checkout blew up"))

    assert result["status"] == "pr_created"
    description = engine.tasks[0].description
    assert "Previously verified fixes" in description
    assert "if rate == 0: raise ValueError()" in description
    # The alert's own trace still has to reach the model.
    assert "Stack trace:" in description


@pytest.mark.asyncio
async def test_fixer_indexes_a_validated_fix_for_the_next_alert(tmp_path):
    store = _store(tmp_path)
    engine = _RecordingEngine(store)
    github = _github_with_mock()
    fixer = _memory_fixer(engine, github)

    result = await fixer.handle_incident(_alert("abc123", "div by zero"))
    assert result["status"] == "pr_created"

    assert [f.pattern_id for f in store.failures] == ["abc123"]
    assert store.failures[0].error_signature == error_signature(
        parse_incident(_alert("abc123", ""))
    )
    assert "app/fix.py" in store.failures[0].verified_patch_diff
    assert "print('fixed')" in store.failures[0].verified_patch_diff

    # The recorded lesson has to answer the next alert, not just sit on disk. A
    # decoy pattern joins the index first: if the lesson were keyed on anything
    # other than the failure it describes, the decoy would outrank it.
    decoy = FailurePattern(
        pattern_id="unrelated",
        error_signature="TypeError: unsupported | File 'app/pdf.py', line 7, in render",
        unsat_core=(),
        verified_patch_diff="unrelated repair",
        z3_contract="",
    )
    store.index_failure_fix(decoy)
    hits = store.lookup_similar_failures(
        error_signature(parse_incident(_alert("abc123", ""))), top_k=2
    )
    assert [pattern.pattern_id for pattern, _score in hits][0] == "abc123"

    # And the next alert of the same bug gets the patch as priors.
    engine2 = _RecordingEngine(store)
    fixer2 = _memory_fixer(engine2, _github_with_mock())
    await fixer2.handle_incident(_alert("zzz999", "checkout blew up"))
    assert "print('fixed')" in engine2.tasks[0].description


@pytest.mark.asyncio
async def test_fixer_keeps_a_gate_blocked_fix_out_of_the_index(tmp_path):
    store = _store(tmp_path)
    engine = _RecordingEngine(store)
    github = _github_with_mock()
    fixer = _memory_fixer(
        engine, github, gate_report=GateReport(passed=False, changed_files=1, blocked_by=["judge"])
    )

    result = await fixer.handle_incident(_alert("abc123", "div by zero"))

    assert result["status"] == "pr_created"
    assert result["merge_gate_passed"] is False
    assert store.failures == []


@pytest.mark.asyncio
async def test_fixer_keeps_a_gate_errored_fix_out_of_the_index(tmp_path):
    """An unusable gate means "unverified", not "allowed": no lesson is recorded."""
    store = _store(tmp_path)

    async def crashing_gate(files):
        raise RuntimeError("lean toolchain broken")

    async def factory():
        return _RecordingEngine(store)

    fixer = IncidentFixer(
        github=_github_with_mock(),
        engine_factory=factory,
        gate_runner=crashing_gate,
        auto_approve=True,
    )
    result = await fixer.handle_incident(_alert("abc123", "div by zero"))

    assert result["merge_gate_error"] == "lean toolchain broken"
    assert store.failures == []


@pytest.mark.asyncio
async def test_fixer_still_ships_the_pr_when_memory_is_unusable(tmp_path):
    """Recall and indexing are optimisations: a dead store must not stop the fix."""

    class _BrokenStore:
        def lookup_similar_failures(self, *args, **kwargs):
            raise RuntimeError("hnsw index corrupt")

        def index_failure_fix(self, *args, **kwargs):
            raise RuntimeError("disk full")

    github = _github_with_mock()
    fixer = _memory_fixer(_RecordingEngine(_BrokenStore()), github)

    result = await fixer.handle_incident(SENTRY_PAYLOAD)

    assert result["status"] == "pr_created"
    github.submit_fix_pr.assert_awaited_once()


# ── Worker wiring ──────────────────────────────────────────────────────


def test_worker_registers_fix_incident_task():
    from noema.workers.arq_worker import NoemaWorkerSettings

    assert any(f.__name__ == "fix_incident_task" for f in NoemaWorkerSettings.functions)
