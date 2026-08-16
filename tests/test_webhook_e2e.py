"""E2E: Sentry webhook → validated fix → PR → merge gate → auto-approve/merge.

Exercises the full closed loop through the real HTTP endpoint:
1. A fake Sentry incident is POSTed to ``/webhooks/incident`` with a valid
   ``X-Noema-Signature`` HMAC over the raw body.
2. The engine produces a fix and the passing-run gate validates it.
3. ``IncidentFixer`` opens a PR via the (mocked) GitHub API.
4. The merge gate runs over the fix files; on pass the PR is approved and —
   with ``auto_merge`` — merged (Evolution Auto-Apply).
5. The generation cost is attributed per module (``noema_pr_cost_usd``).

Redis is stubbed out so the inline fallback path is exercised deterministically;
everything else (HMAC, payload parsing, fixer, gate wiring) is real.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from noema.api.server import app
from noema.billing.cost_tracker import CostTracker
from noema.config.settings import get_settings, reset_settings
from noema.core.types import CodeBlock, Solution
from noema.experiments.gate import GateConfig, GateReport, run_merge_gate

SENTRY_INCIDENT = {
    "event": "incident",
    "payload": {
        "event": {
            "id": "e2e-1",
            "title": "ZeroDivisionError: division by zero",
            "message": "div by zero in payments.total()",
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
    },
}

FIX_FILE = ("app/billing.py", "def total(amount, rate):\n    return amount * rate\n")


def _sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _mock_solution(task_id: str = "t1") -> Solution:
    return Solution(
        id="sol-e2e",
        task_id=task_id,
        title="fix: division by zero",
        summary="guard against zero rate",
        code_blocks=[CodeBlock(filename=FIX_FILE[0], language="python", content=FIX_FILE[1])],
        quality="good",
        confidence=0.9,
        metadata={"judge_passed": True},
    )


def _github_transport(calls: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(
            {"method": request.method, "url": str(request.url), "content": request.content}
        )
        path = request.url.path
        if request.method == "GET" and path.endswith("/git/ref/heads/main"):
            return httpx.Response(200, json={"object": {"sha": "base-sha"}})
        if request.method == "POST" and path.endswith("/git/refs"):
            return httpx.Response(201, json={"ref": "refs/heads/branch"})
        if request.method == "GET" and "/contents/" in path:
            return httpx.Response(404, json={"message": "Not Found"})
        if request.method == "PUT" and "/contents/" in path:
            return httpx.Response(201, json={"content": {"path": path.rsplit("/", 1)[-1]}})
        if request.method == "POST" and path.endswith("/pulls"):
            return httpx.Response(
                201, json={"number": 7, "html_url": "https://github.com/acme/payments/pull/7"}
            )
        if request.method == "POST" and path.endswith("/pulls/7/reviews"):
            return httpx.Response(201, json={"id": 1, "state": "APPROVED"})
        if request.method == "PUT" and path.endswith("/pulls/7/merge"):
            return httpx.Response(200, json={"merged": True, "sha": "deadbeef"})
        return httpx.Response(404, json={"message": "no route"})

    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _autonomy_env(monkeypatch):
    monkeypatch.setenv("NOEMA_API__WEBHOOK_SECRET", "sentry-hmac-secret")
    monkeypatch.setenv("NOEMA_AUTONOMY__GITHUB_TOKEN", "ghp_e2e")
    monkeypatch.setenv("NOEMA_AUTONOMY__GITHUB_REPO", "acme/payments")
    monkeypatch.setenv("NOEMA_AUTONOMY__GITHUB_BASE_BRANCH", "main")
    monkeypatch.setenv("NOEMA_AUTONOMY__AUTO_APPROVE", "true")
    monkeypatch.setenv("NOEMA_AUTONOMY__AUTO_MERGE", "true")
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def e2e_mocks(monkeypatch):
    """Stub Redis, the engine, GitHub transport, and the merge gate."""
    calls: list[dict] = []

    async def _fail_enqueue(*args, **kwargs):
        raise RuntimeError("redis unavailable (test)")

    monkeypatch.setattr(
        "noema.workers.arq_worker.enqueue_fix_incident", AsyncMock(side_effect=_fail_enqueue)
    )

    engine = AsyncMock()
    engine.think = AsyncMock(return_value=(_mock_solution(), AsyncMock()))
    run = SimpleNamespace(all_valid=True, summary="all ok")
    engine.validate_solution = AsyncMock(return_value=run)

    async def _engine_factory():
        return engine

    monkeypatch.setattr("noema.autonomy.fixer._default_engine_factory", _engine_factory)

    from noema.autonomy.github import GitHubClient

    monkeypatch.setattr(
        "noema.autonomy.fixer.build_github_client_from_settings",
        lambda: GitHubClient(
            token="ghp_e2e", repo="acme/payments", transport=_github_transport(calls)
        ),
    )

    gate_report = GateReport(passed=True, changed_files=1, judge_score=0.9)
    gate_runner = AsyncMock(return_value=gate_report)
    monkeypatch.setattr("noema.autonomy.fixer._default_gate_runner", gate_runner)

    return SimpleNamespace(calls=calls, engine=engine, gate_runner=gate_runner)


@pytest.mark.asyncio
async def test_sentry_webhook_full_loop_to_auto_merged_pr(e2e_mocks):
    tracker = CostTracker()
    await tracker.record(
        tenant_id="acme",
        task_id="t1",
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=100_000,
        output_tokens=0,
    )
    app.state.cost_tracker = tracker

    body = json.dumps(SENTRY_INCIDENT, ensure_ascii=False).encode()
    secret = get_settings().api.webhook_secret.get_secret_value()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/webhooks/incident",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Noema-Signature": _sign(secret, body),
            },
        )

    assert resp.status_code == 200, resp.text
    result = resp.json()
    assert result["status"] == "pr_created"
    assert result["pr_number"] == 7
    assert result["incident_id"] == "e2e-1"
    assert result["judge_passed"] is True
    assert result["merge_gate_passed"] is True
    assert result["merge_gate_approved"] is True
    assert result["merged"] is True
    assert result["merge_sha"] == "deadbeef"

    e2e_mocks.gate_runner.assert_awaited_once()
    gate_files = e2e_mocks.gate_runner.await_args.args[0]
    assert gate_files == [FIX_FILE]

    methods = [c["method"] for c in e2e_mocks.calls]
    assert "POST" in methods  # branch refs + PR
    assert any("pulls/7/reviews" in c["url"] for c in e2e_mocks.calls)
    assert any("pulls/7/merge" in c["url"] for c in e2e_mocks.calls)

    pr_cost = tracker.get_pr_cost("acme/payments", 7)
    assert pr_cost["total_usd"] == pytest.approx(0.015)
    assert pr_cost["modules"] == {"app/billing.py": 0.015}

    from noema.observability.metrics import _HAS_PROMETHEUS

    if _HAS_PROMETHEUS:
        from prometheus_client import generate_latest

        metrics_body = generate_latest().decode()
        assert "noema_pr_cost_usd" in metrics_body
        assert 'module="app/billing.py"' in metrics_body


@pytest.mark.asyncio
async def test_webhook_returns_queued_when_redis_available(monkeypatch):
    monkeypatch.setattr(
        "noema.workers.arq_worker.enqueue_fix_incident", AsyncMock(return_value="job-42")
    )
    body = json.dumps(SENTRY_INCIDENT, ensure_ascii=False).encode()
    secret = get_settings().api.webhook_secret.get_secret_value()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/webhooks/incident",
            content=body,
            headers={"X-Noema-Signature": _sign(secret, body)},
        )

    assert resp.status_code == 200
    assert resp.json() == {"status": "queued", "job_id": "job-42"}


@pytest.mark.asyncio
async def test_webhook_rejects_missing_signature():
    body = json.dumps(SENTRY_INCIDENT, ensure_ascii=False).encode()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/webhooks/incident", content=body)
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_webhook_rejects_wrong_signature():
    body = json.dumps(SENTRY_INCIDENT, ensure_ascii=False).encode()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/webhooks/incident",
            content=body,
            headers={"X-Noema-Signature": _sign("wrong-secret", body)},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_gate_failure_blocks_auto_approve_and_merge(e2e_mocks):
    e2e_mocks.gate_runner.return_value = GateReport(
        passed=False, changed_files=1, blocked_by=["sandbox"]
    )

    body = json.dumps(SENTRY_INCIDENT, ensure_ascii=False).encode()
    secret = get_settings().api.webhook_secret.get_secret_value()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/webhooks/incident",
            content=body,
            headers={"X-Noema-Signature": _sign(secret, body)},
        )

    assert resp.status_code == 200
    result = resp.json()
    assert result["status"] == "pr_created"
    assert result["merge_gate_passed"] is False
    assert result["merge_gate_blocked_by"] == ["sandbox"]
    assert "merge_gate_approved" not in result
    assert "merged" not in result

    urls = [c["url"] for c in e2e_mocks.calls]
    assert not any("reviews" in u for u in urls)
    assert not any("/merge" in u for u in urls)


@pytest.mark.asyncio
async def test_real_merge_gate_approves_the_fix_files():
    """The actual ``run_merge_gate`` passes on the same files the fixer gated."""
    from noema.judge import JudgeScore, JudgeVerdict

    async def _judge(solution):
        return JudgeVerdict(
            passed=True,
            scores=JudgeScore(architecture=0.9, code_quality=0.9, overall=0.9),
            summary="looks good",
        )

    sandbox = SimpleNamespace(all_valid=True, summary="1 file valid")
    sandbox.validate_files = AsyncMock(return_value=sandbox)

    cfg = GateConfig(
        judge_threshold=0.0,
        explicit_files=[{"path": FIX_FILE[0], "content": FIX_FILE[1]}],
        judge=_judge,
        sandbox=sandbox,
    )
    report = await run_merge_gate(cfg)
    assert report.passed is True
    assert report.changed_files == 1
    assert report.sandbox_all_valid is True
    assert report.blocked_by == []


@pytest.mark.asyncio
async def test_real_merge_gate_blocks_broken_fix():
    async def _judge(solution):
        return SimpleNamespace(
            passed=False,
            summary="fake judge",
            scores=SimpleNamespace(overall=0.0),
        )

    sandbox = SimpleNamespace(all_valid=False, summary="broken")
    sandbox.validate_files = AsyncMock(return_value=sandbox)

    cfg = GateConfig(
        explicit_files=[{"path": "app/billing.py", "content": "def broken(:\n"}],
        judge=_judge,
        sandbox=sandbox,
    )
    report = await run_merge_gate(cfg)
    assert report.passed is False
    assert "sandbox" in report.blocked_by
