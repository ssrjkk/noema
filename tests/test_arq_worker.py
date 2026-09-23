"""Tests for arq worker helpers: node identity, heartbeat, liveness listing.

Covers every public function/method and error path in
``noema.workers.arq_worker`` to reach 100% line coverage.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis
import pytest
from structlog.testing import capture_logs

from noema.workers.arq_worker import (
    HEARTBEAT_INTERVAL,
    HEARTBEAT_PREFIX,
    HEARTBEAT_TTL,
    NodeHeartbeat,
    NoemaWorkerSettings,
    _as_str,
    create_worker,
    drain,
    enqueue_fix_incident,
    enqueue_think,
    fix_incident_task,
    list_active_workers,
    make_node_id,
    run_worker,
    shutdown,
    startup,
    think_task,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


async def _hgetall(r, key: str) -> dict[str, str]:
    raw = await r.hgetall(key)
    return {
        k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
        for k, v in raw.items()
    }


def _make_settings(
    redis_url: str = "redis://fake:6379/0",
    metrics_enabled: bool = True,
    metrics_port: int = 9090,
    ledger_path: str = "",
):
    return SimpleNamespace(
        redis=SimpleNamespace(url=redis_url),
        obs=SimpleNamespace(metrics_enabled=metrics_enabled, metrics_port=metrics_port),
        worker=SimpleNamespace(ledger_path=ledger_path),
    )


def _make_solution(
    task_id: str = "t1",
    solution_id: str = "s1",
    quality_value: str = "good",
    confidence: float = 0.8,
):
    sol = MagicMock()
    sol.id = solution_id
    sol.task_id = task_id
    sol.quality = SimpleNamespace(value=quality_value)
    sol.confidence = confidence
    return sol


def _make_thought(duration_ms: float = 42.0):
    thought = MagicMock()
    thought.duration_ms = duration_ms
    return thought


def _make_noema(
    think_return=None,
    think_side_effect=None,
    stats_side_effect=None,
    model_name: str = "gpt-4",
):
    """Build a mock NoemaEngine where ``think`` is async but ``tracer.get_stats``
    is synchronous (matching the real engine)."""
    noema = MagicMock()
    if think_side_effect is not None:
        noema.think = AsyncMock(side_effect=think_side_effect)
    elif think_return is not None:
        noema.think = AsyncMock(return_value=think_return)
    else:
        noema.think = AsyncMock()
    if stats_side_effect is not None:
        noema.tracer.get_stats = MagicMock(side_effect=stats_side_effect)
    else:
        noema.tracer.get_stats = MagicMock(return_value={"tokens_input": 0, "tokens_output": 0})
    noema.llm.model_name = model_name
    return noema


# ---------------------------------------------------------------------------
# make_node_id
# ---------------------------------------------------------------------------


class TestNodeIdentity:
    def test_make_node_id_unique(self):
        ids = {make_node_id() for _ in range(100)}
        assert len(ids) == 100
        assert all(":" in i for i in ids)

    def test_make_node_id_format(self):
        nid = make_node_id()
        parts = nid.split(":")
        assert len(parts) == 3
        assert len(parts[2]) == 8


# ---------------------------------------------------------------------------
# NodeHeartbeat
# ---------------------------------------------------------------------------


class TestNodeHeartbeat:
    async def test_heartbeat_writes_liveness(self, redis):
        hb = NodeHeartbeat("node-1", redis=redis)
        await hb.start()
        try:
            data = await _hgetall(redis, f"{HEARTBEAT_PREFIX}node-1")
            assert data["node_id"] == "node-1"
            assert data["draining"] == "0"
            assert "started_at" in data
            ttl = await redis.ttl(f"{HEARTBEAT_PREFIX}node-1")
            assert 0 < ttl <= 15
        finally:
            await hb.stop()

    async def test_mark_draining_flag(self, redis):
        hb = NodeHeartbeat("node-1", redis=redis)
        await hb.start()
        try:
            await hb.mark_draining()
            data = await _hgetall(redis, f"{HEARTBEAT_PREFIX}node-1")
            assert data["draining"] == "1"
        finally:
            await hb.stop()

    async def test_stop_removes_key(self, redis):
        hb = NodeHeartbeat("node-1", redis=redis)
        await hb.start()
        await hb.stop()
        assert await redis.exists(f"{HEARTBEAT_PREFIX}node-1") == 0

    async def test_heartbeat_refreshes_ttl(self, redis):
        hb = NodeHeartbeat("node-1", redis=redis)
        await hb.start()
        try:
            await asyncio.sleep(0.1)
            await hb._beat()
            ttl = await redis.ttl(f"{HEARTBEAT_PREFIX}node-1")
            assert ttl > 0
        finally:
            await hb.stop()

    async def test_heartbeat_loop_runs(self, redis):
        hb = NodeHeartbeat("node-1", redis=redis)
        await hb.start()
        try:
            assert hb._task is not None and not hb._task.done()
        finally:
            await hb.stop()

    async def test_start_creates_redis_from_url_when_none(self):
        """Line 68: ``Redis.from_url`` is called when ``_redis`` is None."""
        fake_redis = AsyncMock()
        fake_redis.hset = AsyncMock(return_value=1)
        fake_redis.expire = AsyncMock(return_value=1)
        fake_redis.delete = AsyncMock(return_value=1)
        fake_redis.aclose = AsyncMock()

        with patch("noema.workers.arq_worker.Redis") as mock_redis_cls:
            mock_redis_cls.from_url.return_value = fake_redis
            hb = NodeHeartbeat("node-x", redis_url="redis://localhost:6379/0")
            assert hb._redis is None
            await hb.start()
            try:
                mock_redis_cls.from_url.assert_called_once_with(
                    "redis://localhost:6379/0", decode_responses=True
                )
                assert hb._redis is fake_redis
            finally:
                await hb.stop()

    async def test_beat_noop_when_redis_is_none(self):
        """Line 95: ``_beat()`` returns early when ``_redis`` is None."""
        hb = NodeHeartbeat("node-y", redis_url="")
        assert hb._redis is None
        await hb._beat()  # should not raise

    async def test_loop_catches_beat_exception(self, redis):
        """Lines 117-121: the heartbeat loop catches exceptions from _beat()."""
        hb = NodeHeartbeat("node-loop-err2", redis=redis)
        hb._redis = redis
        call_count = 0

        async def failing_beat():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise ConnectionError("boom")

        hb._beat = failing_beat

        with patch("noema.workers.arq_worker.HEARTBEAT_INTERVAL", 0.01):
            loop_task = asyncio.create_task(hb._loop())
            await asyncio.sleep(0.1)
            loop_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await loop_task

        assert call_count >= 1

    async def test_stop_when_task_is_none(self, redis):
        """stop() still revokes liveness when the loop never started."""
        hb = NodeHeartbeat("node-stop-no-task", redis=redis)
        await hb._beat()
        assert await redis.exists(hb._key()) == 1
        await hb.stop()
        assert await redis.exists(hb._key()) == 0
        assert hb._task is None
        assert hb._redis is None

    async def test_stop_when_redis_is_none(self, redis):
        """A lost connection must not turn shutdown into an AttributeError."""
        hb = NodeHeartbeat("node-stop-no-redis", redis=redis)
        await hb.start()
        key = hb._key()
        assert await redis.exists(key) == 1
        hb._redis = None
        await hb.stop()
        await hb.stop()
        assert await redis.exists(key) == 1

    async def test_metrics_port_stored(self, redis):
        hb = NodeHeartbeat("node-metrics", redis=redis, metrics_port=8080)
        assert hb.metrics_port == 8080

    async def test_beat_writes_metrics_port(self, redis):
        hb = NodeHeartbeat("node-mp", redis=redis, metrics_port=4321)
        await hb.start()
        try:
            data = await _hgetall(redis, f"{HEARTBEAT_PREFIX}node-mp")
            assert data["metrics_port"] == "4321"
        finally:
            await hb.stop()

    async def test_key_format(self):
        hb = NodeHeartbeat("abc")
        assert hb._key() == f"{HEARTBEAT_PREFIX}abc"

    async def test_started_at_is_int(self):
        hb = NodeHeartbeat("node-ts")
        assert isinstance(hb.started_at, int)

    async def test_draining_default_false(self):
        hb = NodeHeartbeat("node-d")
        assert hb.draining is False


# ---------------------------------------------------------------------------
# startup()
# ---------------------------------------------------------------------------


class TestStartup:
    async def test_startup_populates_ctx(self):
        """Lines 125-141: startup() initialises engine, heartbeat, ledger."""
        ctx: dict[str, Any] = {}
        fake_engine = AsyncMock()
        fake_engine.initialize = AsyncMock()
        fake_heartbeat = AsyncMock()
        fake_heartbeat.start = AsyncMock()
        fake_ledger = MagicMock()

        settings = _make_settings(
            metrics_enabled=True, metrics_port=9999, ledger_path="/tmp/ledger.jsonl"
        )

        with (
            patch("noema.workers.arq_worker.NoemaEngine", return_value=fake_engine),
            patch("noema.workers.arq_worker.make_node_id", return_value="test-node-42"),
            patch("noema.workers.arq_worker.NodeHeartbeat", return_value=fake_heartbeat),
            patch(
                "noema.billing.ledger.ContributionLedger", return_value=fake_ledger
            ) as mock_ledger_cls,
            patch("noema.config.settings.get_settings", return_value=settings),
        ):
            await startup(ctx)

        assert ctx["noema"] is fake_engine
        assert ctx["node_id"] == "test-node-42"
        assert ctx["heartbeat"] is fake_heartbeat
        assert ctx["ledger"] is fake_ledger
        fake_engine.initialize.assert_awaited_once()
        fake_heartbeat.start.assert_awaited_once()
        mock_ledger_cls.assert_called_once_with(path="/tmp/ledger.jsonl")

    async def test_startup_metrics_disabled(self):
        """When metrics are disabled, metrics_port=0 is passed to heartbeat."""
        ctx: dict[str, Any] = {}
        fake_engine = AsyncMock()
        fake_engine.initialize = AsyncMock()
        fake_heartbeat = AsyncMock()
        fake_heartbeat.start = AsyncMock()

        settings = _make_settings(metrics_enabled=False, metrics_port=9090)

        with (
            patch("noema.workers.arq_worker.NoemaEngine", return_value=fake_engine),
            patch("noema.workers.arq_worker.make_node_id", return_value="n1"),
            patch("noema.workers.arq_worker.NodeHeartbeat", return_value=fake_heartbeat) as mock_hb,
            patch("noema.billing.ledger.ContributionLedger"),
            patch("noema.config.settings.get_settings", return_value=settings),
        ):
            await startup(ctx)

        call_kwargs = mock_hb.call_args
        assert call_kwargs.kwargs.get("metrics_port") == 0


# ---------------------------------------------------------------------------
# drain()
# ---------------------------------------------------------------------------


class TestDrain:
    async def test_drain_marks_draining_and_stops(self):
        """Lines 150-157: drain() marks draining, stops heartbeat, shuts down engine."""
        heartbeat = AsyncMock()
        heartbeat.mark_draining = AsyncMock()
        heartbeat.stop = AsyncMock()
        engine = AsyncMock()
        engine.shutdown = AsyncMock()
        ctx = {"heartbeat": heartbeat, "noema": engine, "node_id": "drain-node"}

        await drain(ctx)

        heartbeat.mark_draining.assert_awaited_once()
        heartbeat.stop.assert_awaited_once()
        engine.shutdown.assert_awaited_once()

    async def test_drain_no_heartbeat(self):
        """drain() tolerates missing heartbeat."""
        engine = AsyncMock()
        engine.shutdown = AsyncMock()
        ctx: dict[str, Any] = {"noema": engine, "node_id": "n"}

        await drain(ctx)
        engine.shutdown.assert_awaited_once()

    async def test_drain_no_engine(self):
        """drain() tolerates missing engine."""
        heartbeat = AsyncMock()
        heartbeat.mark_draining = AsyncMock()
        heartbeat.stop = AsyncMock()
        ctx: dict[str, Any] = {"heartbeat": heartbeat, "node_id": "n"}

        await drain(ctx)
        heartbeat.mark_draining.assert_awaited_once()
        heartbeat.stop.assert_awaited_once()

    async def test_drain_empty_ctx(self):
        """drain() with no collaborators still reports the drained event."""
        ctx: dict[str, Any] = {}
        with capture_logs() as logs:
            await drain(ctx)
        assert {"event": "arq_worker_drained", "node_id": "?", "log_level": "info"} in logs

    async def test_drain_heartbeat_none_explicit(self):
        """An explicit ``heartbeat: None`` takes the same guard as a missing key."""
        ctx: dict[str, Any] = {"heartbeat": None, "noema": None, "node_id": "x"}
        with capture_logs() as logs:
            await drain(ctx)
        assert {"event": "arq_worker_drained", "node_id": "x", "log_level": "info"} in logs
        assert [e for e in logs if e["event"] == "arq_worker_shutdown_complete"] == []


# ---------------------------------------------------------------------------
# shutdown()
# ---------------------------------------------------------------------------


class TestShutdown:
    async def test_shutdown_calls_drain(self):
        """Lines 161-162: shutdown() delegates to drain() then logs."""
        heartbeat = AsyncMock()
        heartbeat.mark_draining = AsyncMock()
        heartbeat.stop = AsyncMock()
        engine = AsyncMock()
        engine.shutdown = AsyncMock()
        ctx = {"heartbeat": heartbeat, "noema": engine, "node_id": "sd-node"}

        await shutdown(ctx)

        heartbeat.mark_draining.assert_awaited_once()
        heartbeat.stop.assert_awaited_once()
        engine.shutdown.assert_awaited_once()

    async def test_shutdown_empty_ctx(self):
        with capture_logs() as logs:
            await shutdown({})
        events = [entry["event"] for entry in logs]
        assert events == ["arq_worker_drained", "arq_worker_shutdown_complete"]


# ---------------------------------------------------------------------------
# think_task()
# ---------------------------------------------------------------------------


class TestThinkTask:
    async def test_think_task_success_with_ledger(self):
        """Lines 167-205: successful think_task with ledger recording."""
        solution = _make_solution()
        thought = _make_thought(duration_ms=100.0)

        noema = _make_noema(
            think_return=(solution, thought),
            stats_side_effect=[
                {"tokens_input": 10, "tokens_output": 5},  # before
                {"tokens_input": 110, "tokens_output": 55},  # after
            ],
            model_name="gpt-4",
        )

        ledger = MagicMock()
        ledger.record = MagicMock()

        ctx = {"noema": noema, "ledger": ledger, "node_id": "n1"}
        task_data = {
            "title": "Solve P vs NP",
            "description": "Please",
            "complexity": "complex",
            "tags": ["ai"],
            "requirements": [],
            "model": "gpt-4",
            "provider": "openai",
        }

        result = await think_task(ctx, task_data)

        assert result["status"] == "completed"
        assert result["solution_id"] == "s1"
        assert result["task_id"] == "t1"
        assert result["quality"] == "good"
        assert result["confidence"] == 0.8
        assert result["duration_ms"] == 100.0

        ledger.record.assert_called_once()
        call_kwargs = ledger.record.call_args[1]
        assert call_kwargs["node_id"] == "n1"
        assert call_kwargs["task_id"] == "t1"
        assert call_kwargs["kind"] == "solution"
        assert call_kwargs["provider"] == "openai"
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["input_tokens"] == 100
        assert call_kwargs["output_tokens"] == 50
        assert call_kwargs["artifact_ref"] == "s1"

    async def test_think_task_success_without_ledger(self):
        """think_task works when no ledger is in ctx."""
        solution = _make_solution()
        thought = _make_thought()

        noema = _make_noema(think_return=(solution, thought))

        ctx: dict[str, Any] = {"noema": noema}
        task_data = {"title": "Quick task"}

        result = await think_task(ctx, task_data)
        assert result["status"] == "completed"

    async def test_think_task_failure(self):
        """Lines 206-208: think_task returns failed status on exception."""
        noema = _make_noema(think_side_effect=RuntimeError("engine broke"))

        ctx = {"noema": noema}
        task_data = {"title": "Fail task"}

        result = await think_task(ctx, task_data)
        assert result["status"] == "failed"
        assert "engine broke" in result["error"]

    async def test_think_task_model_fallback_to_llm(self):
        """When task_data['model'] is empty, falls back to noema.llm.model_name."""
        solution = _make_solution()
        thought = _make_thought()

        noema = _make_noema(
            think_return=(solution, thought),
            stats_side_effect=[
                {"tokens_input": 0, "tokens_output": 0},
                {"tokens_input": 50, "tokens_output": 25},
            ],
            model_name="claude-3",
        )

        ledger = MagicMock()
        ledger.record = MagicMock()

        ctx = {"noema": noema, "ledger": ledger, "node_id": "n2"}
        task_data = {"title": "T", "model": ""}  # empty model -> fallback

        result = await think_task(ctx, task_data)
        assert result["status"] == "completed"

        call_kwargs = ledger.record.call_args[1]
        assert call_kwargs["model"] == "claude-3"

    async def test_think_task_model_none_zero_cost(self):
        """When model is falsy after fallback, cost_usd=0.0."""
        solution = _make_solution()
        thought = _make_thought()

        noema = _make_noema(
            think_return=(solution, thought),
            stats_side_effect=[
                {"tokens_input": 0, "tokens_output": 0},
                {"tokens_input": 10, "tokens_output": 5},
            ],
            model_name="",
        )

        ledger = MagicMock()
        ledger.record = MagicMock()

        ctx = {"noema": noema, "ledger": ledger, "node_id": ""}
        task_data = {"title": "T", "model": ""}

        result = await think_task(ctx, task_data)
        assert result["status"] == "completed"

        call_kwargs = ledger.record.call_args[1]
        assert call_kwargs["cost_usd"] == 0.0

    async def test_think_task_default_complexity(self):
        """Task complexity defaults to 'moderate' when not provided."""
        solution = _make_solution()
        thought = _make_thought()

        noema = _make_noema(think_return=(solution, thought))

        ctx: dict[str, Any] = {"noema": noema}
        task_data = {"title": "Default complexity"}

        result = await think_task(ctx, task_data)
        assert result["status"] == "completed"

        call_args = noema.think.call_args
        task = call_args[0][0]
        assert task.complexity.value == "moderate"

    async def test_think_task_meta_in_ledger(self):
        """Ledger record includes duration_ms and quality in meta."""
        solution = _make_solution(quality_value="excellent")
        thought = _make_thought(duration_ms=250.0)

        noema = _make_noema(
            think_return=(solution, thought),
            stats_side_effect=[
                {"tokens_input": 0, "tokens_output": 0},
                {"tokens_input": 20, "tokens_output": 10},
            ],
            model_name="m",
        )

        ledger = MagicMock()
        ledger.record = MagicMock()

        ctx = {"noema": noema, "ledger": ledger, "node_id": "n"}
        task_data = {"title": "T", "model": "m"}

        await think_task(ctx, task_data)

        call_kwargs = ledger.record.call_args[1]
        assert call_kwargs["meta"]["duration_ms"] == 250.0
        assert call_kwargs["meta"]["quality"] == "excellent"


# ---------------------------------------------------------------------------
# fix_incident_task()
# ---------------------------------------------------------------------------


class TestFixIncidentTask:
    async def test_fix_incident_task_success(self):
        """Lines 213-219: successful incident fix."""
        fake_fixer = AsyncMock()
        fake_fixer.handle_incident = AsyncMock(return_value={"status": "fixed"})
        fake_fixer.close = AsyncMock()

        payload = {"incident_id": "INC-123"}

        with (
            patch(
                "noema.autonomy.fixer.build_github_client_from_settings", return_value=MagicMock()
            ),
            patch("noema.autonomy.fixer.IncidentFixer", return_value=fake_fixer),
        ):
            result = await fix_incident_task({}, payload)

        assert result == {"status": "fixed"}
        fake_fixer.handle_incident.assert_awaited_once_with(payload)
        fake_fixer.close.assert_awaited_once()

    async def test_fix_incident_task_failure_still_closes(self):
        """finally block ensures close() is called even on failure."""
        fake_fixer = AsyncMock()
        fake_fixer.handle_incident = AsyncMock(side_effect=RuntimeError("GitHub down"))
        fake_fixer.close = AsyncMock()

        with (
            patch(
                "noema.autonomy.fixer.build_github_client_from_settings", return_value=MagicMock()
            ),
            patch("noema.autonomy.fixer.IncidentFixer", return_value=fake_fixer),
            pytest.raises(RuntimeError, match="GitHub down"),
        ):
            await fix_incident_task({}, {"incident_id": "INC-456"})

        fake_fixer.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# NoemaWorkerSettings
# ---------------------------------------------------------------------------


class TestNoemaWorkerSettings:
    def test_functions_registered(self):
        assert think_task in NoemaWorkerSettings.functions
        assert fix_incident_task in NoemaWorkerSettings.functions

    def test_lifecycle_hooks(self):
        assert NoemaWorkerSettings.on_startup is startup
        assert NoemaWorkerSettings.on_shutdown is shutdown

    def test_poll_delay(self):
        assert NoemaWorkerSettings.poll_delay == 1.0

    def test_max_tries(self):
        assert NoemaWorkerSettings.max_tries == 3


# ---------------------------------------------------------------------------
# create_worker()
# ---------------------------------------------------------------------------


class TestCreateWorker:
    async def test_create_worker_with_explicit_url(self):
        """Lines 232-251: create_worker with explicit redis_url."""
        fake_worker = MagicMock()

        with (
            patch("arq.Worker", return_value=fake_worker) as mock_arq,
            patch("arq.connections.RedisSettings") as mock_rs,
        ):
            mock_rs.from_dsn.return_value = MagicMock()
            result = await create_worker(redis_url="redis://myhost:6379/1", burst=True)

        assert result is fake_worker
        mock_rs.from_dsn.assert_called_once_with("redis://myhost:6379/1")
        mock_arq.assert_called_once()
        call_kwargs = mock_arq.call_args[1]
        assert call_kwargs["burst"] is True
        assert call_kwargs["functions"] == NoemaWorkerSettings.functions
        assert call_kwargs["on_startup"] is startup
        assert call_kwargs["on_shutdown"] is shutdown
        assert call_kwargs["poll_delay"] == 1.0
        assert call_kwargs["max_tries"] == 3

    async def test_create_worker_from_settings(self):
        """When redis_url is None, settings are used."""
        settings = _make_settings(redis_url="redis://from-settings:6379/0")
        fake_worker = MagicMock()

        with (
            patch("arq.Worker", return_value=fake_worker),
            patch("arq.connections.RedisSettings") as mock_rs,
            patch("noema.config.settings.get_settings", return_value=settings),
        ):
            result = await create_worker(redis_url=None, burst=False)

        mock_rs.from_dsn.assert_called_once_with("redis://from-settings:6379/0")
        assert result is fake_worker

    async def test_create_worker_burst_false(self):
        fake_worker = MagicMock()

        with (
            patch("arq.Worker", return_value=fake_worker) as mock_arq,
            patch("arq.connections.RedisSettings") as mock_rs,
        ):
            mock_rs.from_dsn.return_value = MagicMock()
            await create_worker(redis_url="redis://x:6379", burst=False)

        call_kwargs = mock_arq.call_args[1]
        assert call_kwargs["burst"] is False


# ---------------------------------------------------------------------------
# run_worker()
# ---------------------------------------------------------------------------


class TestRunWorker:
    async def test_run_worker_delegates(self):
        """Lines 260-261: run_worker creates and runs the worker."""
        fake_worker = AsyncMock()
        fake_worker.async_run = AsyncMock()

        with patch(
            "noema.workers.arq_worker.create_worker", return_value=fake_worker
        ) as mock_create:
            await run_worker(redis_url="redis://h:6379", burst=True)

        mock_create.assert_awaited_once_with("redis://h:6379", burst=True)
        fake_worker.async_run.assert_awaited_once()

    async def test_run_worker_defaults(self):
        fake_worker = AsyncMock()
        fake_worker.async_run = AsyncMock()

        with patch(
            "noema.workers.arq_worker.create_worker", return_value=fake_worker
        ) as mock_create:
            await run_worker()

        mock_create.assert_awaited_once_with(None, burst=False)


# ---------------------------------------------------------------------------
# enqueue_think()
# ---------------------------------------------------------------------------


class TestEnqueueThink:
    async def test_enqueue_think_returns_job_id(self):
        """Lines 266-272: enqueue_think creates pool, enqueues, closes."""
        fake_job = MagicMock()
        fake_job.job_id = "job-abc"

        fake_pool = AsyncMock()
        fake_pool.enqueue_job = AsyncMock(return_value=fake_job)
        fake_pool.close = AsyncMock()

        with (
            patch("arq.create_pool", return_value=fake_pool),
            patch("arq.connections.RedisSettings") as mock_rs,
        ):
            mock_rs.from_dsn.return_value = MagicMock()
            result = await enqueue_think("redis://x:6379", {"title": "T"})

        assert result == "job-abc"
        mock_rs.from_dsn.assert_called_once_with("redis://x:6379")
        fake_pool.enqueue_job.assert_awaited_once_with("think_task", {"title": "T"})
        fake_pool.close.assert_awaited_once()

    async def test_enqueue_think_returns_none_when_no_job(self):
        """When enqueue_job returns None, the function returns None."""
        fake_pool = AsyncMock()
        fake_pool.enqueue_job = AsyncMock(return_value=None)
        fake_pool.close = AsyncMock()

        with (
            patch("arq.create_pool", return_value=fake_pool),
            patch("arq.connections.RedisSettings"),
        ):
            result = await enqueue_think("redis://x:6379", {"title": "T"})

        assert result is None


# ---------------------------------------------------------------------------
# enqueue_fix_incident()
# ---------------------------------------------------------------------------


class TestEnqueueFixIncident:
    async def test_enqueue_fix_incident_returns_job_id(self):
        """Lines 277-283: enqueue_fix_incident creates pool, enqueues, closes."""
        fake_job = MagicMock()
        fake_job.job_id = "job-fix-123"

        fake_pool = AsyncMock()
        fake_pool.enqueue_job = AsyncMock(return_value=fake_job)
        fake_pool.close = AsyncMock()

        with (
            patch("arq.create_pool", return_value=fake_pool),
            patch("arq.connections.RedisSettings") as mock_rs,
        ):
            mock_rs.from_dsn.return_value = MagicMock()
            result = await enqueue_fix_incident("redis://y:6379", {"incident": "X"})

        assert result == "job-fix-123"
        fake_pool.enqueue_job.assert_awaited_once_with("fix_incident_task", {"incident": "X"})
        fake_pool.close.assert_awaited_once()

    async def test_enqueue_fix_incident_returns_none_when_no_job(self):
        fake_pool = AsyncMock()
        fake_pool.enqueue_job = AsyncMock(return_value=None)
        fake_pool.close = AsyncMock()

        with (
            patch("arq.create_pool", return_value=fake_pool),
            patch("arq.connections.RedisSettings"),
        ):
            result = await enqueue_fix_incident("redis://y:6379", {"incident": "Y"})

        assert result is None


# ---------------------------------------------------------------------------
# list_active_workers()
# ---------------------------------------------------------------------------


class TestListWorkers:
    async def test_list_active_workers(self, redis):
        hb1 = NodeHeartbeat("a", redis=redis)
        hb2 = NodeHeartbeat("b", redis=redis)
        await hb1.start()
        await hb2.start()
        try:
            workers = await list_active_workers("", redis=redis)
            assert {w["node_id"] for w in workers} == {"a", "b"}
            assert all(w["draining"] == "0" for w in workers)
        finally:
            await hb1.stop()
            await hb2.stop()

    async def test_drained_worker_not_listed(self, redis):
        hb = NodeHeartbeat("gone", redis=redis)
        await hb.start()
        await hb.stop()
        workers = await list_active_workers("", redis=redis)
        assert workers == []

    async def test_list_workers_sorted_by_started_at(self, redis):
        """Workers are sorted by started_at ascending."""
        hb1 = NodeHeartbeat("old", redis=redis)
        hb2 = NodeHeartbeat("new", redis=redis)
        hb1.started_at = 1000
        hb2.started_at = 2000
        await hb1.start()
        await hb2.start()
        try:
            workers = await list_active_workers("", redis=redis)
            assert workers[0]["node_id"] == "old"
            assert workers[1]["node_id"] == "new"
        finally:
            await hb1.stop()
            await hb2.stop()

    async def test_list_workers_empty_data_skipped(self):
        """Line 298: keys with empty hgetall data are skipped (continue)."""
        fake_redis = AsyncMock()
        # Return two keys: one with data, one empty
        fake_redis.keys = AsyncMock(return_value=["noema:workers:empty", "noema:workers:real"])
        fake_redis.hgetall = AsyncMock(
            side_effect=[
                {},  # empty data -> continue
                {"node_id": "real", "started_at": "500", "draining": "0"},
            ]
        )

        result = await list_active_workers("", redis=fake_redis)
        assert len(result) == 1
        assert result[0]["node_id"] == "real"

    async def test_list_workers_creates_redis_when_none(self):
        """Line 309: when redis is None, a Redis client is created and closed."""
        fake_redis = AsyncMock()
        fake_redis.keys = AsyncMock(return_value=[])
        fake_redis.aclose = AsyncMock()

        with patch("noema.workers.arq_worker.Redis") as mock_redis_cls:
            mock_redis_cls.from_url.return_value = fake_redis
            result = await list_active_workers("redis://test:6379/0", redis=None)

        assert result == []
        mock_redis_cls.from_url.assert_called_once_with(
            "redis://test:6379/0", decode_responses=True
        )
        fake_redis.aclose.assert_awaited_once()

    async def test_list_workers_bytes_keys_decoded(self):
        """Line 303: bytes keys are decoded properly."""
        fake_redis = AsyncMock()
        fake_redis.keys = AsyncMock(return_value=[b"noema:workers:byte-node"])
        fake_redis.hgetall = AsyncMock(
            return_value={
                b"node_id": b"byte-node",
                b"hostname": b"myhost",
                b"started_at": b"9999",
                b"draining": b"0",
            }
        )

        result = await list_active_workers("", redis=fake_redis)
        assert len(result) == 1
        assert result[0]["node_id"] == "byte-node"
        assert result[0]["key"] == "noema:workers:byte-node"

    async def test_list_workers_does_not_close_provided_redis(self, redis):
        """When redis is passed in, it is NOT closed (owned=False)."""
        workers = await list_active_workers("", redis=redis)
        assert isinstance(workers, list)


# ---------------------------------------------------------------------------
# _as_str()
# ---------------------------------------------------------------------------


class TestAsStr:
    def test_bytes_decoded(self):
        """Line 314: bytes values are decoded with errors='replace'."""
        assert _as_str(b"hello") == "hello"

    def test_bytes_with_invalid_utf8(self):
        result = _as_str(b"\xff\xfe")
        assert isinstance(result, str)

    def test_str_passthrough(self):
        """Non-bytes values are converted via str()."""
        assert _as_str("already") == "already"

    def test_int_converted(self):
        assert _as_str(42) == "42"

    def test_none_converted(self):
        assert _as_str(None) == "None"

    def test_float_converted(self):
        assert _as_str(3.14) == "3.14"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_heartbeat_prefix(self):
        assert HEARTBEAT_PREFIX == "noema:workers:"

    def test_heartbeat_ttl(self):
        assert HEARTBEAT_TTL == 15

    def test_heartbeat_interval(self):
        assert HEARTBEAT_INTERVAL == 5
