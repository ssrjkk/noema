"""Tests for neurosymbolic engine."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from noema.neurosymbolic import (
    CircuitBreaker,
    CircuitOpenError,
    Constraint,
    EvolutionEngine,
    LLMRequest,
    NeuralInterface,
    NeuroSymbolicEngine,
    SymbolicEngine,
    SymbolicVerificationError,
    TaskGraph,
)
from noema.neurosymbolic.neural import CircuitState

try:
    import z3  # noqa: F401

    has_z3 = True
except ImportError:
    has_z3 = False


# ── SymbolicEngine ────────────────────────────────────────────────────


def test_validate_task_structure_valid():
    engine = SymbolicEngine()
    engine._validate_task_structure({"requirements": []})


def test_validate_task_structure_missing_requirements():
    engine = SymbolicEngine()
    with pytest.raises(ValueError, match="Missing required field: requirements"):
        engine._validate_task_structure({})


def test_validate_task_structure_not_a_list():
    engine = SymbolicEngine()
    with pytest.raises(ValueError, match="requirements must be a list"):
        engine._validate_task_structure({"requirements": "not_a_list"})


@pytest.mark.asyncio
async def test_parse_task_valid():
    engine = SymbolicEngine()
    task = {
        "requirements": [
            {"name": "x", "type": "numeric", "min": 0, "max": 10, "priority": 2},
        ],
        "constraints": [
            {"name": "flag", "condition": "flag must be true", "priority": 1},
        ],
        "goals": ["solve it"],
    }
    graph = await engine.parse_task(task)
    assert isinstance(graph, TaskGraph)
    assert len(graph.goals) == 1
    assert "parsed_at" in graph.metadata


@pytest.mark.asyncio
async def test_parse_task_invalid_raises():
    engine = SymbolicEngine()
    with pytest.raises(ValueError):
        await engine.parse_task({"wrong": "data"})


@pytest.mark.skipif(not has_z3, reason="z3 not installed")
@pytest.mark.asyncio
async def test_parse_requirement_numeric():
    engine = SymbolicEngine()
    req = {"name": "x", "type": "numeric", "min": 0, "max": 100}
    constraint = await engine._parse_requirement(req)
    assert constraint is not None
    assert constraint.name == "x"
    assert constraint.priority == 1
    assert constraint.description == "x in [0, 100]"


@pytest.mark.skipif(not has_z3, reason="z3 not installed")
@pytest.mark.asyncio
async def test_parse_requirement_non_numeric_returns_none():
    engine = SymbolicEngine()
    req = {"name": "flag", "type": "boolean"}
    constraint = await engine._parse_requirement(req)
    assert constraint is None


@pytest.mark.skipif(not has_z3, reason="z3 not installed")
@pytest.mark.asyncio
async def test_parse_constraint():
    engine = SymbolicEngine()
    const = {"name": "flag", "condition": "must be true"}
    constraint = await engine._parse_constraint(const)
    assert constraint is not None
    assert constraint.name == "flag"
    assert constraint.description == "must be true"


@pytest.mark.skipif(not has_z3, reason="z3 not installed")
@pytest.mark.asyncio
async def test_verify_solution_valid():
    from z3 import And, Int

    engine = SymbolicEngine()
    await engine.initialize()
    var = Int("x")
    req = Constraint(name="x", expression=And(var >= 0, var <= 10), description="x range")
    tg = TaskGraph(requirements=[req], variables={"x": var})
    is_valid, violations = await engine.verify_solution({"x": 5}, tg)
    assert is_valid is True
    assert violations == []


@pytest.mark.skipif(not has_z3, reason="z3 not installed")
@pytest.mark.asyncio
async def test_verify_solution_invalid():
    from z3 import And, Int

    engine = SymbolicEngine()
    await engine.initialize()
    var = Int("x")
    req = Constraint(name="x", expression=And(var >= 0, var <= 10), description="x range")
    tg = TaskGraph(requirements=[req], variables={"x": var})
    is_valid, violations = await engine.verify_solution({"x": 20}, tg)
    assert is_valid is False
    assert len(violations) > 0


@pytest.mark.skipif(not has_z3, reason="z3 not installed")
@pytest.mark.asyncio
async def test_verify_solution_no_graph():
    engine = SymbolicEngine()
    await engine.initialize()
    is_valid, violations = await engine.verify_solution({}, None)
    assert is_valid is False
    assert "task_graph_not_initialized" in violations


@pytest.mark.skipif(not has_z3, reason="z3 not installed")
@pytest.mark.asyncio
async def test_solver_pool_exhausted():
    engine = SymbolicEngine()
    await engine.initialize()
    solvers = []
    for _ in range(10):
        solver = await asyncio.wait_for(engine._solver_pool.get(), timeout=0.5)
        solvers.append(solver)
    with pytest.raises(SymbolicVerificationError, match="Solver pool exhausted"):
        async with engine._get_solver():
            pass
    for s in solvers:
        engine._solver_pool.put_nowait(s)


@pytest.mark.skipif(not has_z3, reason="z3 not installed")
@pytest.mark.asyncio
async def test_verify_solution_timeout():
    engine = SymbolicEngine(verification_timeout=0.05)
    await engine.initialize()

    async def slow_check(solver, solution, tg):
        await asyncio.sleep(10)
        return True

    with patch.object(engine, "_check_solution", slow_check):
        valid, violations = await engine.verify_solution({}, TaskGraph())
        assert valid is False
        assert "verification_timeout" in violations


# ── CircuitBreaker ────────────────────────────────────────────────────


async def _dummy_async():
    return "ok"


async def _failing_async():
    raise ValueError("fail")


def test_circuit_breaker_initial_state():
    cb = CircuitBreaker()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_circuit_breaker_open_after_failures():
    cb = CircuitBreaker(failure_threshold=3)
    cb._on_failure()
    cb._on_failure()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 2
    cb._on_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.failure_count == 3


def test_circuit_breaker_on_success():
    cb = CircuitBreaker()
    cb.state = CircuitState.HALF_OPEN
    cb.failure_count = 3
    cb._on_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_circuit_breaker_on_failure_no_open():
    cb = CircuitBreaker(failure_threshold=5)
    for _ in range(4):
        cb._on_failure()
    assert cb.state == CircuitState.CLOSED


def test_circuit_breaker_on_failure_opens():
    cb = CircuitBreaker(failure_threshold=1)
    cb._on_failure()
    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_call_raises():
    cb = CircuitBreaker(failure_threshold=1)
    cb.state = CircuitState.OPEN
    cb.last_failure_time = datetime.now(UTC)
    with pytest.raises(CircuitOpenError):
        await cb.call(_dummy_async)


@pytest.mark.asyncio
async def test_circuit_breaker_call_success():
    cb = CircuitBreaker()
    result = await cb.call(_dummy_async)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_call_failure_opens():
    cb = CircuitBreaker(failure_threshold=1)
    with pytest.raises(ValueError, match="fail"):
        await cb.call(_failing_async)
    assert cb.state == CircuitState.OPEN


def test_circuit_breaker_should_attempt_reset_no_failure():
    cb = CircuitBreaker()
    assert cb._should_attempt_reset() is True


def test_circuit_breaker_should_attempt_reset_recent():
    cb = CircuitBreaker(recovery_timeout=60.0)
    cb.last_failure_time = datetime.now(UTC)
    assert cb._should_attempt_reset() is False


def test_circuit_breaker_should_attempt_reset_after_recovery():
    cb = CircuitBreaker(recovery_timeout=0.0)
    cb.last_failure_time = datetime.now(UTC) - timedelta(seconds=1)
    assert cb._should_attempt_reset() is True


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_to_closed():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)
    cb.state = CircuitState.OPEN
    cb.last_failure_time = datetime.now(UTC) - timedelta(seconds=1)
    result = await cb.call(_dummy_async)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


# ── NeuralInterface ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_neural_interface_start_stop():
    ni = NeuralInterface()
    assert ni._batch_processor_task is None
    await ni.start()
    assert ni._batch_processor_task is not None
    assert not ni._batch_processor_task.done()
    await ni.stop()
    assert ni._batch_processor_task.cancelled()


@pytest.mark.asyncio
async def test_generate_hypothesis_returns_dict():
    ni = NeuralInterface()
    ni._client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = '{"hypothesis": "test"}'
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_usage = MagicMock()
    mock_usage.total_tokens = 10
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = mock_usage
    ni._client.chat.completions.create = AsyncMock(return_value=mock_resp)
    result = await ni.generate_hypothesis({"task": "test"})
    assert isinstance(result, dict)
    assert result["hypothesis"] == "test"


@pytest.mark.asyncio
async def test_refine_hypothesis_returns_dict():
    ni = NeuralInterface()
    ni._client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = '{"refined": true}'
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_usage = MagicMock()
    mock_usage.total_tokens = 10
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = mock_usage
    ni._client.chat.completions.create = AsyncMock(return_value=mock_resp)
    result = await ni.refine_hypothesis({"old": 1}, ["v1"], {"task": "test"})
    assert isinstance(result, dict)
    assert result["refined"] is True


@pytest.mark.asyncio
async def test_retry_on_rate_limit_error():
    ni = NeuralInterface(max_retries=3, base_delay=0.01)
    ni._client = MagicMock()
    _rate_limit_error = type("RateLimitError", (Exception,), {})
    mock_msg = MagicMock()
    mock_msg.content = '{"ok": true}'
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_usage = MagicMock()
    mock_usage.total_tokens = 5
    mock_success = MagicMock()
    mock_success.choices = [mock_choice]
    mock_success.usage = mock_usage
    ni._client.chat.completions.create = AsyncMock(
        side_effect=[_rate_limit_error("fast"), _rate_limit_error("slow"), mock_success]
    )
    result = await ni.generate_hypothesis({"task": "test"})
    assert result == {"ok": True}
    assert ni._client.chat.completions.create.await_count == 3


@pytest.mark.asyncio
async def test_retry_exhausted():
    ni = NeuralInterface(max_retries=2, base_delay=0.01)
    ni._client = MagicMock()
    _rate_limit_error = type("RateLimitError", (Exception,), {})
    ni._client.chat.completions.create = AsyncMock(side_effect=_rate_limit_error("nope"))
    with pytest.raises(_rate_limit_error):
        await ni.generate_hypothesis({"task": "test"})
    assert ni._client.chat.completions.create.await_count == 2


def test_calculate_backoff_exponential():
    ni = NeuralInterface(base_delay=1.0, max_delay=10.0)
    delays = sorted(ni._calculate_backoff(i) for i in range(5))
    for i in range(1, len(delays)):
        assert delays[i] >= delays[i - 1]


def test_calculate_backoff_bounded():
    ni = NeuralInterface(base_delay=1.0, max_delay=10.0)
    for i in range(10):
        d = ni._calculate_backoff(i)
        assert d <= 11.0


@pytest.mark.asyncio
async def test_execute_request_timeout():
    try:
        from openai import APITimeoutError as _APITimeoutError
    except ImportError:
        pytest.skip("openai not installed")
    ni = NeuralInterface()
    ni._client = MagicMock()
    ni._client.chat.completions.create = AsyncMock(side_effect=TimeoutError())
    request = LLMRequest(messages=[{"role": "user", "content": "test"}], timeout=0.1)
    with pytest.raises(_APITimeoutError):
        await ni._execute_request(request)


@pytest.mark.asyncio
async def test_generate_hypothesis_fallback_no_client():
    ni = NeuralInterface()
    ni._client = None
    result = await ni.generate_hypothesis({"task": "test"})
    assert isinstance(result, dict)
    assert result.get("fallback") is True


# ── EvolutionEngine ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_outcome():
    evo = EvolutionEngine()
    assert len(evo._outcomes) == 0
    await evo.record_outcome(task="t1", hypothesis={"a": 1}, is_successful=True)
    assert len(evo._outcomes) == 1


@pytest.mark.asyncio
async def test_get_stats_empty():
    evo = EvolutionEngine()
    stats = evo.get_stats()
    assert stats["total"] == 0
    assert stats["success_rate"] == 0.0


@pytest.mark.asyncio
async def test_get_stats_success_rate():
    evo = EvolutionEngine()
    await evo.record_outcome(task="t1", hypothesis={"a": 1}, is_successful=True)
    await evo.record_outcome(task="t2", hypothesis={"b": 2}, is_successful=False)
    await evo.record_outcome(task="t3", hypothesis={"c": 3}, is_successful=True)
    stats = evo.get_stats()
    assert stats["total"] == 3
    assert stats["successful"] == 2
    assert stats["failed"] == 1
    assert stats["success_rate"] == 2 / 3


# ── NeuroSymbolicEngine ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_stop():
    engine = NeuroSymbolicEngine()
    assert engine._started is False
    await engine.start()
    assert engine._started is True
    await engine.stop()
    assert engine._started is False


@pytest.mark.asyncio
async def test_session():
    engine = NeuroSymbolicEngine()
    async with engine.session():
        assert engine._started is True
    assert engine._started is False


@pytest.mark.asyncio
async def test_think_raises_if_not_started():
    engine = NeuroSymbolicEngine()
    with pytest.raises(RuntimeError, match="not started"):
        async for _ in engine.think({"requirements": []}):
            pass


@pytest.mark.asyncio
async def test_think_yields_stages():
    engine = NeuroSymbolicEngine()
    await engine.start()
    engine.symbolic.parse_task = AsyncMock(return_value=TaskGraph())
    engine.neural.generate_hypothesis = AsyncMock(return_value={"result": "ok"})
    engine.symbolic.verify_solution = AsyncMock(return_value=(True, []))
    stages = []
    async for event in engine.think({"requirements": []}):
        stages.append(event["stage"])
    assert "parsing" in stages
    assert "hypothesis_generation" in stages
    assert "verification" in stages
    assert "completed" in stages
    assert stages.index("parsing") < stages.index("hypothesis_generation")
    assert stages.index("hypothesis_generation") < stages.index("verification")
    assert stages.index("verification") < stages.index("completed")
    await engine.stop()


@pytest.mark.asyncio
async def test_think_returns_error_on_exception():
    engine = NeuroSymbolicEngine()
    await engine.start()
    engine.symbolic.parse_task = AsyncMock(side_effect=ValueError("parse failure"))
    events = []
    with pytest.raises(ValueError, match="parse failure"):
        async for event in engine.think({"requirements": []}):
            events.append(event)
    error_events = [e for e in events if e.get("stage") == "error"]
    assert len(error_events) == 1
    assert "parse failure" in error_events[0]["error"]
    await engine.stop()


@pytest.mark.asyncio
async def test_get_metrics():
    engine = NeuroSymbolicEngine()
    await engine.start()
    engine.symbolic.parse_task = AsyncMock(return_value=TaskGraph())
    engine.neural.generate_hypothesis = AsyncMock(return_value={"result": "ok"})
    engine.symbolic.verify_solution = AsyncMock(return_value=(True, []))
    async for _ in engine.think({"requirements": []}):
        pass
    metrics = engine.get_metrics()
    assert metrics["tasks_processed"] == 1
    assert metrics["tasks_successful"] == 1
    assert metrics["tasks_failed"] == 0
    assert metrics["total_llm_calls"] == 1
    assert metrics["success_rate"] == 1.0
    await engine.stop()


@pytest.mark.asyncio
async def test_get_metrics_initial():
    engine = NeuroSymbolicEngine()
    metrics = engine.get_metrics()
    assert metrics["tasks_processed"] == 0
    assert metrics["tasks_successful"] == 0
    assert metrics["tasks_failed"] == 0
    assert metrics["success_rate"] == 0.0
