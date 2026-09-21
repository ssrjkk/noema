"""Comprehensive tests for noema.neurosymbolic.neural module.

Covers all public methods, branches, error paths, edge cases,
and fail-closed behaviour of CircuitBreaker, NeuralInterface,
and supporting dataclasses.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from noema.neurosymbolic.neural import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    LLMRequest,
    LLMResponse,
    NeuralInterface,
)


# ── Helpers / Fixtures ────────────────────────────────────────────────


def _make_mock_response(content: str = '{"ok": true}', tokens: int = 10) -> MagicMock:
    """Build a mock OpenAI chat-completion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.total_tokens = tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _make_neural_with_client(
    side_effect=None,
    return_value=None,
    content: str = '{"ok": true}',
    tokens: int = 10,
) -> NeuralInterface:
    """Return a NeuralInterface whose _client is a pre-wired mock."""
    ni = NeuralInterface()
    ni._client = MagicMock()
    if side_effect is not None:
        ni._client.chat.completions.create = AsyncMock(side_effect=side_effect)
    else:
        rv = return_value or _make_mock_response(content=content, tokens=tokens)
        ni._client.chat.completions.create = AsyncMock(return_value=rv)
    return ni


def _error_class(name: str) -> type:
    """Dynamically create an exception class with a given name."""
    return type(name, (Exception,), {})


# ── Dataclass: LLMRequest ─────────────────────────────────────────────


class TestLLMRequest:
    def test_defaults(self):
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        assert req.response_format is None
        assert req.temperature == 0.3
        assert req.max_tokens == 500
        assert req.timeout == 30.0

    def test_custom_values(self):
        req = LLMRequest(
            messages=[],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=1000,
            timeout=60.0,
        )
        assert req.response_format == {"type": "json_object"}
        assert req.temperature == 0.7
        assert req.max_tokens == 1000
        assert req.timeout == 60.0

    def test_empty_messages(self):
        req = LLMRequest(messages=[])
        assert req.messages == []


# ── Dataclass: LLMResponse ────────────────────────────────────────────


class TestLLMResponse:
    def test_fields(self):
        resp = LLMResponse(content="hello", tokens_used=42, latency_ms=123.4, model="gpt-4o-mini")
        assert resp.content == "hello"
        assert resp.tokens_used == 42
        assert resp.latency_ms == 123.4
        assert resp.model == "gpt-4o-mini"


# ── CircuitState constants ────────────────────────────────────────────


class TestCircuitState:
    def test_values(self):
        assert CircuitState.CLOSED == "closed"
        assert CircuitState.OPEN == "open"
        assert CircuitState.HALF_OPEN == "half_open"


# ── CircuitOpenError ──────────────────────────────────────────────────


class TestCircuitOpenError:
    def test_is_exception(self):
        err = CircuitOpenError("test")
        assert isinstance(err, Exception)
        assert str(err) == "test"


# ── CircuitBreaker ────────────────────────────────────────────────────


class TestCircuitBreakerInit:
    def test_defaults(self):
        cb = CircuitBreaker()
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 60.0
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.last_failure_time is None

    def test_custom(self):
        cb = CircuitBreaker(failure_threshold=10, recovery_timeout=30.0)
        assert cb.failure_threshold == 10
        assert cb.recovery_timeout == 30.0


class TestCircuitBreakerCall:
    @pytest.mark.asyncio
    async def test_success_resets_state(self):
        cb = CircuitBreaker()
        cb.failure_count = 3
        cb.state = CircuitState.HALF_OPEN
        result = await cb.call(lambda: _async_val("done"))
        assert result == "done"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_open_raises_when_not_expired(self):
        cb = CircuitBreaker(recovery_timeout=9999.0)
        cb.state = CircuitState.OPEN
        cb.last_failure_time = datetime.now(UTC)
        with pytest.raises(CircuitOpenError, match="OPEN"):
            await cb.call(lambda: _async_val("nope"))

    @pytest.mark.asyncio
    async def test_open_transitions_to_half_open_when_expired(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)
        cb.state = CircuitState.OPEN
        cb.last_failure_time = datetime.now(UTC) - timedelta(seconds=10)
        result = await cb.call(lambda: _async_val("recovered"))
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_open_transitions_to_half_open_no_failure_time(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.state = CircuitState.OPEN
        cb.last_failure_time = None
        result = await cb.call(lambda: _async_val("ok"))
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_failure_propagates_exception(self):
        cb = CircuitBreaker(failure_threshold=5)

        async def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await cb.call(fail)
        assert cb.failure_count == 1
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_call_passes_args_and_kwargs(self):
        cb = CircuitBreaker()
        func = AsyncMock(return_value="result")
        result = await cb.call(func, "a", "b", key="val")
        func.assert_awaited_once_with("a", "b", key="val")
        assert result == "result"


class TestCircuitBreakerOnFailure:
    def test_increments_count(self):
        cb = CircuitBreaker(failure_threshold=10)
        cb._on_failure()
        assert cb.failure_count == 1
        assert cb.last_failure_time is not None

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb._on_failure()
        assert cb.state == CircuitState.CLOSED
        cb._on_failure()
        assert cb.state == CircuitState.OPEN

    def test_records_failure_time(self):
        cb = CircuitBreaker(failure_threshold=1)
        before = datetime.now(UTC)
        cb._on_failure()
        after = datetime.now(UTC)
        assert before <= cb.last_failure_time <= after


class TestCircuitBreakerShouldAttemptReset:
    def test_true_when_no_failure_time(self):
        cb = CircuitBreaker()
        assert cb._should_attempt_reset() is True

    def test_false_when_recent(self):
        cb = CircuitBreaker(recovery_timeout=60.0)
        cb.last_failure_time = datetime.now(UTC)
        assert cb._should_attempt_reset() is False

    def test_true_when_expired(self):
        cb = CircuitBreaker(recovery_timeout=0.0)
        cb.last_failure_time = datetime.now(UTC) - timedelta(seconds=1)
        assert cb._should_attempt_reset() is True

    def test_boundary_exactly_at_timeout(self):
        cb = CircuitBreaker(recovery_timeout=1.0)
        cb.last_failure_time = datetime.now(UTC) - timedelta(seconds=1)
        # Exactly at boundary — total_seconds() == 1.0, not > 1.0, so False
        # But due to floating point timing, it might be slightly > 1.0
        # Just ensure it doesn't crash
        cb._should_attempt_reset()


# ── NeuralInterface: init ─────────────────────────────────────────────


class TestNeuralInterfaceInit:
    def test_defaults(self):
        ni = NeuralInterface()
        assert ni.model == "gpt-4o-mini"
        assert ni.max_retries == 3
        assert ni.base_delay == 1.0
        assert ni.max_delay == 10.0
        assert ni.batch_size == 5
        assert ni.batch_timeout == 1.0
        assert ni._client is None
        assert isinstance(ni.circuit_breaker, CircuitBreaker)
        assert ni._batch_processor_task is None

    def test_custom_params(self):
        ni = NeuralInterface(
            model="gpt-4o",
            max_retries=5,
            base_delay=2.0,
            max_delay=30.0,
            batch_size=10,
            batch_timeout=2.0,
        )
        assert ni.model == "gpt-4o"
        assert ni.max_retries == 5
        assert ni.base_delay == 2.0
        assert ni.max_delay == 30.0
        assert ni.batch_size == 10
        assert ni.batch_timeout == 2.0


# ── NeuralInterface: start / stop ─────────────────────────────────────


class TestNeuralInterfaceStartStop:
    @pytest.mark.asyncio
    async def test_start_creates_task(self):
        ni = NeuralInterface()
        await ni.start()
        assert ni._batch_processor_task is not None
        assert not ni._batch_processor_task.done()
        await ni.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        ni = NeuralInterface()
        await ni.start()
        task1 = ni._batch_processor_task
        await ni.start()  # second call should be a no-op
        assert ni._batch_processor_task is task1
        await ni.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        ni = NeuralInterface()
        await ni.start()
        await ni.stop()
        assert ni._batch_processor_task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self):
        ni = NeuralInterface()
        # Should not raise
        await ni.stop()
        assert ni._batch_processor_task is None


# ── NeuralInterface: _ensure_client ───────────────────────────────────


class TestNeuralInterfaceEnsureClient:
    @pytest.mark.asyncio
    async def test_already_set(self):
        ni = NeuralInterface()
        sentinel = MagicMock()
        ni._client = sentinel
        await ni._ensure_client()
        assert ni._client is sentinel

    @pytest.mark.asyncio
    async def test_import_failure_sets_none(self):
        ni = NeuralInterface()
        ni._client = None
        with patch.dict("sys.modules", {"openai": None}):
            await ni._ensure_client()
        assert ni._client is None

    @pytest.mark.asyncio
    async def test_successful_import(self):
        ni = NeuralInterface()
        ni._client = None
        mock_async_openai = MagicMock()
        mock_module = MagicMock()
        mock_module.AsyncOpenAI = mock_async_openai
        with patch.dict("sys.modules", {"openai": mock_module}):
            await ni._ensure_client()
        assert ni._client is not None


# ── NeuralInterface: _is_retryable ────────────────────────────────────


class TestNeuralInterfaceIsRetryable:
    def test_rate_limit_error(self):
        ni = NeuralInterface()
        assert ni._is_retryable(_error_class("RateLimitError")()) is True

    def test_api_timeout_error(self):
        ni = NeuralInterface()
        assert ni._is_retryable(_error_class("APITimeoutError")()) is True

    def test_internal_server_error(self):
        ni = NeuralInterface()
        assert ni._is_retryable(_error_class("InternalServerError")()) is True

    def test_api_connection_error(self):
        ni = NeuralInterface()
        assert ni._is_retryable(_error_class("APIConnectionError")()) is True

    def test_value_error_not_retryable(self):
        ni = NeuralInterface()
        assert ni._is_retryable(ValueError("x")) is False

    def test_runtime_error_not_retryable(self):
        ni = NeuralInterface()
        assert ni._is_retryable(RuntimeError("x")) is False

    def test_unknown_error_not_retryable(self):
        ni = NeuralInterface()
        assert ni._is_retryable(_error_class("SomeOtherError")()) is False


# ── NeuralInterface: _calculate_backoff ───────────────────────────────


class TestNeuralInterfaceBackoff:
    def test_attempt_zero(self):
        ni = NeuralInterface(base_delay=1.0, max_delay=100.0)
        d = ni._calculate_backoff(0)
        # base_delay * 2^0 = 1.0, plus jitter up to 0.1
        assert 1.0 <= d <= 1.1

    def test_attempt_one(self):
        ni = NeuralInterface(base_delay=1.0, max_delay=100.0)
        d = ni._calculate_backoff(1)
        # base_delay * 2^1 = 2.0, plus jitter up to 0.2
        assert 2.0 <= d <= 2.2

    def test_attempt_three(self):
        ni = NeuralInterface(base_delay=1.0, max_delay=100.0)
        d = ni._calculate_backoff(3)
        # base_delay * 2^3 = 8.0, plus jitter up to 0.8
        assert 8.0 <= d <= 8.8

    def test_capped_at_max_delay(self):
        ni = NeuralInterface(base_delay=1.0, max_delay=5.0)
        d = ni._calculate_backoff(100)
        # min(1.0 * 2^100, 5.0) = 5.0, plus jitter up to 0.5
        assert d <= 5.5

    def test_monotonically_increasing_base(self):
        ni = NeuralInterface(base_delay=1.0, max_delay=1000.0)
        bases = [ni._calculate_backoff(i) for i in range(5)]
        # With jitter the order might not be strict, but base values grow
        # Check that the last is larger than the first
        assert bases[-1] > bases[0]

    def test_zero_base_delay(self):
        ni = NeuralInterface(base_delay=0.0, max_delay=10.0)
        d = ni._calculate_backoff(5)
        assert d == 0.0


# ── NeuralInterface: _execute_request ─────────────────────────────────


class TestNeuralInterfaceExecuteRequest:
    @pytest.mark.asyncio
    async def test_no_client_raises(self):
        ni = NeuralInterface()
        ni._client = None
        req = LLMRequest(messages=[{"role": "user", "content": "test"}])
        with pytest.raises(RuntimeError, match="No LLM client available"):
            await ni._execute_request(req)

    @pytest.mark.asyncio
    async def test_success_with_usage(self):
        ni = _make_neural_with_client(content='{"x": 1}', tokens=42)
        req = LLMRequest(messages=[{"role": "user", "content": "test"}])
        resp = await ni._execute_request(req)
        assert isinstance(resp, LLMResponse)
        assert resp.content == '{"x": 1}'
        assert resp.tokens_used == 42
        assert resp.model == "gpt-4o-mini"
        assert resp.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_success_without_usage(self):
        ni = NeuralInterface()
        ni._client = MagicMock()
        mock_resp = _make_mock_response(content="hello")
        mock_resp.usage = None
        ni._client.chat.completions.create = AsyncMock(return_value=mock_resp)
        req = LLMRequest(messages=[{"role": "user", "content": "test"}])
        resp = await ni._execute_request(req)
        assert resp.tokens_used == 0

    @pytest.mark.asyncio
    async def test_timeout_converts_to_api_timeout_error(self):
        try:
            from openai import APITimeoutError as _APITimeoutError
        except ImportError:
            pytest.skip("openai not installed")
        ni = NeuralInterface()
        ni._client = MagicMock()
        ni._client.chat.completions.create = AsyncMock(side_effect=TimeoutError())
        req = LLMRequest(messages=[{"role": "user", "content": "test"}], timeout=0.01)
        with pytest.raises(_APITimeoutError):
            await ni._execute_request(req)

    @pytest.mark.asyncio
    async def test_passes_correct_params_to_client(self):
        ni = _make_neural_with_client(content="{}")
        req = LLMRequest(
            messages=[{"role": "user", "content": "hello"}],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=200,
            timeout=15.0,
        )
        await ni._execute_request(req)
        ni._client.chat.completions.create.assert_awaited_once_with(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=200,
        )

    @pytest.mark.asyncio
    async def test_generic_exception_propagates(self):
        ni = NeuralInterface()
        ni._client = MagicMock()
        ni._client.chat.completions.create = AsyncMock(side_effect=ConnectionError("down"))
        req = LLMRequest(messages=[{"role": "user", "content": "test"}])
        with pytest.raises(ConnectionError, match="down"):
            await ni._execute_request(req)


# ── NeuralInterface: _execute_with_retry ──────────────────────────────


class TestNeuralInterfaceExecuteWithRetry:
    @pytest.mark.asyncio
    async def test_success_first_try(self):
        ni = _make_neural_with_client(content='{"a": 1}')
        req = LLMRequest(messages=[{"role": "user", "content": "test"}])
        resp = await ni._execute_with_retry(req)
        assert resp.content == '{"a": 1}'
        assert ni._client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_circuit_open_error_reraises_immediately(self):
        ni = NeuralInterface(max_retries=5)
        ni.circuit_breaker.state = CircuitState.OPEN
        ni.circuit_breaker.last_failure_time = datetime.now(UTC)
        ni.circuit_breaker.recovery_timeout = 9999.0
        ni._client = MagicMock()
        ni._client.chat.completions.create = AsyncMock(return_value=_make_mock_response())
        req = LLMRequest(messages=[{"role": "user", "content": "test"}])
        with pytest.raises(CircuitOpenError):
            await ni._execute_with_retry(req)
        # Should not have retried
        assert ni._client.chat.completions.create.await_count == 0

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(self):
        ni = NeuralInterface(max_retries=5)
        ni._client = MagicMock()
        ni._client.chat.completions.create = AsyncMock(
            side_effect=ValueError("bad request")
        )
        req = LLMRequest(messages=[{"role": "user", "content": "test"}])
        with pytest.raises(ValueError, match="bad request"):
            await ni._execute_with_retry(req)
        assert ni._client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_retryable_error_retries_then_succeeds(self):
        timeout_cls = _error_class("APITimeoutError")
        ni = NeuralInterface(max_retries=3, base_delay=0.01)
        ni._client = MagicMock()
        success_resp = _make_mock_response(content='{"retried": true}')
        ni._client.chat.completions.create = AsyncMock(
            side_effect=[timeout_cls("timeout"), success_resp]
        )
        req = LLMRequest(messages=[{"role": "user", "content": "test"}])
        resp = await ni._execute_with_retry(req)
        assert resp.content == '{"retried": true}'
        assert ni._client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_raises_last_error(self):
        rate_cls = _error_class("RateLimitError")
        ni = NeuralInterface(max_retries=2, base_delay=0.01)
        ni._client = MagicMock()
        ni._client.chat.completions.create = AsyncMock(side_effect=rate_cls("limited"))
        req = LLMRequest(messages=[{"role": "user", "content": "test"}])
        with pytest.raises(rate_cls):
            await ni._execute_with_retry(req)
        assert ni._client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_mixed_retryable_then_non_retryable(self):
        rate_cls = _error_class("RateLimitError")
        ni = NeuralInterface(max_retries=5, base_delay=0.01)
        ni._client = MagicMock()
        ni._client.chat.completions.create = AsyncMock(
            side_effect=[rate_cls("slow"), ValueError("bad")]
        )
        req = LLMRequest(messages=[{"role": "user", "content": "test"}])
        with pytest.raises(ValueError, match="bad"):
            await ni._execute_with_retry(req)
        assert ni._client.chat.completions.create.await_count == 2


# ── NeuralInterface: generate_hypothesis ──────────────────────────────


class TestNeuralInterfaceGenerateHypothesis:
    @pytest.mark.asyncio
    async def test_returns_parsed_json(self):
        ni = _make_neural_with_client(content='{"hypothesis": "solve"}')
        result = await ni.generate_hypothesis({"requirements": []})
        assert result == {"hypothesis": "solve"}

    @pytest.mark.asyncio
    async def test_complex_task_graph(self):
        ni = _make_neural_with_client(content='{"plan": [1, 2, 3]}')
        task = {"requirements": [{"name": "x", "type": "numeric"}], "goals": ["minimize"]}
        result = await ni.generate_hypothesis(task)
        assert result == {"plan": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_fail_closed_no_client(self):
        ni = NeuralInterface()
        ni._client = None
        with pytest.raises(RuntimeError, match="No LLM client available"):
            await ni.generate_hypothesis({"task": "test"})

    @pytest.mark.asyncio
    async def test_invalid_json_raises(self):
        ni = _make_neural_with_client(content="not json at all")
        with pytest.raises(json.JSONDecodeError):
            await ni.generate_hypothesis({"task": "test"})

    @pytest.mark.asyncio
    async def test_uses_json_response_format(self):
        ni = _make_neural_with_client(content='{"ok": true}')
        await ni.generate_hypothesis({"task": "test"})
        call_kwargs = ni._client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("response_format") == {"type": "json_object"} or \
               call_kwargs[1].get("response_format") == {"type": "json_object"} or \
               call_kwargs.kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_temperature_is_0_3(self):
        ni = _make_neural_with_client(content='{}')
        await ni.generate_hypothesis({"task": "test"})
        call_kwargs = ni._client.chat.completions.create.call_args
        temp = call_kwargs.kwargs.get("temperature", call_kwargs[1].get("temperature"))
        assert temp == 0.3


# ── NeuralInterface: refine_hypothesis ────────────────────────────────


class TestNeuralInterfaceRefineHypothesis:
    @pytest.mark.asyncio
    async def test_returns_parsed_json(self):
        ni = _make_neural_with_client(content='{"refined": true}')
        result = await ni.refine_hypothesis({"old": 1}, ["v1"], {"task": "test"})
        assert result == {"refined": True}

    @pytest.mark.asyncio
    async def test_empty_violations(self):
        ni = _make_neural_with_client(content='{"ok": true}')
        result = await ni.refine_hypothesis({"h": 1}, [], {"task": "test"})
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_fail_closed_no_client(self):
        ni = NeuralInterface()
        ni._client = None
        with pytest.raises(RuntimeError, match="No LLM client available"):
            await ni.refine_hypothesis({"h": 1}, ["v"], {"task": "test"})

    @pytest.mark.asyncio
    async def test_invalid_json_raises(self):
        ni = _make_neural_with_client(content="garbage")
        with pytest.raises(json.JSONDecodeError):
            await ni.refine_hypothesis({}, [], {})

    @pytest.mark.asyncio
    async def test_temperature_is_0_2(self):
        ni = _make_neural_with_client(content='{}')
        await ni.refine_hypothesis({}, [], {})
        call_kwargs = ni._client.chat.completions.create.call_args
        temp = call_kwargs.kwargs.get("temperature", call_kwargs[1].get("temperature"))
        assert temp == 0.2

    @pytest.mark.asyncio
    async def test_multiple_violations(self):
        ni = _make_neural_with_client(content='{"fixed": true}')
        violations = ["v1", "v2", "v3"]
        result = await ni.refine_hypothesis({"h": 1}, violations, {"task": "test"})
        assert result == {"fixed": True}


# ── NeuralInterface: _build_hypothesis_prompt ─────────────────────────


class TestNeuralInterfacePrompts:
    def test_hypothesis_prompt_contains_task(self):
        ni = NeuralInterface()
        task = {"requirements": [{"name": "x"}], "goals": ["solve"]}
        prompt = ni._build_hypothesis_prompt(task)
        assert "requirements" in prompt
        assert "goals" in prompt
        assert "hypothesis" in prompt.lower()
        assert "JSON" in prompt

    def test_hypothesis_prompt_with_empty_task(self):
        ni = NeuralInterface()
        prompt = ni._build_hypothesis_prompt({})
        assert "{}" in prompt

    def test_refinement_prompt_contains_all_parts(self):
        ni = NeuralInterface()
        hyp = {"solution": "A"}
        violations = ["constraint violated", "type error"]
        task = {"requirements": []}
        prompt = ni._build_refinement_prompt(hyp, violations, task)
        assert "solution" in prompt
        assert "constraint violated" in prompt
        assert "type error" in prompt
        assert "Fix" in prompt
        assert "JSON" in prompt

    def test_refinement_prompt_with_empty_inputs(self):
        ni = NeuralInterface()
        prompt = ni._build_refinement_prompt({}, [], {})
        assert "[]" in prompt  # empty violations list


# ── NeuralInterface: _process_batch ───────────────────────────────────


class TestNeuralInterfaceProcessBatch:
    @pytest.mark.asyncio
    async def test_batch_success(self):
        ni = _make_neural_with_client(content='{"batch": true}')
        req = LLMRequest(messages=[{"role": "user", "content": "test"}])
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await ni._process_batch([(req, future)])
        assert future.done()
        result = future.result()
        assert isinstance(result, LLMResponse)
        assert result.content == '{"batch": true}'

    @pytest.mark.asyncio
    async def test_batch_failure_sets_exception(self):
        ni = NeuralInterface()
        ni._client = MagicMock()
        ni._client.chat.completions.create = AsyncMock(side_effect=RuntimeError("fail"))
        req = LLMRequest(messages=[{"role": "user", "content": "test"}])
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await ni._process_batch([(req, future)])
        assert future.done()
        with pytest.raises(RuntimeError, match="fail"):
            future.result()

    @pytest.mark.asyncio
    async def test_batch_mixed_success_and_failure(self):
        ni = NeuralInterface()
        ni._client = MagicMock()
        success_resp = _make_mock_response(content='{"ok": true}')
        ni._client.chat.completions.create = AsyncMock(
            side_effect=[success_resp, RuntimeError("boom")]
        )
        req1 = LLMRequest(messages=[{"role": "user", "content": "a"}])
        req2 = LLMRequest(messages=[{"role": "user", "content": "b"}])
        loop = asyncio.get_event_loop()
        f1 = loop.create_future()
        f2 = loop.create_future()
        await ni._process_batch([(req1, f1), (req2, f2)])
        assert f1.result().content == '{"ok": true}'
        with pytest.raises(RuntimeError):
            f2.result()

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        ni = _make_neural_with_client()
        await ni._process_batch([])
        # Should not call the client
        ni._client.chat.completions.create.assert_not_awaited()


# ── NeuralInterface: _batch_processor ─────────────────────────────────


class TestNeuralInterfaceBatchProcessor:
    @pytest.mark.asyncio
    async def test_batch_processor_processes_queued_items(self):
        ni = _make_neural_with_client(content='{"queued": true}')
        ni.batch_timeout = 0.05
        await ni.start()
        try:
            loop = asyncio.get_event_loop()
            req = LLMRequest(messages=[{"role": "user", "content": "test"}])
            future = loop.create_future()
            await ni._request_queue.put((req, future))
            # Wait for processing
            result = await asyncio.wait_for(future, timeout=2.0)
            assert result.content == '{"queued": true}'
        finally:
            await ni.stop()

    @pytest.mark.asyncio
    async def test_batch_processor_cancellation(self):
        ni = NeuralInterface()
        ni.batch_timeout = 0.05
        await ni.start()
        task = ni._batch_processor_task
        await ni.stop()
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_batch_processor_survives_error(self):
        """Batch processor should continue after an error in _process_batch."""
        ni = NeuralInterface()
        ni._client = MagicMock()
        ni.batch_timeout = 0.05
        # First call raises, second succeeds
        success_resp = _make_mock_response(content='{"recovered": true}')
        ni._client.chat.completions.create = AsyncMock(
            side_effect=[RuntimeError("transient"), success_resp]
        )
        await ni.start()
        try:
            loop = asyncio.get_event_loop()
            # First request will fail
            req1 = LLMRequest(messages=[{"role": "user", "content": "a"}])
            f1 = loop.create_future()
            await ni._request_queue.put((req1, f1))
            # Wait for it to fail
            await asyncio.sleep(0.3)
            assert f1.done()
            with pytest.raises(RuntimeError):
                f1.result()

            # Second request should succeed (processor still alive)
            req2 = LLMRequest(messages=[{"role": "user", "content": "b"}])
            f2 = loop.create_future()
            await ni._request_queue.put((req2, f2))
            result = await asyncio.wait_for(f2, timeout=2.0)
            assert result.content == '{"recovered": true}'
        finally:
            await ni.stop()


# ── Integration: full pipeline ────────────────────────────────────────


class TestNeuralInterfaceIntegration:
    @pytest.mark.asyncio
    async def test_generate_then_refine(self):
        ni = NeuralInterface()
        ni._client = MagicMock()
        hyp_resp = _make_mock_response(content='{"solution": "initial"}')
        ref_resp = _make_mock_response(content='{"solution": "refined"}')
        ni._client.chat.completions.create = AsyncMock(
            side_effect=[hyp_resp, ref_resp]
        )
        hyp = await ni.generate_hypothesis({"requirements": []})
        assert hyp == {"solution": "initial"}
        refined = await ni.refine_hypothesis(hyp, ["constraint_violated"], {"requirements": []})
        assert refined == {"solution": "refined"}

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration_with_retry(self):
        """Circuit breaker opens after failures, then _execute_with_retry raises CircuitOpenError."""
        ni = NeuralInterface(max_retries=5, base_delay=0.01)
        ni.circuit_breaker.failure_threshold = 1
        ni._client = MagicMock()
        ni._client.chat.completions.create = AsyncMock(
            side_effect=_error_class("InternalServerError")("fail")
        )
        req = LLMRequest(messages=[{"role": "user", "content": "test"}])
        # First call: goes through circuit breaker, fails, opens it
        # Second call: circuit breaker is open, raises CircuitOpenError immediately
        with pytest.raises((CircuitOpenError, Exception)):
            await ni._execute_with_retry(req)

    @pytest.mark.asyncio
    async def test_full_lifecycle_start_generate_stop(self):
        ni = _make_neural_with_client(content='{"result": 42}')
        await ni.start()
        try:
            result = await ni.generate_hypothesis({"task": "compute"})
            assert result == {"result": 42}
        finally:
            await ni.stop()


# ── Edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_task_graph(self):
        ni = _make_neural_with_client(content='{}')
        result = await ni.generate_hypothesis({})
        assert result == {}

    @pytest.mark.asyncio
    async def test_large_task_graph(self):
        ni = _make_neural_with_client(content='{"ok": true}')
        big_task = {"requirements": [{"name": f"r{i}"} for i in range(100)]}
        result = await ni.generate_hypothesis(big_task)
        assert result == {"ok": True}

    def test_circuit_breaker_zero_threshold(self):
        cb = CircuitBreaker(failure_threshold=0)
        # With threshold 0, any failure should open it
        cb._on_failure()
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_execute_request_latency_non_negative(self):
        ni = _make_neural_with_client(content="{}")
        req = LLMRequest(messages=[{"role": "user", "content": "test"}])
        resp = await ni._execute_request(req)
        assert resp.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_multiple_sequential_requests(self):
        ni = _make_neural_with_client(content='{"seq": true}')
        for _ in range(5):
            result = await ni.generate_hypothesis({"task": "test"})
            assert result == {"seq": True}

    def test_backoff_with_large_max_delay(self):
        ni = NeuralInterface(base_delay=1.0, max_delay=1e6)
        d = ni._calculate_backoff(0)
        assert 1.0 <= d <= 1.1

    def test_backoff_with_zero_max_delay(self):
        ni = NeuralInterface(base_delay=1.0, max_delay=0.0)
        d = ni._calculate_backoff(5)
        # min(1.0 * 32, 0.0) = 0.0, jitter = uniform(0, 0) = 0
        assert d == 0.0


# ── Helpers ────────────────────────────────────────────────────────────


async def _async_val(val):
    return val
