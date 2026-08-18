from __future__ import annotations

import asyncio
import contextlib
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar, cast

import structlog

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = structlog.get_logger(__name__)

_R = TypeVar("_R")


@dataclass
class LLMRequest:
    messages: list[dict]
    response_format: dict | None = None
    temperature: float = 0.3
    max_tokens: int = 500
    timeout: float = 30.0


@dataclass
class LLMResponse:
    content: str
    tokens_used: int
    latency_ms: float
    model: str


class CircuitState:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open and calls are rejected."""


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: datetime | None = None

    async def call(
        self,
        func: Callable[..., Awaitable[_R]],
        *args: Any,
        **kwargs: Any,
    ) -> _R:
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        if not self.last_failure_time:
            return True
        return (datetime.now(UTC) - self.last_failure_time).total_seconds() > self.recovery_timeout

    def _on_success(self) -> None:
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = datetime.now(UTC)

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning("circuit_breaker_opened", failure_count=self.failure_count)


class NeuralInterface:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        batch_size: int = 5,
        batch_timeout: float = 1.0,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

        self._client: Any = None
        self.circuit_breaker = CircuitBreaker()
        self._request_queue: asyncio.Queue[tuple[LLMRequest, asyncio.Future]] = asyncio.Queue()
        self._batch_processor_task: asyncio.Task | None = None

        logger.info(
            "neural_interface_initialized",
            model=model,
            max_retries=max_retries,
            batch_size=batch_size,
        )

    async def start(self) -> None:
        if not self._batch_processor_task:
            self._batch_processor_task = asyncio.create_task(self._batch_processor())
            logger.info("batch_processor_started")

    async def stop(self) -> None:
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._batch_processor_task
            logger.info("batch_processor_stopped")

    async def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI()
        except Exception as e:  # noqa: BLE001 - any client failure degrades to fallback
            logger.warning(
                "openai_unavailable",
                error=str(e),
                error_type=type(e).__name__,
            )
            self._client = None

    async def generate_hypothesis(self, task_graph: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_client()
        prompt = self._build_hypothesis_prompt(task_graph)

        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=500,
        )

        response = await self._execute_with_retry(request)
        return cast("dict[str, Any]", json.loads(response.content))

    async def refine_hypothesis(
        self, hypothesis: dict[str, Any], violations: list[str], task_graph: dict[str, Any]
    ) -> dict[str, Any]:
        await self._ensure_client()
        prompt = self._build_refinement_prompt(hypothesis, violations, task_graph)

        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=500,
        )

        response = await self._execute_with_retry(request)
        return cast("dict[str, Any]", json.loads(response.content))

    async def _execute_with_retry(self, request: LLMRequest) -> LLMResponse:
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = await self.circuit_breaker.call(self._execute_request, request)

                logger.info(
                    "llm_request_completed",
                    attempt=attempt + 1,
                    tokens_used=response.tokens_used,
                    latency_ms=response.latency_ms,
                )

                return response

            except CircuitOpenError:
                logger.error("circuit_breaker_open", attempt=attempt + 1)
                raise

            except Exception as e:
                if isinstance(e, CircuitOpenError):
                    raise
                last_error = e
                if self._is_retryable(e):
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        "llm_request_retry",
                        attempt=attempt + 1,
                        error_type=type(e).__name__,
                        delay=delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("llm_api_error", error=str(e), error_type=type(e).__name__)
                    raise

        logger.error(
            "llm_request_failed_all_retries",
            max_retries=self.max_retries,
            last_error=str(last_error),
        )
        raise last_error or Exception("LLM request failed")

    def _is_retryable(self, error: Exception) -> bool:
        name = type(error).__name__
        return name in (
            "RateLimitError",
            "APITimeoutError",
            "InternalServerError",
            "APIConnectionError",
        )

    async def _execute_request(self, request: LLMRequest) -> LLMResponse:
        if self._client is None:
            return LLMResponse(
                content=json.dumps({"error": "No LLM client available", "fallback": True}),
                tokens_used=0,
                latency_ms=0,
                model=self.model,
            )

        start_time = asyncio.get_event_loop().time()

        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self.model,
                    messages=request.messages,
                    response_format=request.response_format,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ),
                timeout=request.timeout,
            )

            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            return LLMResponse(
                content=response.choices[0].message.content,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                latency_ms=latency_ms,
                model=self.model,
            )

        except TimeoutError:
            logger.error("llm_request_timeout", timeout=request.timeout)
            from openai import APITimeoutError as _APITimeoutError

            _fake_request = cast("Any", None)
            raise _APITimeoutError(_fake_request) from None

    async def _batch_processor(self) -> None:
        while True:
            try:
                batch: list[tuple[LLMRequest, asyncio.Future]] = []

                while len(batch) < self.batch_size:
                    try:
                        request, future = await asyncio.wait_for(
                            self._request_queue.get(), timeout=self.batch_timeout
                        )
                        batch.append((request, future))
                    except TimeoutError:
                        break

                if batch:
                    await self._process_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("batch_processor_error", error=str(e))
                await asyncio.sleep(1.0)

    async def _process_batch(self, batch: list[tuple[LLMRequest, asyncio.Future]]) -> None:
        for request, future in batch:
            try:
                response = await self._execute_request(request)
                future.set_result(response)
            except Exception as e:
                future.set_exception(e)

    def _calculate_backoff(self, attempt: int) -> float:
        delay = min(self.base_delay * (2**attempt), self.max_delay)
        jitter = random.uniform(0, delay * 0.1)
        return cast("float", delay) + jitter

    def _build_hypothesis_prompt(self, task_graph: dict[str, Any]) -> str:
        return f"""Task in formal form:
{json.dumps(task_graph, indent=2)}

Generate a hypothesis solution.
Return ONLY valid JSON."""

    def _build_refinement_prompt(
        self, hypothesis: dict[str, Any], violations: list[str], task_graph: dict[str, Any]
    ) -> str:
        return f"""Current hypothesis:
{json.dumps(hypothesis, indent=2)}

Violations:
{json.dumps(violations, indent=2)}

Task:
{json.dumps(task_graph, indent=2)}

Fix the hypothesis.
Return ONLY valid JSON."""
