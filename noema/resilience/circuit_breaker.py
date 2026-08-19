"""Resilience patterns: Circuit Breaker, Retry with Exponential Backoff."""

from __future__ import annotations

import asyncio
import random
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar, cast

from noema.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

log = get_logger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit is open."""

    def __init__(self, recovery_at: float) -> None:
        self.recovery_at = recovery_at
        remaining = max(0, recovery_at - time.monotonic())
        super().__init__(f"Circuit open, retry in {remaining:.1f}s")


class CircuitBreaker:
    """Circuit breaker with configurable thresholds.

    States:
    - CLOSED: calls pass through. Failures increment counter.
    - OPEN: calls rejected immediately. After recovery_timeout → HALF_OPEN.
    - HALF_OPEN: one test call allowed. Success → CLOSED, failure → OPEN.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max: int = 1,
        name: str = "default",
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        if (
            self._state == CircuitState.OPEN
            and time.monotonic() - self._last_failure_time >= self.recovery_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            log.info("circuit_half_open", circuit=self.name)
        return self._state

    def _on_success(self) -> None:
        self._failure_count = 0
        self._success_count += 1
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            log.info("circuit_closed", circuit=self.name)

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        self._success_count = 0

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            log.warning("circuit_reopened", circuit=self.name)
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            log.warning(
                "circuit_opened",
                circuit=self.name,
                failures=self._failure_count,
                recovery_s=self.recovery_timeout,
            )

    async def execute(
        self, func: Callable[..., Coroutine[Any, Any, T]], *args: Any, **kwargs: Any
    ) -> T:
        """Execute function through the circuit breaker."""
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitBreakerError(self._last_failure_time + self.recovery_timeout)

        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max:
                raise CircuitBreakerError(self._last_failure_time + self.recovery_timeout)
            self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except CircuitBreakerError:
            raise
        except Exception:
            self._on_failure()
            raise

    def reset(self) -> None:
        """Force reset to CLOSED."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }


class RetryPolicy:
    """Exponential backoff with jitter.

    Delay = min(base * 2^attempt + jitter, max_delay)
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        jitter: float = 0.5,
        retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
        non_retryable_exceptions: tuple[type[Exception], ...] = (),
        name: str = "default",
    ) -> None:
        self.name = name
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
        self.non_retryable_exceptions = non_retryable_exceptions

    def _calc_delay(self, attempt: int) -> float:
        delay = self.base_delay * (2**attempt)
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        return cast("float", min(max(delay, 0.1), self.max_delay))

    async def execute(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute with retries. Raises last exception if all retries fail."""
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except self.retryable_exceptions as exc:
                if isinstance(exc, self.non_retryable_exceptions):
                    raise
                last_exc = exc

                if attempt < self.max_retries:
                    delay = self._calc_delay(attempt)
                    log.warning(
                        "retry_attempt",
                        policy=self.name,
                        attempt=attempt + 1,
                        max_retries=self.max_retries,
                        delay_s=round(delay, 2),
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
                else:
                    log.error(
                        "retry_exhausted",
                        policy=self.name,
                        attempts=attempt + 1,
                        error=str(exc),
                    )

        raise last_exc  # type: ignore[misc]

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
        }


class ResilientExecutor:
    """Combines CircuitBreaker + RetryPolicy for LLM calls."""

    def __init__(
        self,
        circuit_breaker: CircuitBreaker | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.circuit = circuit_breaker or CircuitBreaker()
        self.retry = retry_policy or RetryPolicy()

    async def execute(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute with circuit breaker wrapping retry."""

        async def _protected() -> T:
            return await self.circuit.execute(func, *args, **kwargs)

        return await self.retry.execute(_protected)

    def stats(self) -> dict[str, Any]:
        return {
            "circuit": self.circuit.stats(),
            "retry": self.retry.stats(),
        }
