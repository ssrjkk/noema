"""Chaos: Circuit breaker — verify auto-recovery after failures."""

import asyncio
import contextlib

import pytest

from noema.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
)


class TestCircuitBreakerRecovery:
    async def test_circuit_opens_and_recovers(self):
        """Circuit breaker opens after N failures and recovers."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)

        async def failing_func():
            raise ValueError("simulated failure")

        async def succeeding_func():
            return "ok"

        # Exhaust failures → circuit opens
        for _ in range(3):
            with contextlib.suppress(ValueError):
                await cb.execute(failing_func)

        assert cb.state == CircuitState.OPEN, "Should be OPEN after 3 failures"

        # Circuit is OPEN — calls raise CircuitBreakerError immediately
        with pytest.raises(CircuitBreakerError):
            await cb.execute(failing_func)

        # Wait for recovery timeout (state property auto-transitions to HALF_OPEN)
        await asyncio.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN, "Should transition to HALF_OPEN"

        # Half-open succeeds → back to CLOSED
        result = await cb.execute(succeeding_func)
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED, "Should return to CLOSED on success"

    async def test_circuit_stays_open_if_half_open_fails(self):
        """If half-open request fails, circuit stays OPEN."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        async def failing_func():
            raise ValueError("simulated failure")

        # Open the circuit
        for _ in range(2):
            with contextlib.suppress(ValueError):
                await cb.execute(failing_func)

        assert cb.state == CircuitState.OPEN

        # Wait for recovery → HALF_OPEN
        await asyncio.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        # Half-open attempt also fails → reopens (original exception propagates)
        with pytest.raises(ValueError):
            await cb.execute(failing_func)

        assert cb.state == CircuitState.OPEN, "Should stay OPEN if half-open fails"
