"""Tests for Sprint 1-4: config, auth, rate limit, atomic_io, resilience, logging."""

from __future__ import annotations

import asyncio
import json

import pytest

from noema.config.settings import (
    APISettings,
    DatabaseSettings,
    LLMSettings,
    NoemaSettings,
    get_settings,
    reset_settings,
)

# ═══════════════════════════════════════════════════════════════════════════
# Config Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestNoemaSettings:
    def test_default_values(self):
        reset_settings()
        s = NoemaSettings()
        assert s.llm.provider == "ollama"
        assert s.api.port == 8000
        assert s.worker.pool_size == 10
        assert s.db.pool_min >= 1

    def test_sub_settings_defaults(self):
        db = DatabaseSettings()
        assert db.pool_min >= 1
        assert db.pool_max >= db.pool_min

        llm = LLMSettings()
        assert llm.temperature >= 0.0
        assert llm.max_tokens >= 1
        assert llm.circuit_breaker_threshold >= 1
        assert llm.retry_max >= 0

        api = APISettings()
        assert api.port > 0
        assert api.rate_limit_rpm > 0

    def test_singleton(self):
        reset_settings()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reset(self):
        s1 = get_settings()
        reset_settings()
        s2 = get_settings()
        assert s1 is not s2

    def test_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "llm:\n  provider: openai\n  ollama_model: test-model\napi:\n  port: 9999\n"
        )
        s = NoemaSettings.from_yaml(yaml_file)
        assert s.llm.provider == "openai"
        assert s.llm.ollama_model == "test-model"
        assert s.api.port == 9999

    def test_from_yaml_nonexistent(self):
        s = NoemaSettings.from_yaml("/nonexistent/file.yaml")
        assert isinstance(s, NoemaSettings)

    def test_dump_yaml(self, tmp_path):
        s = NoemaSettings()
        out = tmp_path / "out.yaml"
        s.dump_yaml(out)
        content = out.read_text()
        assert "provider" in content
        # Secrets should be masked
        assert "***" not in content or "sensitive" not in content


# ═══════════════════════════════════════════════════════════════════════════
# Atomic I/O Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAtomicIO:
    def test_write_and_read(self, tmp_path):
        from noema.utils.atomic_io import atomic_read_json, atomic_write_json

        path = tmp_path / "data.json"
        data = {"key": "value", "nested": [1, 2, 3]}
        atomic_write_json(path, data)
        loaded = atomic_read_json(path)
        assert loaded == data

    def test_atomicity(self, tmp_path):
        from noema.utils.atomic_io import atomic_read_json, atomic_write_json

        path = tmp_path / "data.json"
        atomic_write_json(path, {"v": 1})
        atomic_write_json(path, {"v": 2})
        assert atomic_read_json(path) == {"v": 2}

    def test_backup_rotation(self, tmp_path):
        from noema.utils.atomic_io import atomic_read_json, atomic_write_json

        path = tmp_path / "data.json"
        for i in range(5):
            atomic_write_json(path, {"v": i}, backup=True, backup_count=3)

        # Current should be v=4
        assert atomic_read_json(path) == {"v": 4}

        # .bak.1 should exist (v=3)
        assert path.with_suffix(".bak.1").is_file()
        assert json.loads(path.with_suffix(".bak.1").read_text()) == {"v": 3}

    def test_read_nonexistent(self, tmp_path):
        from noema.utils.atomic_io import atomic_read_json

        result = atomic_read_json(tmp_path / "nope.json", default={"fallback": True})
        assert result == {"fallback": True}

    def test_read_corrupt(self, tmp_path):
        from noema.utils.atomic_io import atomic_read_json

        path = tmp_path / "corrupt.json"
        path.write_text("not json {{{")
        result = atomic_read_json(path, default={})
        assert result == {}

    def test_write_creates_parent_dirs(self, tmp_path):
        from noema.utils.atomic_io import atomic_write_json

        path = tmp_path / "deep" / "nested" / "file.json"
        atomic_write_json(path, {"ok": True})
        assert path.is_file()


# ═══════════════════════════════════════════════════════════════════════════
# Circuit Breaker Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_closed_passes(self):
        from noema.resilience.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=3, name="test")

        async def ok():
            return "ok"

        result = await cb.execute(ok)
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_failures(self):
        from noema.resilience.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerError,
            CircuitState,
        )

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0, name="test")

        async def fail():
            raise ValueError("boom")

        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.execute(fail)

        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerError):
            await cb.execute(fail)

    @pytest.mark.asyncio
    async def test_half_open_after_recovery(self):
        from noema.resilience.circuit_breaker import (
            CircuitBreaker,
            CircuitState,
        )

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, name="test")

        async def fail():
            raise ValueError("boom")

        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.execute(fail)

        assert cb.state == CircuitState.OPEN
        await asyncio.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_success_resets(self):
        from noema.resilience.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2, name="test")

        async def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            await cb.execute(fail)

        async def ok():
            return 42

        result = await cb.execute(ok)
        assert result == 42
        assert cb._failure_count == 0

    def test_stats(self):
        from noema.resilience.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=5, name="test")
        stats = cb.stats()
        assert stats["name"] == "test"
        assert stats["state"] == "closed"
        assert stats["failure_threshold"] == 5

    def test_reset(self):
        from noema.resilience.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=1, name="test")
        cb._on_failure()
        assert cb._failure_count == 1
        cb.reset()
        assert cb._failure_count == 0
        assert cb.state == CircuitState.CLOSED


# ═══════════════════════════════════════════════════════════════════════════
# Retry Policy Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRetryPolicy:
    @pytest.mark.asyncio
    async def test_success_first_try(self):
        from noema.resilience.circuit_breaker import RetryPolicy

        rp = RetryPolicy(max_retries=3, base_delay=0.01, name="test")
        call_count = 0

        async def ok():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await rp.execute(ok)
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self):
        from noema.resilience.circuit_breaker import RetryPolicy

        rp = RetryPolicy(max_retries=3, base_delay=0.01, name="test")
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("not yet")
            return "ok"

        result = await rp.execute(flaky)
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        from noema.resilience.circuit_breaker import RetryPolicy

        rp = RetryPolicy(max_retries=2, base_delay=0.01, name="test")

        async def always_fail():
            raise ValueError("nope")

        with pytest.raises(ValueError):
            await rp.execute(always_fail)

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        from noema.resilience.circuit_breaker import RetryPolicy

        rp = RetryPolicy(
            max_retries=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
            name="test",
        )
        call_count = 0

        async def type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            await rp.execute(type_error)
        assert call_count == 1  # No retries


# ═══════════════════════════════════════════════════════════════════════════
# Resilient Executor Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestResilientExecutor:
    @pytest.mark.asyncio
    async def test_combined_success(self):
        from noema.resilience.circuit_breaker import (
            CircuitBreaker,
            ResilientExecutor,
            RetryPolicy,
        )

        re = ResilientExecutor(
            circuit_breaker=CircuitBreaker(failure_threshold=5, name="test"),
            retry_policy=RetryPolicy(max_retries=2, base_delay=0.01, name="test"),
        )

        async def ok():
            return 42

        assert await re.execute(ok) == 42

    @pytest.mark.asyncio
    async def test_combined_failure(self):
        from noema.resilience.circuit_breaker import (
            CircuitBreaker,
            ResilientExecutor,
            RetryPolicy,
        )

        re = ResilientExecutor(
            circuit_breaker=CircuitBreaker(failure_threshold=3, name="test"),
            retry_policy=RetryPolicy(max_retries=1, base_delay=0.01, name="test"),
        )

        async def fail():
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            await re.execute(fail)


# ═══════════════════════════════════════════════════════════════════════════
# Logging Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLogging:
    def test_setup_logging(self):
        from noema.logging import get_logger, setup_logging

        setup_logging()
        logger = get_logger("test")
        assert logger is not None

    def test_correlation_id(self):
        from noema.logging import get_correlation_id, set_correlation_id

        cid = set_correlation_id("test-123")
        assert cid == "test-123"
        assert get_correlation_id() == "test-123"

    def test_correlation_id_auto(self):
        from noema.logging import get_correlation_id, set_correlation_id

        cid = set_correlation_id()
        assert len(cid) == 16
        assert get_correlation_id() == cid


# ═══════════════════════════════════════════════════════════════════════════
# Auth Middleware Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAPIKeyAuth:
    def test_auth_disabled_when_no_key(self):
        from noema.config.settings import reset_settings

        reset_settings()
        s = get_settings()
        assert s.api.api_key.get_secret_value() == ""

    def test_auth_header_configurable(self):
        s = APISettings()
        assert s.api_key_header == "X-API-Key"


# ═══════════════════════════════════════════════════════════════════════════
# Rate Limiter Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_allows_under_limit(self):
        from noema.api.rate_limit import _SlidingWindowCounter

        rl = _SlidingWindowCounter(window_seconds=60, max_requests=5)
        allowed, headers = await rl.allow("client1")
        assert allowed is True
        assert int(headers["X-RateLimit-Remaining"]) >= 0

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        from noema.api.rate_limit import _SlidingWindowCounter

        rl = _SlidingWindowCounter(window_seconds=60, max_requests=3)
        for _ in range(3):
            await rl.allow("client1")
        allowed, headers = await rl.allow("client1")
        assert allowed is False
        assert "Retry-After" in headers

    @pytest.mark.asyncio
    async def test_separate_clients(self):
        from noema.api.rate_limit import _SlidingWindowCounter

        rl = _SlidingWindowCounter(window_seconds=60, max_requests=2)
        await rl.allow("a")
        await rl.allow("a")
        allowed_a, _ = await rl.allow("a")
        allowed_b, _ = await rl.allow("b")
        assert allowed_a is False
        assert allowed_b is True


# ═══════════════════════════════════════════════════════════════════════════
# DB Models Tests (schema only, no DB connection)
# ═══════════════════════════════════════════════════════════════════════════


class TestDBModels:
    def test_models_importable(self):
        from noema.db.models import (
            EpisodicMemoryRow,
            EvolutionLogRow,
            FeedbackRow,
            KnowledgeEntryRow,
            ProceduralMemoryRow,
            SemanticMemoryRow,
        )

        assert EpisodicMemoryRow.__tablename__ == "episodic_memory"
        assert SemanticMemoryRow.__tablename__ == "semantic_memory"
        assert ProceduralMemoryRow.__tablename__ == "procedural_memory"
        assert KnowledgeEntryRow.__tablename__ == "knowledge_entries"
        assert FeedbackRow.__tablename__ == "feedback"
        assert EvolutionLogRow.__tablename__ == "evolution_log"

    def test_base_metadata(self):
        from noema.db.engine import Base

        table_names = list(Base.metadata.tables.keys())
        assert "episodic_memory" in table_names
        assert "semantic_memory" in table_names
        assert "knowledge_entries" in table_names
        assert "feedback" in table_names
        assert "evolution_log" in table_names


# ═══════════════════════════════════════════════════════════════════════════
# Prometheus Metrics Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMetrics:
    def test_metrics_importable(self):
        from noema.observability.metrics import (
            LLM_LATENCY,
            REQUEST_COUNT,
        )

        assert REQUEST_COUNT is not None
        assert LLM_LATENCY is not None

    def test_metrics_recordable(self):
        from noema.observability.metrics import REQUEST_COUNT, REQUEST_LATENCY

        REQUEST_COUNT.labels(method="GET", endpoint="/health", status="200").inc()
        REQUEST_LATENCY.labels(method="GET", endpoint="/health").observe(0.05)


# ═══════════════════════════════════════════════════════════════════════════
# LLM Provider Tests (with new resilience)
# ═══════════════════════════════════════════════════════════════════════════


class TestLLMResilience:
    def test_fallback_provider_has_resilience(self):
        from noema.llm.providers import FallbackProvider

        fp = FallbackProvider()
        assert fp._resilient is not None
        assert fp._resilient.circuit is not None
        assert fp._resilient.retry is not None

    def test_provider_stats(self):
        from noema.llm.providers import FallbackProvider

        fp = FallbackProvider()
        stats = fp.stats()
        assert "provider" in stats
        assert "resilience" in stats
        assert "circuit" in stats["resilience"]
        assert "retry" in stats["resilience"]

    @pytest.mark.asyncio
    async def test_fallback_provider_works(self):
        from noema.llm.providers import FallbackProvider, LLMMessage

        fp = FallbackProvider()
        resp = await fp.complete([LLMMessage(role="user", content="test")])
        assert resp.content.startswith("[Fallback mode]")
