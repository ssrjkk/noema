"""Additional tests for llm/providers.py — covering complete(), cache, tracing, error paths."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from noema.llm.providers import (
    BaseLLMProvider,
    FallbackProvider,
    LLMMessage,
    LLMProviderError,
    LLMResponse,
)


class _TestLLM(BaseLLMProvider):
    """Minimal provider for testing complete() logic."""

    def __init__(self, response: LLMResponse | None = None, error: Exception | None = None):
        self._response = response
        self._error = error
        super().__init__()

    @property
    def name(self) -> str:
        return "test"

    @property
    def model_name(self) -> str:
        return "test-model"

    async def _complete(self, messages, temperature=0.7, max_tokens=4096):
        if self._error:
            raise self._error
        return self._response or LLMResponse(content='{"ok": true}', model="test-model", tokens_used=10)


def _msgs() -> list[LLMMessage]:
    return [LLMMessage(role="user", content="hello")]


# ── complete() with cache ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_returns_cached_response():
    provider = _TestLLM()
    mock_cache = MagicMock()
    mock_cache.get.return_value = '{"cached": true}'
    mock_tracer = MagicMock()
    mock_tracer.trace_llm_call = MagicMock()

    with patch("noema.llm.providers.get_cache", return_value=mock_cache):
        with patch("noema.llm.providers.get_tracer", return_value=mock_tracer):
            resp = await provider.complete(_msgs(), tenant_id="t1")

    assert resp.content == '{"cached": true}'
    assert resp.model == "test-model"
    mock_cache.get.assert_called_once()
    mock_tracer.trace_llm_call.assert_called_once()


@pytest.mark.asyncio
async def test_complete_caches_successful_response():
    response = LLMResponse(content='{"result": 1}', model="test-model", tokens_used=50)
    provider = _TestLLM(response=response)
    mock_cache = MagicMock()
    mock_cache.get.return_value = None
    mock_tracer = MagicMock()

    with patch("noema.llm.providers.get_cache", return_value=mock_cache):
        with patch("noema.llm.providers.get_tracer", return_value=mock_tracer):
            resp = await provider.complete(_msgs(), tenant_id="t1")

    assert resp.content == '{"result": 1}'
    mock_cache.set.assert_called_once()


@pytest.mark.asyncio
async def test_complete_does_not_cache_zero_tokens():
    response = LLMResponse(content='{"result": 1}', model="test-model", tokens_used=0)
    provider = _TestLLM(response=response)
    mock_cache = MagicMock()
    mock_cache.get.return_value = None

    with patch("noema.llm.providers.get_cache", return_value=mock_cache):
        with patch("noema.llm.providers.get_tracer", return_value=MagicMock()):
            await provider.complete(_msgs())

    mock_cache.set.assert_not_called()


# ── complete() error handling ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_raises_on_response_error():
    response = LLMResponse(content="", model="test", error="API key invalid")
    provider = _TestLLM(response=response)

    with patch("noema.llm.providers.get_cache", return_value=MagicMock(get=MagicMock(return_value=None))):
        with patch("noema.llm.providers.get_tracer", return_value=MagicMock()):
            with pytest.raises(LLMProviderError, match="API key invalid"):
                await provider.complete(_msgs())


@pytest.mark.asyncio
async def test_complete_traces_error_on_exception():
    provider = _TestLLM(error=RuntimeError("network down"))
    mock_tracer = MagicMock()

    with patch("noema.llm.providers.get_cache", return_value=MagicMock(get=MagicMock(return_value=None))):
        with patch("noema.llm.providers.get_tracer", return_value=mock_tracer):
            with pytest.raises(Exception):
                await provider.complete(_msgs())

    mock_tracer.trace_llm_call.assert_called_once()
    call_kwargs = mock_tracer.trace_llm_call.call_args[1]
    assert call_kwargs["error"] != ""


# ── complete() tenant resolution ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_uses_explicit_tenant_id():
    provider = _TestLLM()
    mock_cache = MagicMock()
    mock_cache.get.return_value = None

    with patch("noema.llm.providers.get_cache", return_value=mock_cache):
        with patch("noema.llm.providers.get_tracer", return_value=MagicMock()):
            await provider.complete(_msgs(), tenant_id="explicit-tenant")

    call_args = mock_cache.get.call_args
    assert call_args[1]["tenant_id"] == "explicit-tenant"


@pytest.mark.asyncio
async def test_complete_falls_back_to_context_tenant():
    provider = _TestLLM()
    mock_cache = MagicMock()
    mock_cache.get.return_value = None

    with patch("noema.llm.providers.get_cache", return_value=mock_cache):
        with patch("noema.llm.providers.get_tracer", return_value=MagicMock()):
            with patch("noema.llm.providers.get_tenant_id", return_value="context-tenant"):
                await provider.complete(_msgs())

    call_args = mock_cache.get.call_args
    assert call_args[1]["tenant_id"] == "context-tenant"


# ── FallbackProvider edge cases ──────────────────────────────────────────────


def test_fallback_classify_empty_messages():
    provider = FallbackProvider()
    result = provider._classify([])
    assert result == "general"


def test_fallback_classify_no_user_messages():
    provider = FallbackProvider()
    msgs = [LLMMessage(role="system", content="You are helpful")]
    result = provider._classify(msgs)
    assert result == "general"


@pytest.mark.asyncio
async def test_fallback_complete_with_empty_content():
    provider = FallbackProvider()
    msgs = [LLMMessage(role="user", content="")]
    resp = await provider._complete(msgs)
    assert resp.content
    assert resp.finish_reason == "stop"


# ── LLMResponse model ────────────────────────────────────────────────────────


def test_llm_response_defaults():
    resp = LLMResponse(content="test")
    assert resp.model == ""
    assert resp.tokens_used == 0
    assert resp.tokens_input == 0
    assert resp.tokens_output == 0
    assert resp.error == ""
    assert resp.latency_ms == 0.0
    assert resp.finish_reason == ""


def test_llm_response_with_all_fields():
    resp = LLMResponse(
        content='{"ok": true}',
        model="gpt-4",
        tokens_used=100,
        tokens_input=50,
        tokens_output=50,
        error="",
        latency_ms=123.45,
        finish_reason="stop",
    )
    assert resp.content == '{"ok": true}'
    assert resp.model == "gpt-4"
    assert resp.tokens_used == 100
    assert resp.latency_ms == 123.45


# ── LLMMessage model ─────────────────────────────────────────────────────────


def test_llm_message_creation():
    msg = LLMMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_llm_message_dump():
    msg = LLMMessage(role="system", content="You are helpful")
    d = msg.model_dump()
    assert d == {"role": "system", "content": "You are helpful"}
