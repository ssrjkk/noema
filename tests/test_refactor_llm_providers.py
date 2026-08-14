"""Coverage for LLM providers in noema.llm.providers (no network, no API keys)."""

import sys

import pytest

from noema.llm.providers import (
    AnthropicProvider,
    BaseLLMProvider,
    FallbackProvider,
    LLMMessage,
    OllamaProvider,
    OpenAIProvider,
    create_llm_provider,
)


def _messages() -> list[LLMMessage]:
    return [LLMMessage(role="user", content="hello")]


# ── OpenAI ──────────────────────────────────────────────────────────────


def test_openai_provider_metadata():
    provider = OpenAIProvider()
    assert isinstance(provider, BaseLLMProvider)
    assert provider.name == "openai"
    assert provider.model_name == "gpt-4o"


def test_openai_provider_custom_model():
    provider = OpenAIProvider(model="gpt-4o-mini")
    assert provider.model_name == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_openai_provider_not_installed_stub():
    provider = OpenAIProvider()
    response = await provider._complete(_messages())
    assert response.content == "OpenAI not installed: pip install openai"
    assert response.model == ""


# ── Anthropic ───────────────────────────────────────────────────────────


def test_anthropic_provider_metadata():
    provider = AnthropicProvider()
    assert isinstance(provider, BaseLLMProvider)
    assert provider.name == "anthropic"
    assert provider.model_name == "claude-sonnet-4-20250514"


def test_anthropic_provider_custom_model():
    provider = AnthropicProvider(model="claude-haiku-4-20250514")
    assert provider.model_name == "claude-haiku-4-20250514"


@pytest.mark.asyncio
async def test_anthropic_provider_not_installed_stub(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    provider = AnthropicProvider()
    response = await provider._complete(_messages())
    assert response.content == "Anthropic not installed: pip install anthropic"
    assert response.model == ""


# ── Ollama ──────────────────────────────────────────────────────────────


def test_ollama_provider_metadata():
    provider = OllamaProvider(model="llama3")
    assert isinstance(provider, BaseLLMProvider)
    assert provider.name == "ollama"
    assert provider.model_name == "llama3"


def test_ollama_provider_default_model():
    provider = OllamaProvider()
    assert provider.model_name  # falls back to settings.llm.ollama_model


@pytest.mark.asyncio
async def test_ollama_provider_connection_refused():
    provider = OllamaProvider(model="llama3", base_url="http://127.0.0.1:1")
    with pytest.raises(OSError):
        await provider._complete(_messages())


# ── Factory ─────────────────────────────────────────────────────────────


def test_create_llm_provider_openai():
    provider = create_llm_provider("openai")
    assert isinstance(provider, OpenAIProvider)
    assert provider.model_name == "gpt-4o"


def test_create_llm_provider_anthropic():
    provider = create_llm_provider("anthropic")
    assert isinstance(provider, AnthropicProvider)


def test_create_llm_provider_unknown_returns_fallback():
    provider = create_llm_provider("unknown-vendor")
    assert isinstance(provider, FallbackProvider)


def test_create_llm_provider_ollama():
    provider = create_llm_provider("ollama")
    assert isinstance(provider, OllamaProvider)
