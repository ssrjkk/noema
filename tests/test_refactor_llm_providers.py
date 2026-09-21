"""Coverage for LLM providers in noema.llm.providers (no network, no API keys)."""

import sys
import time

import pytest

from noema.llm.providers import (
    AnthropicProvider,
    BaseLLMProvider,
    FallbackProvider,
    LLMMessage,
    LLMProviderError,
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
async def test_openai_provider_unavailable_fails_closed(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    provider = OpenAIProvider()
    with pytest.raises(LLMProviderError):
        await provider._complete(_messages())


# ── Anthropic ───────────────────────────────────────────────────────────


def test_anthropic_provider_metadata():
    provider = AnthropicProvider()
    assert isinstance(provider, BaseLLMProvider)
    assert provider.name == "anthropic"
    assert provider.model_name == "claude-sonnet-4-5-20250929"


def test_anthropic_provider_custom_model():
    provider = AnthropicProvider(model="claude-haiku-4-20250514")
    assert provider.model_name == "claude-haiku-4-20250514"


@pytest.mark.asyncio
async def test_anthropic_provider_not_installed_fails_closed(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    provider = AnthropicProvider()
    with pytest.raises(LLMProviderError):
        await provider._complete(_messages())


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
async def test_ollama_provider_connection_refused_fails_closed():
    provider = OllamaProvider(model="llama3", base_url="http://127.0.0.1:1")
    with pytest.raises(LLMProviderError):
        await provider._complete(_messages())


@pytest.mark.asyncio
async def test_ollama_provider_health_check_self_heals(monkeypatch):
    """A failed health check must not poison the provider forever.

    After the recovery window elapses, the next call re-attempts health and
    succeeds, so a long-lived worker recovers when Ollama comes back.
    """

    class _FakeResponse:
        def __init__(self, status: int) -> None:
            self.status = status

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def text(self) -> str:
            return "service unavailable"

    class _FakeClientSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return _FakeSession()

        async def __aexit__(self, *args):
            return False

    healthy = {"flag": False}

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def get(self, url: str) -> _FakeResponse:
            return _FakeResponse(200 if healthy["flag"] else 503)

    monkeypatch.setattr("aiohttp.ClientSession", _FakeClientSession)
    provider = OllamaProvider(model="llama3", base_url="http://127.0.0.1:9")

    # Downstream is unhealthy: the health check fails closed.
    with pytest.raises(LLMProviderError):
        await provider._health_check()
    assert not provider._health_ok

    # Within the recovery window the cached failure is reused (no network).
    with pytest.raises(LLMProviderError):
        await provider._health_check()

    # Service recovers after the window elapses → the provider self-heals.
    provider._health_failed_at = time.monotonic() - provider._health_recovery - 1
    healthy["flag"] = True
    await provider._health_check()
    assert provider._health_ok


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


# ── Fallback (template-based demo engine) ───────────────────────────────


def _fallback_messages(domain: str) -> list[LLMMessage]:
    """Messages whose trailing user content anchors the desired fallback domain."""
    if domain == "terraform":
        text = "Provision an AWS VPC and subnets with Terraform HCL"
    elif domain == "docker":
        text = "Write a Dockerfile to containerize a FastAPI service"
    elif domain == "database":
        text = "Design a PostgreSQL schema with SQL migrations"
    elif domain == "auth":
        text = "Implement RBAC auth with JWT and login flows"
    elif domain == "pipeline":
        text = "Build an Airflow ETL pipeline DAG"
    elif domain == "api":
        text = "Create a FastAPI REST backend with endpoints"
    elif domain == "frontend":
        text = "Build a React UI component list"
    elif domain == "security":
        text = "Run a security vulnerability scan for OWASP threats"
    elif domain == "test":
        text = "Write unit tests with pytest and asserts"
    else:
        text = "Build a general-purpose solution"
    return [
        LLMMessage(role="system", content="You are a senior engineer."),
        LLMMessage(role="user", content=text),
    ]


@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        ("terraform", "terraform"),
        ("docker", "docker"),
        ("database", "database"),
        ("auth", "auth"),
        ("pipeline", "pipeline"),
        ("api", "api"),
        ("frontend", "frontend"),
        ("security", "security"),
        ("test", "test"),
        ("general", "general"),
    ],
)
def test_fallback_classify(domain, expected):
    provider = FallbackProvider()
    assert provider._classify(_fallback_messages(domain)) == expected


@pytest.mark.parametrize(
    "domain",
    ["api", "docker", "terraform", "database", "auth", "pipeline", "frontend", "security", "test"],
)
def test_fallback_template_structure(domain):
    provider = FallbackProvider()
    payload = provider._template(domain)

    assert isinstance(payload, dict)
    assert isinstance(payload["summary"], str)
    assert isinstance(payload["architecture"], dict)
    assert isinstance(payload["architecture"]["pattern"], dict)
    assert isinstance(payload["architecture"]["pattern"]["pros"], list)
    assert isinstance(payload["stack"], dict)
    assert isinstance(payload["stack"]["languages"], list)
    assert isinstance(payload["code"], dict)
    assert isinstance(payload["code"]["files"], list)
    assert len(payload["code"]["files"]) > 0
    for f in payload["code"]["files"]:
        assert isinstance(f, dict)
        assert "path" in f and "content" in f and "description" in f
        assert isinstance(f["content"], str)
        assert f["content"].strip() != ""


def test_fallback_template_unknown_kind_falls_back_to_general():
    provider = FallbackProvider()
    general_files = {f["path"] for f in provider._template("general")["code"]["files"]}
    unknown_files = {f["path"] for f in provider._template("totally-unknown")["code"]["files"]}
    assert unknown_files == general_files


def test_fallback_template_deterministic():
    provider = FallbackProvider()
    for domain in ("api", "terraform", "database"):
        assert provider._template(domain) == provider._template(domain)


@pytest.mark.asyncio
async def test_fallback_complete_returns_fenced_json():
    provider = FallbackProvider()
    resp = await provider._complete(_fallback_messages("api"))

    assert resp.model == "fallback/template-based"
    assert resp.finish_reason == "stop"
    assert resp.tokens_input == 0
    assert resp.tokens_output == len(resp.content)
    assert resp.latency_ms >= 0

    # The response must round-trip through the same parser the pipeline uses.
    from noema.utils.json_utils import extract_fenced_json

    payload = extract_fenced_json(resp.content, default=None)
    assert payload is not None
    assert isinstance(payload, dict)
    assert "code" in payload and "architecture" in payload
    assert resp.content.startswith("```json\n")


@pytest.mark.asyncio
async def test_fallback_unknown_domain_still_parses():
    provider = FallbackProvider()
    await provider._complete(_fallback_messages("general"))
    payload = provider._template("general")
    assert "LLM unavailable" in payload["review"]["weaknesses"][0]
    assert "template" in payload["review"]["final_summary"]
