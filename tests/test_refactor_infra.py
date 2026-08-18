"""Coverage tests for ``noema.errors``, ``noema.routing.model_router``,
``noema.vault.client``, ``noema.debug.replay`` and ``noema.observability.sentry``.
"""

import json

import pytest

import noema.debug.replay as replay_module
from noema.debug.replay import ReplayEngine
from noema.errors import NoemaError
from noema.llm.providers import BaseLLMProvider, LLMMessage, LLMResponse
from noema.observability.sentry import init_sentry
from noema.resilience.circuit_breaker import CircuitBreaker
from noema.routing.model_router import AllProvidersFailedError, ModelRouter
from noema.vault.client import VaultClient

# ── NoemaError ──────────────────────────────────────────────────────────────


class _DomainError(NoemaError):
    pass


def test_noema_error_to_dict():
    err = NoemaError("boom", context={"module": "kernels", "attempts": 3})
    assert err.message == "boom"
    assert err.context == {"module": "kernels", "attempts": 3}
    assert err.to_dict() == {
        "error": "NoemaError",
        "message": "boom",
        "module": "kernels",
        "attempts": 3,
    }


def test_noema_error_default_context_and_subclass_name():
    err = _DomainError("nope")
    assert err.context == {}
    assert err.to_dict() == {"error": "_DomainError", "message": "nope"}


# ── ModelRouter ─────────────────────────────────────────────────────────────


class _FailProvider(BaseLLMProvider):
    @property
    def name(self) -> str:
        return "fail"

    @property
    def model_name(self) -> str:
        return "fail"

    async def _complete(self, messages, temperature=0.7, max_tokens=4096):
        raise RuntimeError("provider down")


class _ScriptProvider(BaseLLMProvider):
    def __init__(self, name: str, content: str) -> None:
        self._name = name
        self._content = content
        super().__init__()

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._name

    async def _complete(self, messages, temperature=0.7, max_tokens=4096):
        return LLMResponse(content=self._content, model=self._name, tokens_used=0)


def test_router_select_by_complexity_and_tags():
    router = ModelRouter()
    assert router.select("trivial").name == "openai"
    assert router.select("moderate").name == "openai"
    assert router.select("complex").name == "anthropic"
    assert router.select("extreme").name == "openai"
    assert router.select("moderate", tags=["security"]).name == "openai"
    assert router.select("moderate", degraded=True).name == "openai"
    assert router.select("unknown-complexity").name == "openai"


def test_router_estimated_cost():
    router = ModelRouter()
    assert router.estimated_cost("simple", 1_000_000, 0) == 0.15
    assert router.estimated_cost("moderate", 1_000_000, 1_000_000) == 12.5
    assert router.estimated_cost("nope", 1_000_000, 1_000_000) == 0.0


def test_router_tiers_and_health():
    router = ModelRouter()
    tiers = router.tiers()
    assert len(tiers) == 4
    assert any(t["model"] == "gpt-4o-mini" for t in tiers)

    health = router.provider_health()
    assert "openai" in health
    assert "state" in health["openai"]


@pytest.mark.asyncio
async def test_router_complete_all_providers_fail():
    router = ModelRouter()
    chain = ["openai", "anthropic", "ollama", "fallback"]
    router._providers = {name: _FailProvider() for name in chain}
    router._breakers = {name: CircuitBreaker(name=f"provider.{name}") for name in chain}

    messages = [LLMMessage(role="user", content="hello")]
    with pytest.raises(AllProvidersFailedError):
        await router.complete(messages)


@pytest.mark.asyncio
async def test_router_complete_falls_through_to_healthy_provider():
    router = ModelRouter()
    chain = ["openai", "anthropic", "ollama", "fallback"]
    router._providers = {
        "openai": _FailProvider(),
        "anthropic": _ScriptProvider("anthropic", "from anthropic"),
        "ollama": _ScriptProvider("ollama", "from ollama"),
        "fallback": _ScriptProvider("fallback", "from fallback"),
    }
    router._breakers = {name: CircuitBreaker(name=f"provider.{name}") for name in chain}

    messages = [LLMMessage(role="user", content="hello")]
    content, provider = await router.complete(messages)
    assert content == "from anthropic"
    assert provider.name == "anthropic"


# ── VaultClient ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_vault_fallback_when_hvac_missing():
    client = VaultClient(url="http://vault:8200", token="hvs.test")
    assert await client.get_secret("noema/llm/openai-api-key") is None
    assert await client.set_secret("noema/llm/foo", "bar") is False
    assert await client.delete_secret("noema/llm/foo") is False
    assert await client.list_secrets() == []
    assert await client.get_llm_api_key("openai") is None
    assert await client.get_llm_api_key("unknown-provider") is None


class _FakeKV2:
    def __init__(self, values: dict[str, str] | None = None, fail: bool = False) -> None:
        self._values = values or {}
        self._fail = fail

    def read_secret_version(self, path, mount_point="secret"):
        if self._fail:
            raise RuntimeError("vault read error")
        return {"data": {"data": {"value": self._values.get(path)}}}

    def create_or_update_secret(self, path, secret, mount_point="secret"):
        if self._fail:
            raise RuntimeError("vault write error")

    def delete_metadata_and_all_versions(self, path, mount_point="secret"):
        if self._fail:
            raise RuntimeError("vault delete error")

    def list_secrets(self, path="", mount_point="secret"):
        if self._fail:
            raise RuntimeError("vault list error")
        return {"data": {"keys": ["key-a", "key-b"]}}


class _FakeVault:
    def __init__(self, values=None, fail=False):
        self.secrets = type("S", (), {})()
        self.secrets.kv = type("K", (), {})()
        self.secrets.kv.v2 = _FakeKV2(values=values, fail=fail)


@pytest.mark.asyncio
async def test_vault_fake_client_success_paths():
    client = VaultClient(url="http://vault:8200", token="hvs.test")
    client._client = _FakeVault(values={"noema/llm/openai-api-key": "sk-123"})

    assert await client.get_secret("noema/llm/openai-api-key") == "sk-123"
    assert await client.set_secret("noema/llm/x", "y") is True
    assert await client.delete_secret("noema/llm/x") is True
    assert await client.list_secrets() == ["key-a", "key-b"]
    assert await client.get_llm_api_key("openai") == "sk-123"
    assert await client.get_llm_api_key("anthropic") is None


@pytest.mark.asyncio
async def test_vault_fake_client_error_paths():
    client = VaultClient(url="http://vault:8200", token="hvs.test")
    client._client = _FakeVault(values={}, fail=True)

    assert await client.get_secret("noema/llm/anything") is None
    assert await client.set_secret("noema/llm/x", "y") is False
    assert await client.delete_secret("noema/llm/x") is False
    assert await client.list_secrets() == []


# ── ReplayEngine ────────────────────────────────────────────────────────────


class _FakeTracer:
    def __init__(self, trace):
        self._trace = trace

    def get_trace(self):
        return self._trace


@pytest.mark.asyncio
async def test_replay_no_tracer(monkeypatch):
    monkeypatch.setattr(replay_module, "get_tracer", lambda: None)
    result = await ReplayEngine().replay_trace("trace-1")
    assert result.error == "Tracer not initialized"


@pytest.mark.asyncio
async def test_replay_empty_trace(monkeypatch):
    monkeypatch.setattr(replay_module, "get_tracer", lambda: _FakeTracer([]))
    result = await ReplayEngine().replay_trace("trace-1")
    assert result.error == "No trace data available"


@pytest.mark.asyncio
async def test_replay_llm_span(monkeypatch):
    span = {
        "kind": "llm",
        "name": "cot.step",
        "trace_id": "trace-1",
        "attributes": {
            "llm.provider": "fallback",
            "llm.model": "fallback",
            "llm.prompt": json.dumps([{"role": "user", "content": "hello"}]),
            "llm.response": "original response",
        },
    }
    monkeypatch.setattr(replay_module, "get_tracer", lambda: _FakeTracer([span]))

    result = await ReplayEngine().replay_trace("trace-1")
    assert result.original_steps == 1
    assert result.new_steps == 1
    assert len(result.diffs) == 1
    diff = result.diffs[0]
    assert diff.step_name == "cot.step"
    assert diff.original_preview == "original response"
    assert "[Fallback mode]" in diff.new_preview
    assert diff.identical is False


@pytest.mark.asyncio
async def test_replay_non_json_prompt(monkeypatch):
    span = {
        "kind": "llm",
        "name": "cot.step",
        "trace_id": "trace-1",
        "attributes": {
            "llm.provider": "fallback",
            "llm.model": "fallback",
            "llm.prompt": "raw text prompt",
            "llm.response": "",
        },
    }
    monkeypatch.setattr(replay_module, "get_tracer", lambda: _FakeTracer([span]))

    result = await ReplayEngine().replay_trace("trace-1")
    assert result.diffs[0].original_preview == "(no response)"
    assert "[Fallback mode]" in result.diffs[0].new_preview


# ── Sentry ──────────────────────────────────────────────────────────────────


def test_sentry_disabled_without_dsn():
    assert init_sentry("") is False


def test_sentry_with_dsn_does_not_crash():
    # either sentry_sdk is absent (ImportError) or the DSN is invalid (Exception) —
    # both must yield False, never raise.
    assert init_sentry("https://example.invalid/not-a-real-project") is False
