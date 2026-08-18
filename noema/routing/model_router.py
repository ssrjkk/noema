"""Model Router — выбирает LLM по сложности, бюджету и здоровью провайдеров."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from noema.llm.providers import BaseLLMProvider, LLMMessage, create_llm_provider
from noema.logging import get_logger
from noema.resilience.circuit_breaker import CircuitBreaker

log = get_logger(__name__)


class AllProvidersFailedError(Exception):
    """Raised when all providers in the fallback chain have failed."""


@dataclass
class ModelTier:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    cost_per_1m_input: float = 0.15
    cost_per_1m_output: float = 0.60
    max_tokens: int = 16_000
    suitable_for: list[str] = field(default_factory=lambda: ["simple", "trivial"])


TIERS: list[ModelTier] = [
    ModelTier(
        provider="openai",
        model="gpt-4o-mini",
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
        suitable_for=["trivial", "simple"],
    ),
    ModelTier(
        provider="openai",
        model="gpt-4o",
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
        suitable_for=["moderate"],
    ),
    ModelTier(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        cost_per_1m_input=3.00,
        cost_per_1m_output=15.00,
        suitable_for=["complex"],
    ),
    ModelTier(
        provider="openai",
        model="o3",
        cost_per_1m_input=10.00,
        cost_per_1m_output=40.00,
        suitable_for=["extreme", "security_critical"],
    ),
]


class ModelRouter:
    """Routes tasks to LLM models with per-provider circuit breakers and fallback chain."""

    def __init__(self) -> None:
        self._tiers = list(TIERS)
        self._fallback_chain = ["openai", "anthropic", "ollama", "fallback"]
        self._breakers: dict[str, CircuitBreaker] = {
            name: CircuitBreaker(name=f"provider.{name}") for name in self._fallback_chain
        }
        self._providers: dict[str, BaseLLMProvider] = {}
        for name in self._fallback_chain:
            try:
                self._providers[name] = create_llm_provider(name)
            except Exception as e:
                log.warning("provider_init_failed", provider=name, error=str(e))

    def _get_provider(self, name: str, model: str | None = None) -> BaseLLMProvider:
        key = f"{name}@{model}" if model else name
        if key not in self._providers:
            try:
                self._providers[key] = create_llm_provider(name, model=model)
            except Exception as e:
                log.warning("provider_init_failed", provider=name, model=model, error=str(e))
                if key != name:
                    return self._get_provider(name)
                raise
            if key != name and name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name=f"provider.{name}")
        return self._providers[key]

    def select(
        self,
        complexity: str = "moderate",
        tags: list[str] | None = None,
        degraded: bool = False,
    ) -> BaseLLMProvider:
        if degraded:
            return self._get_provider("openai")

        tags_lower = {t.lower() for t in (tags or [])}
        tier_name = (
            "extreme"
            if ("security" in tags_lower or "critical" in tags_lower)
            else complexity.lower()
        )

        # Honor the tier's model, not just the provider: otherwise an
        # "extreme" task would silently run on the same default model as a
        # "trivial" one. Providers are cached per (provider, model).
        for tier in reversed(self._tiers):
            if tier_name in tier.suitable_for:
                return self._get_provider(tier.provider, model=tier.model)
        return self._get_provider("openai")

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tenant_id: str = "",
    ) -> tuple[str, BaseLLMProvider]:
        """Try each provider in fallback chain through its circuit breaker."""
        last_error: Exception | None = None
        for provider_name in self._fallback_chain:
            if provider_name not in self._providers:
                continue
            breaker = self._breakers[provider_name]
            provider = self._providers[provider_name]
            try:
                response = await breaker.execute(
                    provider.complete,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tenant_id=tenant_id,
                )
                return response.content, provider
            except Exception as e:
                last_error = e
                continue
        raise AllProvidersFailedError("All LLM providers failed") from last_error

    def provider_health(self) -> dict[str, dict[str, Any]]:
        return {
            name: self._breakers[name].stats()
            for name in self._fallback_chain
            if name in self._providers
        }

    def estimated_cost(self, complexity: str, input_tokens: int, output_tokens: int) -> float:
        tier_name = complexity.lower()
        for tier in self._tiers:
            if tier_name in tier.suitable_for:
                input_cost = (input_tokens / 1_000_000) * tier.cost_per_1m_input
                output_cost = (output_tokens / 1_000_000) * tier.cost_per_1m_output
                return round(input_cost + output_cost, 6)
        return 0.0

    def tiers(self) -> list[dict[str, Any]]:
        return [
            {"provider": t.provider, "model": t.model, "suitable_for": t.suitable_for}
            for t in self._tiers
        ]
