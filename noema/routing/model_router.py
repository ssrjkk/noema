"""Model Router — selects an LLM by complexity, budget, and provider health.

Two routing policies share this single router:

- ``static`` (default): tier table + per-provider circuit breakers.
- ``linucb`` (opt-in via ``NOEMA_LLM__ROUTER=linucb``): the contextual bandit
  from :mod:`noema.llm.router_bandit` picks the arm; verified task outcomes
  feed back through :meth:`ModelRouter.record_outcome`, and an exhausted USD
  budget raises :class:`BudgetExceededError` (fail-closed, never a silent
  fallback to a paid arm).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from noema.config.settings import LLMSettings, get_settings
from noema.llm.providers import BaseLLMProvider, LLMMessage, create_llm_provider
from noema.llm.router_bandit import (
    BudgetExceededError,
    LinUCBModelRouter,
    ModelArm,
    TaskContext,
)
from noema.logging import get_logger
from noema.resilience.circuit_breaker import CircuitBreaker

log = get_logger(__name__)

__all__ = [
    "TIERS",
    "AllProvidersFailedError",
    "BudgetExceededError",
    "ModelRouter",
    "ModelTier",
]


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
    #: Latency estimate (seconds) used only by the linucb utility function.
    avg_latency_sec: float = 2.0


TIERS: list[ModelTier] = [
    ModelTier(
        provider="openai",
        model="gpt-4o-mini",
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
        suitable_for=["trivial", "simple"],
        avg_latency_sec=1.0,
    ),
    ModelTier(
        provider="openai",
        model="gpt-4o",
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
        suitable_for=["moderate"],
        avg_latency_sec=2.5,
    ),
    ModelTier(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        cost_per_1m_input=3.00,
        cost_per_1m_output=15.00,
        suitable_for=["complex"],
        avg_latency_sec=3.0,
    ),
    ModelTier(
        provider="openai",
        model="o3",
        cost_per_1m_input=10.00,
        cost_per_1m_output=40.00,
        suitable_for=["extreme", "security_critical"],
        avg_latency_sec=10.0,
    ),
]

#: Repo complexity labels mapped onto the bandit's 3-level feature scale.
_BANDIT_COMPLEXITY: dict[str, str] = {
    "trivial": "simple",
    "simple": "simple",
    "moderate": "medium",
    "complex": "complex",
    "extreme": "complex",
    "security_critical": "complex",
}


class ModelRouter:
    """Routes tasks to LLM models with per-provider circuit breakers and fallback chain."""

    def __init__(self, bandit: LinUCBModelRouter | None = None) -> None:
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

        self._bandit = bandit
        self._pending_outcomes: list[tuple[str, TaskContext, float]] = []
        if self._bandit is None:
            cfg = get_settings().llm
            policy = cfg.router.strip().lower()
            if policy not in ("static", "linucb"):
                raise ValueError(f"Unknown NOEMA_LLM__ROUTER policy: {cfg.router!r}")
            if policy == "linucb":
                self._bandit = self._build_bandit(cfg)

    def _build_bandit(self, cfg: LLMSettings) -> LinUCBModelRouter:
        arms = [
            ModelArm(
                arm_id=f"{t.provider}/{t.model}",
                provider=t.provider,
                model=t.model,
                cost_per_1k_tokens_usd=(t.cost_per_1m_input + t.cost_per_1m_output) / 2000.0,
                avg_latency_sec=t.avg_latency_sec,
            )
            for t in self._tiers
        ]
        arms.append(
            ModelArm(
                arm_id=f"ollama/{cfg.ollama_model}",
                provider="ollama",
                model=cfg.ollama_model,
                cost_per_1k_tokens_usd=0.0,
                avg_latency_sec=0.8,
                is_local=True,
            )
        )
        checkpoint = cfg.router_checkpoint.strip()
        if checkpoint:
            path = Path(checkpoint)
            if path.is_file():
                # Validated load: a corrupt checkpoint raises instead of
                # silently resetting learned weights.
                return LinUCBModelRouter.load(path)
            log.warning("router_checkpoint_missing", path=str(path))
        return LinUCBModelRouter(arms)

    @property
    def bandit_active(self) -> bool:
        """True when the linucb policy is enabled on this router."""
        return self._bandit is not None

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
        *,
        budget_remaining_usd: float | None = None,
        prompt_tokens: int = 0,
    ) -> BaseLLMProvider:
        """Pick a provider for a task.

        Under the linucb policy the bandit chooses the arm and the decision is
        queued for :meth:`record_outcome`. ``budget_remaining_usd`` is enforced
        fail-closed: when no arm fits, ``BudgetExceededError`` propagates.
        """
        if degraded:
            return self._get_provider("openai")

        tags_lower = {t.lower() for t in (tags or [])}
        tier_name = (
            "extreme"
            if ("security" in tags_lower or "critical" in tags_lower)
            else complexity.lower()
        )

        if self._bandit is not None:
            context = TaskContext(
                prompt_tokens=max(0, prompt_tokens),
                tag_count=len(tags or []),
                domain_modules_count=0,
                complexity_level=_BANDIT_COMPLEXITY.get(tier_name, "medium"),
                requires_formal_z3=get_settings().neurosymbolic.enabled,
            )
            decision = self._bandit.select_arm(context, budget_remaining_usd)
            self._pending_outcomes.append(
                (decision.arm.arm_id, context, decision.estimated_cost_usd)
            )
            return self._get_provider(decision.arm.provider, model=decision.arm.model)

        # Honor the tier's model, not just the provider: otherwise an
        # "extreme" task would silently run on the same default model as a
        # "trivial" one. Providers are cached per (provider, model).
        for tier in reversed(self._tiers):
            if tier_name in tier.suitable_for:
                return self._get_provider(tier.provider, model=tier.model)
        return self._get_provider("openai")

    def record_outcome(self, verified_pass: bool, actual_cost_usd: float | None = None) -> int:
        """Feed a task's ground-truth verification verdict back to the bandit.

        Every arm selected since the previous call is updated once with the
        task-level verdict (Z3 / judge / sandbox ground truth). Cost defaults to
        the per-decision estimate when no actuals are available. No-op under the
        static policy. Returns the number of arms updated.
        """
        if self._bandit is None or not self._pending_outcomes:
            return 0
        pending = self._pending_outcomes
        self._pending_outcomes = []
        updated: set[str] = set()
        for arm_id, context, est_cost in pending:
            if arm_id in updated:
                continue
            updated.add(arm_id)
            self._bandit.record_outcome(
                arm_id,
                context,
                verified_pass,
                est_cost if actual_cost_usd is None else actual_cost_usd,
            )
        return len(updated)

    def discard_pending_outcomes(self) -> None:
        """Drop queued bandit decisions (called at the start of each task).

        Prevents a crashed task's selections from being credited with the next
        task's verification verdict.
        """
        self._pending_outcomes = []

    def bandit_summary(self) -> list[dict[str, object]] | None:
        """Empirical Pass@1 vs. cost per arm, or None under the static policy."""
        if self._bandit is None:
            return None
        return self._bandit.pareto_summary()

    def save_bandit(self, path: str | Path) -> None:
        """Persist bandit weights; no-op under the static policy."""
        if self._bandit is not None:
            self._bandit.save(path)

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
