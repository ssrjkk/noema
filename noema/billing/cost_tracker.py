"""Cost Attribution & Billing — отслеживание затрат по tenant, задаче, шагу."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from noema.logging import get_logger

log = get_logger(__name__)


# Pricing per 1M tokens (USD)
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "o3": {"input": 10.00, "output": 40.00},
    "fallback": {"input": 0.0, "output": 0.0},
}

DEFAULT_PRICING = {"input": 2.50, "output": 10.00}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING.get(model, DEFAULT_PRICING)
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


@dataclass
class CostRecord:
    tenant_id: str = ""
    task_id: str = ""
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    timestamp: float = 0.0
    step_name: str = ""


class CostTracker:
    """Tracks LLM costs per tenant/task/step with Redis + in-memory backends."""

    def __init__(self, redis_url: str = "") -> None:
        self._records: list[CostRecord] = []
        self._daily: dict[str, float] = {}
        self._monthly: dict[str, float] = {}
        self._redis = None
        if redis_url:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(redis_url, decode_responses=True)
            except Exception as e:
                log.warning("redis_connect_failed", error=str(e))

    async def record(
        self,
        tenant_id: str,
        task_id: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        step_name: str = "",
    ) -> CostRecord:
        if input_tokens < 0 or output_tokens < 0:
            log.warning("negative_tokens", input_tokens=input_tokens, output_tokens=output_tokens)
            input_tokens = max(input_tokens, 0)
            output_tokens = max(output_tokens, 0)
        cost = calculate_cost(model, input_tokens, output_tokens)
        record = CostRecord(
            tenant_id=tenant_id,
            task_id=task_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            timestamp=time.time(),
            step_name=step_name,
        )
        self._records.append(record)

        day_key = f"cost:d:{tenant_id}"
        month_key = f"cost:m:{tenant_id}"
        cost_int = int(cost * 10000)

        if self._redis:
            try:
                await self._redis.incrby(day_key, cost_int)
                await self._redis.expire(day_key, 86400 * 2)
                await self._redis.incrby(month_key, cost_int)
                await self._redis.expire(month_key, 86400 * 32)
            except Exception as e:
                log.warning("redis_cost_write_failed", error=str(e))

        self._daily[day_key] = self._daily.get(day_key, 0) + cost
        self._monthly[month_key] = self._monthly.get(month_key, 0) + cost
        return record

    def get_tenant_cost(self, tenant_id: str) -> dict[str, float]:
        day_key = f"cost:d:{tenant_id}"
        month_key = f"cost:m:{tenant_id}"
        return {
            "daily_usd": round(self._daily.get(day_key, 0), 6),
            "monthly_usd": round(self._monthly.get(month_key, 0), 6),
        }

    def get_task_cost(self, task_id: str) -> float:
        return round(sum(r.cost_usd for r in self._records if r.task_id == task_id), 6)

    def get_breakdown(self, tenant_id: str = "") -> dict[str, Any]:
        filtered = self._records
        if tenant_id:
            filtered = [r for r in self._records if r.tenant_id == tenant_id]
        if not filtered:
            return {"total_usd": 0, "by_provider": {}, "by_model": {}, "by_step": {}}
        by_provider: dict[str, float] = {}
        by_model: dict[str, float] = {}
        by_step: dict[str, float] = {}
        total = 0.0
        for r in filtered:
            total += r.cost_usd
            by_provider[r.provider] = by_provider.get(r.provider, 0) + r.cost_usd
            by_model[r.model] = by_model.get(r.model, 0) + r.cost_usd
            by_step[r.step_name or "unknown"] = (
                by_step.get(r.step_name or "unknown", 0) + r.cost_usd
            )
        return {
            "total_usd": round(total, 6),
            "by_provider": {k: round(v, 6) for k, v in by_provider.items()},
            "by_model": {k: round(v, 6) for k, v in by_model.items()},
            "by_step": {k: round(v, 6) for k, v in by_step.items()},
        }

    def stats(self) -> dict[str, Any]:
        return {
            "total_records": len(self._records),
            "total_cost_usd": round(sum(r.cost_usd for r in self._records), 6),
            "total_input_tokens": sum(r.input_tokens for r in self._records),
            "total_output_tokens": sum(r.output_tokens for r in self._records),
        }
