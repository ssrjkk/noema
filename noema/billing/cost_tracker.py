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

# Schema consumed by QuotaManager (``cost_records`` lookups in billing/quotas.py).
CREATE_COST_RECORDS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cost_records (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    task_id TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    input_tokens BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    step_name TEXT NOT NULL DEFAULT '',
    recorded_at TIMESTAMPTZ NOT NULL
)
"""


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


@dataclass
class FileCost:
    """Line-attributed cost of one changed file (module) in a PR."""

    file_path: str
    lines_added: int
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    model: str = ""
    task_id: str = ""
    repo: str = ""
    pr_number: str = ""


def calculate_line_costs(total_cost: float, lines_by_file: dict[str, int]) -> dict[str, float]:
    """Attribute a task's total cost across changed files, weighted by line count.

    Files with zero lines (deletions/renames) get no weight: the cost of
    *producing* new code lands on the modules that grew. The weights always
    sum back to ``total_cost``.
    """
    total_lines = sum(max(lines, 0) for lines in lines_by_file.values())
    if total_lines <= 0 or total_cost <= 0:
        return dict.fromkeys(lines_by_file, 0.0)
    return {
        path: round(total_cost * max(lines, 0) / total_lines, 6)
        for path, lines in lines_by_file.items()
    }


class CostTracker:
    """Tracks LLM costs per tenant/task/step with Redis + in-memory backends.

    ``pg`` (an asyncpg-style pool/connection factory) enables best-effort
    write-through to the ``cost_records`` table that ``QuotaManager`` reads
    for monthly/hourly enforcement.
    """

    def __init__(self, redis_url: str = "", pg: Any = None) -> None:
        self._records: list[CostRecord] = []
        self._daily: dict[str, float] = {}
        self._monthly: dict[str, float] = {}
        self._file_costs: list[FileCost] = []
        self._redis = None
        self._pg = pg
        if redis_url:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(redis_url, decode_responses=True)
            except Exception as e:
                log.warning("redis_connect_failed", error=str(e))

    @staticmethod
    def _day_key(tenant_id: str) -> str:
        return f"cost:d:{tenant_id}:{time.strftime('%Y%m%d')}"

    @staticmethod
    def _month_key(tenant_id: str) -> str:
        return f"cost:m:{tenant_id}:{time.strftime('%Y%m')}"

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

        day_key = self._day_key(tenant_id)
        month_key = self._month_key(tenant_id)
        cost_int = int(cost * 10000)

        if self._redis:
            try:
                await self._redis.incrby(day_key, cost_int)
                await self._redis.expire(day_key, 86400 * 2)
                await self._redis.incrby(month_key, cost_int)
                await self._redis.expire(month_key, 86400 * 32)
            except Exception as e:
                log.warning("redis_cost_write_failed", error=str(e))

        if self._pg is not None:
            await self._write_cost_record(record)

        self._daily[day_key] = self._daily.get(day_key, 0) + cost
        self._monthly[month_key] = self._monthly.get(month_key, 0) + cost
        return record

    async def _write_cost_record(self, record: CostRecord) -> None:
        if self._pg is None:
            return
        try:
            await self._pg.execute(CREATE_COST_RECORDS_TABLE_SQL)
            await self._pg.execute(
                """INSERT INTO cost_records
                   (tenant_id, task_id, provider, model, input_tokens, output_tokens,
                    cost_usd, step_name, recorded_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,to_timestamp($9))""",
                record.tenant_id,
                record.task_id,
                record.provider,
                record.model,
                record.input_tokens,
                record.output_tokens,
                record.cost_usd,
                record.step_name,
                record.timestamp,
            )
        except Exception as e:  # noqa: BLE001 - billing must never break inference
            log.warning("cost_record_pg_write_failed", error=str(e))

    async def get_tenant_cost(self, tenant_id: str) -> dict[str, float]:
        day_key = self._day_key(tenant_id)
        month_key = self._month_key(tenant_id)
        daily_usd = self._daily.get(day_key, 0.0)
        monthly_usd = self._monthly.get(month_key, 0.0)
        if self._redis:
            try:
                daily_cents = int(await self._redis.get(day_key) or 0)
                month_cents = int(await self._redis.get(month_key) or 0)
                # Redis wins (authoritative across pods); local dict is the
                # fallback when Redis is down.
                daily_usd = max(daily_usd, daily_cents / 10000)
                monthly_usd = max(monthly_usd, month_cents / 10000)
            except Exception as e:
                log.warning("redis_cost_read_failed", error=str(e))
        return {
            "daily_usd": round(daily_usd, 6),
            "monthly_usd": round(monthly_usd, 6),
        }

    def get_task_cost(self, task_id: str) -> float:
        return round(sum(r.cost_usd for r in self._records if r.task_id == task_id), 6)

    def attribute_pr_cost(
        self,
        repo: str,
        pr_number: int,
        task_id: str,
        files: list[tuple[str, str]],
        model: str = "",
    ) -> list[FileCost]:
        """Attribute a task's generation cost onto the PR's changed modules.

        ``files`` are ``(path, content)`` pairs as produced by the fixer; the
        cost of the task (see :meth:`get_task_cost`) is distributed across
        files weighted by their line counts, then pushed into the
        ``noema_pr_cost_usd`` / ``noema_code_cost_per_module`` metrics.

        Returns the per-file attribution records.
        """
        if not files:
            return []
        total_cost = self.get_task_cost(task_id)
        lines_by_file = {path: len(content.splitlines()) for path, content in files}
        costs = calculate_line_costs(total_cost, lines_by_file)
        pr_key = str(pr_number)
        attributed: list[FileCost] = []
        for path, content in files:
            fc = FileCost(
                file_path=path,
                lines_added=len(content.splitlines()),
                cost_usd=costs.get(path, 0.0),
                model=model,
                task_id=task_id,
                repo=repo,
                pr_number=pr_key,
            )
            attributed.append(fc)
            self._file_costs.append(fc)
            try:
                from noema.observability.metrics import CODE_COST_PER_MODULE, PR_COST_USD

                PR_COST_USD.labels(repo=repo, pr=pr_key, module=path).inc(fc.cost_usd)
                CODE_COST_PER_MODULE.labels(repo=repo, module=path).set(fc.cost_usd)
            except Exception as e:  # noqa: BLE001 - metrics must never break billing
                log.debug("pr_cost_metric_failed", error=str(e))
        log.info(
            "pr_cost_attributed",
            repo=repo,
            pr=pr_key,
            task_id=task_id,
            total_usd=round(total_cost, 6),
            modules=len(attributed),
        )
        return attributed

    def get_pr_cost(self, repo: str, pr_number: int) -> dict[str, Any]:
        """Per-module cost breakdown of one PR."""
        pr_key = str(pr_number)
        records = [f for f in self._file_costs if f.repo == repo and f.pr_number == pr_key]
        modules = {f.file_path: round(f.cost_usd, 6) for f in records}
        return {
            "repo": repo,
            "pr_number": pr_number,
            "total_usd": round(sum(modules.values()), 6),
            "modules": modules,
        }

    def module_cost_stats(self, repo: str = "") -> dict[str, Any]:
        """Aggregate per-module cost across all attributed PRs."""
        records = self._file_costs
        if repo:
            records = [f for f in records if f.repo == repo]
        modules: dict[str, float] = {}
        for f in records:
            modules[f.file_path] = modules.get(f.file_path, 0.0) + f.cost_usd
        return {
            "modules": {k: round(v, 6) for k, v in modules.items()},
            "total_usd": round(sum(modules.values()), 6),
        }

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
