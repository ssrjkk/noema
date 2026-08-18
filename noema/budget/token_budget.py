"""Token Budget Manager — жёсткий лимит токенов на задачу с graceful degradation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BudgetAction(Enum):
    ALLOW = "allow"
    DEGRADE = "degrade"  # switch to cheaper model
    SKIP = "skip"  # skip non-critical steps
    REJECT = "reject"  # hard cap reached


@dataclass
class TokenBudget:
    max_tokens: int = 100_000
    warn_at: float = 0.7  # fraction of max — switch to degrade mode
    hard_cap_at: float = 0.95  # fraction of max — reject new expensive calls

    _used: int = 0
    _skipped_steps: list[str] = field(default_factory=list, repr=False)
    _degraded: bool = False

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self._used)

    @property
    def fraction_used(self) -> float:
        return self._used / max(self.max_tokens, 1)

    def record(self, tokens: int) -> None:
        self._used += tokens

    def check(self, estimated_cost: int = 1000, step_name: str = "") -> BudgetAction:
        projected = self._used + estimated_cost
        frac = projected / max(self.max_tokens, 1)

        if frac >= self.hard_cap_at:
            self._skipped_steps.append(step_name)
            return BudgetAction.REJECT
        if frac >= self.warn_at:
            if not self._degraded:
                self._degraded = True
            return BudgetAction.DEGRADE
        return BudgetAction.ALLOW

    def should_degrade(self) -> bool:
        if self._degraded:
            return True
        result = self.check(step_name="")
        return result in (BudgetAction.DEGRADE, BudgetAction.SKIP, BudgetAction.REJECT)

    def skipped_steps(self) -> list[str]:
        return list(self._skipped_steps)

    def reset(self) -> None:
        self._used = 0
        self._skipped_steps = []
        self._degraded = False

    def stats(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "used": self._used,
            "remaining": self.remaining,
            "fraction_used": round(self.fraction_used, 3),
            "degraded": self._degraded,
            "skipped_steps": len(self._skipped_steps),
        }
