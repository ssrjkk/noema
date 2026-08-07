from __future__ import annotations

import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class EvolutionEngine:
    def __init__(self) -> None:
        self._outcomes: list[dict[str, Any]] = []

    async def record_outcome(
        self,
        task: Any,
        hypothesis: dict,
        is_successful: bool,
        violations: list[str] | None = None,
    ) -> None:
        self._outcomes.append(
            {
                "timestamp": time.time(),
                "is_successful": is_successful,
                "violations": violations or [],
                "hypothesis_keys": list(hypothesis.keys()),
            }
        )
        logger.info(
            "neurosymbolic_outcome_recorded",
            is_successful=is_successful,
            total_outcomes=len(self._outcomes),
        )

    def get_stats(self) -> dict[str, Any]:
        if not self._outcomes:
            return {"total": 0, "success_rate": 0.0}
        successful = sum(1 for o in self._outcomes if o["is_successful"])
        return {
            "total": len(self._outcomes),
            "successful": successful,
            "failed": len(self._outcomes) - successful,
            "success_rate": successful / len(self._outcomes),
        }
