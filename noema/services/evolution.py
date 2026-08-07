"""Evolution service — wraps EvolutionEngine + GitEvolution with events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from noema.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from noema.core.events import EventBus
    from noema.evolution.engine import EvolutionEngine

log = get_logger(__name__)


class EvolutionService:
    """Manages self-evolution cycles with git-based proposals."""

    def __init__(
        self,
        engine: EvolutionEngine,
        event_bus: EventBus | None = None,
    ) -> None:
        self.engine = engine
        self.event_bus = event_bus

    async def analyze_self(self) -> dict[str, Any]:
        return await self.engine.analyze_self()

    async def run_cycle(self, test_runner: Callable | None = None) -> dict[str, Any]:
        result = await self.engine.run_evolution_cycle(test_runner=test_runner)
        if self.event_bus:
            await self.event_bus.emit(
                "evolution.cycle_completed",
                {
                    "cycle": result.evolution_cycle,
                    "generated": result.patches_generated,
                    "applied": result.patches_applied,
                    "rejected": result.patches_rejected,
                },
                source="evolution_service",
            )
        return result.model_dump()

    def get_stats(self) -> dict[str, Any]:
        return self.engine.get_stats()
