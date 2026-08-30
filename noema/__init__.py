"""Noema — фреймворк генерации технических решений."""

from noema.core.engine import NoemaEngine
from noema.core.types import (
    ArchitecturePattern,
    Solution,
    Task,
    TechStack,
    ThoughtProcess,
)

__version__ = "1.2.0"
__all__ = [
    "NoemaEngine",
    "Task",
    "Solution",
    "TechStack",
    "ArchitecturePattern",
    "ThoughtProcess",
]
