"""Sandbox — изолированное исполнение и верификация сгенерированного кода.

The :class:`~noema.sandbox.environment.Environment` abstraction is the seam
for future non-code media (physics engines, hardware simulators, molecular
dynamics): implement lint/run/tests over your artifact model and plug it in.
"""

from noema.sandbox.engine import (
    CodeValidationResult,
    SandboxConfig,
    SandboxEngine,
    SandboxResult,
)
from noema.sandbox.environment import (
    DockerEnvironment,
    Environment,
    LocalEnvironment,
)

__all__ = [
    "SandboxEngine",
    "SandboxResult",
    "SandboxConfig",
    "CodeValidationResult",
    "Environment",
    "LocalEnvironment",
    "DockerEnvironment",
]
