"""Framework-wide exception hierarchy.

Every domain-specific error derives from :class:`NoemaError` so that
infrastructure and application layers can handle any framework failure with a
single, typed base class while never accidentally swallowing
:class:`KeyboardInterrupt` / :class:`SystemExit`.

The base error carries an optional structured ``context`` payload that is safe
to expose through logs or APIs (never contains secrets or stack traces).
"""

from __future__ import annotations

from typing import Any


class NoemaError(Exception):
    """Base class for all Noema errors.

    Args:
        message: Human-readable error message.
        context: Optional structured, JSON-serializable metadata about the
            failure. This payload is intended to be surfaced in logs/APIs.

    Attributes:
        message: The human-readable error message.
        context: Structured error metadata (never secrets or tracebacks).
    """

    def __init__(self, message: str = "", *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context or {}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for logging or API responses.

        Returns:
            A dict with ``error`` (class name), ``message`` and every key from
            the structured ``context`` payload.
        """
        return {"error": type(self).__name__, "message": self.message, **self.context}
