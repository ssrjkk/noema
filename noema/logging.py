"""Structured logging with structlog + correlation IDs."""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from datetime import UTC
from typing import Any, cast

import structlog

from noema.config.settings import get_settings

# ─── Correlation ID ──────────────────────────────────────────────────────
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    return _correlation_id.get()


def set_correlation_id(cid: str | None = None) -> str:
    cid = cid or uuid.uuid4().hex[:16]
    _correlation_id.set(cid)
    return cid


# ─── Processor chain ─────────────────────────────────────────────────────
def _add_correlation_id(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    cid = get_correlation_id()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


def _add_log_level(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    event_dict["log_level"] = method_name.upper()
    return event_dict


def _timestamper(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    from datetime import datetime

    event_dict["timestamp"] = datetime.now(UTC).isoformat()
    return event_dict


def setup_logging() -> None:
    """Configure structlog + stdlib logging."""
    settings = get_settings()
    level = settings.obs.logging_level.upper()

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        _add_correlation_id,
        _add_log_level,
        _timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.obs.logging_format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.INFO)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a bound logger with optional name."""
    return cast("structlog.stdlib.BoundLogger", structlog.get_logger(name))
