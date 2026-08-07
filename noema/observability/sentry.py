"""Sentry integration with structured breadcrumbs."""

from __future__ import annotations

import logging

import structlog

logger = structlog.get_logger(__name__)


def init_sentry(
    dsn: str,
    environment: str = "production",
    traces_sample_rate: float = 0.1,
    profiles_sample_rate: float = 0.05,
) -> bool:
    """Initialize Sentry SDK. Returns True if initialized."""
    if not dsn:
        logger.info("sentry_not_configured")
        return False
    try:
        import sentry_sdk
        from sentry_sdk.integrations.asyncio import AsyncioIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
        from sentry_sdk.integrations.structlog import StructlogIntegration

        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            traces_sample_rate=traces_sample_rate,
            profiles_sample_rate=profiles_sample_rate,
            integrations=[
                StructlogIntegration(),
                AsyncioIntegration(),
                LoggingIntegration(level=logging.WARNING, event_level=logging.ERROR),
            ],
            send_default_pii=False,
            attach_stacktrace=True,
        )
        logger.info("sentry_initialized", environment=environment)
        return True
    except ImportError:
        logger.warning("sentry_sdk not installed")
        return False
    except Exception as e:
        logger.error("sentry_init_failed", error=str(e))
        return False
