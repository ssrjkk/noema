"""Tests for noema/observability/sentry.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from noema.observability.sentry import init_sentry


class TestInitSentry:
    def test_empty_dsn_returns_false(self):
        result = init_sentry("")
        assert result is False

    def test_none_dsn_returns_false(self):
        result = init_sentry(None)
        assert result is False

    def test_successful_init(self):
        mock_sentry_sdk = MagicMock()
        mock_asyncio_integration = MagicMock()
        mock_logging_integration = MagicMock()
        mock_structlog_integration = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "sentry_sdk": mock_sentry_sdk,
                "sentry_sdk.integrations.asyncio": MagicMock(
                    AsyncioIntegration=mock_asyncio_integration
                ),
                "sentry_sdk.integrations.logging": MagicMock(
                    LoggingIntegration=mock_logging_integration
                ),
                "sentry_sdk.integrations.structlog": MagicMock(
                    StructlogIntegration=mock_structlog_integration
                ),
            },
        ):
            result = init_sentry(
                dsn="https://example@sentry.io/123",
                environment="test",
                traces_sample_rate=0.2,
                profiles_sample_rate=0.1,
            )

        assert result is True
        mock_sentry_sdk.init.assert_called_once()
        call_kwargs = mock_sentry_sdk.init.call_args[1]
        assert call_kwargs["dsn"] == "https://example@sentry.io/123"
        assert call_kwargs["environment"] == "test"
        assert call_kwargs["traces_sample_rate"] == 0.2
        assert call_kwargs["profiles_sample_rate"] == 0.1
        assert call_kwargs["send_default_pii"] is False
        assert call_kwargs["attach_stacktrace"] is True

    def test_import_error_returns_false(self):
        with patch.dict("sys.modules", {"sentry_sdk": None}):
            result = init_sentry("https://example@sentry.io/123")
        assert result is False

    def test_general_exception_returns_false(self):
        mock_sentry_sdk = MagicMock()
        mock_sentry_sdk.init.side_effect = RuntimeError("init failed")

        with patch.dict(
            "sys.modules",
            {
                "sentry_sdk": mock_sentry_sdk,
                "sentry_sdk.integrations.asyncio": MagicMock(),
                "sentry_sdk.integrations.logging": MagicMock(),
                "sentry_sdk.integrations.structlog": MagicMock(),
            },
        ):
            result = init_sentry("https://example@sentry.io/123")

        assert result is False

    def test_default_parameters(self):
        mock_sentry_sdk = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "sentry_sdk": mock_sentry_sdk,
                "sentry_sdk.integrations.asyncio": MagicMock(),
                "sentry_sdk.integrations.logging": MagicMock(),
                "sentry_sdk.integrations.structlog": MagicMock(),
            },
        ):
            result = init_sentry("https://example@sentry.io/123")

        assert result is True
        call_kwargs = mock_sentry_sdk.init.call_args[1]
        assert call_kwargs["environment"] == "production"
        assert call_kwargs["traces_sample_rate"] == 0.1
        assert call_kwargs["profiles_sample_rate"] == 0.05
