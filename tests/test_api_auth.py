"""Tests for API key authentication middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.responses import JSONResponse, Response

from noema.api.auth import APIKeyAuthMiddleware, _PUBLIC_PATHS


def _make_request(path: str = "/api/tasks", headers: dict | None = None) -> MagicMock:
    req = MagicMock()
    req.url.path = path
    req.headers = MagicMock()
    req.headers.get = MagicMock(side_effect=lambda key, default="": (headers or {}).get(key, default))
    return req


@pytest.fixture
def call_next():
    return AsyncMock(return_value=Response(status_code=200, content="OK"))


@pytest.mark.asyncio
async def test_public_path_bypasses_auth(call_next):
    mw = APIKeyAuthMiddleware(MagicMock())
    for path in _PUBLIC_PATHS:
        req = _make_request(path)
        resp = await mw.dispatch(req, call_next)
        assert resp.status_code == 200
    assert call_next.call_count >= len(_PUBLIC_PATHS)


@pytest.mark.asyncio
async def test_no_auth_configured_passes_through(call_next):
    settings = MagicMock()
    settings.api.api_keys = []
    settings.api.api_key.get_secret_value.return_value = ""
    settings.api.api_key_header = "X-API-Key"

    mw = APIKeyAuthMiddleware(MagicMock())
    req = _make_request("/api/data")

    with patch("noema.api.auth.get_settings", return_value=settings):
        resp = await mw.dispatch(req, call_next)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_single_master_key_valid(call_next):
    settings = MagicMock()
    settings.api.api_keys = []
    settings.api.api_key.get_secret_value.return_value = "secret-key-123"
    settings.api.api_key_header = "X-API-Key"

    mw = APIKeyAuthMiddleware(MagicMock())
    req = _make_request("/api/data", {"X-API-Key": "secret-key-123"})

    with patch("noema.api.auth.get_settings", return_value=settings):
        with patch("noema.api.auth.set_tenant_id", return_value="token"):
            with patch("noema.api.auth.reset_tenant_id"):
                resp = await mw.dispatch(req, call_next)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_single_master_key_missing(call_next):
    settings = MagicMock()
    settings.api.api_keys = []
    settings.api.api_key.get_secret_value.return_value = "secret-key-123"
    settings.api.api_key_header = "X-API-Key"

    mw = APIKeyAuthMiddleware(MagicMock())
    req = _make_request("/api/data", {})

    with patch("noema.api.auth.get_settings", return_value=settings):
        resp = await mw.dispatch(req, call_next)
    assert resp.status_code == 401
    body = resp.body
    assert b"missing_api_key" in body


@pytest.mark.asyncio
async def test_single_master_key_invalid(call_next):
    settings = MagicMock()
    settings.api.api_keys = []
    settings.api.api_key.get_secret_value.return_value = "secret-key-123"
    settings.api.api_key_header = "X-API-Key"

    mw = APIKeyAuthMiddleware(MagicMock())
    req = _make_request("/api/data", {"X-API-Key": "wrong-key"})

    with patch("noema.api.auth.get_settings", return_value=settings):
        resp = await mw.dispatch(req, call_next)
    assert resp.status_code == 403
    assert b"invalid_api_key" in resp.body


@pytest.mark.asyncio
async def test_multi_tenant_valid_key(call_next):
    binding = MagicMock()
    binding.key.get_secret_value.return_value = "tenant-key-A"
    binding.tenant_id = "tenant-A"

    settings = MagicMock()
    settings.api.api_keys = [binding]
    settings.api.api_key_header = "X-API-Key"

    mw = APIKeyAuthMiddleware(MagicMock())
    req = _make_request("/api/data", {"X-API-Key": "tenant-key-A"})

    with patch("noema.api.auth.get_settings", return_value=settings):
        with patch("noema.api.auth.set_tenant_id", return_value="token") as mock_set:
            with patch("noema.api.auth.reset_tenant_id") as mock_reset:
                resp = await mw.dispatch(req, call_next)
    assert resp.status_code == 200
    mock_set.assert_called_once_with("tenant-A")
    mock_reset.assert_called_once()


@pytest.mark.asyncio
async def test_multi_tenant_missing_key(call_next):
    binding = MagicMock()
    binding.key.get_secret_value.return_value = "tenant-key-A"
    binding.tenant_id = "tenant-A"

    settings = MagicMock()
    settings.api.api_keys = [binding]
    settings.api.api_key_header = "X-API-Key"

    mw = APIKeyAuthMiddleware(MagicMock())
    req = _make_request("/api/data", {})

    with patch("noema.api.auth.get_settings", return_value=settings):
        resp = await mw.dispatch(req, call_next)
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_multi_tenant_invalid_key(call_next):
    binding = MagicMock()
    binding.key.get_secret_value.return_value = "tenant-key-A"
    binding.tenant_id = "tenant-A"

    settings = MagicMock()
    settings.api.api_keys = [binding]
    settings.api.api_key_header = "X-API-Key"

    mw = APIKeyAuthMiddleware(MagicMock())
    req = _make_request("/api/data", {"X-API-Key": "wrong-tenant-key"})

    with patch("noema.api.auth.get_settings", return_value=settings):
        resp = await mw.dispatch(req, call_next)
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_multi_tenant_tenant_context_restored_on_error():
    binding = MagicMock()
    binding.key.get_secret_value.return_value = "tenant-key-A"
    binding.tenant_id = "tenant-A"

    settings = MagicMock()
    settings.api.api_keys = [binding]
    settings.api.api_key_header = "X-API-Key"

    failing_call_next = AsyncMock(side_effect=RuntimeError("boom"))

    mw = APIKeyAuthMiddleware(MagicMock())
    req = _make_request("/api/data", {"X-API-Key": "tenant-key-A"})

    with patch("noema.api.auth.get_settings", return_value=settings):
        with patch("noema.api.auth.set_tenant_id", return_value="token"):
            with patch("noema.api.auth.reset_tenant_id") as mock_reset:
                with pytest.raises(RuntimeError, match="boom"):
                    await mw.dispatch(req, failing_call_next)
    mock_reset.assert_called_once_with("token")
