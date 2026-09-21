"""API key authentication middleware for FastAPI."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from noema.config.settings import get_settings
from noema.context import reset_tenant_id, set_tenant_id

if TYPE_CHECKING:
    from fastapi import Request, Response

# Paths that NEVER require authentication
_PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/",
        "/health",
        "/ready",
        "/docs",
        "/openapi.json",
        "/redoc",
    }
)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Validates API key from header on every request.

    Supports two modes:
    1. Single master key (settings.api.api_key) — all requests use tenant "default"
    2. Multi-tenant keys (settings.api.api_keys) — each key maps to a specific tenant_id

    If both are empty, auth is disabled (dev mode).
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        settings = get_settings()

        # Public paths bypass auth
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Extract key
        provided_key = request.headers.get(settings.api.api_key_header, "")

        # Multi-tenant mode: check api_keys list first
        if settings.api.api_keys:
            if not provided_key:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "missing_api_key",
                        "message": f"Provide key via {settings.api.api_key_header} header",
                    },
                )

            # Find matching key binding
            tenant_id = None
            for binding in settings.api.api_keys:
                expected = binding.key.get_secret_value()
                if expected and secrets.compare_digest(provided_key, expected):
                    tenant_id = binding.tenant_id
                    break

            if tenant_id is None:
                return JSONResponse(
                    status_code=403,
                    content={"error": "invalid_api_key", "message": "API key is invalid"},
                )

            # Set tenant context for this request
            token = set_tenant_id(tenant_id)
            try:
                response = await call_next(request)
                return response
            finally:
                reset_tenant_id(token)

        # Single master key mode (legacy)
        master_key = settings.api.api_key.get_secret_value()
        if master_key:
            if not provided_key:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "missing_api_key",
                        "message": f"Provide key via {settings.api.api_key_header} header",
                    },
                )

            if not secrets.compare_digest(provided_key, master_key):
                return JSONResponse(
                    status_code=403,
                    content={"error": "invalid_api_key", "message": "API key is invalid"},
                )

            # Master key uses "default" tenant
            token = set_tenant_id("default")
            try:
                response = await call_next(request)
                return response
            finally:
                reset_tenant_id(token)

        # Auth disabled (dev mode)
        return await call_next(request)
