"""API key authentication middleware for FastAPI."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from noema.config.settings import get_settings

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

    If ``settings.api.api_key`` is empty, auth is disabled (dev mode).
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        settings = get_settings()

        # Auth disabled in dev
        if not settings.api.api_key.get_secret_value():
            return await call_next(request)

        # Public paths bypass auth
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Extract key
        provided_key = request.headers.get(settings.api.api_key_header, "")

        if not provided_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "missing_api_key",
                    "message": f"Provide key via {settings.api.api_key_header} header",
                },
            )

        # Constant-time comparison to prevent timing attacks
        expected = settings.api.api_key.get_secret_value()
        if not secrets.compare_digest(provided_key, expected):
            return JSONResponse(
                status_code=403,
                content={"error": "invalid_api_key", "message": "API key is invalid"},
            )

        return await call_next(request)
