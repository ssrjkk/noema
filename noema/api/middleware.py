"""Request/response middleware: correlation ID, CORS, body size limit."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from noema.config.settings import get_settings
from noema.logging import set_correlation_id

if TYPE_CHECKING:
    from fastapi import Request, Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Injects correlation ID into every request context."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        cid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:16])
        set_correlation_id(cid)
        request.state.correlation_id = cid

        response = await call_next(request)
        response.headers["X-Request-ID"] = cid
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects requests exceeding configured body size."""

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self._max_body = get_settings().api.max_request_body

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > self._max_body:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "payload_too_large",
                    "message": f"Max body size: {self._max_body} bytes",
                },
            )
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds standard security headers to every response."""

    _HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Cache-Control": "no-store, no-cache, must-revalidate",
    }

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        for key, val in self._HEADERS.items():
            response.headers[key] = val
        return response
