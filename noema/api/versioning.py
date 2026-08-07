"""API versioning middleware with Sunset/Deprecation headers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

if TYPE_CHECKING:
    from fastapi import Request, Response


class APIVersionMiddleware(BaseHTTPMiddleware):
    """Adds API version headers to all responses.

    Current version: v1 (stable)
    Sunset header for deprecated versions.
    """

    CURRENT_VERSION = "v1"
    SUPPORTED_VERSIONS = ["v1"]
    DEPRECATED_VERSIONS: dict[str, str] = {}
    SUNSET_DATES: dict[str, str] = {}

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)

        response.headers["X-API-Version"] = self.CURRENT_VERSION
        response.headers["X-API-Supported-Versions"] = ", ".join(self.SUPPORTED_VERSIONS)

        path = str(request.url.path)
        for version, sunset_date in self.SUNSET_DATES.items():
            if path.startswith(f"/api/{version}/"):
                response.headers["Sunset"] = sunset_date
                response.headers["Deprecation"] = "true"
                break

        return response
