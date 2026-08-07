"""Cache-Control and ETag middleware for API responses."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

if TYPE_CHECKING:
    from fastapi import Request, Response


class CacheControlMiddleware(BaseHTTPMiddleware):
    """Adds Cache-Control and ETag headers to cacheable GET responses.

    Configuration (path → max_age in seconds):
        /health: 10s
        /admin/metrics: 5s
        /knowledge/stats: 30s
        /features: 30s
        /kernels: 60s
        /agents: 60s
    """

    CACHE_CONFIG: dict[str, int] = {
        "/health": 10,
        "/admin/metrics": 5,
        "/knowledge/stats": 30,
        "/features": 30,
        "/kernels": 60,
        "/agents": 60,
    }

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)

        if request.method != "GET":
            return response

        path = str(request.url.path)

        # Check if this path has a cache config
        max_age = None
        for cached_path, age in self.CACHE_CONFIG.items():
            if path == cached_path or path.startswith(cached_path + "/"):
                max_age = age
                break

        if max_age is not None and response.status_code < 400:
            # Generate ETag from response body
            body_iterator = getattr(response, "body_iterator", None)
            if body_iterator is None:
                return response
            body = b""
            async for chunk in body_iterator:
                body += chunk
            etag = hashlib.md5(body).hexdigest()

            # Check If-None-Match
            if_none_match = request.headers.get("If-None-Match", "")
            if if_none_match.strip('"') == etag:
                from starlette.responses import Response as StarResponse

                return StarResponse(
                    status_code=304,
                    headers={
                        "ETag": f'"{etag}"',
                        "Cache-Control": f"public, max-age={max_age}",
                    },
                )

            # Rebuild response with body consumed from iterator
            from starlette.responses import Response as StarResponse

            new_resp = StarResponse(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )
            new_resp.headers["Cache-Control"] = f"public, max-age={max_age}, must-revalidate"
            new_resp.headers["ETag"] = f'"{etag}"'
            return new_resp

        return response
