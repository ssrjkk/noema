"""Sliding-window rate limiter for FastAPI (Redis or in-memory fallback)."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from noema.config.settings import get_settings

if TYPE_CHECKING:
    from fastapi import Request, Response


def _ip_in_trusted(ip: str, trusted: list[str]) -> bool:
    """True if ``ip`` falls in any trusted proxy IP/CIDR range."""
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for entry in trusted:
        try:
            if ipaddress.ip_address(entry) == address:
                return True
        except ValueError:
            pass
        try:
            if address in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue
    return False


class _SlidingWindowCounter:
    """In-memory sliding window counter per client key.

    For production with Redis, swap implementation to use sorted sets.
    This fallback is sufficient for single-process deployments.
    """

    def __init__(self, window_seconds: int = 60, max_requests: int = 60) -> None:
        self.window = window_seconds
        self.max_requests = max_requests
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    def _prune(self, key: str, now: float) -> None:
        cutoff = now - self.window
        hits = self._hits.get(key)
        if hits:
            self._hits[key] = [t for t in hits if t > cutoff]
            if not self._hits[key]:
                del self._hits[key]

    async def _periodic_cleanup(self) -> None:
        while True:
            await asyncio.sleep(self.window * 2)
            async with self._lock:
                now = time.monotonic()
                cutoff = now - self.window * 2
                stale_keys = [k for k, v in self._hits.items() if not v or max(v) < cutoff]
                for k in stale_keys:
                    del self._hits[k]

    def start_cleanup(self) -> None:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    def stop_cleanup(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None

    async def allow(self, key: str) -> tuple[bool, dict[str, str]]:
        """Return (allowed, headers)."""
        async with self._lock:
            now = time.monotonic()
            self._prune(key, now)
            current = len(self._hits.get(key, []))

            remaining = max(0, self.max_requests - current)
            reset_at = int(now + self.window)

            if current >= self.max_requests:
                retry_after = self.window - (now - self._hits[key][0])
                return False, {
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at),
                    "Retry-After": str(int(retry_after) + 1),
                }

            self._hits[key].append(now)
            return True, {
                "X-RateLimit-Limit": str(self.max_requests),
                "X-RateLimit-Remaining": str(remaining - 1),
                "X-RateLimit-Reset": str(reset_at),
            }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter.

    Client is identified by API key header or IP address.
    """

    _PUBLIC_PATHS: frozenset[str] = frozenset(
        {"/", "/health", "/ready", "/docs", "/openapi.json", "/redoc"}
    )

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        settings = get_settings()
        self._limiter = _SlidingWindowCounter(
            window_seconds=60,
            max_requests=settings.api.rate_limit_rpm,
        )
        self._enabled = settings.api.rate_limit_enabled
        self._key_header = settings.api.api_key_header
        self._trusted_proxies = list(settings.api.trusted_proxies)

    def _client_key(self, request: Request) -> str:
        """Extract client identifier: API key or peer IP.

        ``X-Forwarded-For`` is only trusted when the direct peer is a configured
        reverse proxy; otherwise a spoofed header could trivially bypass limits.
        """
        api_key = request.headers.get(self._key_header, "")
        if api_key:
            # Hash key to avoid storing raw secrets in memory
            return "k:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]
        peer = request.client.host if request.client else ""
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded and peer and _ip_in_trusted(peer, self._trusted_proxies):
            return "ip:" + forwarded.split(",")[0].strip()
        return "ip:" + (peer or "unknown")

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self._enabled:
            return await call_next(request)

        if self._limiter._cleanup_task is None:
            self._limiter.start_cleanup()

        if request.url.path in self._PUBLIC_PATHS:
            return await call_next(request)

        key = self._client_key(request)
        allowed, headers = await self._limiter.allow(key)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Max {self._limiter.max_requests} requests per minute",
                },
                headers=headers,
            )

        response = await call_next(request)
        for h, v in headers.items():
            response.headers[h] = v
        return response
