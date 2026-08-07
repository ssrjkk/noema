"""Comprehensive tests for API infrastructure modules: middleware, rate limiting,
problem details, versioning, diagnostics, and cache headers."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from noema.api.cache_headers import CacheControlMiddleware
from noema.api.middleware import (
    RequestIDMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)
from noema.api.problem import ProblemResponse, problem_response
from noema.api.rate_limit import RateLimitMiddleware, _SlidingWindowCounter
from noema.api.versioning import APIVersionMiddleware
from noema.config.settings import NoemaSettings, reset_settings

_saved_settings = None


def _override_api_settings(**kwargs):
    """Replace the settings singleton with one that has overridden API values."""
    global _saved_settings
    import noema.config.settings as mod

    _saved_settings = mod._settings
    reset_settings()
    s = NoemaSettings()
    for key, val in kwargs.items():
        setattr(s.api, key, val)
    mod._settings = s


def _restore_settings():
    """Restore the original settings singleton."""
    global _saved_settings
    if _saved_settings is not None:
        import noema.config.settings as mod

        mod._settings = _saved_settings
        _saved_settings = None


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _restore_settings_after():
    """Restore global settings after each test that uses _override_api_settings."""
    yield
    _restore_settings()


@pytest.fixture
def app():
    """Minimal FastAPI app with all test routes."""
    app = FastAPI()

    @app.get("/hello")
    async def hello():
        return {"message": "hello"}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/admin/metrics")
    async def metrics():
        return {"cpu": 0.5, "mem": 0.3}

    @app.get("/knowledge/stats")
    async def knowledge_stats():
        return {"documents": 10}

    @app.get("/features")
    async def features():
        return {"feature_x": True}

    @app.get("/kernels")
    async def kernels():
        return {"kernels": []}

    @app.get("/agents")
    async def agents():
        return {"agents": []}

    @app.post("/echo")
    async def echo(request: Request):
        body = await request.body()
        return {"received": body.decode()}

    return app


@pytest.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── RequestSizeLimitMiddleware ───────────────────────────────────────────────


class TestRequestSizeLimitMiddleware:
    async def test_rejects_body_exceeding_limit(self, app: FastAPI):
        _override_api_settings(max_request_body=10)
        app.add_middleware(RequestSizeLimitMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/echo", content=b"x" * 11)
        assert resp.status_code == 413
        body = resp.json()
        assert body["error"] == "payload_too_large"
        assert "10 bytes" in body["message"]

    async def test_allows_body_within_limit(self, app: FastAPI):
        _override_api_settings(max_request_body=100)
        app.add_middleware(RequestSizeLimitMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/echo", content=b"hello")
        assert resp.status_code == 200
        assert resp.json()["received"] == "hello"

    async def test_missing_content_length_is_allowed(self, app: FastAPI):
        """The middleware must not crash when content-length header is missing."""
        _override_api_settings(max_request_body=10)
        app.add_middleware(RequestSizeLimitMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/echo", content=None)
        assert resp.status_code == 200

    async def test_get_request_no_body(self, app: FastAPI):
        """GET requests with no body should pass through."""
        _override_api_settings(max_request_body=10)
        app.add_middleware(RequestSizeLimitMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/hello")
        assert resp.status_code == 200


# ── Sliding Window Counter (unit tests) ──────────────────────────────────────


class TestSlidingWindowCounter:
    async def test_increment_and_decrement(self):
        counter = _SlidingWindowCounter(window_seconds=60, max_requests=5)
        allowed, headers = await counter.allow("client1")
        assert allowed is True
        assert int(headers["X-RateLimit-Remaining"]) == 4
        assert headers["X-RateLimit-Limit"] == "5"

        allowed2, headers2 = await counter.allow("client1")
        assert allowed2 is True
        assert int(headers2["X-RateLimit-Remaining"]) == 3

    async def test_rate_limit_exceeded_returns_false(self):
        counter = _SlidingWindowCounter(window_seconds=60, max_requests=3)
        for _ in range(3):
            await counter.allow("client2")
        allowed, headers = await counter.allow("client2")
        assert allowed is False
        assert headers["X-RateLimit-Remaining"] == "0"
        assert "Retry-After" in headers

    async def test_rate_limit_allowed_within_window(self):
        counter = _SlidingWindowCounter(window_seconds=60, max_requests=5)
        for _ in range(5):
            allowed, _ = await counter.allow("client3")
            assert allowed is True

    async def test_different_clients_independent(self):
        counter = _SlidingWindowCounter(window_seconds=60, max_requests=2)
        assert (await counter.allow("alice"))[0] is True
        assert (await counter.allow("alice"))[0] is True
        assert (await counter.allow("alice"))[0] is False
        assert (await counter.allow("bob"))[0] is True
        assert (await counter.allow("bob"))[0] is True

    async def test_stale_key_cleanup(self):
        counter = _SlidingWindowCounter(window_seconds=0.1, max_requests=5)
        now = time.monotonic()
        counter._hits["stale_client"] = [now - 0.5]  # 500ms old, well beyond cleanup cutoff
        counter.start_cleanup()
        import asyncio

        await asyncio.sleep(0.25)  # wait for at least one cleanup cycle (0.2s)
        counter.stop_cleanup()
        assert "stale_client" not in counter._hits

    async def test_prune_removes_expired_entries(self):
        counter = _SlidingWindowCounter(window_seconds=1, max_requests=5)
        now = time.monotonic()
        # Insert a stale timestamp manually (1.5x window in the past)
        counter._hits["quick"] = [now - counter.window * 1.5]
        counter._prune("quick", now)
        assert "quick" not in counter._hits


# ── RateLimitMiddleware ──────────────────────────────────────────────────────


class TestRateLimitMiddleware:
    async def test_rate_limit_allowed(self, app: FastAPI):
        _override_api_settings(rate_limit_enabled=True, rate_limit_rpm=100)
        app.add_middleware(RateLimitMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/hello", headers={"X-API-Key": "test-key-1"})
        assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers
        assert "X-RateLimit-Reset" in resp.headers

    async def test_rate_limit_exceeded_returns_429(self, app: FastAPI):
        _override_api_settings(rate_limit_enabled=True, rate_limit_rpm=2)
        app.add_middleware(RateLimitMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp1 = await client.get("/hello", headers={"X-API-Key": "limited-key"})
            assert resp1.status_code == 200
            resp2 = await client.get("/hello", headers={"X-API-Key": "limited-key"})
            assert resp2.status_code == 200
            resp3 = await client.get("/hello", headers={"X-API-Key": "limited-key"})
            assert resp3.status_code == 429
            body = resp3.json()
            assert body["error"] == "rate_limit_exceeded"

    async def test_public_paths_not_rate_limited(self, app: FastAPI):
        _override_api_settings(rate_limit_enabled=True, rate_limit_rpm=0)
        app.add_middleware(RateLimitMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200

    async def test_disabled_rate_limit_passes_all(self, app: FastAPI):
        _override_api_settings(rate_limit_enabled=False)
        app.add_middleware(RateLimitMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for _ in range(10):
                resp = await client.get("/hello")
                assert resp.status_code == 200

    async def test_ip_based_client_key(self, app: FastAPI):
        _override_api_settings(rate_limit_enabled=True, rate_limit_rpm=1)
        app.add_middleware(RateLimitMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/hello", headers={"X-Forwarded-For": "10.0.0.1"})
            assert resp.status_code == 200
            resp2 = await client.get("/hello", headers={"X-Forwarded-For": "10.0.0.2"})
            assert resp2.status_code == 200
            resp3 = await client.get("/hello", headers={"X-Forwarded-For": "10.0.0.1"})
            assert resp3.status_code == 429


# ── ProblemResponse ──────────────────────────────────────────────────────────


class TestProblemResponse:
    def test_status_code_mapping(self):
        resp = ProblemResponse(status=404, title="Not Found", detail="Resource missing")
        assert resp.status_code == 404
        assert resp.media_type == "application/problem+json"

    def test_media_type(self):
        assert ProblemResponse.media_type == "application/problem+json"
        resp = ProblemResponse(status=400, title="Bad Request")
        assert resp.media_type == "application/problem+json"

    def test_rfc_7807_format(self):
        resp = ProblemResponse(
            status=422, title="Validation Error", detail="field required", instance="/api/foo"
        )
        body = resp.body
        assert b'"type":"about:blank"' in body
        assert b'"title":"Validation Error"' in body
        assert b'"status":422' in body
        assert b'"detail":"field required"' in body
        assert b'"instance":"/api/foo"' in body

    def test_extra_fields_included(self):
        resp = ProblemResponse(
            status=403, title="Forbidden", extra={"errors": [{"field": "name", "msg": "required"}]}
        )
        body = resp.body.decode()
        assert '"errors"' in body
        assert '"field":"name"' in body

    def test_problem_response_function(self):
        resp = problem_response(status=500, title="Internal Error", detail="something broke")
        assert isinstance(resp, ProblemResponse)
        assert resp.status_code == 500
        assert b'"title":"Internal Error"' in resp.body

    def test_validation_error_format(self):
        resp = ProblemResponse(
            status=422,
            title="Validation Error",
            detail="Request validation failed",
            extra={
                "errors": [
                    {
                        "loc": ["body", "name"],
                        "msg": "field required",
                        "type": "value_error.missing",
                    },
                ]
            },
        )
        body = resp.body.decode()
        assert "field required" in body
        assert "value_error.missing" in body

    def test_500_error_returns_problem(self):
        resp = problem_response(
            status=500, title="Internal Server Error", detail="unexpected condition"
        )
        assert resp.status_code == 500
        assert b'"type":"about:blank"' in resp.body
        assert b'"status":500' in resp.body


# ── APIVersionMiddleware ─────────────────────────────────────────────────────


class TestAPIVersionMiddleware:
    async def test_version_header_added(self, app: FastAPI):
        app.add_middleware(APIVersionMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/hello")
        assert resp.headers.get("X-API-Version") == "v1"

    async def test_supported_versions_header(self, app: FastAPI):
        app.add_middleware(APIVersionMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/hello")
        assert resp.headers.get("X-API-Supported-Versions") == "v1"

    async def test_sunset_and_deprecation_headers_for_old_versions(self, app: FastAPI):
        APIVersionMiddleware.SUNSET_DATES = {"v0": "Sat, 31 Dec 2025 23:59:59 GMT"}
        APIVersionMiddleware.DEPRECATED_VERSIONS = {"v0": "1.0.0"}
        app.add_middleware(APIVersionMiddleware)

        @app.get("/api/v0/legacy")
        async def legacy():
            return {"legacy": True}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v0/legacy")
        assert resp.headers.get("Sunset") == "Sat, 31 Dec 2025 23:59:59 GMT"
        assert resp.headers.get("Deprecation") == "true"

        APIVersionMiddleware.SUNSET_DATES = {}
        APIVersionMiddleware.DEPRECATED_VERSIONS = {}

    async def test_current_version_no_sunset(self, app: FastAPI):
        APIVersionMiddleware.SUNSET_DATES = {"v0": "old-date"}
        app.add_middleware(APIVersionMiddleware)

        @app.get("/api/v1/current")
        async def current():
            return {"current": True}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/current")
        assert "Sunset" not in resp.headers
        assert "Deprecation" not in resp.headers

        APIVersionMiddleware.SUNSET_DATES = {}


# ── CacheControlMiddleware ───────────────────────────────────────────────────


class TestCacheControlMiddleware:
    async def test_cache_control_added_for_get_cached_path(self, app: FastAPI):
        app.add_middleware(CacheControlMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        cc = resp.headers.get("Cache-Control", "")
        assert "public" in cc
        assert "max-age=10" in cc

    async def test_etag_computed_from_body(self, app: FastAPI):
        app.add_middleware(CacheControlMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.headers.get("ETag", "").startswith('"')
        assert resp.headers["ETag"].endswith('"')

    async def test_if_none_match_returns_304(self, app: FastAPI):
        app.add_middleware(CacheControlMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp1 = await client.get("/health")
        etag = resp1.headers["ETag"]

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp2 = await client.get("/health", headers={"If-None-Match": etag})
        assert resp2.status_code == 304
        assert resp2.headers["ETag"] == etag

    async def test_non_cached_path_no_cache_control(self, app: FastAPI):
        app.add_middleware(CacheControlMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/hello")
        assert "Cache-Control" not in resp.headers

    async def test_non_get_methods_bypass_caching(self, app: FastAPI):
        app.add_middleware(CacheControlMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/echo", content=b"test")
        assert "Cache-Control" not in resp.headers

    async def test_error_responses_not_cached(self, app: FastAPI):
        @app.get("/error")
        async def error():
            return JSONResponse(status_code=500, content={"error": "fail"})

        app.add_middleware(CacheControlMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/error")
        assert resp.status_code == 500
        assert "Cache-Control" not in resp.headers


# ── RequestIDMiddleware ──────────────────────────────────────────────────────


class TestRequestIDMiddleware:
    async def test_correlation_id_injected(self, app: FastAPI):
        app.add_middleware(RequestIDMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/hello")
        assert "X-Request-ID" in resp.headers
        assert len(resp.headers["X-Request-ID"]) == 16

    async def test_request_id_preserved(self, app: FastAPI):
        app.add_middleware(RequestIDMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/hello", headers={"X-Request-ID": "my-custom-id"})
        assert resp.headers["X-Request-ID"] == "my-custom-id"


# ── SecurityHeadersMiddleware ────────────────────────────────────────────────


class TestSecurityHeadersMiddleware:
    async def test_security_headers_added(self, app: FastAPI):
        app.add_middleware(SecurityHeadersMiddleware)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/hello")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("X-XSS-Protection") == "1; mode=block"
        assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        assert resp.headers.get("Cache-Control") == "no-store, no-cache, must-revalidate"
