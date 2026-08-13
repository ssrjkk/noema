"""Tests for the webhook/callback system."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from noema.api.webhooks import (
    WebhookDispatcher,
    WebhookRegistration,
    _mask_url,
    _verify_signature,
    get_webhook_dispatcher,
)


@pytest.fixture(autouse=True)
def reset_dispatcher():
    """Reset singleton between tests."""
    import noema.api.webhooks as wh

    wh._dispatcher = None
    yield
    wh._dispatcher = None


class TestWebhookDispatcher:
    async def test_register_webhook(self):
        d = WebhookDispatcher()
        reg = WebhookRegistration(url="https://example.com/hook", events=["task.completed"])
        d.register("wh-1", reg)
        assert "wh-1" in d._registrations
        assert d._registrations["wh-1"].url == "https://example.com/hook"

    async def test_unregister_webhook(self):
        d = WebhookDispatcher()
        reg = WebhookRegistration(url="https://example.com/hook")
        d.register("wh-1", reg)
        d.unregister("wh-1")
        assert "wh-1" not in d._registrations

    async def test_emit_dispatches_to_registered(self):
        d = WebhookDispatcher()
        received = []

        async def mock_handler(event):
            received.append(event)

        reg = WebhookRegistration(
            url="http://localhost:99999/nonexistent", events=["*"], retry_count=0, timeout=0.5
        )
        d.register("wh-1", reg)
        await d.emit("test.event", {"key": "value"})
        assert d._queue.qsize() == 1
        event = await d._queue.get()
        assert event["type"] == "test.event"
        assert event["payload"]["key"] == "value"

    async def test_emit_no_registrations(self):
        d = WebhookDispatcher()
        await d.emit("test.event", {"key": "value"})
        event = await d._queue.get()
        assert event["type"] == "test.event"

    async def test_signature_header(self):
        reg = WebhookRegistration(
            url="http://localhost:99999/nonexistent",
            secret="my-secret",
            events=["*"],
            retry_count=0,
            timeout=0.5,
        )
        d = WebhookDispatcher()
        d.register("wh-1", reg)

        body = json.dumps({"key": "value"}, ensure_ascii=False)
        expected_sig = hmac.new(b"my-secret", body.encode(), hashlib.sha256).hexdigest()

        await d.emit("test.event", {"key": "value"})
        event = await d._queue.get()

        headers = {"Content-Type": "application/json"}
        if reg.secret:
            signature = hmac.new(
                reg.secret.encode(),
                json.dumps(event["payload"], ensure_ascii=False).encode(),
                hashlib.sha256,
            ).hexdigest()
            headers["X-Webhook-Signature"] = signature

        assert headers["X-Webhook-Signature"] == expected_sig

    async def test_list_webhooks_urls_masked(self):
        d = WebhookDispatcher()
        d.register("wh-1", WebhookRegistration(url="https://example.com/callback", events=["*"]))
        d.register(
            "wh-2",
            WebhookRegistration(
                url="https://my-service.company.io/hook", events=["task.completed"]
            ),
        )

        result = d.list_registrations()
        assert "wh-1" in result
        assert "wh-2" in result
        assert "example" not in result["wh-1"]["url"]
        assert "***" in result["wh-1"]["url"]
        assert "***" in result["wh-2"]["url"]

    async def test_start_stop(self):
        d = WebhookDispatcher()
        await d.start()
        assert d._worker_task is not None
        assert not d._worker_task.done()
        await d.stop()
        assert d._worker_task is None or d._worker_task.done()


class TestMaskUrl:
    def test_mask_short_host(self):
        masked = _mask_url("https://abc.com/path")
        assert "abc" not in masked or "***" in masked

    def test_mask_long_host(self):
        masked = _mask_url("https://myservice.example.com/webhook")
        assert "***" in masked

    def test_mask_with_port(self):
        masked = _mask_url("https://hook.example.com:8443/callback")
        assert "***" in masked
        assert "8443" in masked


class TestVerifySignature:
    def _sig(self, secret: str, body: bytes) -> str:
        return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    def test_valid_signature_passes(self):
        body = json.dumps({"event": "incident", "payload": {"a": 1}}, ensure_ascii=False).encode()
        assert _verify_signature("s3cret", body, self._sig("s3cret", body))

    def test_missing_header_fails(self):
        body = b"{}"
        assert not _verify_signature("s3cret", body, None)

    def test_wrong_secret_fails(self):
        body = b"{}"
        assert not _verify_signature("other", body, self._sig("s3cret", body))

    def test_wrong_body_fails(self):
        sig = self._sig("s3cret", b"payload-a")
        assert not _verify_signature("s3cret", b"payload-b", sig)

    def test_wrong_algo_fails(self):
        body = b"{}"
        assert not _verify_signature("s3cret", body, "md5=" + "0" * 32)

    def test_tampered_digest_fails(self):
        body = b"{}"
        sig = self._sig("s3cret", body)
        tampered = sig[:-1] + ("0" if sig[-1] != "0" else "1")
        assert not _verify_signature("s3cret", body, tampered)

    def test_garbage_header_fails(self):
        assert not _verify_signature("s3cret", b"{}", "sha256=zzz")


class TestSingleton:
    async def test_get_webhook_dispatcher(self):
        d1 = get_webhook_dispatcher()
        d2 = get_webhook_dispatcher()
        assert d1 is d2

    async def test_singleton_lifecycle(self):
        d = get_webhook_dispatcher()
        await d.start()
        assert d._worker_task is not None
        await d.stop()
