"""Webhook/callback system for task completion events."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog
from fastapi import APIRouter

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@dataclass
class WebhookRegistration:
    url: str
    secret: str = ""
    events: list[str] = field(default_factory=lambda: ["*"])
    retry_count: int = 3
    timeout: float = 10.0


class WebhookDispatcher:
    def __init__(self) -> None:
        self._registrations: dict[str, WebhookRegistration] = {}
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        if not self._worker_task:
            self._worker_task = asyncio.create_task(self._dispatch_loop())
            logger.info("webhook_dispatcher_started")

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            logger.info("webhook_dispatcher_stopped")

    def register(self, webhook_id: str, registration: WebhookRegistration) -> None:
        self._registrations[webhook_id] = registration
        logger.info(
            "webhook_registered",
            webhook_id=webhook_id,
            url=registration.url,
            events=registration.events,
        )

    def unregister(self, webhook_id: str) -> None:
        self._registrations.pop(webhook_id, None)

    def list_registrations(self) -> dict[str, dict[str, Any]]:
        result = {}
        for wid, reg in self._registrations.items():
            result[wid] = {
                "url": _mask_url(reg.url),
                "events": reg.events,
                "retry_count": reg.retry_count,
                "timeout": reg.timeout,
            }
        return result

    async def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        await self._queue.put({"type": event_type, "payload": payload, "timestamp": time.time()})

    async def _dispatch_loop(self) -> None:
        while True:
            try:
                event = await self._queue.get()
                await self._dispatch_event(event)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("webhook_dispatch_loop_error", error=str(e))

    async def _dispatch_event(self, event: dict[str, Any]) -> None:
        event_type = event["type"]
        payload = event["payload"]
        body = json.dumps(payload, ensure_ascii=False)

        import aiohttp  # noqa: PLC0415

        async with aiohttp.ClientSession() as session:
            for wid, reg in self._registrations.items():
                if "*" not in reg.events and event_type not in reg.events:
                    continue
                for attempt in range(reg.retry_count):
                    try:
                        headers = {"Content-Type": "application/json"}
                        if reg.secret:
                            signature = hmac.new(
                                reg.secret.encode(),
                                body.encode(),
                                hashlib.sha256,
                            ).hexdigest()
                            headers["X-Webhook-Signature"] = signature

                        async with session.post(
                            reg.url,
                            data=body,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=reg.timeout),
                        ) as resp:
                            if resp.status < 500:
                                logger.info(
                                    "webhook_delivered",
                                    webhook_id=wid,
                                    event_type=event_type,
                                    status=resp.status,
                                    attempt=attempt + 1,
                                )
                                break
                            logger.warning(
                                "webhook_retry",
                                webhook_id=wid,
                                event_type=event_type,
                                status=resp.status,
                                attempt=attempt + 1,
                            )
                    except TimeoutError:
                        logger.warning(
                            "webhook_timeout",
                            webhook_id=wid,
                            event_type=event_type,
                            attempt=attempt + 1,
                        )
                    except Exception as e:
                        logger.error(
                            "webhook_error",
                            webhook_id=wid,
                            event_type=event_type,
                            error=str(e),
                            attempt=attempt + 1,
                        )


def _mask_url(url: str) -> str:
    """Mask the middle portion of a URL for safe listing."""
    try:
        from urllib.parse import urlparse  # noqa: PLC0415

        parsed = urlparse(url)
        host = parsed.hostname or ""
        if len(host) <= 6:
            masked_host = host[:2] + "***" + host[-2:] if len(host) > 4 else host[:1] + "***"
        else:
            masked_host = host[:3] + "***" + host[-3:]
        return f"{parsed.scheme}://{masked_host}{'@' + masked_host if parsed.username else ''}{':' + str(parsed.port) if parsed.port else ''}{parsed.path or '/'}"
    except Exception:
        return url[:15] + "***" if len(url) > 20 else url


# Singleton
_dispatcher: WebhookDispatcher | None = None


def get_webhook_dispatcher() -> WebhookDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = WebhookDispatcher()
    return _dispatcher


# ─── REST Endpoints ───────────────────────────────────────────────────

from pydantic import BaseModel, Field  # noqa: PLC0415, E402


class WebhookRegisterRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048)
    secret: str = Field(default="", max_length=512)
    events: list[str] = Field(default_factory=lambda: ["*"], max_length=50)
    retry_count: int = Field(default=3, ge=1, le=10)


class WebhookRegisterResponse(BaseModel):
    webhook_id: str
    url: str
    events: list[str]
    retry_count: int


@router.post("/register", response_model=WebhookRegisterResponse)
async def register_webhook(body: WebhookRegisterRequest) -> WebhookRegisterResponse:
    """Register a new webhook."""
    dispatcher = get_webhook_dispatcher()
    webhook_id = str(uuid.uuid4())[:8]
    registration = WebhookRegistration(
        url=body.url,
        secret=body.secret,
        events=body.events,
        retry_count=body.retry_count,
    )
    dispatcher.register(webhook_id, registration)
    return WebhookRegisterResponse(
        webhook_id=webhook_id,
        url=body.url,
        events=body.events,
        retry_count=body.retry_count,
    )


@router.delete("/{webhook_id}")
async def unregister_webhook(webhook_id: str) -> dict[str, str]:
    """Unregister a webhook."""
    dispatcher = get_webhook_dispatcher()
    dispatcher.unregister(webhook_id)
    return {"status": "unregistered", "webhook_id": webhook_id}


@router.get("")
async def list_webhooks() -> dict[str, dict[str, Any]]:
    """List all registered webhooks (URLs masked)."""
    dispatcher = get_webhook_dispatcher()
    return dispatcher.list_registrations()


# ─── Incident ingestion (T2.1) ────────────────────────────────────────


class IncidentWebhookRequest(BaseModel):
    """A Sentry alert or generic incident webhook body."""

    event: str = Field(default="incident", max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)


@router.post("/incident")
async def incident_webhook(body: IncidentWebhookRequest) -> dict[str, Any]:
    """Consume an incident (Sentry alert / webhook) and enqueue a fix task.

    Prefers the async worker (arq over Redis) when available; falls back to
    running the incident→PR loop inline so a standalone deployment still
    produces a fix PR. Returns ``{"status": "queued", "job_id"}`` or the
    fixer's result (e.g. ``{"status": "pr_created", "pr_url": ...}``).
    """
    from noema.autonomy.fixer import IncidentFixer, build_github_client_from_settings
    from noema.config.settings import get_settings
    from noema.workers.arq_worker import enqueue_fix_incident

    try:
        job_id = await enqueue_fix_incident(get_settings().redis.url, body.payload)
        return {"status": "queued", "job_id": job_id}
    except Exception:
        logger.warning("incident_enqueue_failed_running_inline")
        fixer = IncidentFixer(github=build_github_client_from_settings())
        try:
            return await fixer.handle_incident(body.payload)
        finally:
            await fixer.close()
