"""Async event bus for decoupled service communication."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from noema.logging import get_logger

log = get_logger(__name__)


@dataclass
class Event:
    """An event published on the bus."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.monotonic)
    source: str = ""


# Handler type: async callable that takes an Event
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """In-process async pub/sub event bus.

    Supports:
    - Multiple subscribers per event type
    - Wildcard subscriptions (`*` receives all events)
    - Typed event routing
    - Dead letter queue for failed handlers
    """

    def __init__(self, max_queue: int = 1000) -> None:
        self._subscribers: dict[str, list[EventHandler]] = {}
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue)
        self._dead_letters: list[tuple[Event, Exception]] = []
        self._running = False
        self._worker_task: asyncio.Task | None = None
        self._published = 0
        self._delivered = 0
        self._failed = 0

    async def start(self) -> None:
        """Start the consumer loop."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._consume_loop())
        log.info("event_bus_started")

    async def stop(self) -> None:
        """Stop the consumer loop gracefully."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
        log.info("event_bus_stopped", published=self._published, delivered=self._delivered)

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe a handler to an event type.

        Use event_type="*" to receive all events.
        """
        self._subscribers.setdefault(event_type, []).append(handler)
        log.debug("event_subscribed", event_type=event_type, handler=handler.__qualname__)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove a subscription."""
        handlers = self._subscribers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, event: Event) -> None:
        """Publish an event to the bus."""
        try:
            self._queue.put_nowait(event)
            self._published += 1
        except asyncio.QueueFull:
            log.warning("event_bus_queue_full", event_type=event.type)
            self._dead_letters.append((event, RuntimeError("Queue full")))
            # Bound the dead-letter backlog: an overloaded bus must not
            # leak memory indefinitely.
            if len(self._dead_letters) > 1000:
                self._dead_letters.pop(0)
            self._failed += 1

    async def emit(
        self, event_type: str, payload: dict[str, Any] | None = None, source: str = ""
    ) -> Event:
        """Convenience: create and publish an event."""
        event = Event(type=event_type, payload=payload or {}, source=source)
        await self.publish(event)
        return event

    async def _consume_loop(self) -> None:
        """Main consumer loop — dispatches events to handlers."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except TimeoutError:
                continue

            handlers = self._subscribers.get(event.type, []) + self._subscribers.get("*", [])

            if not handlers:
                self._delivered += 1
                continue

            for handler in handlers:
                try:
                    await handler(event)
                    self._delivered += 1
                except Exception as exc:
                    self._failed += 1
                    self._dead_letters.append((event, exc))
                    log.error(
                        "event_handler_error",
                        event_type=event.type,
                        handler=handler.__qualname__,
                        error=str(exc),
                    )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "published": self._published,
            "delivered": self._delivered,
            "failed": self._failed,
            "queue_size": self._queue.qsize(),
            "dead_letters": len(self._dead_letters),
            "subscriptions": {k: len(v) for k, v in self._subscribers.items()},
        }


# ─── Singleton ───────────────────────────────────────────────────────────
_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def reset_event_bus() -> None:
    global _bus
    _bus = None


def schedule_event(
    bus: EventBus,
    event_type: str,
    payload: dict[str, Any] | None = None,
    source: str = "",
) -> None:
    """Fire-and-forget publish.

    Keeps a reference to the background task so it cannot be garbage
    collected mid-execution, and logs any unexpected failure. Safe to
    call from synchronous code running inside an event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        log.debug("event_emit_skipped_no_loop", event_type=event_type)
        return

    task = loop.create_task(bus.emit(event_type, payload, source))

    def _log_failure(t: asyncio.Task) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            exc = t.exception()
            if exc is not None:
                log.error("event_emit_failed", event_type=event_type, error=str(exc))

    task.add_done_callback(_log_failure)
