"""OTLP/HTTP exporter — push traces to any OpenTelemetry Collector.

The payload builder produces the standard ``ExportTraceServiceRequest`` in
JSON form (protobuf-JSON mapping), which the collector accepts at
``POST {endpoint}/v1/traces``. The exporter runs its own asyncio loop on a
daemon worker thread so it works from sync engine code (workers, CLI) and
from the async API server alike.

Transport degrades to a logged warning when the collector is unreachable so
instrumented code never fails because telemetry is down.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from noema.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

log = get_logger(__name__)

OTLP_TRACE_PATH = "/v1/traces"

# OpenTelemetry SpanKind values.
SPAN_KIND_INTERNAL = 2
SPAN_KIND_SERVER = 1
SPAN_KIND_CLIENT = 3

# Status codes.
STATUS_UNSET = 0
STATUS_OK = 1
STATUS_ERROR = 2

KIND_TO_OTLP = {
    "internal": SPAN_KIND_INTERNAL,
    "llm": SPAN_KIND_CLIENT,
    "tool": SPAN_KIND_CLIENT,
    "server": SPAN_KIND_SERVER,
    "client": SPAN_KIND_CLIENT,
    "producer": 4,
    "consumer": 5,
}

MAX_QUEUE = 5000
MAX_BATCH = 128
_POST_FN: Callable[[str, dict[str, Any]], Coroutine[Any, Any, bool]] | None = None


def _json_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


def _base64_id(hex_id: str, byte_len: int) -> str:
    """Encode a hex id as base64, zero-padding to ``byte_len`` bytes."""
    if not hex_id:
        return base64.b64encode(b"\x00" * byte_len).decode("ascii")
    raw = hex_id.lower().encode("ascii")
    try:
        padded = raw.rjust(byte_len * 2, b"0")
        return base64.b64encode(binascii.unhexlify(padded)).decode("ascii")
    except (binascii.Error, ValueError):
        return base64.b64encode(b"\x00" * byte_len).decode("ascii")


def _span_times_ns(span: dict[str, Any]) -> tuple[int, int]:
    start_ns = int(span.get("start_epoch_ns") or 0)
    end_ns = int(span.get("end_epoch_ns") or 0)
    now = time.time_ns()
    if start_ns <= 0:
        start_ns = now - int(float(span.get("duration_ms", 0)) * 1_000_000)
    if end_ns <= 0:
        end_ns = start_ns + int(float(span.get("duration_ms", 0)) * 1_000_000)
    return start_ns, end_ns


def _to_otlp_event(event: dict[str, Any]) -> dict[str, Any]:
    attrs = []
    for key, value in (event.get("attributes") or {}).items():
        attrs.append({"key": str(key), "value": _json_value(value)})
    return {
        "timeUnixNano": str(int(event.get("timestamp_ns") or 0)),
        "name": str(event.get("name", "event")),
        "attributes": attrs,
    }


def build_otlp_span(span: dict[str, Any]) -> dict[str, Any]:
    """Convert a ``TraceSpan.to_dict()`` mapping into an OTLP span."""
    start_ns, end_ns = _span_times_ns(span)
    otlp_span: dict[str, Any] = {
        "traceId": _base64_id(str(span.get("trace_id", "")), 16),
        "spanId": _base64_id(str(span.get("span_id", "")), 8),
        "name": str(span.get("name", "")),
        "kind": KIND_TO_OTLP.get(str(span.get("kind", "internal")), SPAN_KIND_INTERNAL),
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "attributes": [
            {"key": str(k), "value": _json_value(v)}
            for k, v in (span.get("attributes") or {}).items()
        ],
        "events": [_to_otlp_event(e) for e in (span.get("events") or [])],
        "status": {"code": STATUS_OK},
    }
    parent_id = span.get("parent_id")
    if parent_id:
        otlp_span["parentSpanId"] = _base64_id(str(parent_id), 8)
    if span.get("status") == "error":
        otlp_span["status"] = {"code": STATUS_ERROR, "message": str(span.get("error", ""))[:512]}
    return otlp_span


def build_otlp_trace_request(
    spans: list[dict[str, Any]], service_name: str = "noema"
) -> dict[str, Any]:
    """Build the ``ExportTraceServiceRequest`` JSON body for a batch."""
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": service_name}}]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "noema.tracing.tracer", "version": ""},
                        "spans": [build_otlp_span(s) for s in spans],
                    }
                ],
            }
        ]
    }


async def _http_post(endpoint: str, payload: dict[str, Any]) -> bool:
    """POST the payload to ``{endpoint}/v1/traces`` (OTLP/HTTP protobuf-JSON)."""
    import aiohttp

    url = endpoint.rstrip("/") + OTLP_TRACE_PATH
    timeout = aiohttp.ClientTimeout(total=5.0)
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.post(url, json=payload) as resp,
        ):
            return 200 <= resp.status < 300
    except (aiohttp.ClientError, TimeoutError, OSError) as exc:
        log.debug("otlp_export_failed", endpoint=endpoint, error=str(exc))
        return False


async def _post_any(endpoint: str, payload: dict[str, Any]) -> bool:
    fn = _POST_FN
    if fn is None:
        return await _http_post(endpoint, payload)
    return await fn(endpoint, payload)


@dataclass
class _OTLPExporter:
    endpoint: str
    service_name: str
    flush_interval: float = 1.0
    batch_size: int = MAX_BATCH
    _queue: deque[dict[str, Any]] = field(default_factory=deque)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _loop: asyncio.AbstractEventLoop | None = None

    def emit(self, span: dict[str, Any]) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._queue.append(span)
        if len(self._queue) > MAX_QUEUE:
            for _ in range(len(self._queue) - MAX_QUEUE):
                self._queue.popleft()

    def pending(self) -> int:
        return len(self._queue)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="otlp-exporter", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.flush_interval + 2.0)
        self._thread = None
        self._loop = None

    def flush(self) -> None:
        """Drain the queue on whatever thread we happen to run on (sync)."""
        batch: list[dict[str, Any]] = []
        for _ in range(self.batch_size):
            try:
                batch.append(self._queue.popleft())
            except IndexError:
                break
        if not batch:
            return
        try:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    _post_any(self.endpoint, build_otlp_trace_request(batch, self.service_name))
                )
            finally:
                loop.close()
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("otlp_flush_failed", error=str(exc))

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            asyncio.set_event_loop(loop)
            while not self._stop.is_set():
                cycle_start = time.monotonic()
                batch: list[dict[str, Any]] = []
                for _ in range(self.batch_size):
                    try:
                        batch.append(self._queue.popleft())
                    except IndexError:
                        break
                if batch:
                    try:
                        loop.run_until_complete(
                            _post_any(
                                self.endpoint, build_otlp_trace_request(batch, self.service_name)
                            )
                        )
                    except Exception as exc:
                        log.debug("otlp_export_failed", error=str(exc))
                dead = self.flush_interval - (time.monotonic() - cycle_start)
                if dead > 0:
                    self._stop.wait(dead)
        finally:
            # Best-effort drain of whatever is still queued at shutdown.
            try:
                batch = list(self._queue)
                self._queue.clear()
                if batch:
                    loop.run_until_complete(
                        _post_any(self.endpoint, build_otlp_trace_request(batch, self.service_name))
                    )
            except Exception as exc:  # pragma: no cover - defensive
                log.debug("otlp_final_flush_failed", error=str(exc))
            loop.close()
            self._loop = None


_exporter: _OTLPExporter | None = None
_lock = threading.Lock()


def set_post_fn(
    fn: Callable[[str, dict[str, Any]], Coroutine[Any, Any, bool]] | None,
) -> Callable[[str, dict[str, Any]], Coroutine[Any, Any, bool]] | None:
    """Override the HTTP POST (test seam). Returns the previous function."""
    global _POST_FN
    prev = _POST_FN
    _POST_FN = fn
    return prev


def start_otlp_exporter(
    endpoint: str,
    service_name: str = "noema",
    flush_interval: float = 1.0,
    batch_size: int = MAX_BATCH,
) -> None:
    global _exporter
    with _lock:
        if _exporter is not None:
            if _exporter.endpoint == endpoint and _exporter.service_name == service_name:
                return
            _exporter.stop()
        exporter = _OTLPExporter(
            endpoint=endpoint,
            service_name=service_name or "noema",
            flush_interval=flush_interval,
            batch_size=batch_size,
        )
        exporter.start()
        _exporter = exporter
        log.info("otlp_exporter_started", endpoint=endpoint, service_name=exporter.service_name)


def stop_otlp_exporter() -> None:
    global _exporter
    with _lock:
        stop = _exporter
        _exporter = None
    if stop is not None:
        stop.stop()
        log.info("otlp_exporter_stopped")


def get_otlp_exporter() -> _OTLPExporter | None:
    with _lock:
        return _exporter


def is_otlp_active() -> bool:
    exp = get_otlp_exporter()
    return exp is not None


def emit_otlp_span(span: dict[str, Any]) -> None:
    exp = get_otlp_exporter()
    if exp is not None:
        exp.emit(span)


def flush_otlp_exporter() -> None:
    exp = get_otlp_exporter()
    if exp is not None:
        exp.flush()
