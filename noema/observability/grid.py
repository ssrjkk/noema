"""Grid dashboard — per-node health view for a fleet of Noema workers (T3.4).

Each worker node publishes a liveness heartbeat to Redis (see
``noema.workers.arq_worker.NodeHeartbeat``) and serves Prometheus exposition
on its ``metrics_port``. :class:`GridDashboard` joins the two: it lists the
live nodes from Redis, scrapes each node's ``/metrics`` endpoint, and folds
the raw counters into a per-node latency/token/error view plus cluster
totals — the data the ``GET /v1/grid`` endpoint and the ``noema grid`` CLI
command render.

Parsing: uses ``prometheus_client.parser`` when available and falls back to
a minimal exposition-format line parser so the dashboard also works with
``prometheus-client`` absent (the same graceful-degradation contract as
:mod:`noema.observability.metrics`).

Transport: the metrics fetch is an injected ``async (url) -> str`` callable;
the default uses aiohttp with a short timeout. An unreachable node is
reported as ``unreachable`` and never fails the snapshot — a dead node is
exactly what the dashboard exists to show.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from noema.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from redis.asyncio import Redis as AsyncRedis

logger = get_logger(__name__)

FETCH_TIMEOUT = 5.0

# Metrics the dashboard folds into the per-node view: metric name → view key.
_REQUEST_TOTAL = "noema_http_requests_total"
_LLM_TOKENS = "noema_llm_tokens_total"
_LLM_LATENCY = "noema_llm_request_duration_seconds"


async def _default_fetch(url: str) -> str:
    """Fetch a metrics endpoint via aiohttp with a hard timeout."""
    import aiohttp

    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT)) as session,
        session.get(url) as resp,
    ):
        resp.raise_for_status()
        return await resp.text()


def parse_exposition(text: str) -> dict[str, list[dict[str, Any]]]:
    """Parse Prometheus exposition text into ``{metric: [{labels, value}]}``.

    Prefers the ``prometheus_client`` parser; falls back to a line parser
    that understands plain ``name{labels} value`` samples (enough for the
    counters/gauges/histograms this project exports).
    """
    try:
        from prometheus_client.parser import text_string_to_metric_families

        samples: dict[str, list[dict[str, Any]]] = {}
        for family in text_string_to_metric_families(text):
            for sample in family.samples:
                samples.setdefault(sample.name, []).append(
                    {"labels": dict(sample.labels), "value": float(sample.value)}
                )
        return samples
    except ImportError:
        return _parse_exposition_lines(text)
    except Exception:  # noqa: BLE001 - malformed exposition must not kill the view
        return _parse_exposition_lines(text)


def _parse_exposition_lines(text: str) -> dict[str, list[dict[str, Any]]]:
    """Minimal exposition parser: ``name{a="b",c="d"} value [timestamp]``."""
    samples: dict[str, list[dict[str, Any]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name, value_token = parts[0], parts[1]
        labels: dict[str, str] = {}
        if "{" in name and name.endswith("}"):
            name, _, label_blob = name.partition("{")
            for chunk in label_blob[:-1].split('",'):
                if "=" not in chunk:
                    continue
                key, _, raw = chunk.partition("=")
                labels[key.strip()] = raw.strip().strip('"')
        try:
            value = float(value_token)
        except ValueError:
            continue
        samples.setdefault(name, []).append({"labels": labels, "value": value})
    return samples


@dataclass
class NodeView:
    """One node's health as rendered on the dashboard."""

    node_id: str = ""
    address: str = ""  # hostname of the worker (from the heartbeat)
    metrics_port: int = 0
    draining: bool = False
    last_heartbeat: int = 0
    reachable: bool = False
    error: str = ""
    http_requests: int = 0
    http_errors: int = 0
    llm_tokens: float = 0.0
    llm_calls: float = 0.0
    llm_latency_sum_s: float = 0.0
    scrape_ms: float = 0.0

    @property
    def llm_latency_avg_ms(self) -> float:
        return round(self.llm_latency_sum_s / self.llm_calls * 1000, 1) if self.llm_calls else 0.0

    @property
    def error_rate(self) -> float:
        return round(self.http_errors / self.http_requests, 4) if self.http_requests else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "address": self.address,
            "metrics_port": self.metrics_port,
            "metrics_url": f"http://{self.address}:{self.metrics_port}/metrics"
            if self.metrics_port
            else "",
            "draining": self.draining,
            "last_heartbeat": self.last_heartbeat,
            "reachable": self.reachable,
            "error": self.error,
            "http_requests": self.http_requests,
            "http_errors": self.http_errors,
            "error_rate": self.error_rate,
            "llm_tokens": self.llm_tokens,
            "llm_calls": self.llm_calls,
            "llm_latency_avg_ms": self.llm_latency_avg_ms,
            "scrape_ms": self.scrape_ms,
        }


class GridDashboard:
    """Aggregates live Prometheus metrics for every node in the grid.

    Args:
        redis_url: Redis DSN holding the worker heartbeat keys.
        redis: Pre-built async Redis client (injectable; used by tests via
            fakeredis). When omitted it is built from ``redis_url``.
        fetch: ``async (url) -> str`` metrics transport (injectable).
    """

    def __init__(
        self,
        redis_url: str = "",
        redis: AsyncRedis | None = None,
        fetch: Any = None,
    ) -> None:
        self.redis_url = redis_url
        self._redis = redis
        self._fetch = fetch or _default_fetch

    async def _get_redis(self) -> AsyncRedis:
        if self._redis is None:
            from redis.asyncio import Redis

            self._redis = Redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    async def aclose(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def snapshot(self) -> dict[str, Any]:
        """Live per-node view + cluster totals. Never raises."""
        nodes = await self._live_nodes()
        views = await asyncio.gather(*(self._view(node) for node in nodes))
        node_dicts = [v.to_dict() for v in views]
        reachable = [v for v in views if v.reachable]
        totals = {
            "nodes_total": len(views),
            "nodes_reachable": len(reachable),
            "nodes_draining": sum(1 for v in views if v.draining),
            "http_requests": sum(v.http_requests for v in reachable),
            "http_errors": sum(v.http_errors for v in reachable),
            "llm_tokens": sum(v.llm_tokens for v in reachable),
            "llm_calls": sum(v.llm_calls for v in reachable),
        }
        return {
            "generated_at": time.time(),
            "nodes": node_dicts,
            "totals": totals,
        }

    async def _live_nodes(self) -> list[dict[str, str]]:
        """Heartbeat records for live nodes (reuse of the worker registry)."""
        from noema.workers.arq_worker import list_active_workers

        r = await self._get_redis()
        return await list_active_workers(self.redis_url, redis=r)

    async def _view(self, node: dict[str, str]) -> NodeView:
        view = NodeView(
            node_id=node.get("node_id", ""),
            address=node.get("hostname", ""),
            draining=node.get("draining") == "1",
            last_heartbeat=int(node.get("last_heartbeat", "0") or 0),
            metrics_port=int(node.get("metrics_port", "0") or 0),
        )
        if not view.address or not view.metrics_port:
            view.error = "no metrics endpoint advertised"
            return view

        url = f"http://{view.address}:{view.metrics_port}/metrics"
        started = time.monotonic()
        try:
            text = await self._fetch(url)
        except Exception as e:  # noqa: BLE001 - a dead node is a data point
            view.error = f"{type(e).__name__}: {e}"
            view.scrape_ms = round((time.monotonic() - started) * 1000, 1)
            return view

        view.scrape_ms = round((time.monotonic() - started) * 1000, 1)
        view.reachable = True
        _fold_samples(view, parse_exposition(text))
        return view


def _fold_samples(view: NodeView, samples: dict[str, list[dict[str, Any]]]) -> None:
    """Fold raw exposition samples into a node view (counters + histograms)."""
    for sample in samples.get(_REQUEST_TOTAL, []):
        view.http_requests += int(sample["value"])
        if str(sample["labels"].get("status", "")).startswith(("4", "5")):
            view.http_errors += int(sample["value"])
    for sample in samples.get(_LLM_TOKENS, []):
        view.llm_tokens += sample["value"]
    # Histograms export ``<name>_count`` / ``<name>_sum`` samples.
    for sample in samples.get(_LLM_LATENCY + "_count", []):
        view.llm_calls += sample["value"]
    for sample in samples.get(_LLM_LATENCY + "_sum", []):
        view.llm_latency_sum_s += sample["value"]


async def iter_grid_snapshots(
    dashboard: GridDashboard, interval: float = 5.0
) -> AsyncIterator[dict[str, Any]]:
    """Endless snapshot stream for long-lived consumers (SSE/dashboard loop)."""
    while True:
        yield await dashboard.snapshot()
        await asyncio.sleep(interval)
