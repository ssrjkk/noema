"""Federation router — split, delegate, re-join sub-tasks across peer nodes.

Protocol:
- ``execute`` fans a list of sub-task descriptions out to peers round-robin,
  skipping peers whose circuit is open (they are failing).
- Each delegation runs through a per-peer :class:`ResilientExecutor`
  (circuit breaker + exponential-backoff retries) with a hard request
  timeout, so one slow or flapping peer can never wedge the hierarchy.
- A sub-task that cannot be delegated (no healthy peers, circuit open,
  retries exhausted) falls back to the ``local_executor`` callback — the
  hierarchy still completes, and the ledger records the fallback.
- Results are re-joined in submission order with a per-subtask status.

Transport: the default client factory builds a real
:class:`~noema.grpc.client.NoemaGRPCClient`; tests inject a fake factory, so
no sockets are needed. Channels are plaintext/loopback — same constraint as
the standalone gRPC server.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from noema.grpc.client import NoemaGRPCClient
from noema.logging import get_logger
from noema.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    ResilientExecutor,
    RetryPolicy,
)

if TYPE_CHECKING:
    from noema.billing.ledger import ContributionLedger

logger = get_logger(__name__)

LocalExecutor = Callable[[str, str], Awaitable[dict[str, Any]]]
ClientFactory = Callable[[str], Any]


@dataclass(frozen=True)
class PeerNode:
    """One peer Noema node, identified by ``host:port``."""

    host: str
    port: int

    @classmethod
    def parse(cls, address: str) -> PeerNode:
        host, _, port = address.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"peer address must be 'host:port', got {address!r}")
        return cls(host=host, port=int(port))

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class DelegationResult:
    """Outcome of one sub-task: where it ran and what it produced."""

    subtask_id: str
    description: str
    peer: str = ""  # "" → executed locally
    status: str = "pending"  # delegated | local | failed
    response: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "description": self.description,
            "peer": self.peer,
            "status": self.status,
            "response": self.response,
            "error": self.error,
            "attempts": self.attempts,
        }


def _default_client_factory(address: str) -> NoemaGRPCClient:
    host, _, port = address.rpartition(":")
    return NoemaGRPCClient(host=host, port=int(port))


class FederationRouter:
    """Distributes sub-tasks across peer nodes and re-joins the results.

    Args:
        peers: Peer addresses (``host:port``). Empty → everything runs local.
        local_executor: ``async (title, description) -> dict`` used when no
            peer is available or a delegation ultimately fails.
        client_factory: gRPC client factory (injectable for tests).
        ledger: Contribution ledger receiving one entry per sub-task.
        node_id: This node's identity, written into every ledger entry.
        max_retries / base_delay: Per-delegation retry policy (short base
            delay by default: peers are redundant, retrying fast is better
            than stalling the hierarchy).
        failure_threshold / recovery_timeout: Per-peer circuit breaker.
        request_timeout: Hard per-attempt gRPC timeout, seconds.
    """

    def __init__(
        self,
        peers: list[str],
        local_executor: LocalExecutor | None = None,
        client_factory: ClientFactory | None = None,
        ledger: ContributionLedger | None = None,
        node_id: str = "local",
        max_retries: int = 2,
        base_delay: float = 0.05,
        max_delay: float = 1.0,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        request_timeout: float = 30.0,
    ) -> None:
        self.peer_nodes = [PeerNode.parse(p) for p in peers]
        self.local_executor = local_executor
        self.client_factory = client_factory or _default_client_factory
        self.ledger = ledger
        self.node_id = node_id
        self.request_timeout = request_timeout
        self._executors: dict[str, ResilientExecutor] = {
            peer.address: ResilientExecutor(
                circuit_breaker=CircuitBreaker(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    name=f"federation:{peer.address}",
                ),
                retry_policy=RetryPolicy(
                    max_retries=max_retries,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    jitter=0.2,
                    name=f"federation:{peer.address}",
                    # An open circuit must fail fast, not burn retries.
                    non_retryable_exceptions=(CircuitBreakerError,),
                ),
            )
            for peer in self.peer_nodes
        }
        self._clients: dict[str, Any] = {}
        self._robin = 0

    # ── Peer plumbing ────────────────────────────────────────────────

    def _client(self, address: str) -> Any:
        if address not in self._clients:
            self._clients[address] = self.client_factory(address)
        return self._clients[address]

    async def aclose(self) -> None:
        """Close pooled peer clients (best-effort)."""
        for address, client in list(self._clients.items()):
            close = getattr(client, "close", None)
            if close is not None:
                try:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:  # noqa: BLE001 - teardown only
                    logger.debug("federation_client_close_failed", peer=address, error=str(e))
        self._clients.clear()

    def healthy_peers(self) -> list[str]:
        """Peers whose circuit is not open (usable for delegation)."""
        return [
            address
            for address, executor in self._executors.items()
            if executor.circuit.state.value != "open"
        ]

    def circuit_stats(self) -> dict[str, dict[str, Any]]:
        return {addr: ex.circuit.stats() for addr, ex in self._executors.items()}

    # ── Delegation ───────────────────────────────────────────────────

    async def delegate(self, subtask_id: str, description: str) -> DelegationResult:
        """Run one sub-task on the next healthy peer, or locally on failure.

        Round-robin starts at the healthy-peer cursor so work spreads across
        the fleet instead of hammering the first node.
        """
        result = DelegationResult(subtask_id=subtask_id, description=description)
        healthy = self.healthy_peers()
        if not healthy:
            return await self._run_local(result, reason="no healthy peers")

        address = healthy[self._robin % len(healthy)]
        self._robin += 1
        executor = self._executors[address]
        attempts = 0

        async def _counted() -> dict[str, Any]:
            nonlocal attempts
            attempts += 1
            return await self._peer_think(address, "federation subtask", description)

        started = time.monotonic()
        try:
            async with asyncio.timeout(self.request_timeout * (executor.retry.max_retries + 1)):
                response = await executor.execute(_counted)
        except CircuitBreakerError as e:
            return await self._run_local(result, reason=f"circuit open on {address}: {e}")
        except TimeoutError:
            return await self._run_local(result, reason=f"timeout delegating to {address}")
        except Exception as e:  # noqa: BLE001 - a dead peer must not kill the run
            return await self._run_local(result, reason=f"{type(e).__name__}: {e}")

        result.peer = address
        result.status = "delegated"
        result.response = response
        result.attempts = attempts
        self._record(result, started)
        return result

    async def _peer_think(self, address: str, title: str, description: str) -> dict[str, Any]:
        """One Think RPC against a peer (the retried/circuit-protected unit)."""
        client = self._client(address)
        result: dict[str, Any] = await client.think(title=title, description=description)
        return result

    async def _run_local(self, result: DelegationResult, reason: str) -> DelegationResult:
        """Fall back to the local executor; record the outcome honestly."""
        result.error = reason
        started = time.monotonic()
        if self.local_executor is None:
            result.status = "failed"
            logger.warning("federation_subtask_failed", subtask=result.subtask_id, reason=reason)
            self._record(result, started)
            return result
        try:
            result.response = await self.local_executor(result.subtask_id, result.description)
            result.status = "local"
            logger.info(
                "federation_subtask_local_fallback",
                subtask=result.subtask_id,
                reason=reason,
            )
        except Exception as e:  # noqa: BLE001
            result.status = "failed"
            result.error = f"{reason}; local: {type(e).__name__}: {e}"
            logger.error("federation_subtask_failed", subtask=result.subtask_id, error=result.error)
        self._record(result, started)
        return result

    def _record(self, result: DelegationResult, started: float) -> None:
        if self.ledger is None:
            return
        response = result.response or {}
        self.ledger.record(
            node_id=self.node_id,
            task_id=result.subtask_id,
            kind="subtask" if result.status == "delegated" else f"delegation-{result.status}",
            input_tokens=int(response.get("tokens_input", 0) or 0),
            output_tokens=int(response.get("tokens_output", 0) or 0),
            cost_usd=float(response.get("cost_usd", 0.0) or 0.0),
            peer=result.peer,
            artifact_ref=str(response.get("solution_id", "")),
            meta={
                "description": result.description[:200],
                "duration_ms": round((time.monotonic() - started) * 1000, 1),
                "quality": str(response.get("quality", "")),
            },
        )

    # ── Split → fan-out → re-join ────────────────────────────────────

    async def execute(
        self,
        title: str,
        subtasks: list[str],
        task_id: str = "",
    ) -> dict[str, Any]:
        """Split → delegate → re-join a hierarchy of sub-tasks.

        Every sub-task runs concurrently on its assigned peer; the join is
        positional, so the caller gets results back in submission order
        regardless of which node executed what.
        """
        started = time.monotonic()
        if not subtasks:
            subtasks = [title]

        results = await asyncio.gather(
            *(
                self.delegate(f"{task_id}:{i}" if task_id else str(i), desc)
                for i, desc in enumerate(subtasks)
            )
        )
        delegated = sum(1 for r in results if r.status == "delegated")
        local = sum(1 for r in results if r.status == "local")
        failed = sum(1 for r in results if r.status == "failed")
        peers_used = sorted({r.peer for r in results if r.peer})
        summary = {
            "title": title,
            "task_id": task_id,
            "subtasks": [r.to_dict() for r in results],
            "total": len(results),
            "delegated": delegated,
            "local": local,
            "failed": failed,
            "peers_used": peers_used,
            "duration_ms": round((time.monotonic() - started) * 1000, 1),
        }
        logger.info(
            "federation_execute_complete",
            title=title,
            total=len(results),
            delegated=delegated,
            local=local,
            failed=failed,
            peers=peers_used,
        )
        return summary


def router_from_settings(
    local_executor: LocalExecutor | None = None,
    ledger: ContributionLedger | None = None,
    node_id: str = "local",
    client_factory: ClientFactory | None = None,
) -> FederationRouter:
    """Build a :class:`FederationRouter` from ``settings.federation``."""
    from noema.config.settings import get_settings

    cfg = get_settings().federation
    return FederationRouter(
        peers=list(cfg.peers),
        local_executor=local_executor,
        client_factory=client_factory,
        ledger=ledger,
        node_id=node_id,
        max_retries=cfg.max_retries,
        base_delay=cfg.base_delay,
        failure_threshold=cfg.failure_threshold,
        recovery_timeout=cfg.recovery_timeout,
        request_timeout=cfg.request_timeout,
    )
