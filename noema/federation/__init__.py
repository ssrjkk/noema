"""Federation protocol — sub-task delegation between Noema nodes (T3.2).

Heavy reasoning is split into sub-tasks and delegated to peer Noema nodes
over the existing gRPC ``Think`` RPC, with per-peer retry (exponential
backoff) and circuit breaking. A peer whose circuit is open is skipped until
its recovery timeout elapses; when no healthy peer remains the sub-task
falls back to the local executor instead of failing the whole hierarchy.
Every delegation lands in the contribution ledger (T3.3), so a distributed
run produces an auditable per-node value trail.
"""

from __future__ import annotations

from noema.federation.router import (
    DelegationResult,
    FederationRouter,
    PeerNode,
)

__all__ = ["DelegationResult", "FederationRouter", "PeerNode"]
