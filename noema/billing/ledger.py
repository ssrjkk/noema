"""Contribution ledger — auditable record of who generated what value (T3.3).

Every value-producing unit of work (a local ``think`` run, a federated
sub-task delegated to a peer node, an incident fix) lands here as a
:class:`LedgerEntry`: which node executed it, for which task/subtask, with
which model, consuming how many tokens and costing how much. Aggregations
answer "which node contributed what value" without touching the raw cost
records; ``audit()`` returns the full ordered trail for review.

Persistence: entries are appended as JSON lines to an optional ``path`` (best
effort — a failed write degrades to memory-only and logs a warning, it never
breaks the run that produced the value). Memory stays bounded via
``max_entries``; the JSONL file is the durable history.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from noema.logging import get_logger

log = get_logger(__name__)


@dataclass
class LedgerEntry:
    """One value-producing unit of work, attributed to a node."""

    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    node_id: str = ""
    task_id: str = ""
    subtask_id: str = ""
    kind: str = "solution"  # solution | subtask | fix | delegation-fallback
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    peer: str = ""  # peer address when the work was delegated off-node
    artifact_ref: str = ""  # solution id / PR number / artifact path
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContributionLedger:
    """Bounded, append-only ledger of per-node contributions.

    Args:
        path: Optional JSONL file for durable, append-only persistence.
        max_entries: In-memory cap; the oldest entries are dropped first
            (the JSONL file, when configured, keeps the full history).
    """

    def __init__(self, path: str | Path = "", max_entries: int = 10_000) -> None:
        self.path = Path(path) if path else None
        self.max_entries = max(1, max_entries)
        self._entries: list[LedgerEntry] = []

    def record(
        self,
        node_id: str,
        task_id: str = "",
        subtask_id: str = "",
        kind: str = "solution",
        provider: str = "",
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        peer: str = "",
        artifact_ref: str = "",
        meta: dict[str, Any] | None = None,
    ) -> LedgerEntry:
        """Append one entry and persist it best-effort. Never raises."""
        entry = LedgerEntry(
            node_id=node_id,
            task_id=task_id,
            subtask_id=subtask_id,
            kind=kind,
            provider=provider,
            model=model,
            input_tokens=max(input_tokens, 0),
            output_tokens=max(output_tokens, 0),
            total_tokens=max(input_tokens, 0) + max(output_tokens, 0),
            cost_usd=max(cost_usd, 0.0),
            peer=peer,
            artifact_ref=artifact_ref,
            meta=meta or {},
        )
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]
        self._append_to_file(entry)
        return entry

    def _append_to_file(self, entry: LedgerEntry) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        except OSError as e:
            # The ledger must never break the run it is auditing.
            log.warning("ledger_write_failed", error=str(e))

    def per_node(self, task_id: str = "") -> dict[str, Any]:
        """Aggregate tokens/cost/contributions grouped by node.

        With ``task_id`` the aggregation covers only that task's entries.
        """
        entries = [e for e in self._entries if not task_id or e.task_id == task_id]
        nodes: dict[str, dict[str, Any]] = {}
        for e in entries:
            agg = nodes.setdefault(
                e.node_id,
                {
                    "entries": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "delegated_to_peers": 0,
                },
            )
            agg["entries"] += 1
            agg["input_tokens"] += e.input_tokens
            agg["output_tokens"] += e.output_tokens
            agg["total_tokens"] += e.total_tokens
            agg["cost_usd"] += e.cost_usd
            if e.peer:
                agg["delegated_to_peers"] += 1
        return {
            "nodes": {
                node: {
                    **agg,
                    "cost_usd": round(agg["cost_usd"], 6),
                }
                for node, agg in sorted(nodes.items())
            },
            "total_cost_usd": round(sum(e.cost_usd for e in entries), 6),
            "total_tokens": sum(e.total_tokens for e in entries),
        }

    def entries_for(self, task_id: str) -> list[dict[str, Any]]:
        """Full ordered trail for one task (audit view)."""
        return [e.to_dict() for e in self._entries if e.task_id == task_id]

    def audit(self) -> dict[str, Any]:
        """The complete ledger: every entry, oldest first."""
        return {
            "entries": [e.to_dict() for e in self._entries],
            "count": len(self._entries),
        }

    def load(self) -> int:
        """Re-load in-memory entries from the JSONL file. Returns the count.

        Used by a fresh process (e.g. a reporting CLI) to audit a ledger
        written by worker nodes without re-running anything.
        """
        self._entries = []
        if self.path is None or not self.path.is_file():
            return 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("ledger_corrupt_line_skipped")
                    continue
                self._entries.append(LedgerEntry(**data))
        return len(self._entries)
