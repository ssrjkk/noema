"""CLI commands for the Noema grid: live fleet health (T3.4)."""

from __future__ import annotations

import asyncio
import json

import typer

from noema.cli.ui import data_table, kv_panel, ok

grid_app = typer.Typer(help="Grid federation commands", rich_markup_mode="rich")


@grid_app.command("status")
def status(
    redis: str = typer.Option("redis://localhost:6379/0", help="Redis DSN with worker heartbeats"),
) -> None:
    """Render live grid health: per-node latency/tokens/errors + totals."""
    from noema.observability.grid import GridDashboard

    async def _run() -> dict:
        dashboard = GridDashboard(redis_url=redis)
        try:
            return await dashboard.snapshot()
        finally:
            await dashboard.aclose()

    snap = asyncio.run(_run())
    totals = snap.get("totals", {})
    data_table(
        "Grid nodes",
        [
            "Node",
            "Address",
            "Reachable",
            "Draining",
            "LLM tokens",
            "LLM calls",
            "Avg LLM ms",
            "Errors",
        ],
        [
            [
                n.get("node_id", "?"),
                n.get("address", "?") or "?",
                "yes" if n.get("reachable") else f"no ({n.get('error', '')[:40]})",
                "yes" if n.get("draining") else "no",
                n.get("llm_tokens", 0),
                n.get("llm_calls", 0),
                n.get("llm_latency_avg_ms", 0),
                n.get("http_errors", 0),
            ]
            for n in snap.get("nodes", [])
        ],
    )
    kv_panel("Cluster totals", [(k, str(v)) for k, v in totals.items()])
    print(json.dumps(snap))


@grid_app.command("federate")
def federate(
    title: str = typer.Option(..., help="Root task title"),
    subtasks: list[str] = typer.Option(..., "--subtask", help="Sub-task description (repeatable)"),
    peers: list[str] = typer.Option(None, "--peer", help="Peer address host:port (repeatable)"),
    timeout: float = typer.Option(30.0, help="Per-attempt peer timeout, seconds"),
) -> None:
    """Split sub-tasks across peer nodes and re-join the results (T3.2).

    Without ``--peer`` everything runs through the local engine fallback.
    """
    from noema.billing.ledger import ContributionLedger
    from noema.config.settings import get_settings
    from noema.core.engine import NoemaEngine
    from noema.federation.router import FederationRouter
    from noema.workers.arq_worker import make_node_id

    async def _run() -> dict:
        settings = get_settings()
        engine = NoemaEngine()
        await engine.initialize()
        try:

            async def local_executor(subtask_id: str, description: str) -> dict:
                from noema.core.types import Task

                solution, _ = await engine.think(
                    Task(title=description[:80], description=description)
                )
                return {
                    "solution_id": solution.id,
                    "quality": solution.quality.value,
                }

            router = FederationRouter(
                peers=list(peers) if peers else list(settings.federation.peers),
                local_executor=local_executor,
                ledger=ContributionLedger(),
                node_id=make_node_id(),
                request_timeout=timeout,
            )
            try:
                return await router.execute(title, list(subtasks))
            finally:
                await router.aclose()
        finally:
            await engine.shutdown()

    summary = asyncio.run(_run())
    ok(
        f"Federated run complete: {summary['delegated']} delegated, "
        f"{summary['local']} local, {summary['failed']} failed "
        f"({summary['duration_ms']:.0f} ms)"
    )
    print(json.dumps(summary, indent=2))
