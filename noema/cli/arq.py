"""CLI commands for Arq worker management."""

from __future__ import annotations

import asyncio
import json

import typer

from noema.cli.ui import console, data_table, ok
from noema.workers.arq_worker import create_worker

arq_app = typer.Typer(help="Background worker commands", rich_markup_mode="rich")


@arq_app.command()
def worker(redis: str = "redis://localhost:6379/0") -> None:
    """Start the Arq background worker."""

    async def _run() -> None:
        console.print(f"[accent]Starting Arq worker[/accent] on [path]{redis}[/path]")
        w = await create_worker(redis)
        await w.async_run()

    asyncio.run(_run())


@arq_app.command()
def enqueue(
    title: str,
    description: str = "",
    complexity: str = "moderate",
    redis: str = "redis://localhost:6379/0",
) -> None:
    """Enqueue a think task."""
    from noema.workers.arq_worker import enqueue_think

    async def _run() -> None:
        job_id = await enqueue_think(
            redis,
            {
                "title": title,
                "description": description,
                "complexity": complexity,
            },
        )
        ok(f"Task queued: job_id={job_id}")
        print(json.dumps({"job_id": job_id, "status": "queued"}))

    asyncio.run(_run())


@arq_app.command("workers")
def workers(redis: str = "redis://localhost:6379/0") -> None:
    """List live worker nodes from their Redis heartbeats."""

    from noema.workers.arq_worker import list_active_workers

    async def _run() -> list[dict[str, str]]:
        return await list_active_workers(redis)

    fleet = asyncio.run(_run())
    data_table(
        "Worker nodes",
        ["Node", "Started", "Heartbeat", "Draining", "Metrics port"],
        [
            [
                w.get("node_id", "?"),
                w.get("started_at", "?"),
                w.get("last_heartbeat", "?"),
                "yes" if w.get("draining") == "1" else "no",
                w.get("metrics_port", "0"),
            ]
            for w in fleet
        ],
    )
    print(json.dumps(fleet))


@arq_app.command("ledger")
def ledger(
    path: str = typer.Argument(..., help="JSONL ledger file written by a worker node"),
    task_id: str = typer.Option("", help="Show only this task's entries"),
) -> None:
    """Audit a worker node's contribution ledger (who generated what value)."""
    from noema.billing.ledger import ContributionLedger

    box = ContributionLedger(path=path)
    loaded = box.load()
    per_node = box.per_node(task_id=task_id) if task_id else box.per_node()
    entries = box.entries_for(task_id) if task_id else box.audit()["entries"]
    ok(f"Ledger {path}: {loaded} entries")
    print(json.dumps({"per_node": per_node, "entries": entries}, indent=2))
