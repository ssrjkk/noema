"""CLI commands for Arq worker management."""

from __future__ import annotations

import asyncio
import json

import typer

from noema.cli.ui import console, ok
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
