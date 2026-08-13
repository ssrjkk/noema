"""CLI commands for the gRPC server."""

from __future__ import annotations

import asyncio
import json

import typer

from noema.cli.ui import console, ok

grpc_app = typer.Typer(help="gRPC server commands", rich_markup_mode="rich")


@grpc_app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(50051, "--port"),
) -> None:
    """Start the gRPC server with a live NoemaEngine."""

    async def _run() -> None:
        from noema.core.engine import NoemaEngine
        from noema.grpc.server import serve_grpc, stop_grpc

        console.print(f"[accent]Starting gRPC server[/accent] on [path]{host}:{port}[/path]")
        noema = NoemaEngine()
        await noema.initialize()
        server = await serve_grpc(noema, host=host, port=port)
        try:
            await server.wait_for_termination()
        finally:
            await stop_grpc(server)
            await noema.shutdown()

    asyncio.run(_run())


@grpc_app.command()
def health(
    host: str = typer.Option("localhost", "--host"),
    port: int = typer.Option(50051, "--port"),
) -> None:
    """Check gRPC server liveness."""

    async def _run() -> None:
        from noema.grpc.client import NoemaGRPCClient

        client = NoemaGRPCClient(host=host, port=port)
        await client.connect()
        try:
            result = await client.health()
            print(json.dumps(result))
            ok(f"gRPC healthy: {result['status']} v{result['version']}")
        finally:
            await client.close()

    asyncio.run(_run())
