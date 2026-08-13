"""Tests for the gRPC servicer and client (in-process, fake engine)."""

from __future__ import annotations

import grpc
import pytest

from noema.core.types import Solution, SolutionQuality, ThoughtProcess
from noema.grpc.client import NoemaGRPCClient
from noema.grpc.noema_engine_pb2 import (
    CancelRequest,
    EvolveRequest,
)
from noema.grpc.noema_engine_pb2_grpc import (
    NoemaEngineServiceStub,
    add_NoemaEngineServiceServicer_to_server,
)
from noema.grpc.server import NoemaEngineServicer


class FakeNoema:
    def __init__(self) -> None:
        self.neurosymbolic = None
        self.seen_steps: list[str] = []

    async def think(self, task, on_step_start=None, on_step_end=None):
        for name, label, done in (("architecture", "Designing", 1), ("codegen", "Writing", 2)):
            self.seen_steps.append(name)
            if on_step_start:
                await on_step_start(name, label, done, 3)
            if on_step_end:
                await on_step_end(name, "OK " + label, done, 3)
        solution = Solution(
            task_id=task.id,
            title=task.title,
            summary="fake solution",
            quality=SolutionQuality.EXCELLENT,
            confidence=0.9,
        )
        return solution, ThoughtProcess(task_id=task.id, duration_ms=12.5)

    async def evolve(self):
        return {
            "patches_generated": 1,
            "patches_applied": 1,
            "summary": "evolved",
            "improvements": ["fast path"],
        }


@pytest.fixture
async def client():
    fake = FakeNoema()
    server = grpc.aio.server()
    add_NoemaEngineServiceServicer_to_server(NoemaEngineServicer(fake), server)
    port = server.add_insecure_port("[::]:0")
    await server.start()
    c = NoemaGRPCClient(host="localhost", port=port)
    await c.connect()
    try:
        yield c, fake
    finally:
        await c.close()
        await server.stop(0)


async def test_health(client):
    c, _ = client
    result = await c.health()
    assert result["status"] == "ok"
    assert result["version"] == "1.0.0"


async def test_think_unary(client):
    c, _ = client
    result = await c.think("Test task", "desc", tags=["a"])
    assert result["quality"] == "excellent"
    assert result["task_id"]
    assert not result["error"]


async def test_think_stream_yields_real_steps(client):
    c, fake = client
    updates = await c.think_stream("Stream task", "desc")
    stages = [u["stage"] for u in updates]
    assert "architecture" in stages
    assert "codegen" in stages
    assert stages[-1] == "completed"
    assert updates[0]["status"] == "running"
    assert fake.seen_steps == ["architecture", "codegen"]


async def test_metrics_unary(client):
    c, _ = client
    result = await c.metrics()
    assert result["tasks_processed"] == 0


async def test_cancel_without_manager_returns_not_found(client):
    c, _ = client
    request = CancelRequest(task_id="nope")
    stub = NoemaEngineServiceStub(c._channel)
    response = await stub.CancelTask(request)
    assert response.cancelled is False
    assert response.status == "not_found"


async def test_evolve(client):
    c, _ = client
    stub = NoemaEngineServiceStub(c._channel)
    response = await stub.Evolve(EvolveRequest())
    assert response.patches_generated == 1
    assert response.improvements == ["fast path"]


async def test_client_requires_connect():
    c = NoemaGRPCClient()
    with pytest.raises(RuntimeError):
        await c.health()


async def test_stream_error_is_surfaced():
    class BrokenNoema:
        neurosymbolic = None

        async def think(self, task, on_step_start=None, on_step_end=None):
            raise RuntimeError("boom")

    server = grpc.aio.server()
    add_NoemaEngineServiceServicer_to_server(NoemaEngineServicer(BrokenNoema()), server)
    port = server.add_insecure_port("[::]:0")
    await server.start()
    c = NoemaGRPCClient(host="localhost", port=port)
    await c.connect()
    try:
        updates = await c.think_stream("x", "y")
        assert updates[-1]["stage"] == "error"
        assert "boom" in updates[-1]["message"]
    finally:
        await c.close()
        await server.stop(0)


async def test_think_unary_error_sets_grpc_status():
    class BrokenNoema:
        neurosymbolic = None

        async def think(self, task, on_step_start=None, on_step_end=None):
            raise RuntimeError("boom")

    server = grpc.aio.server()
    add_NoemaEngineServiceServicer_to_server(NoemaEngineServicer(BrokenNoema()), server)
    port = server.add_insecure_port("[::]:0")
    await server.start()
    c = NoemaGRPCClient(host="localhost", port=port)
    await c.connect()
    try:
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await c.think("x", "y")
        assert exc_info.value.code() == grpc.StatusCode.INTERNAL
        assert "boom" in exc_info.value.details()
    finally:
        await c.close()
        await server.stop(0)
