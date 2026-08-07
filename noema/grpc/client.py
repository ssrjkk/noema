"""Async gRPC client for NoemaEngine service."""

from __future__ import annotations

from typing import Any

import grpc
import structlog

from noema.grpc.noema_engine_pb2 import (
    HealthRequest,
    MetricsRequest,
    ThinkRequest,
)
from noema.grpc.noema_engine_pb2_grpc import NoemaEngineServiceStub

logger = structlog.get_logger(__name__)


class NoemaGRPCClient:
    """Async gRPC client for NoemaEngine."""

    def __init__(self, host: str = "localhost", port: int = 50051) -> None:
        self.address = f"{host}:{port}"
        self._channel: grpc.aio.Channel | None = None
        self._stub: NoemaEngineServiceStub | None = None

    async def connect(self) -> None:
        self._channel = grpc.aio.insecure_channel(self.address)
        self._stub = NoemaEngineServiceStub(self._channel)
        logger.info("grpc_client_connected", address=self.address)

    async def close(self) -> None:
        if self._channel:
            await self._channel.close()

    async def think(
        self,
        title: str,
        description: str,
        complexity: str = "moderate",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        if not self._stub:
            raise RuntimeError("Not connected. Call connect() first.")
        request = ThinkRequest(
            title=title, description=description, complexity=complexity, tags=tags or []
        )
        response = await self._stub.Think(request)
        return {
            "solution_id": response.solution_id,
            "task_id": response.task_id,
            "title": response.title,
            "quality": response.quality,
            "confidence": response.confidence,
            "error": response.error,
        }

    async def health(self) -> dict:
        if not self._stub:
            raise RuntimeError("Not connected.")
        response = await self._stub.Health(HealthRequest())
        return {
            "status": response.status,
            "version": response.version,
            "uptime_s": response.uptime_s,
        }

    async def metrics(self, tenant_id: str = "") -> dict:
        if not self._stub:
            raise RuntimeError("Not connected.")
        response = await self._stub.GetMetrics(MetricsRequest(tenant_id=tenant_id))
        return {
            "tasks_processed": response.tasks_processed,
            "tasks_successful": response.tasks_successful,
            "success_rate": response.success_rate,
        }
