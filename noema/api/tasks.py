"""Async task API endpoints using Arq."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from noema.config.settings import get_settings
from noema.logging import get_logger
from noema.workers.arq_worker import enqueue_think

log = get_logger(__name__)
router = APIRouter(prefix="/tasks", tags=["tasks"])


class TaskEnqueueRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=10000)
    complexity: str = Field(default="moderate", pattern="^(simple|moderate|complex|extreme)$")
    tags: list[str] = Field(default_factory=list, max_length=20)


class TaskEnqueueResponse(BaseModel):
    job_id: str | None
    status: str


@router.post("/enqueue", response_model=TaskEnqueueResponse)
async def enqueue_task(request: TaskEnqueueRequest) -> TaskEnqueueResponse:
    """Enqueue a think task for background processing."""
    settings = get_settings()
    redis_url = settings.redis.url
    try:
        job_id = await enqueue_think(
            redis_url,
            {
                "title": request.title,
                "description": request.description,
                "complexity": request.complexity,
                "tags": request.tags,
            },
        )
        log.info("task_enqueued", job_id=job_id, title=request.title)
        return TaskEnqueueResponse(job_id=job_id, status="queued")
    except Exception as e:
        log.error("task_enqueue_failed", error=str(e))
        raise HTTPException(status_code=503, detail=f"Failed to enqueue task: {str(e)}") from e
