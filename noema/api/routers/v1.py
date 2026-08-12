"""v1 router — mounts all core endpoints under /api/v1/ prefix."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1")


def mount_main_router() -> None:
    """Include the core + experiments routers (lazy import avoids circular dependency)."""
    from noema.api.experiments import router as experiments_router
    from noema.api.server import router as main_router

    router.include_router(main_router)
    router.include_router(experiments_router)


mount_main_router()
