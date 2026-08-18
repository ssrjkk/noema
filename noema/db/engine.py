"""Async PostgreSQL engine with connection pooling and pre-ping health checks."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from noema.config.settings import get_settings
from noema.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

log = get_logger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""


class Database:
    """Async PostgreSQL database manager.

    Usage:
        db = Database()
        await db.init()
        async with db.session() as session:
            result = await session.execute(select(User))
        await db.close()
    """

    def __init__(self) -> None:
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    async def init(self) -> None:
        """Initialize engine and session factory."""
        settings = get_settings()
        self._engine = create_async_engine(
            settings.db.url,
            pool_size=settings.db.pool_min,
            max_overflow=settings.db.pool_max - settings.db.pool_min,
            pool_timeout=settings.db.pool_timeout,
            pool_pre_ping=True,  # Verify connections before use
            echo=settings.db.echo,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        log.info("db_initialized", url=settings.db.url.split("@")[-1])

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a session with automatic commit/rollback."""
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call init() first.")

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def create_tables(self) -> None:
        """Create all tables (for development)."""
        if self._engine is None:
            raise RuntimeError("Database not initialized.")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.info("db_tables_created")

    async def drop_tables(self) -> None:
        """Drop all tables (for testing only!)."""
        if self._engine is None:
            raise RuntimeError("Database not initialized.")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        log.info("db_tables_dropped")

    async def health_check(self) -> bool:
        """Verify database connectivity."""
        if self._engine is None:
            return False
        try:
            async with self._engine.connect() as conn:
                await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
            return True
        except Exception as exc:
            log.error("db_health_check_failed", error=str(exc))
            return False

    async def close(self) -> None:
        """Dispose engine and close pool."""
        if self._engine:
            await self._engine.dispose()
            log.info("db_closed")


# ─── Singleton ───────────────────────────────────────────────────────────
_db: Database | None = None


def get_db() -> Database:
    """Get or create the global database singleton."""
    global _db
    if _db is None:
        _db = Database()
    return _db


async def init_db() -> Database:
    """Initialize the global database."""
    db = get_db()
    await db.init()
    return db


async def close_db() -> None:
    """Close the global database."""
    if _db:
        await _db.close()
