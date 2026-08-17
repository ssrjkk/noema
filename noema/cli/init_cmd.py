"""noema init - bootstrap a new noema project."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import structlog
import typer

from noema.cli.ui import ELLIPSIS, console, ok, panel, warn
from noema.config.settings import NoemaSettings, get_settings

logger = structlog.get_logger(__name__)
init_app = typer.Typer(help="Bootstrap commands", rich_markup_mode="rich")


@init_app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
    db_url: str = typer.Option(
        "postgresql+asyncpg://noema:noema@localhost:5432/noema", help="PostgreSQL DSN"
    ),
    redis_url: str = typer.Option("redis://localhost:6379/0", help="Redis URL"),
    llm_provider: str = typer.Option("ollama", help="LLM provider"),
    env: str = typer.Option("development", help="Environment name"),
) -> None:
    """Initialize noema project - config, DB, seed data."""
    project_root = Path.cwd()
    settings_path = project_root / "settings.yaml"

    if settings_path.exists() and not force:
        warn(f"File exists: {settings_path}. Use --force to overwrite.")
        raise typer.Exit(1)

    panel(f"Initializing noema in [path]{project_root}[/path]", title="Noema Init")

    settings = NoemaSettings()
    settings.db.url = db_url
    settings.redis.url = redis_url
    settings.llm.provider = llm_provider
    settings.obs.sentry_environment = env

    settings.dump_yaml(settings_path)
    ok(f"Created {settings_path}")

    console.print(f"  Running database migrations{ELLIPSIS}")
    import alembic.command as alembic_cmd
    from alembic.config import Config as AlembicConfig

    alembic_cfg = AlembicConfig(project_root / "alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    try:
        alembic_cmd.upgrade(alembic_cfg, "head")
        ok("Migrations applied")
    except Exception as e:
        warn(f"Migration failed: {e}")
        console.print("  You can run later: [path]alembic upgrade head[/path]")

    console.print("")
    panel(
        "Start the API with:\n"
        "  [path]uvicorn noema.api.server:app --reload[/path]\n\n"
        "Or with Docker:\n"
        "  [path]docker-compose up[/path]",
        title="Next Steps",
        border="ok",
    )


@init_app.command()
def seed() -> None:
    """Seed database with initial data."""

    async def _seed() -> None:
        from noema.db.engine import init_db

        db = await init_db()
        async with db.session() as session:
            from sqlalchemy import text

            await session.execute(
                text("""
                INSERT INTO feature_flags (id, flag_key, tenant_id, value, created_at, updated_at)
                VALUES
                    ('f1', 'cost_tracking', 'default', true, NOW(), NOW()),
                    ('f2', 'neurosymbolic', 'default', false, NOW(), NOW()),
                    ('f3', 'streaming', 'default', true, NOW(), NOW())
                ON CONFLICT (id) DO NOTHING
            """)
            )
            await session.commit()
        ok("Seed data inserted")

    asyncio.run(_seed())
    ok("Database seeded with initial feature flags")


@init_app.command()
def show_config() -> None:
    """Print current configuration."""
    settings = get_settings()
    typer.echo(json.dumps(settings.model_dump(), indent=2, default=str))
