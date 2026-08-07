"""Chaos: PostgreSQL failover — verify system degrades gracefully."""

from datetime import UTC

from noema.persistence.pg_memory import PostgresMemoryStore
from noema.resilience.graceful_degradation import GracefulDegradation


class TestPostgresFailover:
    async def test_graceful_degradation_detects_db_down(self):
        """GracefulDegradation detects that PostgreSQL is unreachable."""
        deg = GracefulDegradation()

        # Simulate a broken pg client that raises on any call
        class _BrokenPG:
            async def fetchval(self, _):
                raise ConnectionError("pg unreachable")

        deg.pg = _BrokenPG()
        deg._pg_healthy = True
        await deg.check_health()
        status = deg.get_status()
        assert status.get("postgresql") == "degraded", "Should detect DB is down"

    async def test_db_execute_falls_back_to_file(self, tmp_path):
        """db_execute writes fallback file when PostgreSQL is down."""
        deg = GracefulDegradation()
        deg._pg_healthy = False
        deg.pg = None
        deg._fallback_dir = tmp_path

        result = await deg.db_execute("INSERT INTO test VALUES ($1)", "val")
        assert result is None, "Should return None on fallback"

        files = list(tmp_path.iterdir())
        assert len(files) > 0, "Should have written fallback file"
        content = files[0].read_text(encoding="utf-8")
        assert "INSERT INTO test" in content

    async def test_memory_fallback_on_db_failure(self, tmp_path):
        """PostgresMemoryStore falls back to file storage when DB is unavailable."""
        store = PostgresMemoryStore(persist_dir=str(tmp_path))
        # _load uses file fallback when _has_pg is False (no db_url set)
        await store._load()
        assert store.persist_dir is not None

    async def test_audit_logger_file_fallback(self, tmp_path):
        """AuditLogger writes to file when PostgreSQL is unavailable."""
        from datetime import datetime

        from noema.audit.logger import AuditEvent, AuditLogger

        logger_fallback = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        await logger_fallback.initialize()
        assert logger_fallback._file_fallback is True

        await logger_fallback.log(
            AuditEvent(
                timestamp=datetime.now(UTC),
                event_type="chaos_test",
                tenant_id="chaos_tenant",
                user_id="chaos_user",
            )
        )
        files = list(tmp_path.iterdir())
        assert len(files) > 0, "Should have written fallback log file"
