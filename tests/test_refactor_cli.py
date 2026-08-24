"""CLI smoke coverage — noema cli commands via Typer's CliRunner.

Only commands that avoid a live LLM/DB/network are exercised; the heavy
``think`` / ``pipeline`` / ``evolve`` / ``serve`` paths stay out.
"""

import asyncio
import json
import re
from datetime import UTC, datetime

import pytest
from typer.testing import CliRunner

from noema.audit.logger import AuditEvent, AuditLogger
from noema.cli.audit import audit_app
from noema.cli.health import health_app
from noema.cli.init_cmd import init_app
from noema.cli.main import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def test_app_no_args_shows_help():
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage: noema" in _strip_ansi(result.output)


def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage: noema" in _strip_ansi(result.output)


def test_graph_stats():
    result = runner.invoke(app, ["graph", "stats"])
    assert result.exit_code == 0


def test_graph_suggest_with_tags():
    result = runner.invoke(app, ["graph", "suggest", "--tags", "python,fastapi"])
    assert result.exit_code == 0


def test_graph_compatible_with_tech():
    result = runner.invoke(app, ["graph", "compatible", "--tech", "fastapi"])
    assert result.exit_code == 0


def test_graph_unknown_action_shows_usage():
    result = runner.invoke(app, ["graph", "nonsense"])
    assert result.exit_code == 0
    assert "Usage: noema graph" in result.output


def test_knowledge_stats():
    result = runner.invoke(app, ["knowledge", "stats"])
    assert result.exit_code == 0


def test_knowledge_search_requires_query():
    result = runner.invoke(app, ["knowledge", "search"])
    assert result.exit_code == 0
    assert "Specify --query" in result.output


def test_modules_list():
    result = runner.invoke(app, ["modules", "list"])
    assert result.exit_code == 0
    assert "Noema Modules" in result.output


def test_modules_stats():
    result = runner.invoke(app, ["modules", "stats"])
    assert result.exit_code == 0
    assert "Module Stats" in result.output


def test_modules_run_requires_name():
    result = runner.invoke(app, ["modules", "run"])
    assert result.exit_code == 0
    assert "Specify --name" in result.output


def test_memory_stats():
    result = runner.invoke(app, ["memory", "stats"])
    assert result.exit_code == 0
    assert "Memory Stats" in result.output


def test_feedback_stats():
    result = runner.invoke(app, ["feedback", "stats"])
    assert result.exit_code == 0


def test_agents_command():
    result = runner.invoke(app, ["agents"])
    assert result.exit_code == 0
    assert "architect" in result.output.lower()


def test_kernels_command():
    result = runner.invoke(app, ["kernels"])
    assert result.exit_code == 0
    assert "Available Kernels" in result.output


def test_hierarchy_command():
    result = runner.invoke(app, ["hierarchy"])
    assert result.exit_code == 0
    assert "Worker Hierarchy" in result.output


def test_discover_command():
    result = runner.invoke(app, ["discover"])
    assert result.exit_code == 0
    assert "Resource Discovery" in result.output


def test_health_check_llm():
    result = runner.invoke(health_app, ["check-llm"])
    assert result.exit_code == 0
    assert "ollama" in result.output


def test_health_check_all():
    result = runner.invoke(health_app, ["check"])
    assert result.exit_code in (0, 1)
    assert "System Health" in result.output


def test_init_show_config():
    result = runner.invoke(init_app, ["show-config"])
    assert result.exit_code == 0
    body = json.loads(result.output)
    assert "db" in body


def test_init_creates_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(init_app, ["init", "--db-url", "sqlite:///test.db"])
    assert result.exit_code == 0
    assert (tmp_path / "settings.yaml").exists()
    content = (tmp_path / "settings.yaml").read_text(encoding="utf-8")
    assert "test.db" in content


def test_init_second_run_without_force_exits_1(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = runner.invoke(init_app, ["init"])
    assert first.exit_code == 0
    second = runner.invoke(init_app, ["init"])
    assert second.exit_code == 1
    assert "Use --force" in second.output


def test_init_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(init_app, ["init", "--db-url", "sqlite:///first.db"])
    forced = runner.invoke(init_app, ["init", "--force", "--db-url", "sqlite:///second.db"])
    assert forced.exit_code == 0
    content = (tmp_path / "settings.yaml").read_text(encoding="utf-8")
    assert "second.db" in content


def test_audit_chain_stats():
    result = runner.invoke(audit_app, ["chain", "stats"])
    assert result.exit_code == 0
    assert "Merkle Chain" in result.output


def test_audit_chain_append_requires_event():
    result = runner.invoke(audit_app, ["chain", "append"])
    assert result.exit_code == 1
    assert "Specify --event" in result.output


def test_audit_chain_unknown_action():
    result = runner.invoke(audit_app, ["chain", "nope"])
    assert result.exit_code == 1
    assert "Unknown action" in result.output


def test_audit_chain_append_export_import(tmp_path):
    export = tmp_path / "chain.json"
    append = runner.invoke(audit_app, ["chain", "append", "--event", '{"x": 1}'])
    assert append.exit_code == 0
    assert "Block appended" in append.output

    export_result = runner.invoke(audit_app, ["chain", "export", "--export", str(export)])
    assert export_result.exit_code == 0
    assert export.exists()

    import_result = runner.invoke(audit_app, ["chain", "import", "--import", str(export)])
    assert import_result.exit_code == 0
    assert "verified=True" in import_result.output


def test_audit_verify_missing_file(tmp_path):
    result = runner.invoke(audit_app, ["verify", str(tmp_path / "nope.json")])
    assert result.exit_code == 1
    assert "File not found" in result.output


def _write_proof_file(tmp_path, proof_dict, with_leaf=True):
    doc = {"proof": proof_dict}
    if with_leaf:
        doc["leaf_data"] = proof_dict["leaf_data"]
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.fixture
def audit_proof(tmp_path):
    async def _setup():
        logger = AuditLogger(pg_pool=None, fallback_dir=str(tmp_path))
        await logger.initialize()
        for i in range(6):
            await logger.log(
                AuditEvent(
                    datetime.now(UTC),
                    "task.completed",
                    "tenant-a",
                    f"u{i}",
                    task_id=f"task-{i}",
                    details={"result": f"r{i}"},
                )
            )
        return await logger.get_proof_for_task(tenant_id="tenant-a", task_id="task-2")

    return asyncio.run(_setup())


def test_audit_verify_valid_proof(tmp_path, audit_proof):
    path = _write_proof_file(tmp_path, audit_proof)
    result = runner.invoke(audit_app, ["verify", str(path)])
    assert result.exit_code == 0
    assert "VALID" in result.output


def test_audit_verify_tampered_proof(tmp_path, audit_proof):
    tampered = dict(audit_proof)
    leaf = dict(audit_proof["leaf_data"])
    leaf["details"] = {"result": "HACKED"}
    tampered["leaf_data"] = leaf
    path = _write_proof_file(tmp_path, tampered)
    result = runner.invoke(audit_app, ["verify", str(path)])
    assert result.exit_code == 1
    assert "INVALID" in result.output


def test_audit_verify_missing_leaf(tmp_path, audit_proof):
    no_leaf = dict(audit_proof)
    no_leaf.pop("leaf_data", None)
    path = _write_proof_file(tmp_path, no_leaf, with_leaf=False)
    result = runner.invoke(audit_app, ["verify", str(path)])
    assert result.exit_code == 1
    assert "Provide --leaf-data" in result.output
