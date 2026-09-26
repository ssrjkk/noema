"""Tests for CLI arq commands — worker, enqueue, workers, ledger."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from noema.cli.arq import arq_app

runner = CliRunner()


def test_enqueue():
    with patch("noema.workers.arq_worker.enqueue_think", new_callable=AsyncMock, return_value="job-abc-123"):
        result = runner.invoke(
            arq_app,
            ["enqueue", "Build a REST API", "--description", "with auth", "--complexity", "simple"],
        )
        assert result.exit_code == 0
        output = result.stdout
        assert "job-abc-123" in output
        parsed = json.loads(output.strip().split("\n")[-1])
        assert parsed["job_id"] == "job-abc-123"
        assert parsed["status"] == "queued"


def test_enqueue_default_complexity():
    with patch("noema.workers.arq_worker.enqueue_think", new_callable=AsyncMock, return_value="job-xyz"):
        result = runner.invoke(arq_app, ["enqueue", "Simple task"])
        assert result.exit_code == 0
        assert "job-xyz" in result.stdout


def test_workers_list():
    fleet = [
        {
            "node_id": "worker-1",
            "started_at": "2026-09-25T10:00:00Z",
            "last_heartbeat": "2026-09-25T10:05:00Z",
            "draining": "0",
            "metrics_port": "9090",
        },
        {
            "node_id": "worker-2",
            "started_at": "2026-09-25T09:00:00Z",
            "last_heartbeat": "2026-09-25T10:04:59Z",
            "draining": "1",
            "metrics_port": "9091",
        },
    ]

    with patch("noema.workers.arq_worker.list_active_workers", new_callable=AsyncMock, return_value=fleet):
        result = runner.invoke(arq_app, ["workers"])
        assert result.exit_code == 0
        output = result.stdout
        assert "worker-1" in output
        assert "worker-2" in output
        assert "draining" in output.lower() or "yes" in output
        lines = output.strip().split("\n")
        parsed = json.loads(lines[-1])
        assert len(parsed) == 2


def test_workers_empty():
    with patch("noema.workers.arq_worker.list_active_workers", new_callable=AsyncMock, return_value=[]):
        result = runner.invoke(arq_app, ["workers"])
        assert result.exit_code == 0
        lines = result.stdout.strip().split("\n")
        parsed = json.loads(lines[-1])
        assert parsed == []


def test_ledger(tmp_path):
    ledger_file = tmp_path / "ledger.jsonl"
    ledger_file.write_text("")

    mock_ledger = MagicMock()
    mock_ledger.load.return_value = 5
    mock_ledger.per_node.return_value = {"node-1": 3, "node-2": 2}
    mock_ledger.audit.return_value = {"entries": [{"task_id": "t1", "value": 100}]}
    mock_ledger.entries_for.return_value = [{"task_id": "t1", "value": 100}]

    with patch("noema.billing.ledger.ContributionLedger", return_value=mock_ledger):
        result = runner.invoke(arq_app, ["ledger", str(ledger_file)])
        assert result.exit_code == 0
        assert "5" in result.stdout
        assert "entries" in result.stdout


def test_ledger_with_task_filter(tmp_path):
    ledger_file = tmp_path / "ledger.jsonl"
    ledger_file.write_text("")

    mock_ledger = MagicMock()
    mock_ledger.load.return_value = 3
    mock_ledger.per_node.return_value = {"node-1": 3}
    mock_ledger.entries_for.return_value = [{"task_id": "t1", "value": 50}]

    with patch("noema.billing.ledger.ContributionLedger", return_value=mock_ledger):
        result = runner.invoke(
            arq_app, ["ledger", str(ledger_file), "--task-id", "t1"]
        )
        assert result.exit_code == 0
        mock_ledger.entries_for.assert_called_once_with("t1")
        mock_ledger.per_node.assert_called_once_with(task_id="t1")
