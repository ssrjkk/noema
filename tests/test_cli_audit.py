"""Tests for CLI audit commands."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from noema.cli.audit import audit_app

runner = CliRunner()


def test_audit_proof_prints_to_console(tmp_path):
    """proof() without --output should print JSON to console."""
    mock_audit = AsyncMock()
    mock_proof = {
        "block_index": 5,
        "leaf_hash": "a" * 64,
        "root_hash": "b" * 64,
        "path": [],
    }
    mock_audit.get_proof_for_task = AsyncMock(return_value=mock_proof)
    mock_audit.initialize = AsyncMock()

    with patch("noema.audit.logger.AuditLogger", return_value=mock_audit):
        result = runner.invoke(
            audit_app,
            ["proof", "--tenant", "tenant1", "--task", "task1", "--fallback-dir", str(tmp_path)],
        )

    assert result.exit_code == 0
    assert "task1" in result.stdout
    assert "tenant1" in result.stdout


def test_audit_proof_saves_to_file(tmp_path):
    """proof() with --output should write proof JSON to file."""
    mock_audit = AsyncMock()
    mock_proof = {"block_index": 5, "leaf_hash": "a" * 64, "root_hash": "b" * 64, "path": []}
    mock_audit.get_proof_for_task = AsyncMock(return_value=mock_proof)
    mock_audit.initialize = AsyncMock()

    output_file = tmp_path / "proof.json"

    with patch("noema.audit.logger.AuditLogger", return_value=mock_audit):
        result = runner.invoke(
            audit_app,
            [
                "proof",
                "--tenant",
                "tenant1",
                "--task",
                "task1",
                "--output",
                str(output_file),
                "--fallback-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0
    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert data["task_id"] == "task1"
    assert data["tenant_id"] == "tenant1"
    assert "proof" in data


def test_audit_verify_file_not_found():
    """verify() with nonexistent file should exit with code 1."""
    result = runner.invoke(audit_app, ["verify", "/nonexistent/proof.json"])
    assert result.exit_code == 1
    assert "File not found" in result.stdout


def test_audit_verify_no_leaf_data(tmp_path):
    """verify() without leaf_data should exit with code 1."""
    proof_file = tmp_path / "proof.json"
    proof_file.write_text(json.dumps({"block_index": 5, "root_hash": "abc"}))

    result = runner.invoke(audit_app, ["verify", str(proof_file)])
    assert result.exit_code == 1
    assert "leaf_data" in result.stdout


def test_audit_verify_valid_proof(tmp_path):
    """verify() with valid proof should show VALID."""
    mock_proof = MagicMock()
    mock_proof.block_index = 5
    mock_proof.leaf_hash = b"\x00" * 32
    mock_proof.root_hash = b"\x01" * 32
    mock_proof.path = []

    proof_data = {
        "proof": {"block_index": 5, "leaf_hash": "00" * 32, "root_hash": "01" * 32, "path": []},
        "leaf_data": {"task_id": "task1"},
    }

    proof_file = tmp_path / "proof.json"
    proof_file.write_text(json.dumps(proof_data))

    with patch("noema.cli.audit.InclusionProof") as mock_inclusion, patch(
        "noema.cli.audit.verify_inclusion_proof", return_value=True
    ):
        mock_inclusion.from_dict.return_value = mock_proof
        result = runner.invoke(audit_app, ["verify", str(proof_file)])

    assert result.exit_code == 0
    assert "VALID" in result.stdout


def test_audit_chain_stats():
    """chain stats should display chain information."""
    with patch("noema.audit.merkle.MerkleChainAudit") as mock_chain:
        mock_chain_obj = MagicMock()
        mock_chain_obj.to_dict.return_value = {
            "chain_id": "test-chain",
            "height": 10,
            "root": "a" * 64,
        }
        mock_chain.return_value = mock_chain_obj

        result = runner.invoke(audit_app, ["chain", "stats"])

    assert result.exit_code == 0
    assert "Merkle Chain" in result.stdout


def test_audit_chain_append_no_event():
    """chain append without --event should exit with code 1."""
    result = runner.invoke(audit_app, ["chain", "append"])
    assert result.exit_code == 1
    assert "--event" in result.stdout


def test_audit_chain_append_with_event():
    """chain append with --event should append block."""
    with patch("noema.audit.merkle.MerkleChainAudit") as mock_chain:
        mock_block = MagicMock()
        mock_block.index = 1
        mock_block.block_hash = b"\x00" * 32

        mock_chain_obj = MagicMock()
        mock_chain_obj.append.return_value = mock_block
        mock_chain.return_value = mock_chain_obj

        result = runner.invoke(
            audit_app, ["chain", "append", "--event", '{"action": "test"}']
        )

    assert result.exit_code == 0
    assert "Block appended" in result.stdout


def test_audit_chain_export_no_path():
    """chain export without --export should exit with code 1."""
    result = runner.invoke(audit_app, ["chain", "export"])
    assert result.exit_code == 1
    assert "--export" in result.stdout


def test_audit_chain_export_with_path(tmp_path):
    """chain export with --export should write chain to file."""
    export_file = tmp_path / "chain.json"

    with patch("noema.audit.merkle.MerkleChainAudit") as mock_chain:
        mock_chain_obj = MagicMock()
        mock_chain_obj.chain_id = "test-chain"
        mock_chain_obj.export_blocks.return_value = [{"index": 0, "data": {}}]
        mock_chain.return_value = mock_chain_obj

        result = runner.invoke(audit_app, ["chain", "export", "--export", str(export_file)])

    assert result.exit_code == 0
    assert export_file.exists()
    data = json.loads(export_file.read_text())
    assert data["chain_id"] == "test-chain"


def test_audit_chain_import_no_path():
    """chain import without --import should exit with code 1."""
    result = runner.invoke(audit_app, ["chain", "import"])
    assert result.exit_code == 1
    assert "--import" in result.stdout


def test_audit_chain_import_with_path(tmp_path):
    """chain import with --import should load and verify chain."""
    import_file = tmp_path / "chain.json"
    import_file.write_text(
        json.dumps({"chain_id": "test-chain", "blocks": [{"index": 0, "data": {}}]})
    )

    with patch("noema.audit.merkle.MerkleChainAudit") as mock_chain:
        mock_chain_obj = MagicMock()
        mock_chain_obj.chain_id = "test-chain"
        mock_chain_obj.height = 1
        mock_chain_obj.verify_chain.return_value = True
        mock_chain.import_blocks.return_value = mock_chain_obj

        result = runner.invoke(audit_app, ["chain", "import", "--import", str(import_file)])

    assert result.exit_code == 0
    assert "verified=True" in result.stdout


def test_audit_chain_unknown_action():
    """chain with unknown action should exit with code 1."""
    result = runner.invoke(audit_app, ["chain", "unknown"])
    assert result.exit_code == 1
    assert "Unknown action" in result.stdout
