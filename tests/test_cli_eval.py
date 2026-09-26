"""Tests for CLI eval commands — leaderboard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from noema.cli.eval import eval_app
from noema.eval.leaderboard import LeaderboardRow

runner = CliRunner()


def test_leaderboard_no_results_dir(tmp_path: Path):
    nonexistent = tmp_path / "does_not_exist"
    result = runner.invoke(eval_app, ["--out", str(nonexistent)])
    assert result.exit_code == 0
    assert "not found" in result.stdout.lower() or "Results dir not found" in result.stdout


def test_leaderboard_empty_dir(tmp_path: Path):
    empty = tmp_path / "empty_results"
    empty.mkdir()
    result = runner.invoke(eval_app, ["--out", str(empty)])
    assert result.exit_code == 0
    assert "No experiment summaries" in result.stdout


def test_leaderboard_with_summaries(tmp_path: Path):
    run_dir = tmp_path / "exp1" / "run1"
    run_dir.mkdir(parents=True)
    summary = {
        "summary": [
            {
                "provider": "openai",
                "model": "gpt-4",
                "n": 5,
                "mean_judge_score": 0.85,
                "mean_latency_ms": 1200.0,
                "mean_total_tokens": 3500.0,
                "mean_cost": 0.05,
                "error_rate": 0.02,
                "sandbox_pass_rate": 0.95,
            }
        ]
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))

    result = runner.invoke(eval_app, ["--out", str(tmp_path)])
    assert result.exit_code == 0
    assert "openai" in result.stdout
    assert "gpt-4" in result.stdout


def test_leaderboard_markdown_output(tmp_path: Path):
    run_dir = tmp_path / "exp1" / "run1"
    run_dir.mkdir(parents=True)
    summary = {
        "summary": [
            {
                "provider": "anthropic",
                "model": "claude-3",
                "n": 3,
                "mean_judge_score": 0.92,
                "mean_latency_ms": 800.0,
                "mean_total_tokens": 2000.0,
                "mean_cost": 0.03,
                "error_rate": 0.01,
            }
        ]
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))

    result = runner.invoke(eval_app, ["--out", str(tmp_path), "--markdown"])
    assert result.exit_code == 0
    output = result.stdout
    assert "| Provider |" in output
    assert "anthropic" in output
    assert "claude-3" in output


def test_leaderboard_top_n(tmp_path: Path):
    for i in range(5):
        run_dir = tmp_path / f"exp{i}" / "run1"
        run_dir.mkdir(parents=True)
        summary = {
            "summary": [
                {
                    "provider": f"provider_{i}",
                    "model": f"model_{i}",
                    "n": 1,
                    "mean_judge_score": 0.5 + i * 0.1,
                    "mean_latency_ms": 1000.0,
                    "mean_total_tokens": 1000.0,
                    "mean_cost": 0.01,
                    "error_rate": 0.0,
                }
            ]
        }
        (run_dir / "summary.json").write_text(json.dumps(summary))

    result = runner.invoke(eval_app, ["--out", str(tmp_path), "--top", "2"])
    assert result.exit_code == 0
    assert "2 entries" in result.stdout
