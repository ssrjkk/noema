"""Tests for the eval leaderboard (aggregation over experiment summaries)."""

from __future__ import annotations

import json

import pytest

from noema.eval.leaderboard import (
    compute_leaderboard,
    render_markdown,
    to_dict,
)


def _write_summary(root, parts, payload) -> None:
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _summary_rows(provider: str, model: str, score: float, n: int = 2, **over) -> dict:
    row = {
        "provider": provider,
        "model": model,
        "n": n,
        "mean_judge_score": score,
        "stddev_judge_score": 0.0,
        "mean_latency_ms": 100.0,
        "stddev_latency_ms": 0.0,
        "mean_total_tokens": 1000.0,
        "mean_cost": 0.001,
        "sandbox_pass_rate": 1.0,
        "error_rate": 0.0,
    }
    row.update(over)
    return row


def _run_summary(run_id: str, rows: list[dict]) -> dict:
    return {"run_id": run_id, "generated_at": "2026-01-01T00:00:00Z", "n_records": sum(r["n"] for r in rows), "summary": rows}


@pytest.fixture
def results_root(tmp_path):
    exp_a = tmp_path / "exp_a"
    _write_summary(
        exp_a,
        ["run1", "summary.json"],
        _run_summary("run1", [_summary_rows("openai", "gpt-4", 0.90), _summary_rows("ollama", "gemma", 0.60)]),
    )
    _write_summary(
        exp_a,
        ["run2", "summary.json"],
        _run_summary("run2", [_summary_rows("openai", "gpt-4", 0.80), _summary_rows("ollama", "gemma", 0.70)]),
    )
    # second experiment for the same provider/model → merged
    exp_b = tmp_path / "exp_b"
    _write_summary(
        exp_b,
        ["run3", "summary.json"],
        _run_summary("run3", [_summary_rows("openai", "gpt-4", 0.85, n=2, mean_latency_ms=50.0)]),
    )
    return tmp_path


def test_compute_leaderboard_merges_and_ranks(results_root):
    rows = compute_leaderboard(results_root)
    by_key = {r.key: r for r in rows}

    # openai/gpt-4 across 3 runs / 6 cells, weighted mean (0.9*2+0.8*2+0.85*2)/6
    openai = by_key[("openai", "gpt-4")]
    assert openai.runs == 3
    assert openai.cells == 6
    assert openai.mean_judge_score == pytest.approx(0.85, abs=1e-4)
    assert openai.best_score == 0.90
    assert openai.worst_score == 0.80

    ollama = by_key[("ollama", "gemma")]
    assert ollama.runs == 2
    assert ollama.mean_judge_score == pytest.approx(0.65, abs=1e-4)

    # ranked: openai first (higher score)
    assert rows[0].key == ("openai", "gpt-4")


def test_weighted_by_cell_count(results_root):
    rows = compute_leaderboard(results_root)
    openai = next(r for r in rows if r.key == ("openai", "gpt-4"))
    # weight uses cells: run with n=2 counts double
    assert openai.mean_latency_ms == pytest.approx((100 * 2 + 100 * 2 + 50 * 2) / 6, abs=1e-3)


def test_top_n_truncation(results_root):
    rows = compute_leaderboard(results_root, top_n=1)
    assert len(rows) == 1
    assert rows[0].key == ("openai", "gpt-4")


def test_empty_root_returns_empty_list(tmp_path):
    assert compute_leaderboard(tmp_path) == []


def test_corrupt_summary_is_skipped(results_root):
    corrupted = results_root / "corrupt_exp" / "run9"
    corrupted.mkdir(parents=True, exist_ok=True)
    (corrupted / "summary.json").write_text("{ nope", encoding="utf-8")
    rows = compute_leaderboard(results_root)
    assert "corrupt_exp" not in {r.provider for r in rows}
    assert [r.key for r in rows]  # others still present


def test_missing_optional_fields_do_not_crash(tmp_path):
    path = tmp_path / "x" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"run_id": "x", "summary": [{"provider": "p", "model": "m"}]}),
        encoding="utf-8",
    )
    rows = compute_leaderboard(tmp_path)
    assert len(rows) == 1
    assert rows[0].mean_judge_score == 0.0


def test_empty_provider_model_skipped(tmp_path):
    _write_summary(tmp_path, ["x", "summary.json"], _run_summary("x", [_summary_rows("", "m", 0.5)]))
    assert compute_leaderboard(tmp_path) == []


def test_render_markdown(results_root):
    rows = compute_leaderboard(results_root)
    md = render_markdown(rows)
    assert "| Provider | Model |" in md
    assert "openai" in md
    assert "| gpt-4 |" in md
    assert "~%" in md or "%" in md
    assert md.endswith("\n")


def test_render_markdown_empty(tmp_path):
    assert compute_leaderboard(tmp_path / "empty") == []
    md = render_markdown([])
    assert "No experiment runs found" in md


def test_to_dict_json_serializable(results_root):
    import json as _json

    data = to_dict(compute_leaderboard(results_root))
    assert data and isinstance(data[0]["mean_judge_score"], float)
    _json.dumps(data)


def test_flat_run_layout_supported(tmp_path):
    # `--out results` without experiment grouping → results/<run_id>/summary.json
    _write_summary(tmp_path, ["runA", "summary.json"], _run_summary("runA", [_summary_rows("openai", "gpt-4", 0.7)]))
    rows = compute_leaderboard(tmp_path)
    assert rows[0].key == ("openai", "gpt-4")
