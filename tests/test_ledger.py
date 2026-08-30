"""Tests for the contribution ledger (T3.3): record, aggregate, persist."""

from __future__ import annotations

import json

from noema.billing.ledger import ContributionLedger


def test_record_and_per_node_aggregation(tmp_path):
    ledger = ContributionLedger(path=tmp_path / "ledger.jsonl")
    ledger.record(node_id="node-a", task_id="t1", input_tokens=100, output_tokens=50, cost_usd=0.5)
    ledger.record(node_id="node-a", task_id="t2", input_tokens=10, output_tokens=5, cost_usd=0.05)
    ledger.record(node_id="node-b", task_id="t1", input_tokens=30, output_tokens=20, cost_usd=0.2)

    agg = ledger.per_node()
    assert agg["nodes"]["node-a"]["entries"] == 2
    assert agg["nodes"]["node-a"]["total_tokens"] == 165
    assert agg["nodes"]["node-a"]["cost_usd"] == 0.55
    assert agg["nodes"]["node-b"]["total_tokens"] == 50
    assert agg["total_tokens"] == 215
    assert agg["total_cost_usd"] == 0.75

    only_t1 = ledger.per_node(task_id="t1")
    assert only_t1["nodes"]["node-a"]["entries"] == 1
    assert "node-b" in only_t1["nodes"]


def test_delegated_entries_flagged(tmp_path):
    ledger = ContributionLedger()
    ledger.record(node_id="n", task_id="t", peer="10.0.0.2:50051", input_tokens=5, output_tokens=5)
    agg = ledger.per_node()
    assert agg["nodes"]["n"]["delegated_to_peers"] == 1


def test_task_trail_and_audit(tmp_path):
    ledger = ContributionLedger()
    ledger.record(node_id="a", task_id="t1", kind="solution")
    ledger.record(node_id="a", task_id="t1", kind="subtask")
    ledger.record(node_id="b", task_id="t2")
    trail = ledger.entries_for("t1")
    assert len(trail) == 2
    assert all(e["task_id"] == "t1" for e in trail)
    assert ledger.audit()["count"] == 3


def test_jsonl_persistence_roundtrip(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = ContributionLedger(path=path)
    ledger.record(node_id="a", task_id="t", model="gpt-4o", input_tokens=1, output_tokens=2)
    ledger.record(node_id="b", task_id="t", input_tokens=3, output_tokens=4)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["node_id"] == "a"

    fresh = ContributionLedger(path=path)
    assert fresh.load() == 2
    assert fresh.audit()["entries"][0]["model"] == "gpt-4o"


def test_corrupt_lines_skipped_on_load(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        '{"node_id": "a", "task_id": "t"}\nnot-json\n\n',
        encoding="utf-8",
    )
    ledger = ContributionLedger(path=path)
    assert ledger.load() == 1


def test_memory_bounded(tmp_path):
    ledger = ContributionLedger(max_entries=3)
    for i in range(10):
        ledger.record(node_id=f"n{i % 2}", task_id=f"t{i}")
    assert ledger.audit()["count"] == 3


def test_write_failure_never_raises(tmp_path):
    # A directory as the ledger path makes every append fail with
    # IsADirectoryError (an OSError) — which the ledger must swallow: it can
    # never break the run it is auditing.
    ledger = ContributionLedger(path=tmp_path)  # tmp_path is a directory
    entry = ledger.record(node_id="a", task_id="t")
    assert entry.node_id == "a"
    assert ledger.audit()["count"] == 1
