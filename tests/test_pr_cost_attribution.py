"""Line-level cost attribution: task cost → per-module PR cost + metrics."""

from __future__ import annotations

import pytest

from noema.billing.cost_tracker import CostTracker, calculate_line_costs


class TestCalculateLineCosts:
    def test_weighted_by_lines(self):
        costs = calculate_line_costs(1.0, {"a.py": 10, "b.py": 30})
        assert costs["a.py"] == pytest.approx(0.25)
        assert costs["b.py"] == pytest.approx(0.75)
        assert sum(costs.values()) == pytest.approx(1.0)

    def test_zero_total_cost(self):
        costs = calculate_line_costs(0.0, {"a.py": 10})
        assert costs == {"a.py": 0.0}

    def test_zero_lines_gets_no_weight(self):
        costs = calculate_line_costs(1.0, {"a.py": 10, "b.py": 0})
        assert costs["a.py"] == pytest.approx(1.0)
        assert costs["b.py"] == pytest.approx(0.0)

    def test_empty_input(self):
        assert calculate_line_costs(5.0, {}) == {}


@pytest.mark.asyncio
class TestPrCostAttribution:
    async def test_attribute_pr_cost_distributes_task_cost(self):
        tracker = CostTracker()
        await tracker.record(
            tenant_id="acme",
            task_id="t1",
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=1_000_000,
            output_tokens=0,
        )
        assert tracker.get_task_cost("t1") == pytest.approx(0.15)

        files = [("src/payments/api.py", "x = 1\n" * 10), ("src/worker/job.py", "y = 2\n" * 30)]
        attributed = tracker.attribute_pr_cost("acme/payments", 7, "t1", files, model="gpt-4o-mini")

        assert len(attributed) == 2
        by_path = {f.file_path: f for f in attributed}
        assert by_path["src/payments/api.py"].cost_usd == pytest.approx(0.0375)
        assert by_path["src/worker/job.py"].cost_usd == pytest.approx(0.1125)
        total = sum(f.cost_usd for f in attributed)
        assert total == pytest.approx(0.15)

    async def test_get_pr_cost_and_module_stats(self):
        tracker = CostTracker()
        await tracker.record(
            tenant_id="acme",
            task_id="t1",
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=200_000,
            output_tokens=0,
        )
        tracker.attribute_pr_cost("acme/payments", 7, "t1", [("app/billing.py", "a\nb\nc\n")])

        pr_cost = tracker.get_pr_cost("acme/payments", 7)
        assert pr_cost["total_usd"] == pytest.approx(0.03)
        assert pr_cost["modules"] == {"app/billing.py": 0.03}

        other = tracker.get_pr_cost("acme/payments", 99)
        assert other["total_usd"] == 0.0

        stats = tracker.module_cost_stats("acme/payments")
        assert stats["modules"] == {"app/billing.py": 0.03}
        assert stats["total_usd"] == pytest.approx(0.03)

    async def test_attribute_with_no_task_cost_is_zero(self):
        tracker = CostTracker()
        attributed = tracker.attribute_pr_cost("o/r", 1, "missing-task", [("a.py", "x\n")])
        assert attributed[0].cost_usd == 0.0
        assert tracker.get_pr_cost("o/r", 1)["total_usd"] == 0.0

    async def test_attribute_empty_files(self):
        tracker = CostTracker()
        assert tracker.attribute_pr_cost("o/r", 1, "t", []) == []

    async def test_exposes_prometheus_metrics(self):
        from noema.observability.metrics import _HAS_PROMETHEUS

        if not _HAS_PROMETHEUS:
            pytest.skip("prometheus_client not installed")

        tracker = CostTracker()
        await tracker.record(
            tenant_id="acme",
            task_id="t-metrics",
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=100_000,
            output_tokens=0,
        )
        tracker.attribute_pr_cost("acme/payments", 42, "t-metrics", [("app/api.py", "x\n" * 5)])

        from prometheus_client import generate_latest

        body = generate_latest().decode()
        assert "noema_pr_cost_usd" in body
        assert "noema_code_cost_per_module" in body
        assert 'module="app/api.py"' in body
        assert 'pr="42"' in body
