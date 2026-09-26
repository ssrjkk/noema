"""Tests for causal/engine.py — covering gaps: analyze_all, reset, constraints."""

from __future__ import annotations

from noema.causal.engine import CausalEngine
from noema.causal.graph import CausalGraph


def _simple_engine_and_graph():
    engine = CausalEngine(enabled=True, max_counterfactuals=3)
    reqs = [
        {"category": "perf", "description": "fast", "priority": 8},
        {"category": "security", "description": "safe", "priority": 9},
        {"category": "ux", "description": "easy", "priority": 7},
    ]
    graph = engine.build_graph(reqs)
    return engine, graph


def test_build_graph_with_constraints():
    engine = CausalEngine(enabled=True)
    reqs = [{"category": "a", "description": "first", "priority": 5}]
    constraints = [{"name": "budget"}, {"name": "time"}]
    graph = engine.build_graph(reqs, constraints)
    assert graph.node_count == 3  # 1 req + 2 constraints


def test_build_graph_with_dependencies():
    engine = CausalEngine(enabled=True)
    reqs = [
        {"category": "a", "description": "first", "priority": 5},
        {"category": "b", "description": "second", "priority": 7},
    ]
    deps = [("a_first", "b_second", 0.5)]
    graph = engine.build_graph(reqs, dependencies=deps)
    assert graph.edge_count == 1


def test_analyze_counterfactual_missing_treatment():
    engine = CausalEngine(enabled=True)
    reqs = [{"category": "a", "description": "test", "priority": 5}]
    graph = engine.build_graph(reqs)
    result = engine.analyze_counterfactual(graph, "nonexistent", "also_missing")
    assert result is None


def test_analyze_counterfactual_missing_outcome():
    engine = CausalEngine(enabled=True)
    reqs = [{"category": "a", "description": "test", "priority": 5}]
    graph = engine.build_graph(reqs)
    nodes = graph.get_all_nodes()
    result = engine.analyze_counterfactual(graph, nodes[0].id, "nonexistent")
    assert result is None


def test_analyze_counterfactual_disabled():
    engine = CausalEngine(enabled=False)
    reqs = [{"category": "a", "description": "test", "priority": 5}]
    graph = engine.build_graph(reqs)
    nodes = graph.get_all_nodes()
    result = engine.analyze_counterfactual(graph, nodes[0].id, nodes[0].id)
    assert result is None


def test_analyze_all_counterfactuals():
    engine, graph = _simple_engine_and_graph()
    results = engine.analyze_all_counterfactuals(graph, "test task")
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert "variable" in r
        assert "intervention_1x" in r
        assert "intervention_2x" in r
        assert "backdoor_variables" in r
        assert "mediator_variables" in r


def test_analyze_all_counterfactuals_disabled():
    engine = CausalEngine(enabled=False)
    reqs = [{"category": "a", "description": "test", "priority": 5}]
    graph = engine.build_graph(reqs)
    results = engine.analyze_all_counterfactuals(graph, "test")
    assert results == []


def test_analyze_all_counterfactuals_empty_graph():
    engine = CausalEngine(enabled=True)
    graph = CausalGraph()
    results = engine.analyze_all_counterfactuals(graph, "test")
    assert results == []


def test_analyze_all_counterfactuals_respects_max():
    engine = CausalEngine(enabled=True, max_counterfactuals=1)
    reqs = [
        {"category": "a", "description": "first", "priority": 5},
        {"category": "b", "description": "second", "priority": 7},
        {"category": "c", "description": "third", "priority": 3},
    ]
    graph = engine.build_graph(reqs)
    results = engine.analyze_all_counterfactuals(graph, "test")
    assert len(results) <= 1


def test_metrics_tracking():
    engine, graph = _simple_engine_and_graph()
    metrics = engine.get_metrics()
    assert metrics["graphs_built"] == 1

    nodes = graph.get_all_nodes()
    engine.analyze_counterfactual(graph, nodes[0].id, nodes[-1].id)
    metrics = engine.get_metrics()
    assert metrics["interventions_run"] == 1
    assert metrics["counterfactuals_estimated"] == 1
    assert metrics["mean_confidence"] > 0.0


def test_reset_metrics():
    engine, graph = _simple_engine_and_graph()
    nodes = graph.get_all_nodes()
    engine.analyze_counterfactual(graph, nodes[0].id, nodes[-1].id)
    assert engine.get_metrics()["interventions_run"] > 0

    engine.reset_metrics()
    metrics = engine.get_metrics()
    assert metrics["graphs_built"] == 0
    assert metrics["interventions_run"] == 0
    assert metrics["counterfactuals_estimated"] == 0
    assert metrics["mean_confidence"] == 0.0
    assert metrics["total_confidence"] == 0.0


def test_get_metrics_returns_copy():
    engine = CausalEngine()
    m1 = engine.get_metrics()
    m1["graphs_built"] = 999
    m2 = engine.get_metrics()
    assert m2["graphs_built"] == 0
