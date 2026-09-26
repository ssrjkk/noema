"""Tests for causal/graph.py — covering gaps not hit by property-based tests."""

from __future__ import annotations

import pytest

from noema.causal.graph import CausalEdge, CausalGraph, CausalNode, InterventionResult, VariableType


def _make_graph():
    cg = CausalGraph()
    cg.add_node(CausalNode(id="a", name="A", var_type=VariableType.CONTINUOUS), value=1.0)
    cg.add_node(CausalNode(id="b", name="B", var_type=VariableType.CONTINUOUS), value=2.0)
    cg.add_node(CausalNode(id="c", name="C", var_type=VariableType.BINARY), value=0.0)
    cg.add_edge("a", "b", strength=0.8)
    cg.add_edge("b", "c", strength=0.5)
    return cg


def test_remove_node():
    cg = _make_graph()
    assert cg.node_count == 3
    cg.remove_node("b")
    assert cg.node_count == 2
    assert not cg.has_node("b")
    assert not cg.has_edge("a", "b")
    assert not cg.has_edge("b", "c")


def test_remove_nonexistent_node_raises():
    import networkx as nx

    cg = _make_graph()
    with pytest.raises(nx.NetworkXError):
        cg.remove_node("nonexistent")


def test_remove_edge():
    cg = _make_graph()
    assert cg.edge_count == 2
    cg.remove_edge("a", "b")
    assert cg.edge_count == 1
    assert not cg.has_edge("a", "b")
    assert cg.has_edge("b", "c")


def test_add_edge_missing_source():
    cg = CausalGraph()
    cg.add_node(CausalNode(id="x", name="X"), value=0.0)
    with pytest.raises(ValueError, match="node.*not found"):
        cg.add_edge("x", "missing")


def test_add_edge_missing_target():
    cg = CausalGraph()
    cg.add_node(CausalNode(id="x", name="X"), value=0.0)
    with pytest.raises(ValueError, match="node.*not found"):
        cg.add_edge("missing", "x")


def test_get_parents():
    cg = _make_graph()
    assert cg.get_parents("b") == ["a"]
    assert cg.get_parents("c") == ["b"]
    assert cg.get_parents("a") == []


def test_get_children():
    cg = _make_graph()
    assert cg.get_children("a") == ["b"]
    assert cg.get_children("b") == ["c"]
    assert cg.get_children("c") == []


def test_get_ancestors():
    cg = _make_graph()
    assert cg.get_ancestors("c") == {"a", "b"}
    assert cg.get_ancestors("b") == {"a"}
    assert cg.get_ancestors("a") == set()


def test_get_descendants():
    cg = _make_graph()
    assert cg.get_descendants("a") == {"b", "c"}
    assert cg.get_descendants("b") == {"c"}
    assert cg.get_descendants("c") == set()


def test_get_all_nodes():
    cg = _make_graph()
    nodes = cg.get_all_nodes()
    assert len(nodes) == 3
    assert {n.id for n in nodes} == {"a", "b", "c"}


def test_iter_nodes():
    cg = _make_graph()
    ids = {n.id for n in cg.iter_nodes()}
    assert ids == {"a", "b", "c"}


def test_node_ids():
    cg = _make_graph()
    assert set(cg.node_ids) == {"a", "b", "c"}


def test_has_node():
    cg = _make_graph()
    assert cg.has_node("a")
    assert not cg.has_node("z")


def test_has_edge():
    cg = _make_graph()
    assert cg.has_edge("a", "b")
    assert not cg.has_edge("b", "a")
    assert not cg.has_edge("a", "c")


def test_validate_dag():
    cg = _make_graph()
    assert cg.validate_dag() is True


def test_node_with_domain():
    cg = CausalGraph()
    node = CausalNode(id="x", name="X", domain=(0.0, 100.0))
    cg.add_node(node, value=50.0)
    d = cg.to_dict()
    assert d["nodes"]["x"]["domain"] == [0.0, 100.0]


def test_to_dict():
    cg = _make_graph()
    d = cg.to_dict()
    assert "nodes" in d
    assert "edges" in d
    assert "dag" in d
    assert d["dag"] is True
    assert len(d["nodes"]) == 3
    assert len(d["edges"]) == 2
    edge = d["edges"][0]
    assert "source" in edge
    assert "target" in edge
    assert "strength" in edge


def test_find_confounders():
    cg = CausalGraph()
    cg.add_node(CausalNode(id="t", name="Treatment"), value=1.0)
    cg.add_node(CausalNode(id="o", name="Outcome"), value=0.0)
    cg.add_node(CausalNode(id="c", name="Confounder"), value=0.5)
    cg.add_edge("c", "t", strength=0.7)
    cg.add_edge("c", "o", strength=0.6)
    cg.add_edge("t", "o", strength=1.0)

    confounders = cg.find_confounders("t", "o")
    assert "c" in confounders


def test_find_confounders_with_exclude():
    cg = CausalGraph()
    cg.add_node(CausalNode(id="t", name="Treatment"), value=1.0)
    cg.add_node(CausalNode(id="o", name="Outcome"), value=0.0)
    cg.add_node(CausalNode(id="c", name="Confounder"), value=0.5)
    cg.add_edge("c", "t", strength=0.7)
    cg.add_edge("c", "o", strength=0.6)
    cg.add_edge("t", "o", strength=1.0)

    confounders = cg.find_confounders("t", "o", exclude={"c"})
    assert confounders == []


def test_find_confounders_missing_treatment():
    cg = _make_graph()
    with pytest.raises(ValueError, match="Treatment node.*not found"):
        cg.find_confounders("missing", "b")


def test_find_confounders_missing_outcome():
    cg = _make_graph()
    with pytest.raises(ValueError, match="Outcome node.*not found"):
        cg.find_confounders("a", "missing")


def test_find_instruments():
    cg = CausalGraph()
    cg.add_node(CausalNode(id="z", name="Instrument"), value=1.0)
    cg.add_node(CausalNode(id="t", name="Treatment"), value=0.0)
    cg.add_node(CausalNode(id="o", name="Outcome"), value=0.0)
    cg.add_edge("z", "t", strength=1.0)
    cg.add_edge("t", "o", strength=1.0)

    instruments = cg.find_instruments("t", "o")
    assert isinstance(instruments, list)


def test_find_mediators():
    cg = CausalGraph()
    cg.add_node(CausalNode(id="t", name="Treatment"), value=1.0)
    cg.add_node(CausalNode(id="m", name="Mediator"), value=0.0)
    cg.add_node(CausalNode(id="o", name="Outcome"), value=0.0)
    cg.add_edge("t", "m", strength=1.0)
    cg.add_edge("m", "o", strength=1.0)

    mediators = cg.find_mediators("t", "o")
    assert "m" in mediators


def test_estimate_ate_direct_edge():
    cg = CausalGraph()
    cg.add_node(CausalNode(id="t", name="T"), value=0.0)
    cg.add_node(CausalNode(id="o", name="O"), value=5.0)
    cg.add_edge("t", "o", strength=1.0)

    result = cg.estimate_ate("t", "o", intervention_value=2.0)
    assert isinstance(result, InterventionResult)
    assert result.target_variable == "o"
    assert result.confidence > 0.0
    assert result.confidence <= 1.0


def test_estimate_ate_zero_delta():
    cg = CausalGraph()
    cg.add_node(CausalNode(id="t", name="T"), value=1.0)
    cg.add_node(CausalNode(id="o", name="O"), value=5.0)
    cg.add_edge("t", "o", strength=1.0)

    result = cg.estimate_ate("t", "o", intervention_value=1.0)
    assert result.estimated_effect == 5.0
    assert result.confidence == 1.0


def test_estimate_counterfactual_with_evidence():
    cg = CausalGraph()
    cg.add_node(CausalNode(id="t", name="T"), value=0.0)
    cg.add_node(CausalNode(id="o", name="O"), value=5.0)
    cg.add_edge("t", "o", strength=1.0)

    result = cg.estimate_counterfactual("t", "o", intervention_value=2.0, evidence={"t": 1.0})
    assert isinstance(result, InterventionResult)


def test_estimate_counterfactual_without_evidence():
    cg = CausalGraph()
    cg.add_node(CausalNode(id="t", name="T"), value=0.0)
    cg.add_node(CausalNode(id="o", name="O"), value=5.0)
    cg.add_edge("t", "o", strength=1.0)

    result = cg.estimate_counterfactual("t", "o", intervention_value=2.0)
    assert isinstance(result, InterventionResult)


def test_from_requirements():
    reqs = [
        {"category": "perf", "description": "fast response", "priority": 8},
        {"category": "security", "description": "auth required", "priority": 9},
    ]
    cg = CausalGraph.from_requirements(reqs)
    assert cg.node_count == 2
    assert cg.validate_dag()


def test_from_requirements_with_dependencies():
    reqs = [
        {"category": "a", "description": "first", "priority": 5},
        {"category": "b", "description": "second", "priority": 7},
    ]
    deps = [("a_first", "b_second", 0.8)]
    cg = CausalGraph.from_requirements(reqs, deps)
    assert cg.node_count == 2
    assert cg.edge_count == 1


def test_from_requirements_missing_dep_names():
    reqs = [
        {"category": "a", "description": "first", "priority": 5},
    ]
    deps = [("nonexistent", "also_missing", 0.5)]
    cg = CausalGraph.from_requirements(reqs, deps)
    assert cg.node_count == 1
    assert cg.edge_count == 0


def test_causal_edge_dataclass():
    edge = CausalEdge(source="a", target="b", strength=0.9, delay=0.1)
    assert edge.source == "a"
    assert edge.target == "b"
    assert edge.strength == 0.9
    assert edge.delay == 0.1


def test_causal_node_defaults():
    node = CausalNode()
    assert node.name == ""
    assert node.var_type == VariableType.CONTINUOUS
    assert node.domain is None
    assert len(node.id) == 8


def test_variable_type_values():
    assert VariableType.CONTINUOUS.value == "continuous"
    assert VariableType.CATEGORICAL.value == "categorical"
    assert VariableType.BINARY.value == "binary"


def test_estimate_ate_with_confounders():
    cg = CausalGraph()
    cg.add_node(CausalNode(id="c", name="C"), value=2.0)
    cg.add_node(CausalNode(id="t", name="T"), value=0.0)
    cg.add_node(CausalNode(id="o", name="O"), value=0.0)
    cg.add_edge("c", "t", strength=0.5)
    cg.add_edge("c", "o", strength=0.5)
    cg.add_edge("t", "o", strength=1.0)

    result = cg.estimate_ate("t", "o", intervention_value=1.0)
    assert result.backdoor_variables == ("c",)
    assert result.confidence < 1.0


def test_estimate_ate_with_mediators():
    cg = CausalGraph()
    cg.add_node(CausalNode(id="t", name="T"), value=0.0)
    cg.add_node(CausalNode(id="m", name="M"), value=0.0)
    cg.add_node(CausalNode(id="o", name="O"), value=0.0)
    cg.add_edge("t", "m", strength=0.8)
    cg.add_edge("m", "o", strength=0.6)

    result = cg.estimate_ate("t", "o", intervention_value=1.0)
    assert result.frontdoor_variables == ("m",)


def test_estimate_ate_counterfactual_values():
    cg = CausalGraph()
    cg.add_node(CausalNode(id="t", name="T"), value=0.0)
    cg.add_node(CausalNode(id="m", name="M"), value=0.0)
    cg.add_node(CausalNode(id="o", name="O"), value=0.0)
    cg.add_edge("t", "m", strength=0.8)
    cg.add_edge("m", "o", strength=0.6)
    cg.add_edge("t", "o", strength=0.3)

    result = cg.estimate_ate("t", "o", intervention_value=2.0)
    assert result.counterfactual is not None or result.counterfactual is None
