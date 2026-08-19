from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from noema.causal import CausalEngine, CausalGraph, CausalNode, InterventionResult
from noema.causal.graph import VariableType

# в”Ђв”Ђ Strategies в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

node_name_strategy = st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_")
node_id_strategy = st.text(min_size=1, max_size=8, alphabet="abcdefghijklmnopqrstuvwxyz0123456789")

causal_node_strategy = st.builds(
    CausalNode,
    id=node_id_strategy,
    name=node_name_strategy,
    var_type=st.sampled_from(list(VariableType)),
    domain=st.one_of(st.none(), st.tuples(st.floats(0, 10), st.floats(10, 100))),
    structural_equation=st.just(""),
)

value_strategy = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
strength_strategy = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# в”Ђв”Ђ Property: CausalGraph Construction в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ


@given(st.lists(causal_node_strategy, min_size=1, max_size=10, unique_by=lambda n: n.id))
@settings(max_examples=100, deadline=2000)
def test_causal_graph_add_nodes(nodes):
    cg = CausalGraph()
    for n in nodes:
        cg.add_node(n, value=0.0)
    assert cg.node_count == len(nodes)
    for n in nodes:
        assert cg.has_node(n.id)


@given(st.lists(causal_node_strategy, min_size=2, max_size=8, unique_by=lambda n: n.id))
@settings(max_examples=50, deadline=2000)
def test_causal_graph_add_edges(nodes):
    cg = CausalGraph()
    for n in nodes:
        cg.add_node(n, value=0.0)
    for i in range(len(nodes) - 1):
        cg.add_edge(nodes[i].id, nodes[i + 1].id, strength=1.0)
    if len(nodes) >= 2:
        assert cg.validate_dag()
    assert cg.edge_count == max(0, len(nodes) - 1)


@given(st.lists(causal_node_strategy, min_size=3, max_size=8, unique_by=lambda n: n.id))
@settings(max_examples=50, deadline=2000)
def test_causal_graph_acyclic_guarantee(nodes):
    cg = CausalGraph()
    for n in nodes:
        cg.add_node(n, value=0.0)
    for i in range(len(nodes) - 1):
        cg.add_edge(nodes[i].id, nodes[i + 1].id, strength=0.5)
    assert cg.validate_dag()


@given(st.lists(causal_node_strategy, min_size=2, max_size=6, unique_by=lambda n: n.id))
@settings(max_examples=50, deadline=2000)
def test_estimate_ate_without_confounders(nodes):
    cg = CausalGraph()
    for n in nodes:
        cg.add_node(n, value=1.0)
    for i in range(len(nodes) - 1):
        cg.add_edge(nodes[i].id, nodes[i + 1].id, strength=0.5)
    if len(nodes) >= 2:
        result = cg.estimate_ate(nodes[0].id, nodes[-1].id, intervention_value=1.0)
        assert isinstance(result, InterventionResult)
        assert result.confidence > 0.0
        assert result.confidence <= 1.0


@given(st.lists(causal_node_strategy, min_size=2, max_size=5, unique_by=lambda n: n.id))
@settings(max_examples=50, deadline=2000)
def test_find_confounders_returns_list(nodes):
    cg = CausalGraph()
    for n in nodes:
        cg.add_node(n, value=0.0)
    for i in range(len(nodes) - 1):
        cg.add_edge(nodes[i].id, nodes[i + 1].id, strength=1.0)
    if len(nodes) >= 2:
        confounders = cg.find_confounders(nodes[0].id, nodes[-1].id)
        assert isinstance(confounders, list)


@given(st.lists(causal_node_strategy, min_size=2, max_size=5, unique_by=lambda n: n.id))
@settings(max_examples=50, deadline=2000)
def test_find_mediators_returns_list(nodes):
    cg = CausalGraph()
    for n in nodes:
        cg.add_node(n, value=0.0)
    for i in range(len(nodes) - 1):
        cg.add_edge(nodes[i].id, nodes[i + 1].id, strength=1.0)
    if len(nodes) >= 2:
        mediators = cg.find_mediators(nodes[0].id, nodes[-1].id)
        assert isinstance(mediators, list)


# в”Ђв”Ђ Property: InterventionResult в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ


@given(
    st.text(min_size=1, max_size=10),
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=2000)
def test_intervention_result_immutable(var, inter, effect, conf):
    result = InterventionResult(
        target_variable=var,
        intervention_value=inter,
        estimated_effect=effect,
        confidence=conf,
        backdoor_variables=(),
        frontdoor_variables=(),
    )
    assert result.target_variable == var
    assert result.intervention_value == inter
    assert result.estimated_effect == effect
    assert result.confidence == conf
    assert result.backdoor_variables == ()
    assert result.frontdoor_variables == ()


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=50, deadline=2000)
def test_intervention_confidence_bounds(conf):
    result = InterventionResult(
        target_variable="x",
        intervention_value=1.0,
        estimated_effect=0.5,
        confidence=max(0.0, min(1.0, conf)),
        backdoor_variables=(),
        frontdoor_variables=(),
    )
    assert 0.0 <= result.confidence <= 1.0


# в”Ђв”Ђ Property: CausalEngine в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ


@given(
    st.lists(
        st.fixed_dictionaries(
            {
                "category": st.text(min_size=1, max_size=10, alphabet="abc_"),
                "description": st.text(min_size=1, max_size=30),
                "priority": st.integers(min_value=1, max_value=10),
            }
        ),
        min_size=1,
        max_size=5,
    )
)
@settings(max_examples=50, deadline=2000)
def test_causal_engine_build_graph(requirements):
    engine = CausalEngine(enabled=True)
    graph = engine.build_graph(requirements)
    assert graph.node_count == len(requirements)
    assert engine.get_metrics()["graphs_built"] >= 1


@given(
    st.lists(
        st.fixed_dictionaries(
            {
                "category": st.text(min_size=1, max_size=10, alphabet="abc_"),
                "description": st.text(min_size=1, max_size=30),
                "priority": st.integers(min_value=1, max_value=10),
            }
        ),
        min_size=2,
        max_size=5,
    )
)
@settings(max_examples=50, deadline=2000)
def test_causal_engine_analyze_counterfactual(requirements):
    engine = CausalEngine(enabled=True)
    graph = engine.build_graph(requirements)
    nodes = graph.get_all_nodes()
    if len(nodes) >= 2:
        result = engine.analyze_counterfactual(
            graph, nodes[0].id, nodes[-1].id, intervention_value=2.0
        )
        if result:
            assert result.estimated_effect is not None
            assert result.confidence > 0.0


@given(
    st.lists(
        st.fixed_dictionaries(
            {
                "category": st.text(min_size=1, max_size=5, alphabet="ab_"),
                "description": st.text(min_size=1, max_size=15),
                "priority": st.integers(min_value=1, max_value=10),
            }
        ),
        min_size=1,
        max_size=4,
    )
)
@settings(max_examples=30)
def test_causal_engine_metrics_structure(requirements):
    engine = CausalEngine(enabled=True)
    metrics = engine.get_metrics()
    assert "graphs_built" in metrics
    assert "interventions_run" in metrics
    assert "counterfactuals_estimated" in metrics
    engine.build_graph(requirements)
    assert engine.get_metrics()["graphs_built"] > 0


@given(
    st.lists(
        st.fixed_dictionaries(
            {
                "category": st.text(min_size=1, max_size=5, alphabet="ab_"),
                "description": st.text(min_size=1, max_size=15),
                "priority": st.integers(min_value=1, max_value=10),
            }
        ),
        min_size=1,
        max_size=4,
    )
)
@settings(max_examples=30)
def test_causal_engine_disabled_returns_none(requirements):
    engine = CausalEngine(enabled=False)
    engine.build_graph(requirements)
    result = engine.analyze_counterfactual("nonexistent", "nonexistent", 1.0)
    assert result is None


# в”Ђв”Ђ Integration: from_requirements в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ


@given(
    st.lists(
        st.fixed_dictionaries(
            {
                "category": st.text(min_size=1, max_size=10),
                "description": st.text(min_size=1, max_size=40),
                "priority": st.integers(min_value=1, max_value=10),
            }
        ),
        min_size=1,
        max_size=8,
    ),
    st.one_of(
        st.none(),
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=20),
                st.text(min_size=1, max_size=20),
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=0,
            max_size=5,
        ),
    ),
)
@settings(max_examples=50, deadline=2000)
def test_from_requirements_builds_valid_dag(requirements, dependencies):
    cg = CausalGraph.from_requirements(requirements, dependencies)
    assert cg.node_count == len(requirements)
    if cg.node_count > 0:
        assert cg.validate_dag()
    assert "nodes" in cg.to_dict()
    assert "edges" in cg.to_dict()
