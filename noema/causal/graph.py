from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import structlog

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = structlog.get_logger(__name__)


class VariableType(Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BINARY = "binary"


@dataclass(frozen=True)
class CausalNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    var_type: VariableType = VariableType.CONTINUOUS
    domain: tuple[float, float] | None = None
    structural_equation: str = ""


@dataclass(frozen=True)
class CausalEdge:
    source: str
    target: str
    strength: float = 1.0
    delay: float = 0.0


@dataclass(frozen=True)
class InterventionResult:
    target_variable: str
    intervention_value: float
    estimated_effect: float
    confidence: float
    backdoor_variables: tuple[str, ...]
    frontdoor_variables: tuple[str, ...]
    counterfactual: dict[str, float] | None = None


class CausalGraph:
    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: dict[str, CausalNode] = {}
        self._node_values: dict[str, float] = {}
        self._node_domains: dict[str, tuple[float, float]] = {}

    def add_node(self, node: CausalNode, value: float = 0.0) -> None:
        self._nodes[node.id] = node
        self._node_values[node.id] = value
        self._graph.add_node(node.id, name=node.name, var_type=node.var_type.value)
        if node.domain:
            self._node_domains[node.id] = node.domain

    def add_edge(
        self, source_id: str, target_id: str, strength: float = 1.0, delay: float = 0.0
    ) -> None:
        if source_id not in self._nodes or target_id not in self._nodes:
            raise ValueError(f"Cannot add edge {source_id}->{target_id}: node(s) not found")
        self._graph.add_edge(source_id, target_id, strength=strength, delay=delay)

    def remove_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        self._node_values.pop(node_id, None)
        self._node_domains.pop(node_id, None)
        self._graph.remove_node(node_id)

    def remove_edge(self, source_id: str, target_id: str) -> None:
        self._graph.remove_edge(source_id, target_id)

    def get_parents(self, node_id: str, *, direct: bool = True) -> list[str]:
        return list(self._graph.predecessors(node_id))

    def get_children(self, node_id: str, *, direct: bool = True) -> list[str]:
        return list(self._graph.successors(node_id))

    def get_ancestors(self, node_id: str) -> set[str]:
        return set(nx.ancestors(self._graph, node_id))

    def get_descendants(self, node_id: str) -> set[str]:
        return set(nx.descendants(self._graph, node_id))

    def get_all_nodes(self) -> list[CausalNode]:
        return list(self._nodes.values())

    def iter_nodes(self) -> Iterator[CausalNode]:
        return iter(self._nodes.values())

    @property
    def node_ids(self) -> list[str]:
        return list(self._nodes.keys())

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def has_edge(self, source_id: str, target_id: str) -> bool:
        return cast("bool", self._graph.has_edge(source_id, target_id))

    def validate_dag(self) -> bool:
        return cast("bool", nx.is_directed_acyclic_graph(self._graph))

    def find_confounders(
        self, treatment: str, outcome: str, *, exclude: set[str] | None = None
    ) -> list[str]:
        if not self.validate_dag():
            raise ValueError("Causal graph must be a DAG for do-calculus")
        if treatment not in self._nodes:
            raise ValueError(f"Treatment node {treatment} not found")
        if outcome not in self._nodes:
            raise ValueError(f"Outcome node {outcome} not found")
        confounders = []
        excluded = exclude or set()
        for node_id in self._nodes:
            if node_id in (treatment, outcome) or node_id in excluded:
                continue
            parents_outcome = set(self._graph.predecessors(outcome))
            if node_id in parents_outcome and self._graph.has_edge(node_id, treatment):
                confounders.append(node_id)
        return sorted(set(confounders))

    def find_instruments(self, treatment: str, outcome: str) -> list[str]:
        if not self.validate_dag():
            raise ValueError("Causal graph must be a DAG for do-calculus")
        instruments = []
        for node_id in self._nodes:
            if node_id in (treatment, outcome):
                continue
            if self._graph.has_edge(node_id, treatment) and not self._graph.has_edge(
                node_id, outcome
            ):
                ancestors_outcome = nx.ancestors(self._graph, outcome)
                if node_id not in ancestors_outcome:
                    desc_treatment = nx.descendants(self._graph, node_id)
                    if treatment in desc_treatment:
                        instruments.append(node_id)
        return sorted(set(instruments))

    def find_mediators(self, treatment: str, outcome: str) -> list[str]:
        if not self.validate_dag():
            raise ValueError("Causal graph must be a DAG for do-calculus")
        mediators = []
        for node_id in self._nodes:
            if node_id in (treatment, outcome):
                continue
            ancestors = nx.ancestors(self._graph, outcome)
            descendants = nx.descendants(self._graph, treatment)
            if node_id in ancestors and node_id in descendants:
                mediators.append(node_id)
        return sorted(set(mediators))

    def estimate_ate(
        self, treatment: str, outcome: str, intervention_value: float = 1.0
    ) -> InterventionResult:
        if not self.validate_dag():
            raise ValueError("Causal graph must be a DAG for do-calculus")
        backdoor_set = self.find_confounders(treatment, outcome)
        frontdoor_set = self.find_mediators(treatment, outcome)
        current_val = self._node_values.get(outcome, 0.0)
        treatment_current = self._node_values.get(treatment, 0.0)
        delta = intervention_value - treatment_current
        if delta == 0.0:
            return InterventionResult(
                target_variable=outcome,
                intervention_value=intervention_value,
                estimated_effect=current_val,
                confidence=1.0,
                backdoor_variables=tuple(backdoor_set),
                frontdoor_variables=tuple(frontdoor_set),
            )
        direct_paths = 0.0
        for succ in self._graph.successors(treatment):
            edge_data = self._graph.get_edge_data(treatment, succ)
            s = edge_data.get("strength", 1.0) if edge_data else 1.0
            path_effect = 0.0
            if succ == outcome:
                path_effect = s
            elif outcome in nx.descendants(self._graph, succ):
                path_effect = s * 0.5
            direct_paths += path_effect
        confounder_bias = 0.0
        for c in backdoor_set:
            c_val = self._node_values.get(c, 0.0)
            edge_to_t = self._graph.get_edge_data(c, treatment)
            edge_to_o = self._graph.get_edge_data(c, outcome)
            if edge_to_t and edge_to_o:
                confounder_bias += (
                    edge_to_t.get("strength", 0.5) * edge_to_o.get("strength", 0.5) * abs(c_val)
                )
        mediator_effect = 0.0
        for m in frontdoor_set:
            edge_t_m = self._graph.get_edge_data(treatment, m)
            edge_m_o = self._graph.get_edge_data(m, outcome)
            if edge_t_m and edge_m_o:
                mediator_effect += edge_t_m.get("strength", 0.5) * edge_m_o.get("strength", 0.5)
        total_effect = direct_paths * delta + mediator_effect * delta
        # Confounders observed along backdoor paths bias the naive estimate:
        # subtract their contribution (the previous ``max()`` made this a no-op).
        total_effect = max(0.0, total_effect - confounder_bias * 0.1)
        estimated = current_val + total_effect
        confidence = 1.0 / (1.0 + len(backdoor_set) * 0.1 + len(frontdoor_set) * 0.05)
        confidence = max(0.1, min(1.0, confidence))
        counterfactual: dict[str, float] = {}
        for node_id in self._nodes:
            if node_id != outcome and node_id != treatment:
                parents = list(self._graph.predecessors(node_id))
                if parents:
                    base = self._node_values.get(node_id, 0.0)
                    influenced = any(
                        p in (treatment, *backdoor_set, *frontdoor_set) for p in parents
                    )
                    if influenced:
                        parent_delta = sum(
                            self._graph.get_edge_data(p, node_id, {}).get("strength", 0.5) * 0.1
                            for p in parents
                        )
                        counterfactual[self._nodes[node_id].name or node_id] = base + parent_delta
        return InterventionResult(
            target_variable=outcome,
            intervention_value=intervention_value,
            estimated_effect=round(estimated, 4),
            confidence=round(confidence, 4),
            backdoor_variables=tuple(backdoor_set),
            frontdoor_variables=tuple(frontdoor_set),
            counterfactual=counterfactual or None,
        )

    def estimate_counterfactual(
        self,
        treatment: str,
        outcome: str,
        intervention_value: float = 1.0,
        evidence: dict[str, float] | None = None,
    ) -> InterventionResult:
        if evidence:
            for k, v in evidence.items():
                if k in self._node_values:
                    self._node_values[k] = v
        return self.estimate_ate(treatment, outcome, intervention_value)

    @property
    def node_count(self) -> int:
        return cast("int", self._graph.number_of_nodes())

    @property
    def edge_count(self) -> int:
        return cast("int", self._graph.number_of_edges())

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {
                nid: {
                    "name": node.name,
                    "type": node.var_type.value,
                    "domain": list(node.domain) if node.domain else None,
                    "value": self._node_values.get(nid, 0.0),
                }
                for nid, node in self._nodes.items()
            },
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "strength": self._graph.get_edge_data(u, v, {}).get("strength", 1.0),
                }
                for u, v in self._graph.edges()
            ],
            "dag": self.validate_dag(),
        }

    @classmethod
    def from_requirements(
        cls,
        requirements: list[dict[str, Any]],
        dependencies: list[tuple[str, str, float]] | None = None,
    ) -> CausalGraph:
        cg = cls()
        deps = dependencies or []
        req_map: dict[str, str] = {}
        for req in requirements:
            nid = str(uuid.uuid4())[:8]
            name = req.get("category", "unknown") + "_" + req.get("description", "")[:20]
            priority = req.get("priority", 5)
            cg.add_node(
                CausalNode(id=nid, name=name, var_type=VariableType.CONTINUOUS, domain=(0, 10)),
                value=float(priority),
            )
            req_map[name] = nid
        for src_name, tgt_name, strength in deps:
            src_id = req_map.get(src_name)
            tgt_id = req_map.get(tgt_name)
            if src_id and tgt_id:
                cg.add_edge(src_id, tgt_id, strength=strength)
        return cg
