from __future__ import annotations

import uuid
from typing import Any

import structlog

from noema.causal.graph import CausalGraph, CausalNode, InterventionResult, VariableType

logger = structlog.get_logger(__name__)


class CausalEngine:
    def __init__(self, enabled: bool = True, max_counterfactuals: int = 5) -> None:
        self._enabled = enabled
        self._max_counterfactuals = max_counterfactuals
        self._metrics: dict[str, int | float] = {
            "graphs_built": 0,
            "interventions_run": 0,
            "counterfactuals_estimated": 0,
            "mean_confidence": 0.0,
            "total_confidence": 0.0,
        }

    def build_graph(
        self,
        requirements: list[dict[str, Any]],
        constraints: list[dict[str, Any]] | None = None,
        dependencies: list[tuple[str, str, float]] | None = None,
    ) -> CausalGraph:
        graph = CausalGraph.from_requirements(requirements, dependencies)
        if constraints:
            for c in constraints:
                nid = str(uuid.uuid4())[:8]
                c_name = c.get("name", "constraint")
                graph.add_node(
                    CausalNode(
                        id=nid, name=f"constraint_{c_name}", var_type=VariableType.CATEGORICAL
                    ),
                    value=1.0,
                )
        self._metrics["graphs_built"] += 1
        logger.info("causal_graph_built", nodes=graph.node_count, edges=graph.edge_count)
        return graph

    def analyze_counterfactual(
        self,
        graph: CausalGraph,
        treatment: str,
        outcome: str,
        intervention_value: float = 1.0,
    ) -> InterventionResult | None:
        if not self._enabled:
            return None
        if not graph.has_node(treatment):
            logger.warning("treatment_node_not_found", node=treatment)
            return None
        if not graph.has_node(outcome):
            logger.warning("outcome_node_not_found", node=outcome)
            return None
        result = graph.estimate_ate(treatment, outcome, intervention_value)
        self._metrics["interventions_run"] += 1
        self._metrics["counterfactuals_estimated"] += 1
        self._metrics["total_confidence"] += result.confidence
        self._metrics["mean_confidence"] = self._metrics["total_confidence"] / max(
            1, self._metrics["counterfactuals_estimated"]
        )
        logger.info(
            "counterfactual_estimated",
            treatment=treatment,
            outcome=outcome,
            effect=result.estimated_effect,
            confidence=result.confidence,
        )
        return result

    def analyze_all_counterfactuals(
        self,
        graph: CausalGraph,
        task_description: str,
    ) -> list[dict[str, Any]]:
        if not self._enabled:
            return []
        results: list[dict[str, Any]] = []
        nodes = graph.get_all_nodes()
        main_outcome = nodes[-1].id if nodes else ""
        if not main_outcome:
            return results
        count = 0
        for node in nodes:
            if count >= self._max_counterfactuals:
                break
            if node.id == main_outcome:
                continue
            result = self.analyze_counterfactual(graph, node.id, main_outcome, 1.0)
            if result:
                alt = self.analyze_counterfactual(graph, node.id, main_outcome, 2.0)
                results.append(
                    {
                        "variable": node.name or node.id,
                        "intervention_1x": {
                            "effect": result.estimated_effect,
                            "confidence": result.confidence,
                        },
                        "intervention_2x": {
                            "effect": alt.estimated_effect if alt else None,
                            "confidence": alt.confidence if alt else None,
                        },
                        "backdoor_variables": list(result.backdoor_variables),
                        "mediator_variables": list(result.frontdoor_variables),
                    }
                )
                count += 1
        logger.info(
            "all_counterfactuals_computed",
            count=count,
            graph_nodes=graph.node_count,
        )
        return results

    def get_metrics(self) -> dict[str, int | float]:
        return dict(self._metrics)

    def reset_metrics(self) -> None:
        self._metrics = {
            "graphs_built": 0,
            "interventions_run": 0,
            "counterfactuals_estimated": 0,
            "mean_confidence": 0.0,
            "total_confidence": 0.0,
        }
