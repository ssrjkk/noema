"""Ontology — типизированный граф знаний о мире (entities/relations).

Includes the ORL pipeline: hypotheses (:mod:`noema.ontology.hypothesis`),
the epistemic validator (:mod:`noema.ontology.validator`) and the
crystallization driver (:mod:`noema.ontology.orl`).
"""

from noema.ontology.graph import (
    DEFAULT_MAX_ENTITIES,
    DEFAULT_MAX_RELATIONS,
    Entity,
    OntologyError,
    OntologyGraph,
    Relation,
)
from noema.ontology.hypothesis import OntologicalHypothesis, ValidationResult
from noema.ontology.orl import CrystallizationResult, crystallize_axiom
from noema.ontology.validator import EpistemicValidator

__all__ = [
    "Entity",
    "Relation",
    "OntologyGraph",
    "OntologyError",
    "DEFAULT_MAX_ENTITIES",
    "DEFAULT_MAX_RELATIONS",
    "OntologicalHypothesis",
    "ValidationResult",
    "EpistemicValidator",
    "CrystallizationResult",
    "crystallize_axiom",
]
