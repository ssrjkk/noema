"""Ontology — типизированный граф знаний о мире (entities/relations)."""

from noema.ontology.graph import (
    DEFAULT_MAX_ENTITIES,
    DEFAULT_MAX_RELATIONS,
    Entity,
    OntologyError,
    OntologyGraph,
    Relation,
)

__all__ = [
    "Entity",
    "Relation",
    "OntologyGraph",
    "OntologyError",
    "DEFAULT_MAX_ENTITIES",
    "DEFAULT_MAX_RELATIONS",
]
