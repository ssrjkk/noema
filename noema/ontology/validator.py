"""Epistemic validator — the "sandbox for ideas".

An ontological hypothesis is a claim about the world. Before it may mutate
the graph it must survive deterministic, LLM-free checks:

1. **Confidence gate** — below the threshold the claim is noise, not knowledge.
2. **Endpoint existence** — axioms about unknown entities are hallucination.
3. **Duplicate check** — the exact triple already exists → nothing to add.
4. **Contradiction check** — ``requires`` vs an existing ``forbids`` (and vice
   versa) over the same pair → the graph would assert two incompatible laws.
5. **Type rules** — optional per-predicate allowed entity-type pairs
   (``requires`` may only link Service → Database in some domains).
6. **Cycle check** — a candidate that closes a directed cycle would make the
   ontology assert circular causality.

The validator never raises; every violation is a structured rejection reason
the caller can log or surface. It is the epistemic counterweight to the LLM:
the model proposes, the compiler disposes.
"""

from __future__ import annotations

from noema.ontology.graph import OntologyGraph, Relation
from noema.ontology.hypothesis import OntologicalHypothesis, ValidationResult

_OPPOSITES = {"requires": "forbids", "forbids": "requires"}

TypeRules = dict[str, tuple[set[str], set[str]]]


class EpistemicValidator:
    """Deterministic acceptance gate for candidate axioms."""

    def __init__(
        self,
        min_confidence: float = 0.5,
        type_rules: TypeRules | None = None,
    ) -> None:
        self.min_confidence = min_confidence
        self.type_rules = type_rules or {}

    def validate(
        self, hypothesis: OntologicalHypothesis, ontology: OntologyGraph
    ) -> ValidationResult:
        """Return accept/reject; never raises."""
        if hypothesis.confidence < self.min_confidence:
            return ValidationResult.reject(
                f"confidence {hypothesis.confidence:.2f} below threshold {self.min_confidence}"
            )
        if ontology.get_entity(hypothesis.subject) is None:
            return ValidationResult.reject(f"unknown_subject:{hypothesis.subject}")
        if ontology.get_entity(hypothesis.object) is None:
            return ValidationResult.reject(f"unknown_object:{hypothesis.object}")

        for r in ontology.relations():
            if (
                r.subject == hypothesis.subject
                and r.object == hypothesis.object
                and r.predicate == hypothesis.predicate
            ):
                return ValidationResult.reject(
                    f"already_present:{hypothesis.subject} {hypothesis.predicate} {hypothesis.object}"
                )
            if (
                r.subject == hypothesis.subject
                and r.object == hypothesis.object
                and r.predicate == _OPPOSITES[hypothesis.predicate]
            ):
                return ValidationResult.reject(
                    f"contradicts_existing_axiom:{hypothesis.subject} "
                    f"{_OPPOSITES[hypothesis.predicate]} {hypothesis.object}"
                )

        type_rule = self.type_rules.get(hypothesis.predicate)
        if type_rule is not None:
            allowed_subjects, allowed_objects = type_rule
            subject_entity = ontology.get_entity(hypothesis.subject)
            object_entity = ontology.get_entity(hypothesis.object)
            assert subject_entity is not None and object_entity is not None
            if subject_entity.type not in allowed_subjects:
                return ValidationResult.reject(
                    f"invalid_entity_types:subject {hypothesis.subject!r} "
                    f"(type {subject_entity.type!r}) not allowed for {hypothesis.predicate}"
                )
            if object_entity.type not in allowed_objects:
                return ValidationResult.reject(
                    f"invalid_entity_types:object {hypothesis.object!r} "
                    f"(type {object_entity.type!r}) not allowed for {hypothesis.predicate}"
                )

        try:
            creates_cycle = self._would_create_cycle(hypothesis, ontology)
        except Exception:  # noqa: BLE001 - never raise; a full/corrupt graph rejects
            return ValidationResult.reject("validator_error:cycle_check_unavailable")
        if creates_cycle:
            return ValidationResult.reject(
                f"creates_cycle:{hypothesis.subject} {hypothesis.predicate} {hypothesis.object}"
            )

        return ValidationResult.accept()

    def _would_create_cycle(
        self, hypothesis: OntologicalHypothesis, ontology: OntologyGraph
    ) -> bool:
        candidate = OntologyGraph(
            max_entities=ontology.max_entities, max_relations=ontology.max_relations
        )
        for entity in ontology.entities():
            candidate.add_entity(entity)
        for relation in ontology.relations():
            candidate.add_relation(relation)
        candidate.add_relation(
            Relation(
                subject=hypothesis.subject,
                predicate=hypothesis.predicate,
                object=hypothesis.object,
                weight=hypothesis.confidence,
            )
        )
        return candidate.has_cycle()
