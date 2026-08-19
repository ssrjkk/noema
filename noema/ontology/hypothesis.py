"""Ontological hypotheses: candidate axioms proposed by the ORL pipeline.

A hypothesis is an LLM-generated claim about the world ("Payment requires
IdempotencyKey") that must survive :class:`~noema.ontology.validator.EpistemicValidator`
before it may mutate the ontology. Every accepted hypothesis carries
provenance (source + trigger) so the graph stays an auditable journal of
survival, not an anonymous pile of facts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Predicate = Literal["requires", "forbids"]


class OntologicalHypothesis(BaseModel):
    """A candidate axiom: ``subject --predicate--> object``."""

    subject: str = Field(min_length=1)
    predicate: Predicate
    object: str = Field(min_length=1)
    rationale: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = "healer_orl"
    trigger: str = ""


class ValidationResult(BaseModel):
    """The verdict of one epistemic validation pass."""

    valid: bool
    reason: str = ""
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def accept(cls) -> ValidationResult:
        return cls(valid=True, reason="ok")

    @classmethod
    def reject(cls, reason: str) -> ValidationResult:
        return cls(valid=False, reason=reason)
