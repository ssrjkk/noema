"""Ontological Reinforcement Learning — crystallizing knowledge from chaos.

When code fails and a fix succeeds, the difference between the two is a
lesson. This module turns that lesson into an ontology mutation through a
strict pipeline:

1. **Hypothesis generation** — the LLM is asked to formulate the violated law
   as a structured triple (subject, predicate, object) with a rationale and a
   self-assessed confidence. The model proposes; it never decides.
2. **Epistemic validation** — :class:`~noema.ontology.validator.EpistemicValidator`
   applies deterministic, LLM-free checks (confidence gate, endpoint
   existence, duplicates, contradictions, type rules, cycles). A rejected
   hypothesis never touches the graph.
3. **Provenance & mutation** — accepted hypotheses are added with ``source``
   and ``trigger`` metadata and persisted atomically, so the ontology remains
   an auditable journal: every law remembers why it exists.

The pipeline is fail-open-safe in the opposite direction: it never raises and
never blocks the caller; unavailability (fallback LLM, unparsable output,
rejection) degrades to a logged skip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from noema.logging import get_logger
from noema.ontology.graph import OntologyGraph, Relation
from noema.ontology.hypothesis import OntologicalHypothesis
from noema.ontology.validator import EpistemicValidator
from noema.utils.json_utils import extract_fenced_json

if TYPE_CHECKING:
    from noema.llm.providers import BaseLLMProvider, LLMMessage

log = get_logger(__name__)

_MAX_CODE_CHARS = 1500


@dataclass
class CrystallizationResult:
    """Outcome of one crystallization attempt."""

    accepted: bool
    reason: str
    hypothesis: OntologicalHypothesis | None = None


def _llm_available(llm: BaseLLMProvider) -> bool:
    return getattr(llm, "name", "fallback") != "fallback"


async def crystallize_axiom(
    llm: BaseLLMProvider,
    ontology: OntologyGraph,
    failed_code: str,
    fixed_code: str,
    error_summary: str = "",
    task_id: str = "",
    validator: EpistemicValidator | None = None,
    persist_path: Path | str | None = None,
) -> CrystallizationResult:
    """Propose → validate → mutate → persist one ontological axiom.

    Returns :class:`CrystallizationResult`; never raises. The graph is only
    mutated when the hypothesis survives the validator.
    """
    if not _llm_available(llm):
        return CrystallizationResult(False, "llm_unavailable")
    if not ontology.stats()["entities"]:
        return CrystallizationResult(False, "ontology_empty")

    from noema.llm.providers import LLMMessage

    messages: list[LLMMessage] = [
        LLMMessage(
            role="system",
            content=(
                "You extract ontological laws from code failures. Given the failed code, "
                "the fixed code, and the error, formulate the violated law as a JSON object: "
                '{"subject": <entity name>, "predicate": "requires"|"forbids", '
                '"object": <entity name>, "rationale": <why>, "confidence": <0.0-1.0>}. '
                "Return ONLY valid JSON. Use entity names from the code or the domain. "
                "Never invent predicates other than requires/forbids."
            ),
        ),
        LLMMessage(
            role="user",
            content=(
                f"Failed code:\n{failed_code[:_MAX_CODE_CHARS]}\n\n"
                f"Fixed code:\n{fixed_code[:_MAX_CODE_CHARS]}\n\n"
                f"Error: {error_summary[:500]}\n"
                f"Task: {task_id}\n"
            ),
        ),
    ]
    try:
        response = await llm.complete(messages, temperature=0.2, max_tokens=512)
    except Exception as e:  # noqa: BLE001 - ORL must never crash the caller
        log.warning("orl_llm_failed", task=task_id, error=str(e)[:200])
        return CrystallizationResult(False, f"llm_error:{e}")

    parsed = extract_fenced_json(response.content, default=None)
    if not isinstance(parsed, dict):
        log.warning("orl_unparsable_response", task=task_id)
        return CrystallizationResult(False, "unparsable_llm_response")

    try:
        hypothesis = OntologicalHypothesis(
            **{k: v for k, v in parsed.items() if k in OntologicalHypothesis.model_fields},
            source="healer_orl",
            trigger=task_id,
        )
    except Exception as e:  # noqa: BLE001 - malformed hypotheses are rejected, not raised
        log.warning("orl_invalid_hypothesis", task=task_id, error=str(e)[:200])
        return CrystallizationResult(False, f"invalid_hypothesis:{e}")

    check = (validator or EpistemicValidator()).validate(hypothesis, ontology)
    if not check.valid:
        log.info("orl_rejected", task=task_id, reason=check.reason)
        return CrystallizationResult(False, check.reason, hypothesis)

    ontology.add_relation(
        Relation(
            subject=hypothesis.subject,
            predicate=hypothesis.predicate,
            object=hypothesis.object,
            weight=hypothesis.confidence,
            metadata={
                "source": hypothesis.source,
                "trigger": hypothesis.trigger,
                "rationale": hypothesis.rationale[:500],
                "confidence": f"{hypothesis.confidence:.2f}",
            },
        )
    )
    if persist_path is not None:
        ontology.save(Path(persist_path))

    log.info(
        "orl_axiom_crystallized",
        task=task_id,
        axiom=f"{hypothesis.subject} {hypothesis.predicate} {hypothesis.object}",
        confidence=hypothesis.confidence,
    )
    return CrystallizationResult(True, "accepted", hypothesis)
