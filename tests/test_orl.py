"""ORL pipeline tests: epistemic validator, crystallization, healer integration."""

from __future__ import annotations

import json

import pytest

from noema.healer.engine import SelfHealer
from noema.llm.providers import BaseLLMProvider, FallbackProvider, LLMResponse
from noema.ontology import (
    Entity,
    OntologicalHypothesis,
    OntologyGraph,
    Relation,
    crystallize_axiom,
)
from noema.ontology.validator import EpistemicValidator

SERVICES = ("AuthService", "PaymentService", "UserDB", "IdempotencyKey")


def _graph() -> OntologyGraph:
    g = OntologyGraph()
    for name in SERVICES:
        g.add_entity(
            Entity(
                name=name,
                type="service"
                if name.endswith("Service")
                else "db"
                if name.endswith("DB")
                else "key",
            )
        )
    return g


def _h(
    subject: str = "PaymentService",
    predicate: str = "requires",
    object_: str = "IdempotencyKey",
    confidence: float = 0.9,
) -> OntologicalHypothesis:
    return OntologicalHypothesis(
        subject=subject,
        predicate=predicate,
        object=object_,
        rationale="observed in failed code",
        confidence=confidence,
    )


class _ScriptedLLM(BaseLLMProvider):
    """LLM whose next response is scripted; can also blow up."""

    def __init__(self, responses: list[str] | None = None, raise_error: bool = False) -> None:
        super().__init__()
        self._responses = list(responses or [])
        self._raise_error = raise_error
        self.calls = 0

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def model_name(self) -> str:
        return "scripted"

    async def _complete(self, messages, temperature=0.7, max_tokens=4096):
        self.calls += 1
        if self._raise_error:
            raise RuntimeError("scripted boom")
        content = self._responses.pop(0) if self._responses else "{}"
        return LLMResponse(content=content, model="scripted", tokens_used=10)


VALID_LLM_JSON = json.dumps(
    {
        "subject": "PaymentService",
        "predicate": "requires",
        "object": "IdempotencyKey",
        "rationale": "payment must be idempotent",
        "confidence": 0.85,
    }
)


# ── EpistemicValidator ────────────────────────────────────────────────


def test_validator_accepts_valid_hypothesis():
    graph = _graph()
    result = EpistemicValidator().validate(_h(), graph)
    assert result.valid


def test_validator_rejects_contradiction():
    graph = _graph()
    graph.add_relation(
        Relation(subject="PaymentService", predicate="forbids", object="IdempotencyKey")
    )
    result = EpistemicValidator().validate(_h(predicate="requires"), graph)
    assert not result.valid
    assert "contradicts_existing_axiom" in result.reason


def test_validator_rejects_duplicate():
    graph = _graph()
    graph.add_relation(
        Relation(subject="PaymentService", predicate="requires", object="IdempotencyKey")
    )
    result = EpistemicValidator().validate(_h(), graph)
    assert not result.valid
    assert "already_present" in result.reason


def test_validator_rejects_unknown_endpoints():
    graph = _graph()
    result = EpistemicValidator().validate(_h(object_="GhostDB"), graph)
    assert not result.valid
    assert "unknown_object" in result.reason
    result = EpistemicValidator().validate(_h(subject="GhostService"), graph)
    assert not result.valid
    assert "unknown_subject" in result.reason


def test_validator_rejects_low_confidence():
    graph = _graph()
    result = EpistemicValidator().validate(_h(confidence=0.1), graph)
    assert not result.valid
    assert "below threshold" in result.reason


def test_validator_rejects_cycle():
    graph = _graph()
    graph.add_relation(
        Relation(subject="PaymentService", predicate="requires", object="AuthService")
    )
    graph.add_relation(Relation(subject="AuthService", predicate="requires", object="UserDB"))
    result = EpistemicValidator().validate(_h(subject="UserDB", object_="PaymentService"), graph)
    assert not result.valid
    assert "creates_cycle" in result.reason


def test_validator_type_rules():
    graph = _graph()
    validator = EpistemicValidator(type_rules={"requires": ({"service"}, {"db"})})
    result = validator.validate(
        _h(subject="AuthService", object_="UserDB", predicate="requires"), graph
    )
    assert result.valid
    result = validator.validate(
        _h(subject="AuthService", object_="PaymentService", predicate="requires"), graph
    )
    assert not result.valid
    assert "invalid_entity_types" in result.reason


# ── crystallize_axiom ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_crystallize_accepts_and_persists(tmp_path):
    graph = _graph()
    persist = tmp_path / "ontology.json"
    llm = _ScriptedLLM(responses=[VALID_LLM_JSON])

    result = await crystallize_axiom(
        llm,
        graph,
        failed_code="print(payment.id)",
        fixed_code="print(payment.idempotency_key)",
        error_summary="KeyError: idempotency",
        task_id="task-42",
        persist_path=persist,
    )

    assert result.accepted, result.reason
    assert result.hypothesis is not None
    assert result.hypothesis.source == "healer_orl"
    assert result.hypothesis.trigger == "task-42"

    relations = list(graph.relations())
    assert len(relations) == 1
    r = relations[0]
    assert r.subject == "PaymentService"
    assert r.predicate == "requires"
    assert r.object == "IdempotencyKey"
    assert r.metadata.get("source") == "healer_orl"
    assert r.metadata.get("trigger") == "task-42"
    assert r.metadata.get("rationale") == "payment must be idempotent"

    reloaded = OntologyGraph.load(persist)
    assert len(list(reloaded.relations())) == 1
    assert list(reloaded.relations())[0].metadata["source"] == "healer_orl"


@pytest.mark.asyncio
async def test_crystallize_rejects_unparsable_llm_response():
    graph = _graph()
    llm = _ScriptedLLM(responses=["certainly! here is some prose, no json"])

    result = await crystallize_axiom(
        llm, graph, "def broken(): pass", "def fixed(): return 1", task_id="t-unparsable"
    )

    assert not result.accepted
    assert "unparsable" in result.reason
    assert len(list(graph.relations())) == 0


@pytest.mark.asyncio
async def test_crystallize_rejects_violating_hypothesis(tmp_path):
    graph = _graph()
    graph.add_relation(
        Relation(subject="PaymentService", predicate="forbids", object="IdempotencyKey")
    )
    persist = tmp_path / "ontology.json"
    llm = _ScriptedLLM(responses=[VALID_LLM_JSON])

    result = await crystallize_axiom(
        llm, graph, "def a(): pass", "def b(): pass", task_id="t-violating", persist_path=persist
    )

    assert not result.accepted
    assert "contradicts_existing_axiom" in result.reason
    assert len(list(graph.relations())) == 1
    assert not persist.exists()


@pytest.mark.asyncio
async def test_crystallize_skips_fallback_llm():
    graph = _graph()
    result = await crystallize_axiom(FallbackProvider(), graph, "a", "b")
    assert not result.accepted
    assert result.reason == "llm_unavailable"


@pytest.mark.asyncio
async def test_crystallize_skips_empty_ontology():
    result = await crystallize_axiom(_ScriptedLLM(), OntologyGraph(), "a", "b")
    assert not result.accepted
    assert result.reason == "ontology_empty"


@pytest.mark.asyncio
async def test_crystallize_never_raises_on_llm_failure():
    graph = _graph()
    llm = _ScriptedLLM(raise_error=True)
    result = await crystallize_axiom(
        llm, graph, "def crash(): raise", "def ok(): pass", task_id="t-crash"
    )
    assert not result.accepted
    assert "llm_error" in result.reason
    assert len(list(graph.relations())) == 0


@pytest.mark.asyncio
async def test_crystallize_rejects_when_graph_is_full():
    graph = OntologyGraph(max_entities=10, max_relations=1)
    for name in SERVICES:
        graph.add_entity(Entity(name=name, type="x"))
    graph.add_relation(Relation(subject="AuthService", predicate="requires", object="UserDB"))
    llm = _ScriptedLLM(responses=[VALID_LLM_JSON])

    result = await crystallize_axiom(
        llm, graph, "def full(): pass", "def fixed(): pass", task_id="t-full"
    )

    assert not result.accepted
    assert "validator_error" in result.reason
    assert len(list(graph.relations())) == 1


# ── SelfHealer integration ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_healer_proposes_axiom(tmp_path):
    graph = _graph()
    llm = _ScriptedLLM(responses=[VALID_LLM_JSON])
    healer = SelfHealer(llm=llm, ontology=graph, ontology_persist_path=tmp_path / "ontology.json")

    accepted = await healer._propose_ontological_axiom(
        failed_code="payment without idempotency",
        fixed_code="payment with idempotency key",
        error_summary="duplicate submission",
        task_id="task-7",
    )

    assert accepted
    relations = list(graph.relations())
    assert len(relations) == 1
    assert relations[0].predicate == "requires"
    assert relations[0].metadata["source"] == "healer_orl"
    assert (tmp_path / "ontology.json").exists()


@pytest.mark.asyncio
async def test_healer_without_llm_is_noop():
    graph = _graph()
    healer = SelfHealer(ontology=graph)
    assert await healer._propose_ontological_axiom("a", "b") is False
    assert len(list(graph.relations())) == 0


@pytest.mark.asyncio
async def test_healer_never_raises():
    graph = _graph()
    llm = _ScriptedLLM(raise_error=True)
    healer = SelfHealer(llm=llm, ontology=graph)
    assert (
        await healer._propose_ontological_axiom(
            "def crash(): raise", "def ok(): pass", task_id="t-healer-crash"
        )
        is False
    )
    assert len(list(graph.relations())) == 0
