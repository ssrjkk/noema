"""Tests for the ontology knowledge graph (noema/ontology)."""

import json
from pathlib import Path

import pytest

from noema.ontology import Entity, OntologyError, OntologyGraph, Relation


def _sample() -> OntologyGraph:
    g = OntologyGraph()
    for name, type_ in [
        ("alice", "user"),
        ("bob", "user"),
        ("account-1", "account"),
        ("server-eu", "server"),
        ("money-usd", "currency"),
    ]:
        g.add_entity(Entity(name=name, type=type_))
    g.add_relation(Relation("alice", "owns", "account-1", weight=1.0))
    g.add_relation(Relation("account-1", "runs_on", "server-eu", weight=0.8))
    g.add_relation(Relation("money-usd", "backs", "account-1", weight=0.5))
    return g


def test_add_and_get_entity() -> None:
    g = OntologyGraph()
    g.add_entity(Entity(name="server-eu", type="server", attributes={"region": "eu"}))
    entity = g.get_entity("server-eu")
    assert entity is not None
    assert entity.type == "server"
    assert entity.attributes == {"region": "eu"}


def test_entity_types_query() -> None:
    g = _sample()
    assert [e.name for e in g.entities(entity_type="user")] == ["alice", "bob"]
    assert len(g.entities()) == 5


def test_add_relation_unknown_endpoint_raises() -> None:
    g = _sample()
    with pytest.raises(OntologyError, match="Unknown subject"):
        g.add_relation(Relation("ghost", "owns", "account-1"))
    with pytest.raises(OntologyError, match="Unknown object"):
        g.add_relation(Relation("alice", "owns", "ghost"))


def test_empty_and_control_char_names_rejected() -> None:
    g = OntologyGraph()
    with pytest.raises(OntologyError):
        g.add_entity(Entity(name="  ", type="user"))
    with pytest.raises(OntologyError):
        g.add_entity(Entity(name="bad\x00name", type="user"))


def test_duplicate_relation_is_idempotent() -> None:
    g = _sample()
    g.add_relation(Relation("alice", "owns", "account-1", weight=9.0))
    assert len(g.relations()) == 3


def test_entity_cap_enforced() -> None:
    g = OntologyGraph(max_entities=2)
    g.add_entity(Entity("a", "user"))
    g.add_entity(Entity("b", "user"))
    with pytest.raises(OntologyError, match="Entity cap"):
        g.add_entity(Entity("c", "user"))


def test_relation_cap_enforced() -> None:
    g = OntologyGraph(max_relations=1)
    g.add_entity(Entity("a", "user"))
    g.add_entity(Entity("b", "server"))
    g.add_relation(Relation("a", "uses", "b"))
    with pytest.raises(OntologyError, match="Relation cap"):
        g.add_relation(Relation("b", "hosts", "a"))


def test_neighbors_bfs_depth_and_order() -> None:
    g = _sample()
    one_hop = g.neighbors("account-1", depth=1)
    assert {t[0] for t in one_hop} == {"alice", "server-eu", "money-usd"}
    assert one_hop[0][0] == "server-eu"  # heaviest outgoing first
    two_hop = g.neighbors("alice", depth=2)
    assert {t[0] for t in two_hop} == {"account-1", "server-eu", "money-usd"}


def test_neighbors_unknown_entity_empty() -> None:
    assert _sample().neighbors("ghost") == []


def test_find_paths_bounded() -> None:
    g = _sample()
    paths = g.find_paths("alice", "server-eu", max_depth=3)
    assert paths == [["alice", "account-1", "server-eu"]]
    assert g.find_paths("alice", "ghost") == []
    assert g.find_paths("money-usd", "server-eu", max_depth=3) == [
        ["money-usd", "account-1", "server-eu"]
    ]


def test_cycle_detection() -> None:
    g = OntologyGraph()
    g.add_entity(Entity("a", "user"))
    g.add_entity(Entity("b", "server"))
    g.add_entity(Entity("c", "server"))
    g.add_relation(Relation("a", "runs", "b"))
    g.add_relation(Relation("b", "peers", "c"))
    g.add_relation(Relation("c", "peers", "a"))
    assert g.has_cycle() is True
    g2 = _sample()
    assert g2.has_cycle() is False


def test_save_load_roundtrip(tmp_path: Path) -> None:
    g = _sample()
    path = tmp_path / "ontology.json"
    g.save(path)
    loaded = OntologyGraph.load(path)
    assert loaded.stats()["entities"] == 5
    assert loaded.stats()["relations"] == 3
    assert loaded.get_entity("alice") == g.get_entity("alice")
    assert loaded.find_paths("alice", "server-eu") == g.find_paths("alice", "server-eu")


def test_load_corrupt_file_degrades_to_empty(tmp_path: Path) -> None:
    path = tmp_path / "ontology.json"
    path.write_text("{not valid json", encoding="utf-8")
    graph = OntologyGraph.load(path)
    assert graph.stats()["entities"] == 0


def test_load_missing_file_degrades_to_empty(tmp_path: Path) -> None:
    graph = OntologyGraph.load(tmp_path / "missing.json")
    assert graph.stats()["entities"] == 0


def test_load_ignores_malformed_records(tmp_path: Path) -> None:
    path = tmp_path / "ontology.json"
    path.write_text(
        json.dumps(
            {
                "entities": [
                    {"name": "ok", "type": "user"},
                    {"name": "broken"},  # missing type -> skipped
                ],
                "relations": [
                    {"subject": "nope", "predicate": "x", "object": "ok"},
                ],
            }
        ),
        encoding="utf-8",
    )
    graph = OntologyGraph.load(path)
    assert graph.stats()["entities"] == 1
    assert graph.stats()["relations"] == 0


def test_to_context_orders_by_weight_and_limits() -> None:
    g = OntologyGraph()
    for name in ("a", "b", "c", "d"):
        g.add_entity(Entity(name, "user"))
    for i, (s, o) in enumerate((("a", "b"), ("b", "c"), ("c", "d"))):
        g.add_relation(Relation(s, "uses", o, weight=float(10 - i)))
    ctx = g.to_context(limit=2)
    lines = ctx.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("a --uses--> b")
    assert lines[1].startswith("b --uses--> c")
    assert "c --uses--> d" not in ctx


def test_stats() -> None:
    assert _sample().stats() == {
        "entities": 5,
        "relations": 3,
        "has_cycle": False,
    }


# ── Subgraph extraction & axiom rendering ────────────────────────────────────


def test_get_subgraph_bounded_depth() -> None:
    g = _sample()
    one_hop = g.get_subgraph(["alice"], depth=1)
    assert one_hop.stats()["entities"] == 2  # alice + account-1
    assert one_hop.stats()["relations"] == 1
    two_hop = g.get_subgraph(["alice"], depth=2)
    assert two_hop.stats()["entities"] == 4  # alice, account-1, server-eu, money-usd
    assert two_hop.stats()["relations"] == 3


def test_get_subgraph_from_multiple_roots() -> None:
    g = _sample()
    sub = g.get_subgraph(["alice", "server-eu"], depth=1)
    assert sub.stats()["entities"] == 3  # alice, server-eu, account-1
    assert sub.stats()["relations"] == 2


def test_get_subgraph_unknown_root_is_empty() -> None:
    sub = _sample().get_subgraph(["ghost"])
    assert sub.stats()["entities"] == 0
    assert sub.stats()["relations"] == 0


def test_get_subgraph_respects_inherited_caps() -> None:
    g = OntologyGraph(max_entities=3)
    for name in ("a", "b", "c"):
        g.add_entity(Entity(name, "node"))
    g.add_relation(Relation("a", "links", "b"))
    g.add_relation(Relation("b", "links", "c"))
    sub = g.get_subgraph(["a"], depth=2)
    assert sub.max_entities == 3
    assert sub.stats()["entities"] == 3
    assert sub.stats()["relations"] == 2


def test_to_rules_renders_must_and_forbids() -> None:
    g = OntologyGraph()
    for name in ("User", "AuthMethod", "Plaintext"):
        g.add_entity(Entity(name, "domain"))
    g.add_relation(Relation("User", "requires", "AuthMethod", weight=1.0))
    g.add_relation(Relation("User", "forbids", "Plaintext", weight=0.9))
    rules = g.to_rules()
    assert "User MUST be linked to AuthMethod via requires" in rules
    assert "User MUST NOT Plaintext (forbids)" in rules


def test_to_rules_orders_by_weight_and_limits() -> None:
    g = OntologyGraph()
    for name in ("a", "b", "c", "d"):
        g.add_entity(Entity(name, "node"))
    for i, (s, o) in enumerate((("a", "b"), ("b", "c"), ("c", "d"))):
        g.add_relation(Relation(s, "requires", o, weight=float(10 - i)))
    rules = g.to_rules(limit=2)
    assert len(rules.splitlines()) == 2
    assert rules.splitlines()[0].startswith("[AXIOM w=10.0] a")
