"""T1.4 tests: domain knowledge seeding so RAG retrieval answers each domain.

Covers ``noema/knowledge/domains.py`` and ``store.py``:
- every one of the 22 built-in domain modules has a seeded knowledge entry,
- ``KnowledgeStore.search`` returns a relevant hit for each domain query
  without any tuning (done-when: >= 90% domains surface their own entry),
- the seeded corpus does not cross-match badly (sanity scores).
"""

import tempfile
from pathlib import Path

import pytest

from noema.knowledge.domains import DOMAIN_INDEX, DOMAIN_KNOWLEDGE
from noema.knowledge.store import KnowledgeStore
from noema.modules.registry import ModuleRegistry

_N_DOMAINS = 22


@pytest.fixture()
async def fresh_store() -> KnowledgeStore:
    """A store with an empty persist file: only built-in knowledge is indexed."""
    tmp = Path(tempfile.mkdtemp()) / "none.json"
    store = KnowledgeStore(persist_path=str(tmp))
    await store.load()
    return store


def test_twenty_two_domains_seeded() -> None:
    assert len(DOMAIN_KNOWLEDGE) == _N_DOMAINS
    assert len(DOMAIN_INDEX) == _N_DOMAINS


def test_every_registry_module_has_a_seed() -> None:
    registry_names = {m["name"] for m in ModuleRegistry().list_modules()}
    assert registry_names == set(DOMAIN_INDEX.keys())


async def test_search_returns_relevant_hit_for_each_domain(fresh_store: KnowledgeStore) -> None:
    """T1.4 done-when: each domain query returns its own seeded entry on top."""
    hits = 0
    failures: list[str] = []
    for module in sorted(ModuleRegistry().list_modules(), key=lambda m: m["name"]):
        query = f"{module['name']} {module['description']}"
        results = await fresh_store.search(query, top_k=5)
        assert results, f"no hits for domain {module['name']!r}"
        top_title = results[0].get("title")
        expected = DOMAIN_INDEX[module["name"]]
        if top_title == expected:
            hits += 1
        else:
            failures.append(f"{module['name']}: top={top_title!r}")

    rate = hits / len(DOMAIN_INDEX)
    assert rate >= 0.9, (
        f"domain relevance rate {rate:.0%} < 90%; {len(failures)} missed: {failures}"
    )


async def test_every_seed_is_reachable_by_module_name(fresh_store: KnowledgeStore) -> None:
    """A bare module-name query still lands on the right entry."""
    for name, title in DOMAIN_INDEX.items():
        results = await fresh_store.search(name, top_k=3)
        titles = {r.get("title") for r in results}
        assert results, f"no hits for bare module query {name!r}"
        assert title in titles, f"module {name!r} seed not in top-3: {titles}"


async def test_seed_scores_are_sane(fresh_store: KnowledgeStore) -> None:
    """Top scores are finite, positive, and above the tie-noise floor."""
    for module in ModuleRegistry().list_modules():
        query = f"{module['name']} {module['description']}"
        results = await fresh_store.search(query, top_k=1)
        score = results[0].get("score", 0.0)
        assert score == score, f"NaN score for {module['name']!r}"
        assert score > 0.05, f"weak score {score:.3f} for {module['name']!r}"
