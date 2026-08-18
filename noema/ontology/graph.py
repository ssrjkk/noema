"""Ontology — a typed knowledge graph of the world.

The idea is deliberately simple: describe entities and the relations between
them ("User owns Account", "Account runs_on Server") as data, and treat code
as a function applied to that ontology. This module is the storage and query
layer for such a graph:

- typed :class:`Entity` / :class:`Relation` records with attribute bags,
- bounded growth (hard caps instead of unbounded accumulation),
- directed adjacency with BFS neighbors and bounded path search,
- cycle detection for consistency checks,
- durable, tenant-agnostic persistence through :mod:`noema.utils.atomic_io`
  (atomic writes with rotating backups; corrupt files degrade to empty graphs
  instead of raising).

The graph is not a suggestion engine (see ``noema/knowledge/graph.py`` for
that); it is a place to *record* facts about the world so that reasoning
pipelines can later query them.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from noema.logging import get_logger
from noema.utils.atomic_io import atomic_read_json, atomic_write_json

log = get_logger(__name__)

DEFAULT_MAX_ENTITIES = 10_000
DEFAULT_MAX_RELATIONS = 50_000


class OntologyError(ValueError):
    """Raised when an ontology operation violates a structural invariant."""


@dataclass(frozen=True)
class Entity:
    name: str
    type: str
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str
    object: str
    weight: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)


def _clean_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        raise OntologyError("Entity/relation names must not be empty")
    if any(ord(c) < 32 for c in cleaned):
        raise OntologyError(f"Control characters are not allowed in names: {name!r}")
    return cleaned


class OntologyGraph:
    """A bounded, directed, typed knowledge graph with durable persistence.

    Complexity:
    - ``add_entity`` / ``add_relation``: ``O(1)``.
    - ``neighbors``: ``O(V + E)`` for the BFS at the requested depth.
    - ``find_paths``: ``O(E^depth)`` worst case, bounded by ``max_depth``.
    - ``has_cycle``: ``O(V + E)``.
    """

    def __init__(
        self,
        max_entities: int = DEFAULT_MAX_ENTITIES,
        max_relations: int = DEFAULT_MAX_RELATIONS,
    ) -> None:
        self.max_entities = max_entities
        self.max_relations = max_relations
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []
        self._by_subject: dict[str, list[int]] = {}
        self._by_object: dict[str, list[int]] = {}

    # ── Mutation ─────────────────────────────────────────────────────────

    def add_entity(self, entity: Entity) -> None:
        """Add or replace an entity by name. Raises on cap overflow."""
        name = _clean_name(entity.name)
        if name not in self._entities and len(self._entities) >= self.max_entities:
            raise OntologyError(
                f"Entity cap reached ({self.max_entities}); refusing to grow unbounded"
            )
        self._entities[name] = Entity(
            name=name, type=(entity.type or "unknown").strip(), attributes=dict(entity.attributes)
        )

    def add_relation(self, relation: Relation) -> None:
        """Add a relation between two known entities.

        Raises :class:`OntologyError` if either endpoint is unknown or the
        relation cap is reached. Duplicate (subject, predicate, object)
        triples are idempotent (the existing record is kept).
        """
        subject = _clean_name(relation.subject)
        object_ = _clean_name(relation.object)
        if subject not in self._entities:
            raise OntologyError(f"Unknown subject entity: {subject!r}")
        if object_ not in self._entities:
            raise OntologyError(f"Unknown object entity: {object_!r}")
        predicate = (relation.predicate or "").strip()
        if not predicate:
            raise OntologyError("Relation predicate must not be empty")

        for idx in self._by_subject.get(subject, ()):
            existing = self._relations[idx]
            if existing.predicate == predicate and existing.object == object_:
                return
        if len(self._relations) >= self.max_relations:
            raise OntologyError(
                f"Relation cap reached ({self.max_relations}); refusing to grow unbounded"
            )

        idx = len(self._relations)
        self._relations.append(
            Relation(
                subject=subject,
                predicate=predicate,
                object=object_,
                weight=float(relation.weight),
                metadata=dict(relation.metadata),
            )
        )
        self._by_subject.setdefault(subject, []).append(idx)
        self._by_object.setdefault(object_, []).append(idx)

    # ── Query ────────────────────────────────────────────────────────────

    def get_entity(self, name: str) -> Entity | None:
        return self._entities.get(_clean_name(name))

    def entities(self, entity_type: str | None = None) -> list[Entity]:
        if entity_type is None:
            return sorted(self._entities.values(), key=lambda e: e.name)
        return sorted(
            (e for e in self._entities.values() if e.type == entity_type), key=lambda e: e.name
        )

    def relations(self) -> list[Relation]:
        return list(self._relations)

    def neighbors(
        self, name: str, depth: int = 1, max_depth: int = 3
    ) -> list[tuple[str, str, float]]:
        """BFS neighbors within ``depth`` hops, directed.

        Returns ``(other, predicate, weight)`` tuples, outgoing edges first,
        sorted by weight descending. ``depth`` is clamped to ``max_depth``.
        """
        start = _clean_name(name)
        if start not in self._entities:
            return []
        depth = max(1, min(depth, max_depth))
        seen: set[str] = set()
        frontier = {start}
        out: list[tuple[str, str, float]] = []
        incoming: list[tuple[str, str, float]] = []
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                for idx in self._by_subject.get(node, ()):
                    r = self._relations[idx]
                    if r.object not in seen:
                        out.append((r.object, r.predicate, r.weight))
                        next_frontier.add(r.object)
                for idx in self._by_object.get(node, ()):
                    r = self._relations[idx]
                    if r.subject not in seen:
                        incoming.append((r.subject, r.predicate, r.weight))
                        next_frontier.add(r.subject)
            seen.update(frontier)
            frontier = next_frontier
        seen.update(frontier)
        out.sort(key=lambda t: t[2], reverse=True)
        incoming.sort(key=lambda t: t[2], reverse=True)
        return out + incoming

    def find_paths(self, start: str, end: str, max_depth: int = 4) -> list[list[str]]:
        """Bounded DFS over directed edges; returns up to 8 simple paths."""
        start = _clean_name(start)
        end = _clean_name(end)
        if start not in self._entities or end not in self._entities:
            return []
        paths: list[list[str]] = []

        def _dfs(node: str, path: list[str]) -> None:
            if len(paths) >= 8 or len(path) > max_depth:
                return
            for idx in self._by_subject.get(node, ()):
                nxt = self._relations[idx].object
                if nxt in path:
                    continue
                cand = [*path, nxt]
                if nxt == end:
                    paths.append(cand)
                    if len(paths) >= 8:
                        return
                else:
                    _dfs(nxt, cand)

        _dfs(start, [start])
        return paths

    def has_cycle(self) -> bool:
        """Detect directed cycles reachable from any node (iterative DFS)."""
        white, gray, black = 0, 1, 2
        color: dict[str, int] = dict.fromkeys(self._entities, white)
        for start in self._entities:
            if color[start] != white:
                continue
            stack: list[tuple[str, bool]] = [(start, False)]
            while stack:
                node, expanded = stack.pop()
                if expanded:
                    color[node] = black
                    continue
                if color[node] == gray:
                    return True
                color[node] = gray
                stack.append((node, True))
                for idx in self._by_subject.get(node, ()):
                    nxt = self._relations[idx].object
                    if color[nxt] == white or color[nxt] == gray:
                        stack.append((nxt, False))
        return False

    # ── Rendering for prompt context ─────────────────────────────────────

    def to_context(self, limit: int = 50) -> str:
        """Render the graph as compact triples, heaviest relations first."""
        ordered = sorted(self._relations, key=lambda r: r.weight, reverse=True)[:limit]
        return "\n".join(
            f"{r.subject} --{r.predicate}--> {r.object} (w={r.weight:.1f})" for r in ordered
        )

    def stats(self) -> dict[str, Any]:
        return {
            "entities": len(self._entities),
            "relations": len(self._relations),
            "has_cycle": self.has_cycle(),
        }

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist atomically (tmp + rename + rotating backups)."""
        payload = {
            "entities": [asdict(e) for e in self._entities.values()],
            "relations": [asdict(r) for r in self._relations],
        }
        atomic_write_json(Path(path), payload)

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> OntologyGraph:
        """Load a graph; corrupt or missing files degrade to an empty graph."""
        data = atomic_read_json(Path(path), default=None)
        graph = cls(**kwargs)
        if not isinstance(data, dict):
            return graph
        with contextlib.suppress(Exception):
            for raw in data.get("entities", []):
                graph.add_entity(
                    Entity(**{k: v for k, v in raw.items() if k in ("name", "type", "attributes")})
                )
            for raw in data.get("relations", []):
                graph.add_relation(
                    Relation(
                        **{
                            k: v
                            for k, v in raw.items()
                            if k in ("subject", "predicate", "object", "weight", "metadata")
                        }
                    )
                )
        return graph

    def __len__(self) -> int:
        return len(self._entities)
