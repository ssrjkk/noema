"""Hybrid Neuro-Cognitive Memory: BM25 + Dense + RRF + Cross-Encoder + Failure Index.

Implements:
- Bi-temporal fact versioning (valid_from / superseded_at) with strict supersede().
- Non-destructive Ebbinghaus retention weighting: R(dt) = exp(-lambda * dt_days).
- Three-stage retrieval: Sparse BM25 + Dense cosine + Reciprocal Rank Fusion (RRF)
  with an optional Cross-Encoder reranker hook.
- Failure-Indexed Memory mapping (stacktrace + Z3 unsat_core) -> verified_patch_diff.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Extract lowercase alphanumeric/identifier tokens."""
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def default_hash_embedding(text: str, dim: int = 64) -> list[float]:
    """Deterministic lightweight token dense projection (zero-dep fallback)."""
    vec = [0.0] * dim
    tokens = tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 1e-12 else vec


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two equal-length dense vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


class CrossEncoderReranker(Protocol):
    """Pluggable cross-encoder (e.g., BAAI/bge-reranker-v2-m3) for top-K precision."""

    def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return relevance scores for each document in documents."""


@dataclass
class MemoryRecord:
    """Bi-temporal memory node supporting episodic, semantic, and procedural strata."""

    record_id: str
    content: str
    stratum: str = "semantic"
    tags: tuple[str, ...] = ()
    valid_from: datetime = field(default_factory=lambda: datetime.now(UTC))
    superseded_at: datetime | None = None
    decay_lambda: float = 0.05
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def is_valid_at(self, at_time: datetime) -> bool:
        """Return True iff the fact is active at ``at_time``."""
        if at_time < self.valid_from:
            return False
        return not (self.superseded_at is not None and at_time >= self.superseded_at)

    def retention_weight(self, at_time: datetime) -> float:
        """Compute Ebbinghaus retention multiplier in (0, 1]."""
        delta_days = max(0.0, (at_time - self.valid_from).total_seconds() / 86400.0)
        return math.exp(-self.decay_lambda * delta_days)


@dataclass(frozen=True)
class RetrievalHit:
    """Single scored memory retrieval result."""

    record: MemoryRecord
    score: float
    bm25_score: float
    dense_score: float
    retention: float


@dataclass(frozen=True)
class FailurePattern:
    """Verified fix indexed by Sentry stacktrace and Z3 unsat_core signature."""

    pattern_id: str
    error_signature: str
    unsat_core: tuple[str, ...]
    verified_patch_diff: str
    z3_contract: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class HybridCognitiveMemory:
    """Bi-temporal BM25 + Dense + RRF + Cross-Encoder memory with Failure Index."""

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]] | None = None,
        reranker: CrossEncoderReranker | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.embed_fn = embed_fn or default_hash_embedding
        self.reranker = reranker
        self.rrf_k = rrf_k
        self._records: dict[str, MemoryRecord] = {}
        self._doc_tokens: dict[str, list[str]] = {}
        self._failures: dict[str, FailurePattern] = {}

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def failure_count(self) -> int:
        return len(self._failures)

    def add(self, record: MemoryRecord, *, overwrite: bool = False) -> None:
        """Insert a memory record and refresh its sparse/dense indices.

        Refuses silent overwrite of an existing ``record_id``; route content
        updates through ``supersede()`` (bi-temporal versioning). ``overwrite=True``
        is reserved for internally derived, idempotent re-indexed ids.
        """
        if record.record_id in self._records and not overwrite:
            raise ValueError(
                f"Memory record_id {record.record_id!r} already exists; "
                f"use supersede() or add(overwrite=True)"
            )
        if not record.embedding:
            record.embedding = self.embed_fn(record.content)
        self._records[record.record_id] = record
        self._doc_tokens[record.record_id] = tokenize(record.content + " " + " ".join(record.tags))

    def supersede(
        self,
        old_record_id: str,
        new_record: MemoryRecord,
        at_time: datetime | None = None,
        *,
        strict: bool = True,
    ) -> None:
        """Close the validity interval of an outdated fact and insert its replacement.

        Raises KeyError if ``strict=True`` and ``old_record_id`` does not exist.
        """
        ts = at_time or datetime.now(UTC)
        if old_record_id in self._records:
            self._records[old_record_id].superseded_at = ts
        elif strict:
            raise KeyError(f"Cannot supersede unknown record_id: {old_record_id!r}")
        new_record.valid_from = ts
        self.add(new_record)

    def _compute_bm25(
        self,
        query_tokens: list[str],
        candidate_ids: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> dict[str, float]:
        n_docs = len(candidate_ids)
        if n_docs == 0 or not query_tokens:
            return dict.fromkeys(candidate_ids, 0.0)

        avg_dl = sum(len(self._doc_tokens[rid]) for rid in candidate_ids) / max(1, n_docs)
        df: Counter[str] = Counter()
        for rid in candidate_ids:
            for term in set(self._doc_tokens[rid]):
                df[term] += 1

        scores: dict[str, float] = {}
        for rid in candidate_ids:
            doc_toks = self._doc_tokens[rid]
            dl = len(doc_toks)
            tf = Counter(doc_toks)
            score = 0.0
            for q in query_tokens:
                freq = tf.get(q, 0)
                if freq == 0:
                    continue
                idf = math.log(1.0 + (n_docs - df[q] + 0.5) / (df[q] + 0.5))
                denom = freq + k1 * (1.0 - b + b * (dl / max(1e-6, avg_dl)))
                score += idf * ((freq * (k1 + 1.0)) / denom)
            scores[rid] = score
        return scores

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        at_time: datetime | None = None,
        min_retention: float = 0.0,
        include_superseded: bool = False,
    ) -> list[RetrievalHit]:
        """Hybrid BM25 + Dense + RRF retrieval with bi-temporal filtering and soft decay.

        By default ``min_retention=0.0`` so older valid semantic facts are never silently
        erased; Ebbinghaus decay acts purely as a ranking weight unless pruning is requested.
        """
        now = at_time or datetime.now(UTC)
        valid_ids = [
            rid
            for rid, rec in self._records.items()
            if (include_superseded or rec.is_valid_at(now))
            and rec.retention_weight(now) >= min_retention
        ]
        if not valid_ids:
            return []

        q_tokens = tokenize(query)
        q_emb = self.embed_fn(query)
        bm25_scores = self._compute_bm25(q_tokens, valid_ids)
        dense_scores = {
            rid: max(0.0, cosine_similarity(q_emb, self._records[rid].embedding))
            for rid in valid_ids
        }

        bm25_ranked = sorted(valid_ids, key=lambda rid: bm25_scores[rid], reverse=True)
        dense_ranked = sorted(valid_ids, key=lambda rid: dense_scores[rid], reverse=True)
        bm25_rank_map = {rid: idx + 1 for idx, rid in enumerate(bm25_ranked)}
        dense_rank_map = {rid: idx + 1 for idx, rid in enumerate(dense_ranked)}

        rrf_scores: dict[str, float] = {}
        for rid in valid_ids:
            rec = self._records[rid]
            retention = rec.retention_weight(now)
            rrf = (1.0 / (self.rrf_k + bm25_rank_map[rid])) + (
                1.0 / (self.rrf_k + dense_rank_map[rid])
            )
            rrf_scores[rid] = rrf * (0.75 + 0.25 * retention)

        shortlist = sorted(valid_ids, key=lambda rid: rrf_scores[rid], reverse=True)[
            : max(top_k * 3, 15)
        ]

        if self.reranker is not None and shortlist:
            ce_scores = self.reranker.score_pairs(
                query, [self._records[rid].content for rid in shortlist]
            )
            for rid, ce in zip(shortlist, ce_scores, strict=True):
                rrf_scores[rid] = 0.65 * ce + 0.35 * rrf_scores[rid]
            shortlist.sort(key=lambda rid: rrf_scores[rid], reverse=True)

        return [
            RetrievalHit(
                record=self._records[rid],
                score=rrf_scores[rid],
                bm25_score=bm25_scores[rid],
                dense_score=dense_scores[rid],
                retention=self._records[rid].retention_weight(now),
            )
            for rid in shortlist[:top_k]
        ]

    def index_failure_fix(self, pattern: FailurePattern) -> None:
        """Store a verified repair indexed by its failure signature and unsat_core."""
        self._failures[pattern.pattern_id] = pattern
        searchable = (
            f"FAILURE_FIX {pattern.error_signature} "
            f"UNSAT_CORE {' '.join(pattern.unsat_core)} "
            f"CONTRACT {pattern.z3_contract}"
        )
        self.add(
            MemoryRecord(
                record_id=f"fail::{pattern.pattern_id}",
                content=searchable,
                stratum="procedural",
                tags=("failure_fix", *pattern.unsat_core),
                valid_from=pattern.created_at,
                decay_lambda=0.005,
                metadata={"pattern_id": pattern.pattern_id},
            ),
            overwrite=True,
        )

    def lookup_similar_failures(
        self,
        error_signature: str,
        unsat_core: Sequence[str] = (),
        top_k: int = 3,
    ) -> list[tuple[FailurePattern, float]]:
        """Retrieve past verified patches matching an incoming incident or Z3 failure."""
        query = f"FAILURE_FIX {error_signature} {' '.join(unsat_core)}"
        hits = self.retrieve(query, top_k=top_k * 2)
        matched: list[tuple[FailurePattern, float]] = []
        for hit in hits:
            pid = hit.record.metadata.get("pattern_id")
            if isinstance(pid, str) and pid in self._failures:
                matched.append((self._failures[pid], hit.score))
            if len(matched) >= top_k:
                break
        return matched
