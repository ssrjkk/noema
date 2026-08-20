"""Dense embeddings — optional sentence-transformers with n-gram fallback."""

from __future__ import annotations

from functools import cache
from typing import Any, cast

import numpy as np

from noema.logging import get_logger

log = get_logger(__name__)


@cache
def get_embedder(dim: int = 128) -> DenseEmbedder:
    """Process-wide shared embedder (one model load / projection matrix).

    Every consumer (memory indexes, semantic cache) reuses the same instance,
    so a SentenceTransformer model is loaded at most once per dimension.
    """
    return DenseEmbedder(dim=dim)


class DenseEmbedder:
    """Produces dense vector embeddings for text.

    Uses sentence-transformers if available; falls back to character
    n-gram hashing with random projection (128-dim).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 128) -> None:
        self._model: Any = None
        self._dim = dim
        self._rng = np.random.RandomState(42)
        self._proj = self._rng.randn(10000, dim).astype(np.float32)
        self._model_name = model_name
        self._try_load_model()

    def _try_load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        except ImportError:
            log.debug("sentence_transformers_unavailable_fallback_to_ngrams")

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def is_semantic(self) -> bool:
        return self._model is not None

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        if self._model is not None:
            return cast("np.ndarray", self._model.encode(texts, normalize_embeddings=True)).astype(
                np.float32
            )
        return self._fallback_embed(texts)

    def embed_one(self, text: str) -> np.ndarray:
        return cast("np.ndarray", self.embed([text])[0]).reshape(1, -1)

    def _fallback_embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.empty((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            text_lower = text.lower()
            if not text_lower:
                continue
            codes = np.frombuffer(text_lower.encode("utf-32-le"), dtype="<u4")
            buckets: set[int] = set()
            for n in (3, 4):
                if codes.size < n:
                    continue
                windows = np.lib.stride_tricks.sliding_window_view(codes, n)
                if n == 3:
                    hv = (
                        windows[:, 0] * 257 * 257 + windows[:, 1] * 257 + windows[:, 2]
                    ) & 0xFFFFFFFF
                else:
                    hv = (
                        windows[:, 0] * 257 * 257 * 257
                        + windows[:, 1] * 257 * 257
                        + windows[:, 2] * 257
                        + windows[:, 3]
                    ) & 0xFFFFFFFF
                buckets.update((hv % 10000).astype(np.int32).tolist())
            vec = np.zeros(10000, dtype=np.float32)
            for bucket in buckets:
                vec[bucket] = 1.0
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec = vec / norm
            dense = vec @ self._proj
            dnorm = np.linalg.norm(dense)
            if dnorm > 1e-10:
                dense = dense / dnorm
            vectors[i] = dense
        return vectors

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-10)
        b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-10)
        return cast("np.ndarray", a_norm @ b_norm.T)


class HNSWIndex:
    """FAISS HNSW index with numpy fallback for dense vector search.

    Adds are incremental: new vectors go straight into the FAISS index (HNSW
    supports online insertion) and are staged as chunks for the numpy fallback
    path, so per-add cost is O(1) amortized instead of a full O(N) rebuild.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._ids: list[str] = []
        self._chunks: list[np.ndarray] = []
        self._vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self._concatenated = True
        self._faiss_index: Any = None
        self._try_init_faiss()

    def _try_init_faiss(self) -> None:
        try:
            import faiss

            self._faiss_index = faiss.IndexHNSWFlat(self._dim, 32)
            self._faiss_index.hnsw.efConstruction = 200
        except ImportError:
            log.debug("faiss_unavailable_fallback_to_numpy_search")

    @property
    def size(self) -> int:
        return len(self._ids)

    def _materialize(self) -> np.ndarray:
        """Lazily concatenate staged chunks (once per mutation batch)."""
        if self._concatenated:
            return self._vectors
        if self._chunks:
            self._vectors = np.concatenate(self._chunks, axis=0)
        else:
            self._vectors = np.empty((0, self._dim), dtype=np.float32)
        self._concatenated = True
        return self._vectors

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        batch = vectors.astype(np.float32)
        self._ids.extend(ids)
        self._chunks.append(batch)
        self._concatenated = False
        if self._faiss_index is not None:
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            normalized = batch / np.maximum(norms, 1e-10)
            self._faiss_index.add(normalized)

    def reset(self) -> None:
        self._ids = []
        self._chunks = []
        self._vectors = np.empty((0, self._dim), dtype=np.float32)
        self._concatenated = True
        if self._faiss_index is not None:
            self._faiss_index.reset()

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        q = query_vector.reshape(1, -1).astype(np.float32)
        q = q / np.maximum(np.linalg.norm(q), 1e-10)
        k = min(top_k, len(self._ids))

        if self._faiss_index is not None and self._faiss_index.ntotal > 0:
            distances, indices = self._faiss_index.search(q, k)
            results = []
            for d2, idx in zip(distances[0], indices[0], strict=False):
                if idx >= 0 and idx < len(self._ids):
                    similarity = max(0.0, 1.0 - (d2 * d2) / 2.0)
                    results.append((self._ids[idx], float(similarity)))
            return results

        vectors = self._materialize()
        if len(vectors) == 0:
            return []
        sims = DenseEmbedder.cosine_similarity(q, vectors)[0]
        top_indices = np.argsort(-sims)[:k]
        return [(self._ids[i], float(sims[i])) for i in top_indices]
