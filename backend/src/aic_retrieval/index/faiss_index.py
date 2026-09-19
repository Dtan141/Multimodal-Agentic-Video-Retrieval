"""FAISS vector index.

Uses ``IndexFlatIP`` (exact, brute-force inner product). Since SigLIP2 vectors are
L2-normalized, inner product == cosine similarity, so results are *identical* to
the old ``normalized_vectors @ query`` search in ``AIC2026_retrieval_pipeline.py``
— just wrapped in a persistent, load-once index. Swap to ``IndexHNSWFlat`` /
``IndexIVFPQ`` later if the corpus outgrows exact search.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from aic_retrieval.logging_conf import get_logger

logger = get_logger(__name__)

_INDEX_NAME = "vectors.faiss"


def _as_f32(vectors: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


class FaissIndex:
    def __init__(self, index: faiss.Index) -> None:
        self.index = index

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    @property
    def dim(self) -> int:
        return self.index.d

    @classmethod
    def empty(cls, dim: int) -> FaissIndex:
        return cls(faiss.IndexFlatIP(dim))

    def add(self, vectors: np.ndarray, *, normalize: bool = True) -> None:
        """Add vectors (defensively L2-normalized so IP == cosine)."""
        arr = _as_f32(vectors)
        if normalize:
            faiss.normalize_L2(arr)
        self.index.add(arr)

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        allowed_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, row_ids), each shape [n_queries, top_k].

        If ``allowed_ids`` is given, the search is restricted to those row ids via
        a FAISS IDSelector (exact — no over-fetch). An empty ``allowed_ids`` yields
        an empty result.
        """
        q = _as_f32(query)
        faiss.normalize_L2(q)
        top_k = min(top_k, self.ntotal)

        if allowed_ids is None:
            scores, ids = self.index.search(q, top_k)
            return scores, ids

        ids64 = np.ascontiguousarray(allowed_ids, dtype=np.int64)
        if ids64.size == 0:
            empty = np.empty((q.shape[0], 0), dtype="float32")
            return empty, np.empty((q.shape[0], 0), dtype="int64")

        top_k = min(top_k, ids64.size)
        # keep the selector alive for the duration of the search call
        selector = faiss.IDSelectorBatch(ids64.size, faiss.swig_ptr(ids64))
        params = faiss.SearchParameters()
        params.sel = selector
        scores, ids = self.index.search(q, top_k, params=params)
        return scores, ids

    def save(self, index_dir: Path) -> Path:
        index_dir.mkdir(parents=True, exist_ok=True)
        path = index_dir / _INDEX_NAME
        faiss.write_index(self.index, str(path))
        logger.info("Saved FAISS index: %s (%d vectors, dim=%d)", path, self.ntotal, self.dim)
        return path

    @classmethod
    def load(cls, index_dir: Path) -> FaissIndex:
        path = index_dir / _INDEX_NAME
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")
        try:
            return cls(faiss.read_index(str(path)))
        except (MemoryError, RuntimeError) as exc:
            # low-RAM machines: memory-map the flat index instead of loading it all
            logger.warning("read_index failed (%s); retrying with mmap", exc)
            return cls(faiss.read_index(str(path), faiss.IO_FLAG_MMAP))
