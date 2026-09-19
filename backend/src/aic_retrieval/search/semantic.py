"""Semantic (text->image) search over the FAISS index."""

from __future__ import annotations

import numpy as np

from aic_retrieval.config import Settings
from aic_retrieval.embedding.siglip2 import Siglip2Embedder
from aic_retrieval.index.faiss_index import FaissIndex
from aic_retrieval.index.metadata_store import MetadataStore
from aic_retrieval.logging_conf import get_logger

logger = get_logger(__name__)


class SemanticSearcher:
    def __init__(self, index: FaissIndex, store: MetadataStore, embedder: Siglip2Embedder) -> None:
        self.index = index
        self.store = store
        self.embedder = embedder

    @classmethod
    def load(cls, settings: Settings) -> SemanticSearcher:
        index = FaissIndex.load(settings.index_dir)
        store = MetadataStore.load(settings.index_dir)
        if index.ntotal != len(store):
            logger.warning("index/store length mismatch: %d vs %d", index.ntotal, len(store))
        embedder = Siglip2Embedder(
            model_id=settings.model_id,
            device=settings.device,
            use_fp16=settings.use_fp16,
            text_max_length=settings.text_max_length,
        )
        return cls(index, store, embedder)

    def encode(self, query: str) -> np.ndarray:
        feats = self.embedder.encode_texts([query])
        return feats.detach().float().cpu().numpy().astype(np.float32)

    def search(self, query: str, top_k: int = 100, allowed_ids=None) -> list[dict]:
        qv = self.encode(query)
        scores, ids = self.index.search(qv, top_k, allowed_ids=allowed_ids)
        scores, ids = scores[0], ids[0]

        row_ids = [int(i) for i in ids if i >= 0]
        metas = self.store.get_many(row_ids)

        results: list[dict] = []
        for rank, meta in enumerate(metas, start=1):
            results.append(
                {**meta, "score": float(scores[rank - 1]), "rank": rank, "source": "semantic"}
            )
        return results
