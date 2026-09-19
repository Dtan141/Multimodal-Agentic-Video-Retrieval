"""Cross-encoder reranking.

Scores [query, caption] pairs and combines with the RRF score
(``final = cross + rrf_weight * rrf``), as in ``test/rerank.py:238-243``. The
Qwen2.5-VL final rerank is a Phase-2 hook.
"""

from __future__ import annotations

import torch
from sentence_transformers import CrossEncoder

from aic_retrieval.config import Settings
from aic_retrieval.logging_conf import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self, model: CrossEncoder) -> None:
        self.model = model

    @classmethod
    def load(cls, settings: Settings) -> CrossEncoderReranker:
        device = settings.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info("Loading cross-encoder %s on %s", settings.cross_encoder_model, device)
        return cls(CrossEncoder(settings.cross_encoder_model, device=device))

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        *,
        top_n: int | None = None,
        rrf_weight: float = 0.1,
    ) -> list[dict]:
        scored = [c for c in candidates if (c.get("caption") or "").strip()]
        unscored = [c for c in candidates if not (c.get("caption") or "").strip()]
        if not scored:
            return candidates[:top_n] if top_n else candidates

        pairs = [[query, c["caption"]] for c in scored]
        cross = self.model.predict(pairs)

        out: list[dict] = []
        for c, s in zip(scored, cross, strict=False):
            c = dict(c)
            c["cross_score"] = float(s)
            c["final_score"] = float(s) + rrf_weight * float(c.get("rrf_score", 0.0))
            out.append(c)
        out.sort(key=lambda x: x["final_score"], reverse=True)
        # keep candidates without a caption after the scored ones (no cross signal)
        out.extend(unscored)
        result = out[:top_n] if top_n else out
        for i, c in enumerate(result, start=1):
            c["rank"] = i
        return result
