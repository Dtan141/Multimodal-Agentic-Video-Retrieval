"""Per-module BM25 index (caption / ocr / object / asr).

Each index holds one BM25 corpus plus, per document, the list of frame refs the
document maps to (a caption/asr doc = whole segment -> all its keyframes; an
ocr/object doc = a single keyframe). Search ranks documents then expands to
frames, deduping by (video_id, frame_id) and keeping the best rank — the same
"flatten segments to frames" logic as the old ``test/rerank.py:105-131``.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase + unicode word split (keeps Vietnamese accents for OCR/ASR)."""
    return _TOKEN_RE.findall(str(text).lower())


class ModuleBM25:
    def __init__(self, module: str, bm25: BM25Okapi, refs: list[list[dict]]) -> None:
        self.module = module
        self.bm25 = bm25
        self.refs = refs  # refs[i] = frame dicts for document i

    @classmethod
    def build(cls, module: str, texts: list[str], refs: list[list[dict]]) -> ModuleBM25:
        if len(texts) != len(refs):
            raise ValueError("texts and refs length mismatch")
        corpus = [tokenize(t) for t in texts]
        # BM25Okapi needs a non-empty corpus; guard tiny/empty modules.
        if not corpus:
            corpus = [[""]]
            refs = [[]]
        bm25 = BM25Okapi(corpus)
        return cls(module, bm25, refs)

    def search(self, query: str, top_k: int = 100, keep=None) -> list[dict]:
        """Return frame hits: [{video_id, frame_id, segment_id, score, rank}].

        ``keep``: optional predicate ``(video_id) -> bool`` applied while flattening
        segments to frames (before the top_k cut), so scoping doesn't drop results.
        """
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1]

        results: list[dict] = []
        seen: set[tuple] = set()
        rank = 1
        for di in order:
            if scores[di] <= 0:
                break
            for ref in self.refs[di]:
                if keep is not None and not keep(ref["video_id"]):
                    continue
                key = (ref["video_id"], ref["frame_id"])
                if key in seen:
                    continue
                seen.add(key)
                results.append({**ref, "score": float(scores[di]), "rank": rank})
                rank += 1
                if len(results) >= top_k:
                    return results
        return results

    def save(self, dir_path: Path) -> Path:
        dir_path.mkdir(parents=True, exist_ok=True)
        path = dir_path / f"{self.module}.pkl"
        with path.open("wb") as f:
            pickle.dump({"module": self.module, "bm25": self.bm25, "refs": self.refs}, f)
        return path

    @classmethod
    def load(cls, dir_path: Path, module: str) -> ModuleBM25:
        path = dir_path / f"{module}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found: {path}")
        with path.open("rb") as f:
            data = pickle.load(f)
        return cls(data["module"], data["bm25"], data["refs"])
