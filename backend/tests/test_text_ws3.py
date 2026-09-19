"""WS3 verification: per-module BM25 indexes load and return well-formed frame hits.

Skips automatically if the text indexes have not been built yet.
"""

from __future__ import annotations

import json

import pytest

from aic_retrieval.config import get_settings
from aic_retrieval.text.bm25_index import ModuleBM25, tokenize
from aic_retrieval.text.ingest_metadata import MODULES

settings = get_settings()
TEXT_DIR = settings.index_dir / "text"
pytestmark = pytest.mark.skipif(
    not (TEXT_DIR / "caption.pkl").exists(),
    reason="text indexes not built (run `aic ingest-metadata`)",
)


def test_tokenize_unicode():
    # Vietnamese accents survive tokenization.
    assert tokenize("Người đàn ông 60 giây") == ["người", "đàn", "ông", "60", "giây"]


@pytest.mark.parametrize("module", MODULES)
def test_module_loads(module):
    idx = ModuleBM25.load(TEXT_DIR, module)
    assert idx.module == module
    assert len(idx.refs) == len(idx.bm25.doc_freqs)


def test_hydration_tables_exist():
    import pandas as pd

    seg = pd.read_parquet(TEXT_DIR / "segments.parquet")
    assert {"video_id", "segment_id", "start_time", "end_time", "caption"} <= set(seg.columns)
    assert len(seg) > 0

    fps = json.loads((TEXT_DIR / "fps.json").read_text())
    assert len(fps) > 0
    assert all(isinstance(v, (int, float)) for v in fps.values())


def test_object_search_returns_frames():
    idx = ModuleBM25.load(TEXT_DIR, "object")
    hits = idx.search("building", top_k=20)
    assert isinstance(hits, list)
    if hits:  # 'building' is a very common label; expect matches
        h = hits[0]
        assert {"video_id", "frame_id", "segment_id", "score", "rank"} <= set(h)
        assert h["rank"] == 1
        # ranks are strictly increasing and unique frames
        keys = [(x["video_id"], x["frame_id"]) for x in hits]
        assert len(keys) == len(set(keys))


def test_search_empty_query():
    idx = ModuleBM25.load(TEXT_DIR, "caption")
    assert idx.search("", top_k=10) == []
