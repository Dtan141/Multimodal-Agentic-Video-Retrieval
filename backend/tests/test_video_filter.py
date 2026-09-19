"""Video-scope filter: VideoScope logic, FAISS IDSelector, and service-level scoping."""

from __future__ import annotations

import numpy as np
import pytest

from aic_retrieval.config import get_settings
from aic_retrieval.index.faiss_index import FaissIndex
from aic_retrieval.search.video_filter import VideoScope


def _scope() -> VideoScope:
    vids = np.array(["L21_V001", "L21_V002", "L22_V001", "L22_V002"])
    folders = np.array(["L21", "L21", "L22", "L22"])
    return VideoScope(vids, folders)


def test_allowed_row_ids_include_folder():
    assert set(_scope().allowed_row_ids(["L21"], []).tolist()) == {0, 1}


def test_allowed_row_ids_include_exact():
    assert _scope().allowed_row_ids(["L22_V002"], []).tolist() == [3]


def test_allowed_row_ids_exclude_only():
    assert set(_scope().allowed_row_ids([], ["L21"]).tolist()) == {2, 3}


def test_allowed_row_ids_combined():
    assert _scope().allowed_row_ids(["L21"], ["L21_V001"]).tolist() == [1]


def test_allowed_row_ids_none_when_empty():
    assert _scope().allowed_row_ids([], []) is None


def test_keep_fn():
    keep = VideoScope.keep_fn(["L21"], ["L21_V001"])
    assert keep("L21_V002")
    assert not keep("L21_V001")
    assert not keep("L22_V001")
    assert VideoScope.keep_fn([], []) is None


def test_faiss_id_selector():
    idx = FaissIndex.empty(8)
    x = np.random.randn(10, 8).astype("float32")
    idx.add(x)
    allowed = np.array([2, 4, 6], dtype="int64")
    _, ids = idx.search(x[0], top_k=5, allowed_ids=allowed)
    assert {int(i) for i in ids[0] if i >= 0} <= {2, 4, 6}

    _, ids_empty = idx.search(x[0], top_k=5, allowed_ids=np.array([], dtype="int64"))
    assert ids_empty.shape[1] == 0


# ---- service-level (needs the real index) ---------------------------------
settings = get_settings()


@pytest.mark.skipif(
    not (settings.index_dir / "vectors.faiss").exists()
    or not (settings.index_dir / "text" / "caption.pkl").exists(),
    reason="index/text not built",
)
def test_service_scoping():
    from aic_retrieval.search.service import SearchService

    svc = SearchService.load(settings)

    inc = svc.semantic_search("a man", top_k=20, include=["L21"])
    assert inc and all(r["video_id"].startswith("L21_") for r in inc)

    exc = svc.semantic_search("a man", top_k=50, exclude=["L21_V001"])
    assert all(r["video_id"] != "L21_V001" for r in exc)

    # hybrid honours the filter across branches too
    hyb = svc.hybrid_search(
        "a man riding a horse",
        module_queries={"object": "person horse"},
        top_k=20,
        include=["L27"],
    )
    assert all(r["video_id"].startswith("L27_") for r in hyb)
