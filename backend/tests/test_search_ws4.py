"""WS4 integration: SearchService semantic + module + hybrid (no rerank).

Skips if the index/text stores are not built. Loads SigLIP2 (downloads once).
"""

from __future__ import annotations

import pytest

from aic_retrieval.config import get_settings

settings = get_settings()
pytestmark = pytest.mark.skipif(
    not (settings.index_dir / "vectors.faiss").exists()
    or not (settings.index_dir / "text" / "caption.pkl").exists(),
    reason="index/text not built",
)


@pytest.fixture(scope="module")
def service():
    # Imported lazily so CI (without torch) can still collect this module.
    from aic_retrieval.search.service import SearchService

    return SearchService.load(settings)


def test_semantic_scores_descending(service):
    res = service.semantic_search("a man riding a horse", top_k=10)
    assert len(res) == 10
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)
    assert {"video_id", "frame_id", "rank", "source"} <= set(res[0])
    assert res[0]["source"] == "semantic"


def test_module_search(service):
    res = service.module_search("object", "person", top_k=10)
    assert isinstance(res, list)
    if res:
        assert res[0]["source"] == "meta:object"


def test_hybrid_fuses_sources(service):
    res = service.hybrid_search(
        "a man riding a horse",
        module_queries={"object": "person horse", "caption": "man horse"},
        top_k=20,
        rerank=False,
    )
    assert len(res) > 0
    # ranks are contiguous and sorted by rrf_score
    assert [r["rank"] for r in res] == list(range(1, len(res) + 1))
    scores = [r["rrf_score"] for r in res]
    assert scores == sorted(scores, reverse=True)
    # results are drawn from multiple branches (semantic + at least one metadata
    # module). Cross-branch frames rarely coincide within temporal_margin because
    # the vector and metadata pipelines sample different keyframes, so we check
    # the union of sources rather than per-frame corroboration.
    all_sources = {s for r in res for s in r["sources"]}
    assert "semantic" in all_sources
    assert any(s.startswith("meta:") for s in all_sources)


def test_hybrid_modules_only(service):
    # no semantic query -> search by metadata module alone
    res = service.hybrid_search("", module_queries={"ocr": "giây"}, top_k=10, rerank=False)
    assert len(res) > 0
    assert all("semantic" not in r["sources"] for r in res)
    assert all(any(s.startswith("meta:") for s in r["sources"]) for r in res)


def test_hybrid_empty_returns_nothing(service):
    assert service.hybrid_search("", module_queries={}, top_k=10) == []
