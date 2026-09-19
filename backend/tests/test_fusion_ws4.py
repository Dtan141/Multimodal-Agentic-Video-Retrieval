"""WS4 unit tests for RRF fusion + temporal dedup (no torch needed)."""

from __future__ import annotations

from aic_retrieval.search.fusion import rrf_fuse


def test_rrf_sums_across_sources():
    sources = {
        "semantic": [{"video_id": "L21_V001", "frame_id": 100, "rank": 1}],
        "meta:ocr": [{"video_id": "L21_V001", "frame_id": 100, "rank": 1}],
    }
    fused = rrf_fuse(sources, rrf_k=60, temporal_margin=3)
    assert len(fused) == 1
    # both sources rank 1 -> 2 * 1/61
    assert fused[0]["rrf_score"] == 2 * (1.0 / 61)
    assert set(fused[0]["sources"]) == {"semantic", "meta:ocr"}


def test_temporal_dedup_merges_nearby_frames():
    sources = {
        "semantic": [{"video_id": "L21_V001", "frame_id": 150, "rank": 1}],
        "meta:ocr": [{"video_id": "L21_V001", "frame_id": 152, "rank": 2}],  # within margin 3
    }
    fused = rrf_fuse(sources, temporal_margin=3)
    assert len(fused) == 1  # merged
    assert fused[0]["frame_id"] == 150  # first seen wins
    assert fused[0]["ranks"] == {"semantic": 1, "meta:ocr": 2}


def test_distant_frames_not_merged():
    sources = {
        "semantic": [{"video_id": "L21_V001", "frame_id": 150, "rank": 1}],
        "meta:ocr": [{"video_id": "L21_V001", "frame_id": 200, "rank": 1}],
    }
    fused = rrf_fuse(sources, temporal_margin=3)
    assert len(fused) == 2


def test_ranking_and_top_k():
    sources = {
        "semantic": [
            {"video_id": "V", "frame_id": 10, "rank": 1},
            {"video_id": "V", "frame_id": 500, "rank": 2},
        ],
        "meta:ocr": [{"video_id": "V", "frame_id": 10, "rank": 1}],
    }
    fused = rrf_fuse(sources, top_k=1)
    assert len(fused) == 1
    # frame 10 has two sources -> higher score -> rank 1
    assert fused[0]["frame_id"] == 10
    assert fused[0]["rank"] == 1


def test_best_rank_per_source_kept():
    sources = {
        "meta:ocr": [
            {"video_id": "V", "frame_id": 10, "rank": 5},
            {"video_id": "V", "frame_id": 11, "rank": 2},  # merges into 10, better rank
        ],
    }
    fused = rrf_fuse(sources, temporal_margin=3)
    assert len(fused) == 1
    assert fused[0]["ranks"]["meta:ocr"] == 2
