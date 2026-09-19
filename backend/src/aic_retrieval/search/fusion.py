"""Reciprocal Rank Fusion (RRF) with temporal dedup.

Combines any number of ranked sources (semantic + per-module lexical) into one
list. Ports the RRF from ``test/rerank.py:170-217`` and the ±N-frame temporal
dedup from ``test/pipeline_rerank.py:16-71`` so near-duplicate frames from
different branches collapse into one candidate.
"""

from __future__ import annotations


def rrf_fuse(
    sources: dict[str, list[dict]],
    *,
    rrf_k: int = 60,
    temporal_margin: int = 3,
    top_k: int = 100,
) -> list[dict]:
    """Fuse ``{source_name: [hit, ...]}`` into one ranked list.

    Each hit needs ``video_id`` and ``frame_id``; ``rank`` is used if present,
    otherwise the position in the list. Frames within ``temporal_margin`` of an
    existing candidate (same video) are merged, keeping the best rank per source.
    Returns hits with ``rrf_score``, ``sources`` and per-source ``ranks``.
    """
    reps: dict[str, list[dict]] = {}

    def find_rep(video_id: str, frame_id: int) -> dict | None:
        for rep in reps.get(video_id, []):
            if abs(rep["frame_id"] - frame_id) <= temporal_margin:
                return rep
        return None

    for source, hits in sources.items():
        for i, hit in enumerate(hits):
            vid = hit["video_id"]
            fid = int(hit["frame_id"])
            rank = int(hit.get("rank", i + 1))

            rep = find_rep(vid, fid)
            if rep is None:
                rep = {"video_id": vid, "frame_id": fid, "ranks": {}, "members": []}
                reps.setdefault(vid, []).append(rep)

            prev = rep["ranks"].get(source)
            if prev is None or rank < prev:
                rep["ranks"][source] = rank
            rep["members"].append({"source": source, **hit})

    fused: list[dict] = []
    for rlist in reps.values():
        for rep in rlist:
            score = sum(1.0 / (rrf_k + r) for r in rep["ranks"].values())
            fused.append(
                {
                    "video_id": rep["video_id"],
                    "frame_id": rep["frame_id"],
                    "rrf_score": score,
                    "sources": sorted(rep["ranks"].keys()),
                    "ranks": rep["ranks"],
                    "members": rep["members"],
                }
            )

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    for i, item in enumerate(fused[:top_k], start=1):
        item["rank"] = i
    return fused[:top_k]
