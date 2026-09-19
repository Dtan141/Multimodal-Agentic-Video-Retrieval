"""Frame -> timestamp -> segment context hydration.

Given a (video_id, frame_id), find the containing segment (via
``timestamp = frame_id / fps``) and attach its caption/segment_id/timestamp.
Ports ``test/pipeline_rerank.py:76-105`` but backed by the ``segments.parquet``
+ ``fps.json`` written during ingest, so it works for any frame — including
vector-branch frames absent from the metadata keyframes.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import pandas as pd

TEXT_SUBDIR = "text"


class Hydrator:
    def __init__(self, segments_by_video: dict, fps_map: dict[str, float]) -> None:
        # segments_by_video[vid] = (starts_sorted, [ (end, segment_id, caption) ... ])
        self._segs = segments_by_video
        self._fps = fps_map

    @classmethod
    def load(cls, index_dir: Path) -> Hydrator:
        text_dir = index_dir / TEXT_SUBDIR
        df = pd.read_parquet(text_dir / "segments.parquet")
        fps_map = json.loads((text_dir / "fps.json").read_text(encoding="utf-8"))

        segments_by_video: dict = {}
        for vid, grp in df.groupby("video_id"):
            grp = grp.sort_values("start_time")
            starts = grp["start_time"].fillna(0.0).tolist()
            rows = list(
                zip(
                    grp["end_time"].fillna(0.0).tolist(),
                    grp["segment_id"].tolist(),
                    grp["caption"].fillna("").tolist(),
                    strict=False,
                )
            )
            segments_by_video[vid] = (starts, rows)
        return cls(segments_by_video, fps_map)

    def timestamp(self, video_id: str, frame_id: int) -> float:
        fps = self._fps.get(video_id) or 0.0
        return frame_id / fps if fps else 0.0

    def segment_for(self, video_id: str, frame_id: int) -> dict | None:
        entry = self._segs.get(video_id)
        if not entry:
            return None
        starts, rows = entry
        t = self.timestamp(video_id, frame_id)
        # rightmost segment whose start <= t
        idx = bisect.bisect_right(starts, t) - 1
        if idx < 0:
            idx = 0
        end, seg_id, caption = rows[idx]
        return {"segment_id": seg_id, "caption": caption, "timestamp": t, "in_range": starts[idx] <= t <= end}

    def hydrate(self, hit: dict) -> dict:
        seg = self.segment_for(hit["video_id"], int(hit["frame_id"]))
        if seg:
            hit = {**hit, "segment_id": seg["segment_id"], "caption": seg["caption"], "timestamp": seg["timestamp"]}
        return hit
