"""Parse per-video metadata JSON into 4 per-module BM25 indexes + hydration tables.

Modules:
  caption  - segment-level (segment_caption)     -> all keyframes of the segment
  asr      - segment-level (speech[].text_norm)   -> all keyframes of the segment
  ocr      - keyframe-level (keyframe.ocr)        -> that keyframe
  object   - keyframe-level (object.objects[].label) -> that keyframe

Also writes ``segments.parquet`` (video_id, segment_id, start/end, caption) and
``fps.json`` (video_id -> fps) so WS4 can hydrate frame_id -> timestamp -> segment.

Output goes under ``<index_dir>/text/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from aic_retrieval.logging_conf import get_logger
from aic_retrieval.text.bm25_index import ModuleBM25

logger = get_logger(__name__)

TEXT_SUBDIR = "text"
MODULES = ("caption", "ocr", "object", "asr")


def _frame_id_from_name(name: str) -> int | None:
    stem = Path(name).stem
    tail = stem.split("_")[-1]
    try:
        return int(tail)
    except ValueError:
        return None


def _object_labels(keyframe_value: dict) -> list[str]:
    obj = keyframe_value.get("object")
    if not isinstance(obj, dict):
        return []
    labels = [o.get("label") for o in obj.get("objects", []) if isinstance(o, dict)]
    # de-dupe preserving order
    return list(dict.fromkeys(lbl for lbl in labels if lbl))


def _ocr_text(keyframe_value: dict) -> str:
    ocr = keyframe_value.get("ocr") or []
    if isinstance(ocr, str):
        ocr = [ocr]
    return " ".join(str(x).replace("\n", " ").strip() for x in ocr if str(x).strip())


def _speech_text(segment: dict) -> str:
    parts: list[str] = []
    for sp in segment.get("speech", []) or []:
        if not isinstance(sp, dict):
            continue
        txt = sp.get("text_norm") or sp.get("text_raw") or ""
        if txt.strip():
            parts.append(txt.strip())
        parts.extend(str(k) for k in (sp.get("keywords") or []) if str(k).strip())
    return " ".join(parts)


def ingest_metadata(metadata_dir: Path, index_dir: Path) -> dict:
    json_files = sorted(metadata_dir.rglob("Videos_*/*.json"))
    if not json_files:
        raise FileNotFoundError(f"No Videos_*/*.json under {metadata_dir}")

    texts: dict[str, list[str]] = {m: [] for m in MODULES}
    refs: dict[str, list[list[dict]]] = {m: [] for m in MODULES}
    seg_rows: list[dict] = []
    fps_map: dict[str, float] = {}

    n_videos = 0
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skip %s: %s", jf.name, exc)
            continue

        vid = data.get("video_id") or jf.stem
        fps_map[vid] = float(data.get("fps") or 0.0)
        n_videos += 1

        for seg in data.get("segments", []):
            sid = seg.get("segment_id")
            start = seg.get("start_time")
            end = seg.get("end_time")
            caption = (seg.get("segment_caption") or "").strip()
            keyframes = seg.get("keyframe", {}) or {}

            frame_refs: list[dict] = []
            for name in keyframes:
                fid = _frame_id_from_name(name)
                if fid is not None:
                    frame_refs.append({"video_id": vid, "frame_id": fid, "segment_id": sid})

            seg_rows.append(
                {
                    "video_id": vid,
                    "segment_id": sid,
                    "start_time": start,
                    "end_time": end,
                    "caption": caption,
                }
            )

            # segment-level modules
            if caption and frame_refs:
                texts["caption"].append(caption)
                refs["caption"].append(frame_refs)
            speech = _speech_text(seg)
            if speech and frame_refs:
                texts["asr"].append(speech)
                refs["asr"].append(frame_refs)

            # keyframe-level modules
            for name, val in keyframes.items():
                if not isinstance(val, dict):
                    continue
                fid = _frame_id_from_name(name)
                if fid is None:
                    continue
                one = [{"video_id": vid, "frame_id": fid, "segment_id": sid}]
                ocr = _ocr_text(val)
                if ocr:
                    texts["ocr"].append(ocr)
                    refs["ocr"].append(one)
                labels = _object_labels(val)
                if labels:
                    texts["object"].append(" ".join(labels))
                    refs["object"].append(one)

    out_dir = index_dir / TEXT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for m in MODULES:
        idx = ModuleBM25.build(m, texts[m], refs[m])
        idx.save(out_dir)
        counts[m] = len(texts[m])
        logger.info("Module %-8s: %d documents", m, counts[m])

    pd.DataFrame(seg_rows).to_parquet(out_dir / "segments.parquet", index=False)
    (out_dir / "fps.json").write_text(json.dumps(fps_map), encoding="utf-8")

    logger.info("Ingested %d videos, %d segments", n_videos, len(seg_rows))
    return {"videos": n_videos, "segments": len(seg_rows), "module_docs": counts}
