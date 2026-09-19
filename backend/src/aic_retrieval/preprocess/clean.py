"""Clean the raw metadata into a slim, junk-filtered dataset.

Applied transforms:
  * speech: drop ``keep==false`` items; keep only {text_norm, keywords, entities}
    (drops text_raw/flags/keep/start_time/end_time/model/ner noise).
  * ocr: remove stoplist tokens (per-folder auto set + curated global brands /
    OCR-model artifacts); drop entries that become empty.
  * object: kept as-is (boxes are useful for the UI later).

The merged stoplist is written to ``<preprocess>/ocr_stoplist.json`` with a
``_global`` curated key plus per-folder auto keys, so it can be reviewed/edited
and re-applied.
"""

from __future__ import annotations

import json
from pathlib import Path

from aic_retrieval.logging_conf import get_logger
from aic_retrieval.preprocess.analyze import folder_of
from aic_retrieval.text.bm25_index import tokenize

logger = get_logger(__name__)

# Curated global stoplist: broadcaster watermarks (+ OCR misread variants),
# sponsors, and OCR-model JSON artifacts that leaked into the ocr field.
CURATED_GLOBAL: list[str] = [
    # channel / broadcaster watermarks
    "htv", "htv9", "htvonline", "online", "hd", "tv", "vn", "mtv", "mekong",
    "tuoitre", "tuoitretv", "tuổitretv", "tuổiretv", "tuoitretv",
    # sponsors
    "herbalife",
    # OCR-model JSON artifacts
    "json", "text_content", "bbox_2d", "bbox", "objects", "object",
    "label", "score", "region", "confidence",
]


def merge_stoplist(auto_stoplist: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {"_global": sorted({t.lower() for t in CURATED_GLOBAL})}
    for folder, toks in auto_stoplist.items():
        merged[folder] = sorted({t.lower() for t in toks})
    return merged


def _stopset_for(video_id: str, stoplist: dict[str, list[str]]) -> set[str]:
    s = set(stoplist.get("_global", []))
    s.update(stoplist.get(folder_of(video_id), []))
    return s


def _clean_speech(segment: dict) -> list[dict]:
    out: list[dict] = []
    for sp in segment.get("speech", []) or []:
        if not isinstance(sp, dict):
            continue
        if not sp.get("keep", True):  # drop flagged-junk speech; keep if no flag
            continue
        item: dict = {}
        if sp.get("text_norm"):
            item["text_norm"] = sp["text_norm"]
        if sp.get("keywords"):
            item["keywords"] = sp["keywords"]
        if sp.get("entities"):
            item["entities"] = sp["entities"]
        if item:
            out.append(item)
    return out


def _clean_ocr(value: dict, stopset: set[str]) -> tuple[list[str], int]:
    ocr = value.get("ocr") or []
    if isinstance(ocr, str):
        ocr = [ocr]
    cleaned: list[str] = []
    removed = 0
    for s in ocr:
        toks = tokenize(str(s))
        kept = [t for t in toks if t not in stopset]
        removed += len(toks) - len(kept)
        if kept:
            cleaned.append(" ".join(kept))
    return cleaned, removed


def clean_metadata(
    metadata_dir: Path,
    clean_dir: Path,
    stoplist: dict[str, list[str]],
) -> dict:
    files = sorted(metadata_dir.rglob("Videos_*/*.json"))
    if not files:
        raise FileNotFoundError(f"No Videos_*/*.json under {metadata_dir}")

    n_videos = 0
    speech_kept = speech_dropped = ocr_tokens_removed = 0

    for jf in files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skip %s: %s", jf.name, exc)
            continue
        vid = data.get("video_id") or jf.stem
        stopset = _stopset_for(vid, stoplist)
        n_videos += 1

        clean = {k: data.get(k) for k in ("video_id", "type", "video_path",
                                          "keyframes_folder_path", "metadata_path", "fps")}
        clean_segments = []
        for seg in data.get("segments", []):
            raw_speech = seg.get("speech", []) or []
            speech = _clean_speech(seg)
            speech_kept += len(speech)
            speech_dropped += len(raw_speech) - len(speech)

            keyframes = {}
            for name, val in (seg.get("keyframe") or {}).items():
                if not isinstance(val, dict):
                    continue
                ocr, removed = _clean_ocr(val, stopset)
                ocr_tokens_removed += removed
                keyframes[name] = {"object": val.get("object"), "ocr": ocr}

            clean_segments.append(
                {
                    "segment_id": seg.get("segment_id"),
                    "start_time": seg.get("start_time"),
                    "end_time": seg.get("end_time"),
                    "segment_caption": seg.get("segment_caption"),
                    "speech": speech,
                    "keyframe": keyframes,
                }
            )
        clean["segments"] = clean_segments

        # mirror the Videos_Lxx/<video>.json layout under clean_dir
        rel = jf.relative_to(metadata_dir)
        dest = clean_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(clean, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Cleaned %d videos -> %s | speech kept=%d dropped=%d | ocr tokens removed=%d",
        n_videos, clean_dir, speech_kept, speech_dropped, ocr_tokens_removed,
    )
    return {
        "videos": n_videos,
        "speech_kept": speech_kept,
        "speech_dropped": speech_dropped,
        "ocr_tokens_removed": ocr_tokens_removed,
    }
