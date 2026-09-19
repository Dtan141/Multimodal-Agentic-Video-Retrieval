"""Per-folder frequency analysis to detect junk OCR/ASR terms.

Each ``Videos_Lxx`` folder groups videos with a shared theme, so tokens that
appear across *most videos of a folder* are almost always boilerplate — channel
logos/watermarks (e.g. "htv"), on-screen furniture — not content. We rank OCR
tokens by **video coverage** (fraction of the folder's videos they appear in)
and suggest a per-folder stoplist. Read-only: writes a report + suggested
stoplist for review; applying it happens in ``clean.py``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from aic_retrieval.logging_conf import get_logger
from aic_retrieval.text.bm25_index import tokenize

logger = get_logger(__name__)


def folder_of(video_id: str) -> str:
    return str(video_id).split("_")[0]


def _entity_text(e) -> str:
    if isinstance(e, str):
        return e
    if isinstance(e, dict):
        return e.get("text") or e.get("name") or e.get("entity") or ""
    return str(e)


def analyze(
    metadata_dir: Path,
    out_dir: Path,
    kf_fraction: float = 0.30,
    top_n: int = 40,
) -> tuple[dict, dict]:
    """Analyze OCR/object/ASR frequencies per folder.

    Suggested OCR stoplist = tokens whose *keyframe fraction* (share of the
    folder's OCR keyframes containing the token) >= ``kf_fraction``. That targets
    persistent overlays/watermarks (e.g. "htv") while leaving thematic Vietnamese
    content words alone (BM25 IDF already down-weights those).
    """
    files = sorted(metadata_dir.rglob("Videos_*/*.json"))
    if not files:
        raise FileNotFoundError(f"No Videos_*/*.json under {metadata_dir}")

    videos: dict[str, set] = defaultdict(set)
    ocr_kf: Counter = Counter()                       # folder -> #keyframes with ocr
    ocr_tok_kf: dict[str, Counter] = defaultdict(Counter)          # folder -> token -> #keyframes
    ocr_tok_vid: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    obj_kf: dict[str, Counter] = defaultdict(Counter)
    asr_term_ct: dict[str, Counter] = defaultdict(Counter)
    asr_term_vid: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))

    for jf in files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skip %s: %s", jf.name, exc)
            continue
        vid = data.get("video_id") or jf.stem
        fo = folder_of(vid)
        videos[fo].add(vid)

        for seg in data.get("segments", []):
            for _name, val in (seg.get("keyframe") or {}).items():
                if not isinstance(val, dict):
                    continue
                ocr = val.get("ocr") or []
                if isinstance(ocr, str):
                    ocr = [ocr]
                toks = set(tokenize(" ".join(str(x) for x in ocr)))
                if toks:
                    ocr_kf[fo] += 1
                for t in toks:
                    ocr_tok_kf[fo][t] += 1
                    ocr_tok_vid[fo][t].add(vid)
                obj = val.get("object") or {}
                if isinstance(obj, dict):
                    for o in obj.get("objects", []):
                        if isinstance(o, dict) and o.get("label"):
                            obj_kf[fo][o["label"]] += 1

            for sp in seg.get("speech", []) or []:
                if not isinstance(sp, dict):
                    continue
                terms = {str(k).lower() for k in (sp.get("keywords") or []) if str(k).strip()}
                for e in sp.get("entities") or []:
                    t = _entity_text(e).lower().strip()
                    if t:
                        terms.add(t)
                for t in terms:
                    asr_term_ct[fo][t] += 1
                    asr_term_vid[fo][t].add(vid)

    report: dict = {}
    stoplist: dict = {}
    for fo in sorted(videos):
        nvid = len(videos[fo])
        nkf = ocr_kf[fo] or 1
        ocr_rows = []
        for t, vids in ocr_tok_vid[fo].items():
            kf = ocr_tok_kf[fo][t]
            ocr_rows.append([t, kf, round(kf / nkf, 3), len(vids), round(len(vids) / nvid, 3)])
        # rank by keyframe fraction (persistence), then raw keyframe count
        ocr_rows.sort(key=lambda r: (r[2], r[1]), reverse=True)
        stop = [r[0] for r in ocr_rows if r[2] >= kf_fraction]
        stoplist[fo] = stop

        asr_terms = sorted(asr_term_vid[fo], key=lambda t: len(asr_term_vid[fo][t]), reverse=True)
        report[fo] = {
            "videos": nvid,
            "ocr_keyframes": ocr_kf[fo],
            "suggested_stop_count": len(stop),
            # [token, #keyframes, kf_fraction, #videos, video_coverage]
            "top_ocr": ocr_rows[:top_n],
            "top_objects": obj_kf[fo].most_common(top_n),
            "top_asr_terms": [
                [t, asr_term_ct[fo][t], len(asr_term_vid[fo][t])] for t in asr_terms[:top_n]
            ],
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frequency_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "ocr_stoplist.json").write_text(
        json.dumps(stoplist, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Wrote frequency_report.json + ocr_stoplist.json to %s", out_dir)
    return report, stoplist
