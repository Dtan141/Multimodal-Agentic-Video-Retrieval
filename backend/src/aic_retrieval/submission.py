"""Competition submission CSV (KIS / Q&A / TRAKE).

Ported from ``AIC2026_retrieval_pipeline.py:1253-1333`` to operate on plain
row dicts (so the API/frontend can drive it) instead of a Streamlit DataFrame.
Output: UTF-8, comma-delimited, no header, LF, max 100 rows.
"""

from __future__ import annotations

import csv
import io

MAX_SUBMISSION_ROWS = 100
QUERY_TYPES = ("KIS", "Q&A", "TRAKE")


def normalize_video_name(video_name: str) -> str:
    name = str(video_name or "").strip()
    if name.lower().endswith(".mp4"):
        name = name[:-4]
    return name


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _parse_frame_int(value) -> int:
    if _is_blank(value):
        raise ValueError("empty frame")
    if isinstance(value, bool):
        raise ValueError("invalid frame")
    if isinstance(value, int):
        frame = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"frame not an integer: {value}")
        frame = int(value)
    else:
        number = float(str(value).strip())
        if not number.is_integer():
            raise ValueError(f"frame not an integer: {value}")
        frame = int(number)
    if frame < 0:
        raise ValueError(f"negative frame: {frame}")
    return frame


def build_submission_csv(
    query_type: str,
    rows: list[dict],
    event_count: int = 1,
) -> dict:
    """Validate rows and build the CSV. Returns dict with csv/errors/warnings/row_count."""
    if query_type not in QUERY_TYPES:
        return {"csv": None, "errors": [f"Unknown query_type {query_type!r}"], "warnings": [], "row_count": 0}

    errors: list[str] = []
    warnings: list[str] = []
    output_rows: list[list] = []

    non_empty = [r for r in rows if any(not _is_blank(v) for v in r.values())]
    if not non_empty:
        return {"csv": None, "errors": ["Empty answer table."], "warnings": [], "row_count": 0}
    if len(non_empty) > MAX_SUBMISSION_ROWS:
        return {
            "csv": None,
            "errors": [f"{len(non_empty)} rows exceeds limit {MAX_SUBMISSION_ROWS}."],
            "warnings": [],
            "row_count": len(non_empty),
        }

    for i, row in enumerate(non_empty, start=1):
        video_name = normalize_video_name(str(row.get("video_name", "")))
        if not video_name:
            errors.append(f"Row {i}: missing video_name.")
            continue
        try:
            if query_type == "KIS":
                output_rows.append([video_name, _parse_frame_int(row.get("frame_id"))])
            elif query_type == "Q&A":
                frame = _parse_frame_int(row.get("frame_id"))
                answer = "" if row.get("answer") is None else str(row.get("answer"))
                if answer == "":
                    errors.append(f"Row {i}: empty answer.")
                    continue
                if len(answer) > 100:
                    errors.append(f"Row {i}: answer {len(answer)} chars > 100.")
                    continue
                output_rows.append([video_name, frame, answer])
            else:  # TRAKE
                frames = []
                bad = False
                for e in range(event_count):
                    try:
                        frames.append(_parse_frame_int(row.get(f"frame_{e + 1}")))
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"Row {i}: frame_{e + 1} invalid ({exc}).")
                        bad = True
                if bad:
                    continue
                if frames != sorted(frames):
                    warnings.append(f"Row {i}: TRAKE frames not increasing ({frames}).")
                output_rows.append([video_name, *frames])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Row {i}: {exc}.")

    if errors:
        return {"csv": None, "errors": errors, "warnings": warnings, "row_count": len(non_empty)}

    sio = io.StringIO(newline="")
    writer = csv.writer(sio, delimiter=",", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerows(output_rows)
    return {"csv": sio.getvalue(), "errors": [], "warnings": warnings, "row_count": len(output_rows)}
