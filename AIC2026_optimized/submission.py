from __future__ import annotations

import csv
import io

import numpy as np
import pandas as pd

from config import MAX_SUBMISSION_ROWS
from video_player import normalize_video_name


def blank_answer_dataframe(query_type: str, event_count: int, video_name: str = "") -> pd.DataFrame:
    video_name = normalize_video_name(video_name)
    if query_type == "KIS":
        return pd.DataFrame([{"video_name": video_name, "frame_id": None}])
    if query_type == "Q&A":
        return pd.DataFrame([{"video_name": video_name, "frame_id": None, "answer": ""}])
    row: dict[str, object] = {"video_name": video_name}
    for i in range(event_count):
        row[f"frame_{i + 1}"] = None
    return pd.DataFrame([row])


def generate_range_dataframe(query_type: str, video_name: str, start_frame: int, end_frame: int, frame_step: int, default_answer: str) -> tuple[pd.DataFrame | None, str | None]:
    video_name = normalize_video_name(video_name)
    if not video_name:
        return None, "Hãy nhập tên video trước khi tạo bảng."
    if frame_step <= 0:
        return None, "Khoảng cách frame phải > 0."
    if end_frame < start_frame:
        return None, "Frame end phải >= frame start."

    frames = list(range(int(start_frame), int(end_frame) + 1, int(frame_step)))
    if not frames:
        return None, "Không tạo được frame nào."
    truncated = len(frames) > MAX_SUBMISSION_ROWS
    frames = frames[:MAX_SUBMISSION_ROWS]
    rows = []
    for frame in frames:
        row = {"video_name": video_name, "frame_id": int(frame)}
        if query_type == "Q&A":
            row["answer"] = default_answer
        rows.append(row)
    msg = f"Chỉ giữ {MAX_SUBMISSION_ROWS} dòng đầu." if truncated else None
    return pd.DataFrame(rows), msg


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return isinstance(value, str) and value == ""


def _parse_frame_int(value: object) -> int:
    if _is_blank(value):
        raise ValueError("frame rỗng")
    if isinstance(value, (int, np.integer)):
        frame = int(value)
    elif isinstance(value, (float, np.floating)):
        if not float(value).is_integer():
            raise ValueError(f"frame không phải số nguyên: {value}")
        frame = int(value)
    else:
        number = float(str(value).strip())
        if not number.is_integer():
            raise ValueError(f"frame không phải số nguyên: {value}")
        frame = int(number)
    if frame < 0:
        raise ValueError(f"frame âm: {frame}")
    return frame


def build_submission_csv(df: pd.DataFrame, query_type: str, event_count: int) -> tuple[bytes | None, list[str], list[str], int]:
    errors: list[str] = []
    warnings: list[str] = []
    output_rows: list[list[object]] = []
    records = df.to_dict("records")
    non_empty = [row for row in records if any(not _is_blank(v) for v in row.values())]

    if len(non_empty) > MAX_SUBMISSION_ROWS:
        return None, [f"File có {len(non_empty)} dòng, vượt giới hạn {MAX_SUBMISSION_ROWS}."], warnings, len(non_empty)

    for row_idx, row in enumerate(non_empty, start=1):
        video_name = normalize_video_name(str(row.get("video_name", "")))
        if not video_name:
            errors.append(f"Dòng {row_idx}: thiếu video_name.")
            continue
        try:
            if query_type == "KIS":
                output_rows.append([video_name, _parse_frame_int(row.get("frame_id"))])
            elif query_type == "Q&A":
                frame = _parse_frame_int(row.get("frame_id"))
                answer = "" if row.get("answer") is None else str(row.get("answer"))
                if not answer:
                    errors.append(f"Dòng {row_idx}: answer đang rỗng.")
                    continue
                if len(answer) > 100:
                    errors.append(f"Dòng {row_idx}: answer vượt 100 ký tự.")
                    continue
                output_rows.append([video_name, frame, answer])
            else:
                frames = []
                bad = False
                for i in range(event_count):
                    col = f"frame_{i + 1}"
                    try:
                        frames.append(_parse_frame_int(row.get(col)))
                    except Exception as exc:
                        errors.append(f"Dòng {row_idx}: {col} không hợp lệ ({exc}).")
                        bad = True
                if bad:
                    continue
                if frames != sorted(frames):
                    warnings.append(f"Dòng {row_idx}: frame TRAKE chưa tăng theo thời gian ({frames}).")
                output_rows.append([video_name, *frames])
        except Exception as exc:
            errors.append(f"Dòng {row_idx}: {exc}.")

    if not non_empty:
        errors.append("Bảng đáp án đang trống.")
    if errors:
        return None, errors, warnings, len(non_empty)

    sio = io.StringIO(newline="")
    csv.writer(sio, delimiter=",", lineterminator="\n", quoting=csv.QUOTE_MINIMAL).writerows(output_rows)
    return sio.getvalue().encode("utf-8"), errors, warnings, len(output_rows)


def normalize_csv_filename(filename: str, query_type: str) -> str:
    suffix = {"KIS": "kis", "Q&A": "qa", "TRAKE": "trake"}[query_type]
    name = str(filename or "").strip()
    if not name:
        return f"query-1-{suffix}.csv"
    return name if name.lower().endswith(".csv") else name + ".csv"
