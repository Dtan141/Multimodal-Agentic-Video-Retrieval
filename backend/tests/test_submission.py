"""Pure unit tests for submission CSV building (no index/models needed -> runs in CI)."""

from __future__ import annotations

from aic_retrieval.submission import build_submission_csv, normalize_video_name


def test_normalize_video_name():
    assert normalize_video_name("L21_V001.mp4") == "L21_V001"
    assert normalize_video_name("  L21_V001  ") == "L21_V001"


def test_kis_ok():
    r = build_submission_csv("KIS", [{"video_name": "L21_V001.mp4", "frame_id": 100}])
    assert r["errors"] == []
    assert r["csv"].strip() == "L21_V001,100"
    assert r["row_count"] == 1


def test_qa_answer_length_limit():
    r = build_submission_csv("Q&A", [{"video_name": "V", "frame_id": 1, "answer": "x" * 101}])
    assert r["csv"] is None
    assert r["errors"]


def test_qa_ok_with_comma_is_quoted():
    r = build_submission_csv("Q&A", [{"video_name": "V", "frame_id": 1, "answer": "a, b"}])
    assert r["errors"] == []
    assert '"a, b"' in r["csv"]


def test_trake_frames_and_ordering_warning():
    r = build_submission_csv(
        "TRAKE",
        [{"video_name": "V", "frame_1": 30, "frame_2": 10}],
        event_count=2,
    )
    assert r["csv"].strip() == "V,30,10"
    assert r["warnings"]  # not increasing


def test_missing_video_name_errors():
    r = build_submission_csv("KIS", [{"frame_id": 5}])
    assert r["csv"] is None
    assert r["errors"]


def test_row_limit():
    rows = [{"video_name": "V", "frame_id": i} for i in range(101)]
    r = build_submission_csv("KIS", rows)
    assert r["csv"] is None
    assert any("exceeds" in e for e in r["errors"])


def test_empty_table():
    r = build_submission_csv("KIS", [])
    assert r["csv"] is None
