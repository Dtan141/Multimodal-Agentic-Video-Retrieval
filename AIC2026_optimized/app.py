from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from config import (
    DEFAULT_VISIBLE_RESULTS,
    EMBEDDINGS_DIR,
    GRID_COLUMNS,
    HF_IMAGE_REPO_ID,
    HF_VIDEO_REPOS,
    MAX_SUBMISSION_ROWS,
    MODEL_ID,
    SEARCH_INDEX_DIR,
    TEXT_MAX_LENGTH,
    TOP_K,
    USE_FP16,
)
from hf_assets import build_hf_image_url, fetch_images_parallel, get_hf_auth_uncached
from retrieval_engine import RetrievalEngine, build_engine_from_raw, load_compiled_index
from siglip2_embedder import Siglip2Embedder
from submission import (
    blank_answer_dataframe,
    build_submission_csv,
    generate_range_dataframe,
    normalize_csv_filename,
)
from video_player import (
    download_video_from_hf_cached,
    format_seconds,
    get_video_info,
    normalize_video_name,
    render_custom_video_browser,
)

st.set_page_config(page_title="AIC Retrieval + Video Inspector", page_icon="🔎", layout="wide")
st.title("AIC — Retrieval & Submission Helper")


@st.cache_resource(show_spinner=False)
def get_embedder(model_id: str, use_fp16: bool, text_max_length: int) -> Siglip2Embedder:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Search app only needs text encoding. The offline embedding pipeline can still
    # construct Siglip2Embedder with the default load_image_processor=True.
    return Siglip2Embedder(
        model_id=model_id,
        device=device,
        use_fp16=use_fp16,
        text_max_length=text_max_length,
        load_image_processor=False,
    )


@st.cache_resource(show_spinner=False)
def get_hf_auth() -> tuple[str, str]:
    return get_hf_auth_uncached()


@st.cache_resource(show_spinner=False)
def get_retrieval_engine(embeddings_dir_str: str, index_dir_str: str) -> RetrievalEngine:
    embeddings_dir = Path(embeddings_dir_str)
    index_dir = Path(index_dir_str)
    if (index_dir / "vectors.npy").exists() and (index_dir / "metadata.jsonl").exists():
        return load_compiled_index(index_dir)
    return build_engine_from_raw(embeddings_dir)


def encode_query(embedder: Siglip2Embedder, query: str) -> np.ndarray:
    features = embedder.encode_texts([query])
    if isinstance(features, torch.Tensor):
        features = features.detach().float().cpu().numpy()
    arr = np.asarray(features, dtype=np.float32)
    return arr.reshape(1, -1)[0]


def make_results(engine: RetrievalEngine, ranked: list[tuple[int, float]]) -> list[dict]:
    results: list[dict] = []
    for rank, (idx, score) in enumerate(ranked, start=1):
        item = dict(engine.metadata[idx])
        item["_rank"] = rank
        item["_score"] = score
        item["_image_url"] = build_hf_image_url(item)
        results.append(item)
    return results


def render_image_grid(results: list[dict], token: str) -> None:
    urls = [str(item.get("_image_url") or "") for item in results]
    with st.spinner(f"Đang tải {len(results)} keyframe (song song)..."):
        image_map = fetch_images_parallel(urls, token)

    for start in range(0, len(results), GRID_COLUMNS):
        cols = st.columns(GRID_COLUMNS)
        for col, item in zip(cols, results[start:start + GRID_COLUMNS]):
            with col:
                image_url = item.get("_image_url")
                payload = image_map.get(str(image_url)) if image_url else None
                if isinstance(payload, bytes):
                    st.image(payload, width="stretch")
                elif isinstance(payload, Exception):
                    st.warning(f"Không đọc được ảnh: {type(payload).__name__}")
                else:
                    st.warning("Không xác định được HF image URL.")

                st.markdown(f"**#{item['_rank']} · {item['_score']:.4f}**")
                if item.get("video_id"):
                    st.caption(f"video: {item['video_id']}")
                if item.get("frame_id") is not None:
                    st.caption(f"frame: {item['frame_id']}")
                st.caption(f"source: {item.get('_source_npz', '')}")


@st.fragment
def render_retrieval_tab() -> None:
    st.subheader("Global SigLIP2 Retrieval")

    compiled_ready = (SEARCH_INDEX_DIR / "vectors.npy").exists() and (SEARCH_INDEX_DIR / "metadata.jsonl").exists()
    if compiled_ready:
        st.caption("Search dùng compiled index đã normalize sẵn; mỗi query chỉ encode text + dot product Top-K.")
    else:
        st.warning(
            "Chưa có compiled index. App vẫn chạy từ các NPZ nhưng lần load đầu sẽ chậm. "
            "Chạy `python build_search_index.py` một lần để tối ưu."
        )

    if not EMBEDDINGS_DIR.exists() and not compiled_ready:
        st.error(f"Không tồn tại embedding source: {EMBEDDINGS_DIR}")
        return

    query = st.text_input("Query", placeholder="Ví dụ: một người đứng trước màn hình", key="retrieval_query_input")
    search = st.button("Search Top 100", type="primary", width="stretch", key="retrieval_search_btn")

    if search:
        query = query.strip()
        if not query:
            st.warning("Hãy nhập query.")
        else:
            try:
                token, username = get_hf_auth()
                t0 = time.perf_counter()
                engine = get_retrieval_engine(str(EMBEDDINGS_DIR), str(SEARCH_INDEX_DIR))
                t1 = time.perf_counter()
                embedder = get_embedder(MODEL_ID, USE_FP16, TEXT_MAX_LENGTH)
                t2 = time.perf_counter()
                query_vector = encode_query(embedder, query)
                t3 = time.perf_counter()
                ranked = engine.search(query_vector, TOP_K)
                t4 = time.perf_counter()
                results = make_results(engine, ranked)

                st.session_state["retrieval_result_bundle"] = {
                    "query": query,
                    "results": results,
                    "vector_count": engine.size,
                    "dim": engine.dim,
                    "index_source": engine.source,
                    "npz_files": len(engine.summary),
                    "hf_username": username,
                    "timing": {
                        "index_ms": (t1 - t0) * 1000,
                        "model_ms": (t2 - t1) * 1000,
                        "encode_ms": (t3 - t2) * 1000,
                        "search_ms": (t4 - t3) * 1000,
                    },
                }
                st.session_state["retrieval_visible_count"] = DEFAULT_VISIBLE_RESULTS
            except Exception as exc:
                st.error(f"Retrieval lỗi: {exc}")
                return

    bundle = st.session_state.get("retrieval_result_bundle")
    if not bundle:
        st.info("Nhập query rồi bấm Search. Model và index được cache resource sau lần đầu.")
        return

    results = bundle["results"]
    visible_count = min(
        max(DEFAULT_VISIBLE_RESULTS, int(st.session_state.get("retrieval_visible_count", DEFAULT_VISIBLE_RESULTS))),
        len(results),
    )

    st.subheader(f'Top {len(results)} — "{bundle["query"]}"')
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Vectors", f"{bundle['vector_count']:,}")
    c2.metric("Dim", bundle["dim"])
    c3.metric("Encode", f"{bundle['timing']['encode_ms']:.0f} ms")
    c4.metric("Search", f"{bundle['timing']['search_ms']:.1f} ms")
    c5.metric("Best score", f"{results[0]['_score']:.4f}")
    st.caption(
        f"Index: {bundle['index_source']} · HF: {bundle['hf_username']} · "
        f"device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
    )

    with st.expander("Top 100 table"):
        st.dataframe([
            {
                "rank": x["_rank"], "score": round(x["_score"], 6),
                "video_id": x.get("video_id", ""), "frame_id": x.get("frame_id", ""),
                "global_frame_id": x.get("global_frame_id", x.get("_frame_key", "")),
                "source_npz": x.get("_source_npz", ""),
            }
            for x in results
        ], width="stretch", hide_index=True)

    loaded_count = min(visible_count, len(results))
    remaining = len(results) - loaded_count
    st.caption(f"Đang hiển thị **{loaded_count}/{len(results)}** kết quả")
    b10, b20, ball = st.columns(3)
    if b10.button("+10", width="stretch", disabled=remaining <= 0, key="retrieval_load_10"):
        visible_count = min(loaded_count + 10, len(results))
        st.session_state["retrieval_visible_count"] = visible_count
    if b20.button("+20", width="stretch", disabled=remaining <= 0, key="retrieval_load_20"):
        visible_count = min(loaded_count + 20, len(results))
        st.session_state["retrieval_visible_count"] = visible_count
    if ball.button("All", width="stretch", disabled=remaining <= 0, key="retrieval_load_all"):
        visible_count = len(results)
        st.session_state["retrieval_visible_count"] = visible_count

    st.divider()
    try:
        token, _ = get_hf_auth()
        render_image_grid(results[:visible_count], token)
    except Exception as exc:
        st.error(f"Không tải được keyframe từ Hugging Face: {exc}")


def render_video_inspector() -> tuple[str | None, dict | None]:
    st.subheader("Kiểm tra video & tạo CSV đáp án")
    st.markdown("### 1. Cửa sổ kiểm tra video")

    top1, top2 = st.columns([3, 1])
    with top1:
        video_input = st.text_input(
            "Tên video",
            value=st.session_state.get("inspected_video_id", ""),
            placeholder="Ví dụ: L30_V001 hoặc L30_V001.mp4",
            key="video_inspector_input",
        )
    with top2:
        st.write("")
        st.write("")
        open_video = st.button("Mở video", type="primary", width="stretch", key="open_video_btn")

    with st.expander("Nâng cao: nếu app không tự tìm thấy video trên Hugging Face"):
        preferred_repo = st.selectbox("Ưu tiên repo video", options=[repo_id for repo_id, _ in HF_VIDEO_REPOS], key="preferred_video_repo")
        exact_hf_path = st.text_input("HF path chính xác", placeholder="Ví dụ: Videos_L30/L30_V001.mp4", key="exact_hf_video_path")

    if open_video:
        video_id = normalize_video_name(video_input)
        if not video_id:
            st.warning("Hãy nhập tên video.")
        else:
            try:
                token, _ = get_hf_auth()
                with st.spinner(f"Đang tìm / tải {video_id}.mp4 ..."):
                    local_path, repo_id, path_in_repo = download_video_from_hf_cached(video_id, token, exact_hf_path, preferred_repo)
                    info = get_video_info(local_path)
                st.session_state.update({
                    "inspected_video_id": video_id,
                    "inspected_video_path": local_path,
                    "inspected_video_repo": repo_id,
                    "inspected_video_hf_path": path_in_repo,
                    "inspected_video_info": info,
                    "submission_video_name": video_id,
                })
            except Exception as exc:
                st.error(str(exc))

    video_path = st.session_state.get("inspected_video_path")
    video_id = st.session_state.get("inspected_video_id")
    info = st.session_state.get("inspected_video_info")

    if video_path and video_id and info and Path(video_path).exists():
        if int(info["frame_count"]) <= 0:
            st.error("Không đọc được tổng số frame của video.")
        else:
            render_custom_video_browser(str(video_path), str(video_id), info, initial_frame=0)
            st.caption(
                f"{video_id} · {float(info['fps']):.3f} FPS · {int(info['frame_count']):,} frames · "
                f"{format_seconds(float(info['duration']))} · hover preview chạy trong browser; seek thật khi thả thanh tua"
            )
    else:
        st.info("Chưa mở video. Nhập video ID rồi bấm **Mở video**.")
    return video_id, info


@st.fragment
def render_csv_builder(default_video: str = "") -> None:
    st.divider()
    st.markdown("### 2. Tạo file CSV đáp án")

    q1, q2, q3 = st.columns([1, 1.5, 1])
    with q1:
        query_type = st.selectbox("Loại file", options=["KIS", "Q&A", "TRAKE"], key="submission_query_type")
    with q2:
        current_default = st.session_state.get("submission_video_name", default_video or "")
        video_name = st.text_input("Video name dùng cho CSV", value=current_default, placeholder="L30_V001", key="submission_video_name_widget")
        st.session_state["submission_video_name"] = video_name
    with q3:
        event_count = 1
        if query_type == "TRAKE":
            event_count = int(st.number_input("Số events", min_value=1, max_value=20, value=int(st.session_state.get("trake_event_count", 4)), step=1, key="trake_event_count"))
        else:
            st.caption("KIS: video,frame\n\nQ&A: video,frame,answer")

    start_frame, end_frame, frame_step = 0, 0, 1
    if query_type != "TRAKE":
        r1, r2, r3 = st.columns(3)
        with r1:
            start_frame = int(st.number_input("Frame start", min_value=0, value=int(st.session_state.get("answer_start_frame", 0)), step=1, key="answer_start_frame"))
        with r2:
            end_frame = int(st.number_input("Frame end", min_value=0, value=int(st.session_state.get("answer_end_frame", start_frame)), step=1, key="answer_end_frame"))
        with r3:
            frame_step = int(st.number_input("Khoảng cách frame", min_value=1, value=int(st.session_state.get("answer_frame_step", 30)), step=1, key="answer_frame_step"))

    default_answer = ""
    if query_type == "Q&A":
        default_answer = st.text_input("Answer mặc định", value=st.session_state.get("default_qa_answer", ""), max_chars=100, key="default_qa_answer")

    mode_key = f"{query_type}:{event_count}"
    if st.session_state.get("answer_editor_mode") != mode_key:
        st.session_state["answer_editor_mode"] = mode_key
        st.session_state["answer_editor_df"] = blank_answer_dataframe(query_type, event_count, video_name)
        st.session_state["answer_editor_version"] = int(st.session_state.get("answer_editor_version", 0)) + 1

    if query_type != "TRAKE":
        g1, g2 = st.columns([2, 1])
        generate = g1.button("Tạo bảng từ start → end → khoảng cách", type="primary", width="stretch", key="generate_answer_table")
        clear = g2.button("Xóa bảng", width="stretch", key="clear_answer_table")
        if generate:
            generated_df, message = generate_range_dataframe(query_type, video_name, start_frame, end_frame, frame_step, default_answer)
            if generated_df is None:
                st.error(message or "Không tạo được bảng.")
            else:
                st.session_state["answer_editor_df"] = generated_df
                st.session_state["answer_editor_version"] = int(st.session_state.get("answer_editor_version", 0)) + 1
                if message:
                    st.warning(message)
        if clear:
            st.session_state["answer_editor_df"] = blank_answer_dataframe(query_type, event_count, video_name)
            st.session_state["answer_editor_version"] = int(st.session_state.get("answer_editor_version", 0)) + 1
    else:
        st.caption("TRAKE: chỉ chọn **Số events**, sau đó nhập thủ công các Frame ID trong bảng.")
        if st.button("Xóa bảng TRAKE", width="stretch", key="clear_trake_table"):
            st.session_state["answer_editor_df"] = blank_answer_dataframe(query_type, event_count, video_name)
            st.session_state["answer_editor_version"] = int(st.session_state.get("answer_editor_version", 0)) + 1

    editor_df = st.session_state.get("answer_editor_df")
    if not isinstance(editor_df, pd.DataFrame):
        editor_df = blank_answer_dataframe(query_type, event_count, video_name)

    column_config: dict[str, object] = {
        "video_name": st.column_config.TextColumn("video_name", help="Tên video không có .mp4")
    }
    if query_type in {"KIS", "Q&A"}:
        column_config["frame_id"] = st.column_config.NumberColumn("frame_id", min_value=0, step=1, format="%d")
    if query_type == "Q&A":
        column_config["answer"] = st.column_config.TextColumn("answer", help="Tối đa 100 ký tự", max_chars=100)
    if query_type == "TRAKE":
        for i in range(event_count):
            column_config[f"frame_{i + 1}"] = st.column_config.NumberColumn(f"frame_{i + 1}", min_value=0, step=1, format="%d")

    editor_key = f"answer_editor_{mode_key}_{int(st.session_state.get('answer_editor_version', 0))}"
    edited_df = st.data_editor(editor_df, num_rows="dynamic", hide_index=True, width="stretch", column_config=column_config, key=editor_key)
    st.session_state["answer_editor_df"] = edited_df

    file_default = {"KIS": "default-kis.csv", "Q&A": "default-qa.csv", "TRAKE": "default-trake.csv"}[query_type]
    filename_key = f"csv_filename_{query_type}"
    if filename_key not in st.session_state:
        st.session_state[filename_key] = file_default
    st.text_input("Tên file CSV", key=filename_key)
    filename = normalize_csv_filename(st.session_state[filename_key], query_type)

    csv_bytes, errors, warnings, row_count = build_submission_csv(edited_df, query_type, event_count)
    c1, c2 = st.columns([1, 3])
    c1.metric("Số dòng", f"{row_count}/{MAX_SUBMISSION_ROWS}")
    c2.caption(
        "Format: video_name,frame_id" if query_type == "KIS"
        else "Format: video_name,frame_id,answer" if query_type == "Q&A"
        else f"Format: video_name + {event_count} frame IDs theo thứ tự events"
    )
    for warning in warnings:
        st.warning(warning)
    for error in errors:
        st.error(error)

    if csv_bytes is not None:
        with st.expander("Preview CSV thuần túy", expanded=True):
            st.code(csv_bytes.decode("utf-8"), language="text")
        st.download_button("Xuất file CSV", data=csv_bytes, file_name=filename, mime="text/csv; charset=utf-8", type="primary", width="stretch", key="download_submission_csv")


def render_video_csv_tab() -> None:
    video_id, _ = render_video_inspector()
    # CSV controls live in a fragment. Editing the table does not recreate/reset the video component above.
    render_csv_builder(default_video=str(video_id or ""))


PAGE_RETRIEVAL = "🔎 Retrieval"
PAGE_VIDEO = "🎬 Kiểm tra video + CSV"
if "main_nav" not in st.session_state:
    st.session_state["main_nav"] = PAGE_RETRIEVAL

page = st.segmented_control("Chức năng", options=[PAGE_RETRIEVAL, PAGE_VIDEO], key="main_nav", width="stretch")
if page == PAGE_RETRIEVAL:
    render_retrieval_tab()
elif page == PAGE_VIDEO:
    render_video_csv_tab()
