from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import torch

from huggingface_hub import HfApi, get_token, hf_hub_download, hf_hub_url, snapshot_download
from siglip2_embedder import Siglip2Embedder

# ============================================================
# CONFIG
# ============================================================
CURRENT_DIR = Path(__file__).resolve().parent
BASE_WORKSPACE = CURRENT_DIR / "workspace"
EMBEDDINGS_DIR = BASE_WORKSPACE / "vector_embeddings"

HF_IMAGE_REPO_ID = "Chillguy2026/AIC_2026_data"
HF_IMAGE_REPO_TYPE = "dataset"
HF_IMAGE_ROOT = "keyframe_vector"
HF_CAPTION_METADATA_ROOT = "metadata"
HF_KEYFRAME_METADATA_ROOT = "keyframe_metadata"


# App sẽ thử lần lượt các repo này khi mở video.
# Nếu video hiện chỉ nằm ở một repo, bạn có thể xóa repo còn lại.
HF_VIDEO_REPOS: tuple[tuple[str, str], ...] = (
    ("Chillguy2026/dataset_video", "dataset"),
)

# Các thư mục local được ưu tiên kiểm tra trước khi tải từ HF.
LOCAL_VIDEO_ROOTS: tuple[Path, ...] = (
    CURRENT_DIR / "videos",
    CURRENT_DIR / "hf_download_videos",
    BASE_WORKSPACE / "videos",
)

MODEL_ID = "google/siglip2-base-patch16-224"
TEXT_MAX_LENGTH = 64
USE_FP16 = True
TOP_K = 100
GRID_COLUMNS = 5
DEFAULT_VISIBLE_RESULTS = 10
MAX_SUBMISSION_ROWS = 100

CAPTION_TOP_K_DEFAULT = 100
CAPTION_TOP_K_MAX = 100
CAPTION_DEFAULT_VISIBLE_RESULTS = 10

VIDEO_ASSET_REGISTRY: dict[str, dict] = {}
_VIDEO_HTTP_SERVER: ThreadingHTTPServer | None = None
_VIDEO_HTTP_PORT: int | None = None
_VIDEO_HTTP_LOCK = threading.Lock()


# ============================================================
# PAGE
# ============================================================
st.set_page_config(page_title="AIC Retrieval + Video Inspector", page_icon="🔎", layout="wide")
st.title("AIC — Retrieval & Submission Helper")

# ============================================================
# RETRIEVAL CAPTION - KEYWORD SEARCH
# ============================================================

# Tách token theo "từ" Unicode, bỏ punctuation và không phân biệt hoa/thường.
# Ví dụ: "A red-car!" -> ["a", "red", "car"]
_CAPTION_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def normalize_caption_text(text: str) -> str:
    """Chuẩn hóa text để kiểm tra exact phrase / hiển thị logic ổn định."""
    return " ".join(_CAPTION_WORD_RE.findall(str(text or "").lower()))


def tokenize_caption_words(text: str) -> list[str]:
    """Trả về danh sách từ lowercase. Matching là exact word, không phải substring."""
    return _CAPTION_WORD_RE.findall(str(text or "").lower())


def unique_words_keep_order(words: list[str]) -> list[str]:
    """Bỏ từ query bị lặp nhưng giữ thứ tự người dùng nhập."""
    return list(dict.fromkeys(words))


@st.cache_data(show_spinner=False)
def download_caption_metadata(repo_id: str, repo_type: str, token: str) -> str:
    """Chỉ tải JSON metadata; không tải keyframe_metadata ở bước search."""
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=[
            f"{HF_CAPTION_METADATA_ROOT}/*.json",
            f"{HF_CAPTION_METADATA_ROOT}/**/*.json",
        ],
        token=token,
    )

    metadata_dir = Path(snapshot_path) / HF_CAPTION_METADATA_ROOT

    if not metadata_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy metadata root: {HF_CAPTION_METADATA_ROOT}"
        )

    return str(metadata_dir)


@st.cache_resource
def load_caption_segments(metadata_root_str: str) -> tuple[list[dict], list[dict], dict[str, tuple[int, ...]]]:
    """Đọc JSON một lần và dựng inverted index: word -> segment indices.

    Caption search sau đó không cần SentenceTransformer/GPU/vector caption.
    Mỗi query chỉ tra posting list của các từ trong query.
    """
    metadata_root = Path(metadata_root_str)
    json_files = sorted(metadata_root.rglob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"Không tìm thấy file JSON trong {metadata_root}")

    segments: list[dict] = []
    summary: list[dict] = []
    word_index_mutable: dict[str, list[int]] = {}

    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            video_id = str(data.get("video_id", "")).strip()
            fps = float(data.get("fps", 0.0) or 0.0)
            video_path = str(data.get("video_path", "")).strip()
            keyframes_folder_path = str(data.get("keyframes_folder_path", "")).strip()
            raw_segments = data.get("segments", [])

            if not isinstance(raw_segments, list):
                summary.append({
                    "file": str(json_path.relative_to(metadata_root)),
                    "status": "skip: segments is not list",
                    "segments": 0,
                })
                continue

            kept = 0

            for segment in raw_segments:
                if not isinstance(segment, dict):
                    continue

                caption = str(segment.get("segment_caption", "") or "").strip()
                if not caption:
                    continue

                start_time = float(segment.get("start_time", 0.0) or 0.0)
                end_time = float(segment.get("end_time", start_time) or start_time)
                start_frame = int(round(start_time * fps)) if fps > 0 else None
                end_frame = int(round(end_time * fps)) if fps > 0 else None

                keyframe_data = segment.get("keyframe", {})
                keyframe_files = list(keyframe_data.keys()) if isinstance(keyframe_data, dict) else []
                segment_id = str(segment.get("segment_id", "")).strip()

                segment_index = len(segments)
                segments.append({
                    "video_id": video_id,
                    "segment_id": segment_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "fps": fps,
                    "video_path": video_path,
                    "keyframes_folder_path": keyframes_folder_path,
                    "keyframe_files": keyframe_files,
                    "segment_caption": caption,
                    "_metadata_file": str(json_path.relative_to(metadata_root)),
                    "_segment_key": f"{video_id}:{segment_id}",
                })

                # Mỗi word chỉ thêm segment_index một lần vào posting list.
                # Vì ranking chính là số TỪ QUERY KHÁC NHAU match, không phải số lần word lặp.
                for word in set(tokenize_caption_words(caption)):
                    word_index_mutable.setdefault(word, []).append(segment_index)

                kept += 1

            summary.append({
                "file": str(json_path.relative_to(metadata_root)),
                "status": "ok",
                "segments": kept,
            })

        except Exception as exc:
            summary.append({
                "file": str(json_path.relative_to(metadata_root)),
                "status": f"error: {exc}",
                "segments": 0,
            })

    if not segments:
        raise RuntimeError("Không tìm thấy segment nào có `segment_caption`.")

    # Tuple giúp cache ổn định và tránh vô tình mutate posting list khi search.
    word_index = {
        word: tuple(indices)
        for word, indices in word_index_mutable.items()
    }

    return segments, summary, word_index


def build_hf_segment_keyframe_url(item: dict, filename: str) -> str | None:
    keyframes_folder_path = str(item.get("keyframes_folder_path", "")).strip().strip("/")
    filename = str(filename or "").strip()

    if not keyframes_folder_path or not filename:
        return None

    # JSON thường lưu: Videos_L22/L22_V001_keyframes
    # Nếu metadata vô tình đã chứa prefix keyframe_metadata thì không thêm lần 2.
    prefix = f"{HF_KEYFRAME_METADATA_ROOT}/"
    if keyframes_folder_path.startswith(prefix):
        path_in_repo = f"{keyframes_folder_path}/{filename}"
    else:
        path_in_repo = f"{HF_KEYFRAME_METADATA_ROOT}/{keyframes_folder_path}/{filename}"

    return hf_hub_url(
        repo_id=HF_IMAGE_REPO_ID,
        filename=path_in_repo,
        repo_type=HF_IMAGE_REPO_TYPE,
    )


def search_caption_segments(query: str, segments: list[dict], word_index: dict[str, tuple[int, ...]], top_k: int) -> tuple[list[dict], list[str]]:
    """Keyword retrieval.

    Ranking bắt buộc theo thứ tự:
      1) _match_count: số TỪ QUERY KHÁC NHAU xuất hiện trong caption (cao -> thấp)
         => match 5 từ luôn xếp trên match 3 từ.
      2) _phrase_match: caption chứa nguyên cụm query sau normalize.
      3) _occurrence_count: tổng số lần các query words xuất hiện trong caption.
      4) caption ngắn hơn được ưu tiên nhẹ khi ba tiêu chí trên bằng nhau.

    Matching exact token, case-insensitive, punctuation-insensitive.
    """
    query_words = unique_words_keep_order(tokenize_caption_words(query))
    if not query_words:
        return [], []

    # Tính số query words khác nhau match mỗi segment bằng inverted index.
    match_counts: dict[int, int] = {}
    for word in query_words:
        for segment_idx in word_index.get(word, ()):
            match_counts[segment_idx] = match_counts.get(segment_idx, 0) + 1

    if not match_counts:
        return [], query_words

    normalized_query = " ".join(query_words)
    results: list[dict] = []

    for segment_idx, match_count in match_counts.items():
        base_item = segments[segment_idx]
        caption = str(base_item.get("segment_caption", "") or "")
        caption_tokens = tokenize_caption_words(caption)
        caption_token_set = set(caption_tokens)

        matched_words = [word for word in query_words if word in caption_token_set]
        occurrence_count = sum(caption_tokens.count(word) for word in matched_words)
        normalized_caption = " ".join(caption_tokens)
        phrase_match = bool(normalized_query and normalized_query in normalized_caption)

        item = dict(base_item)
        item["_match_count"] = int(match_count)
        item["_query_word_count"] = len(query_words)
        item["_matched_words"] = matched_words
        item["_occurrence_count"] = int(occurrence_count)
        item["_phrase_match"] = phrase_match
        item["_caption_word_count"] = len(caption_tokens)
        results.append(item)

    results.sort(
        key=lambda item: (
            item["_match_count"],
            int(item["_phrase_match"]),
            item["_occurrence_count"],
            -item["_caption_word_count"],
        ),
        reverse=True,
    )

    top_k = min(max(1, int(top_k)), len(results))
    results = results[:top_k]

    for rank, item in enumerate(results, start=1):
        item["_rank"] = rank

    return results, query_words


def render_caption_search_tab() -> None:
    st.subheader("Caption Keyword Retrieval")
    st.caption(
        "Tìm theo từ xuất hiện trực tiếp trong `segment_caption`. "
        "Xếp hạng ưu tiên số từ query trùng khớp: match 5 từ > match 4 từ > match 3 từ. "
        "Không dùng embedding và không cần GPU."
    )

    query = st.text_input(
        "Query caption",
        placeholder="Ví dụ: city water orange buildings lights",
        key="caption_search_query",
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        search = st.button("Search Caption", type="primary", width="stretch", key="caption_search_button")
    with c2:
        top_k = st.number_input(
            "Top K",
            min_value=1,
            max_value=CAPTION_TOP_K_MAX,
            value=CAPTION_TOP_K_DEFAULT,
            step=1,
            key="caption_search_top_k",
        )

    # Thumbnail là tùy chọn để tránh việc request HF ảnh làm block kết quả text.
    show_thumbnails = st.checkbox(
        "Hiển thị thumbnail keyframe (chậm hơn vì phải tải ảnh từ Hugging Face)",
        value=False,
        key="caption_show_thumbnails",
    )

    if search:
        query = query.strip()
        if not query:
            st.warning("Hãy nhập query.")
            return

        try:
            hf_token, hf_username = get_hf_auth()

            with st.spinner("Đang load JSON metadata + keyword index..."):
                metadata_root = download_caption_metadata(
                    repo_id=HF_IMAGE_REPO_ID,
                    repo_type=HF_IMAGE_REPO_TYPE,
                    token=hf_token,
                )
                segments, summary, word_index = load_caption_segments(metadata_root)

            with st.spinner("Đang tìm caption khớp từ..."):
                results, query_words = search_caption_segments(
                    query=query,
                    segments=segments,
                    word_index=word_index,
                    top_k=int(top_k),
                )

            st.session_state["caption_search_bundle"] = {
                "query": query,
                "query_words": query_words,
                "results": results,
                "segment_count": len(segments),
                "json_count": len(summary),
                "vocabulary_size": len(word_index),
                "summary": summary,
                "hf_username": hf_username,
                "hf_token": hf_token,
            }
            st.session_state["caption_visible_count"] = CAPTION_DEFAULT_VISIBLE_RESULTS

        except Exception as exc:
            st.error(f"Caption search lỗi: {exc}")
            return

    bundle = st.session_state.get("caption_search_bundle")
    if not bundle:
        st.info("Nhập query rồi bấm **Search Caption**. Lần đầu app tải/parse JSON; các query sau dùng cache.")
        return

    results = bundle["results"]
    hf_token = bundle["hf_token"]
    query_words = bundle.get("query_words", [])

    st.subheader(f'Top {len(results)} segment — "{bundle["query"]}"')
    st.caption(
        "Query words: " + (", ".join(query_words) if query_words else "-")
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Segments", f'{bundle["segment_count"]:,}')
    m2.metric("JSON files", f'{bundle["json_count"]:,}')
    m3.metric("Vocabulary", f'{bundle["vocabulary_size"]:,}')
    m4.metric(
        "Best match",
        f'{results[0]["_match_count"]}/{results[0]["_query_word_count"]}' if results else "0",
    )

    st.caption(f'HF: {bundle["hf_username"]} · exact keyword matching · case-insensitive')

    if not results:
        st.warning("Không có segment nào chứa các từ trong query.")
        return

    # Bảng text xuất hiện trước; không bị thumbnail HF chặn.
    table_rows = []
    for item in results:
        caption = item.get("segment_caption", "")
        table_rows.append({
            "rank": item["_rank"],
            "match": f'{item["_match_count"]}/{item["_query_word_count"]}',
            "matched_words": ", ".join(item["_matched_words"]),
            "occurrences": item["_occurrence_count"],
            "video_id": item.get("video_id", ""),
            "segment_id": item.get("segment_id", ""),
            "start_time": item.get("start_time", ""),
            "end_time": item.get("end_time", ""),
            "start_frame": item.get("start_frame", ""),
            "end_frame": item.get("end_frame", ""),
            "caption": caption if len(caption) <= 500 else caption[:497] + "...",
        })

    with st.expander("Bảng kết quả", expanded=True):
        st.dataframe(table_rows, width="stretch", hide_index=True)

    visible_count = int(st.session_state.get("caption_visible_count", CAPTION_DEFAULT_VISIBLE_RESULTS))
    visible_count = min(max(CAPTION_DEFAULT_VISIBLE_RESULTS, visible_count), len(results))
    visible_results = results[:visible_count]
    remaining_count = len(results) - len(visible_results)

    st.caption(f"Đang hiển thị **{len(visible_results)}/{len(results)}** segment chi tiết")

    c10, c20, call = st.columns(3)
    with c10:
        if st.button("+10", key="caption_load_10", width="stretch", disabled=remaining_count <= 0):
            st.session_state["caption_visible_count"] = min(len(visible_results) + 10, len(results))
            st.rerun()
    with c20:
        if st.button("+20", key="caption_load_20", width="stretch", disabled=remaining_count <= 0):
            st.session_state["caption_visible_count"] = min(len(visible_results) + 20, len(results))
            st.rerun()
    with call:
        if st.button("All", key="caption_load_all", width="stretch", disabled=remaining_count <= 0):
            st.session_state["caption_visible_count"] = len(results)
            st.rerun()

    st.divider()

    for item in visible_results:
        rank = item["_rank"]
        video_id = item.get("video_id", "")
        segment_id = item.get("segment_id", "")
        start_time = float(item.get("start_time", 0.0) or 0.0)
        end_time = float(item.get("end_time", 0.0) or 0.0)
        start_frame = item.get("start_frame")
        end_frame = item.get("end_frame")
        caption = item.get("segment_caption", "")
        matched_words = item.get("_matched_words", [])
        keyframe_files = item.get("keyframe_files", [])

        with st.container(border=True):
            if show_thumbnails:
                left, right = st.columns([1, 3])
            else:
                left, right = None, st.container()

            if show_thumbnails and left is not None:
                with left:
                    if keyframe_files:
                        image_url = build_hf_segment_keyframe_url(item, keyframe_files[0])
                        if image_url:
                            try:
                                image_bytes = fetch_private_hf_image(image_url=image_url, token=hf_token)
                                st.image(image_bytes, width="stretch")
                            except Exception as exc:
                                st.caption(f"Không load được keyframe: {exc}")
                    else:
                        st.caption("Segment không có keyframe.")

            with right:
                st.markdown(f"**#{rank} · {video_id} · {segment_id}**")
                st.caption(
                    f'Match **{item["_match_count"]}/{item["_query_word_count"]}** từ · '
                    f'occurrences={item["_occurrence_count"]} · '
                    f'{start_time:.2f}s → {end_time:.2f}s · '
                    f'frame {start_frame} → {end_frame}'
                )
                st.markdown("**Matched words:** " + ", ".join(f"`{word}`" for word in matched_words))
                st.write(caption)


# ============================================================
# RETRIEVAL HELPERS
# ============================================================
@st.cache_resource
def load_embedder(model_id: str, use_fp16: bool, text_max_length: int) -> Siglip2Embedder:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return Siglip2Embedder(
        model_id=model_id,
        device=device,
        use_fp16=use_fp16,
        text_max_length=text_max_length,
    )


@st.cache_resource
def get_hf_auth() -> tuple[str, str]:
    token = get_token()

    if not token:
        raise RuntimeError(
            "Chưa đăng nhập Hugging Face. Hãy chạy `hf auth login` trong terminal rồi khởi động lại Streamlit."
        )

    api = HfApi(token=token)
    user = api.whoami()
    username = user.get("name", "authenticated") if isinstance(user, dict) else "authenticated"

    # Xác nhận token có quyền đọc private dataset chứa keyframe.
    api.dataset_info(repo_id=HF_IMAGE_REPO_ID)
    return token, username


@st.cache_data(ttl=600, max_entries=500, show_spinner=False)
def fetch_private_hf_image(image_url: str, token: str) -> bytes:
    response = requests.get(
        image_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
        allow_redirects=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise RuntimeError(f"HF trả về content-type không phải ảnh: {content_type or 'unknown'}")

    return response.content


def infer_metadata_path(npz_path: Path) -> Path:
    if npz_path.name.endswith("_embeddings.npz"):
        return npz_path.with_name(npz_path.name.replace("_embeddings.npz", "_metadata.jsonl"))
    return npz_path.with_suffix(".jsonl")


def l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def get_frame_key(item: dict, fallback: str) -> str:
    if item.get("global_frame_id"):
        return str(item["global_frame_id"])

    if item.get("video_id") is not None and item.get("frame_id") is not None:
        return f"{item['video_id']}_{int(item['frame_id']):05d}"

    return fallback


@st.cache_data(show_spinner=False)
def load_all_embeddings(root_str: str) -> tuple[np.ndarray, list[dict], list[dict]]:
    root = Path(root_str)
    npz_files = sorted(root.rglob("*.npz"))

    if not npz_files:
        raise FileNotFoundError(f"Không tìm thấy .npz trong {root}")

    vector_chunks: list[np.ndarray] = []
    all_metadata: list[dict] = []
    summary: list[dict] = []

    expected_dim: int | None = None
    seen_keys: set[str] = set()

    for npz_path in npz_files:
        metadata_path = infer_metadata_path(npz_path)

        try:
            with np.load(npz_path, allow_pickle=False) as data:
                if "vectors" not in data:
                    summary.append(
                        {
                            "file": str(npz_path.relative_to(root)),
                            "status": "skip: missing vectors",
                            "total": 0,
                            "kept": 0,
                        }
                    )
                    continue

                vectors = np.asarray(data["vectors"], dtype=np.float32)

                if vectors.ndim != 2:
                    summary.append(
                        {
                            "file": str(npz_path.relative_to(root)),
                            "status": f"skip: shape {vectors.shape}",
                            "total": len(vectors),
                            "kept": 0,
                        }
                    )
                    continue

                if expected_dim is None:
                    expected_dim = vectors.shape[1]

                if vectors.shape[1] != expected_dim:
                    summary.append(
                        {
                            "file": str(npz_path.relative_to(root)),
                            "status": f"skip: dim {vectors.shape[1]} != {expected_dim}",
                            "total": len(vectors),
                            "kept": 0,
                        }
                    )
                    continue

                frame_ids = (
                    [str(x) for x in data["frame_ids"].tolist()]
                    if "frame_ids" in data
                    else [f"{npz_path.stem}_{i}" for i in range(len(vectors))]
                )

            metadata = read_jsonl(metadata_path)

            if metadata and len(metadata) != len(vectors):
                summary.append(
                    {
                        "file": str(npz_path.relative_to(root)),
                        "status": f"skip: metadata {len(metadata)} != vectors {len(vectors)}",
                        "total": len(vectors),
                        "kept": 0,
                    }
                )
                continue

            if not metadata:
                metadata = [{"global_frame_id": frame_id} for frame_id in frame_ids]

            kept_vectors: list[np.ndarray] = []
            kept_metadata: list[dict] = []

            for i, item in enumerate(metadata):
                fallback = frame_ids[i] if i < len(frame_ids) else f"{npz_path.stem}_{i}"
                key = get_frame_key(item, fallback)

                # Giữ nguyên logic demo hiện tại: loại duplicate nếu cùng keyframe xuất hiện ở nhiều NPZ.
                if key in seen_keys:
                    continue

                seen_keys.add(key)

                item = dict(item)
                item["_source_npz"] = str(npz_path.relative_to(root))
                item["_frame_key"] = key

                kept_vectors.append(vectors[i])
                kept_metadata.append(item)

            if kept_vectors:
                vector_chunks.append(np.stack(kept_vectors, axis=0))
                all_metadata.extend(kept_metadata)

            summary.append(
                {
                    "file": str(npz_path.relative_to(root)),
                    "status": "ok",
                    "total": len(vectors),
                    "kept": len(kept_vectors),
                }
            )

        except Exception as exc:
            summary.append(
                {
                    "file": str(npz_path.relative_to(root)),
                    "status": f"error: {exc}",
                    "total": 0,
                    "kept": 0,
                }
            )

    if not vector_chunks:
        raise RuntimeError("Không có vector hợp lệ để retrieval.")

    vectors = np.concatenate(vector_chunks, axis=0).astype(np.float32, copy=False)
    return vectors, all_metadata, summary


def encode_query(embedder: Siglip2Embedder, query: str) -> np.ndarray:
    features = embedder.encode_texts([query])

    if isinstance(features, torch.Tensor):
        features = features.detach().float().cpu().numpy()

    features = np.asarray(features, dtype=np.float32)

    if features.ndim == 1:
        features = features[None, :]

    return l2_normalize_rows(features)[0]


def build_hf_image_url(item: dict) -> str | None:
    source_folder = item.get("source_folder")
    video_id = item.get("video_id")
    filename = item.get("filename")

    if not source_folder or not video_id or not filename:
        return None

    path_in_repo = (
        f"{HF_IMAGE_ROOT}/"
        f"{source_folder}/"
        f"{video_id}_keyframes/"
        f"{filename}"
    )

    return hf_hub_url(
        repo_id=HF_IMAGE_REPO_ID,
        filename=path_in_repo,
        repo_type=HF_IMAGE_REPO_TYPE,
    )


# ============================================================
# VIDEO HELPERS
# ============================================================
def normalize_video_name(video_name: str) -> str:
    name = str(video_name or "").strip()
    if name.lower().endswith(".mp4"):
        name = name[:-4]
    return name


def infer_source_folder(video_id: str) -> str:
    # L30_V001 -> Videos_L30
    prefix = video_id.split("_", 1)[0]
    return f"Videos_{prefix}"


def candidate_video_paths(video_id: str) -> list[str]:
    source_folder = infer_source_folder(video_id)
    prefix = video_id.split("_", 1)[0]

    candidates = [
        f"{source_folder}/{video_id}.mp4",
        f"videos/{source_folder}/{video_id}.mp4",
        f"Videos/{source_folder}/{video_id}.mp4",
        f"videos/{prefix}/{video_id}.mp4",
        f"Videos/{prefix}/{video_id}.mp4",
        f"{video_id}.mp4",
    ]

    # Giữ thứ tự nhưng bỏ trùng.
    return list(dict.fromkeys(candidates))


def find_local_video(video_id: str) -> Path | None:
    filename = f"{video_id}.mp4"
    source_folder = infer_source_folder(video_id)

    for root in LOCAL_VIDEO_ROOTS:
        direct_candidates = [
            root / source_folder / filename,
            root / filename,
        ]
        for path in direct_candidates:
            if path.exists() and path.is_file():
                return path.resolve()

    return None


@st.cache_data(show_spinner=False)
def download_video_from_hf(
    video_id: str,
    token: str,
    exact_hf_path: str = "",
    preferred_repo_id: str = "",
) -> tuple[str, str, str]:
    """Return (local_path, repo_id, path_in_repo)."""
    video_id = normalize_video_name(video_id)
    if not video_id:
        raise ValueError("Tên video đang trống.")

    # 1) Local trước, không cần network.
    local = find_local_video(video_id)
    if local is not None:
        return str(local), "LOCAL", str(local)

    # 2) Nếu người dùng nhập path chính xác thì ưu tiên path đó.
    repo_candidates = list(HF_VIDEO_REPOS)
    if preferred_repo_id:
        repo_candidates.sort(key=lambda x: 0 if x[0] == preferred_repo_id else 1)

    paths = [exact_hf_path.strip()] if exact_hf_path.strip() else candidate_video_paths(video_id)

    errors: list[str] = []
    for repo_id, repo_type in repo_candidates:
        for path_in_repo in paths:
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=path_in_repo,
                    repo_type=repo_type,
                    token=token,
                )
                return local_path, repo_id, path_in_repo
            except Exception as exc:
                errors.append(f"{repo_id} :: {path_in_repo} -> {type(exc).__name__}")

    attempted = "\n".join(errors[:12])
    raise FileNotFoundError(
        "Không tìm thấy video theo các path đã thử. "
        "Mở mục Nâng cao và nhập đúng HF path nếu cấu trúc repo khác.\n\n"
        f"Đã thử:\n{attempted}"
    )


@st.cache_data(show_spinner=False)
def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV không mở được video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0:
        fps = 25.0

    duration = frame_count / fps if frame_count > 0 else 0.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "width": width,
        "height": height,
    }


@lru_cache(maxsize=256)
def extract_frame_preview(video_path: str, frame_idx: int, video_id: str) -> bytes:
    """Extract a compact JPEG thumbnail for hover preview.

    This is intentionally small (max width 360 px) because the function may be
    called repeatedly while the mouse moves across the seek bar.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Không mở được video để đọc frame.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Không đọc được frame {frame_idx}.")

    h, w = frame.shape[:2]
    max_width = 360
    if w > max_width:
        scale = max_width / float(w)
        frame = cv2.resize(
            frame,
            (max_width, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    h, w = frame.shape[:2]
    font_scale = max(0.38, min(w, h) / 700.0)
    thickness = max(1, int(round(font_scale * 2)))
    label = f"{video_id} | FRAME {int(frame_idx)}"

    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    pad = max(5, int(round(7 * font_scale)))
    box_w = min(w, tw + pad * 2)
    box_h = min(h, th + baseline + pad * 2)

    cv2.rectangle(frame, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (pad, th + pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    ok, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
    )
    if not ok:
        raise RuntimeError("Không encode được frame preview.")
    return encoded.tobytes()


class _VideoAssetHandler(BaseHTTPRequestHandler):
    server_version = "AICVideoHTTP/0.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        asset_id = query.get("id", [""])[0]
        asset = VIDEO_ASSET_REGISTRY.get(asset_id)

        if not asset:
            self.send_error(404, "Unknown video asset")
            return

        if parsed.path == "/video":
            self._serve_video_file(Path(asset["path"]))
            return

        if parsed.path == "/thumb":
            frame_arg = query.get("frame", [None])[0]
            time_arg = query.get("t", [None])[0]

            try:
                if frame_arg is not None:
                    frame_idx = int(float(frame_arg))
                elif time_arg is not None:
                    frame_idx = int(round(float(time_arg) * float(asset["fps"])))
                else:
                    frame_idx = 0
            except Exception:
                frame_idx = 0

            frame_idx = max(0, min(int(asset["frame_count"]) - 1, frame_idx))
            try:
                payload = extract_frame_preview(str(asset["path"]), frame_idx, str(asset["video_id"]))
            except Exception as exc:
                self.send_error(500, f"thumb error: {exc}")
                return

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_error(404, "Unsupported path")

    def _serve_video_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, "Video file not found")
            return

        file_size = file_path.stat().st_size
        range_header = self.headers.get("Range")

        start = 0
        end = file_size - 1
        status_code = 200

        if range_header and range_header.startswith("bytes="):
            try:
                range_value = range_header.split("=", 1)[1]
                start_str, end_str = range_value.split("-", 1)
                if start_str:
                    start = int(start_str)
                if end_str:
                    end = int(end_str)
                if end >= file_size:
                    end = file_size - 1
                if start > end:
                    start = 0
                    end = file_size - 1
                status_code = 206
            except Exception:
                start = 0
                end = file_size - 1
                status_code = 200

        chunk_len = (end - start) + 1

        self.send_response(status_code)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(chunk_len))
        self.send_header("Cache-Control", "no-store")
        if status_code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        try:
            with file_path.open("rb") as f:
                f.seek(start)
                remaining = chunk_len
                while remaining > 0:
                    data = f.read(min(1024 * 1024, remaining))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError):
            # Normal when the browser aborts an old range request during fast seeking.
            return


def ensure_video_http_server() -> int:
    global _VIDEO_HTTP_SERVER, _VIDEO_HTTP_PORT

    with _VIDEO_HTTP_LOCK:
        if _VIDEO_HTTP_SERVER is not None and _VIDEO_HTTP_PORT is not None:
            return _VIDEO_HTTP_PORT

        server = ThreadingHTTPServer(("0.0.0.0", 0), _VideoAssetHandler)
        port = int(server.server_address[1])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        _VIDEO_HTTP_SERVER = server
        _VIDEO_HTTP_PORT = port
        return port


def register_video_asset(video_path: str, video_id: str, info: dict) -> str:
    raw = f"{Path(video_path).resolve()}::{video_id}"
    asset_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    VIDEO_ASSET_REGISTRY[asset_id] = {
        "path": str(Path(video_path).resolve()),
        "video_id": str(video_id),
        "fps": float(info["fps"]),
        "frame_count": int(info["frame_count"]),
    }
    return asset_id


def render_custom_video_browser(
    video_path: str,
    video_id: str,
    info: dict,
    initial_frame: int = 0,
) -> None:
    """Single-screen browser player with frame overlay + hover thumbnails.

    The local HTTP server supports byte-range requests so the browser can seek
    quickly without loading the entire video into memory.
    """
    port = ensure_video_http_server()
    asset_id = register_video_asset(video_path, video_id, info)

    fps = float(info["fps"])
    frame_count = int(info["frame_count"])

    video_width = max(1, int(info.get("width", 16) or 16))
    video_height = max(1, int(info.get("height", 9) or 9))

    aspect_ratio = video_width / video_height

    max_frame = max(0, frame_count - 1)
    initial_frame = max(0, min(int(initial_frame), max_frame))

    # Kích thước player tối đa
    player_max_height_px = 648
    player_max_width_px = max(
        320,
        int(round(player_max_height_px * aspect_ratio))
    )

    config = {
        "videoId": str(video_id),
        "fps": fps,
        "frameCount": frame_count,
        "maxFrame": max_frame,
        "initialFrame": initial_frame,

        "videoWidth": video_width,
        "videoHeight": video_height,
        "playerMaxWidth": player_max_width_px,

        "port": int(port),
        "assetId": asset_id,
    }

    config_json = json.dumps(config, ensure_ascii=False)
    html_block = f"""
<div id="aic-custom-video-player"></div>
<script>
const CFG = {config_json};
const host = window.location.hostname || '127.0.0.1';
const baseUrl = `http://${{host}}:${{CFG.port}}`;
const videoUrl = `${{baseUrl}}/video?id=${{encodeURIComponent(CFG.assetId)}}`;
const thumbBaseUrl = `${{baseUrl}}/thumb?id=${{encodeURIComponent(CFG.assetId)}}`;

function clamp(value, minValue, maxValue) {{
  return Math.max(minValue, Math.min(maxValue, value));
}}

function frameFromTime(seconds) {{
  const frame = Math.floor((Math.max(0, seconds) * CFG.fps) + 0.0001);
  return clamp(frame, 0, CFG.maxFrame);
}}

function formatTime(seconds) {{
  const totalSec = Math.max(0, Math.floor(Number(seconds) || 0));
  const s = totalSec % 60;
  const totalMin = Math.floor(totalSec / 60);
  const m = totalMin % 60;
  const h = Math.floor(totalMin / 60);
  const sText = String(s).padStart(2, '0');
  const mText = String(m).padStart(2, '0');
  return h > 0
    ? `${{String(h).padStart(2, '0')}}:${{mText}}:${{sText}}`
    : `${{mText}}:${{sText}}`;
}}

const root = document.getElementById('aic-custom-video-player');
root.innerHTML = `
  <style>
    * {{ box-sizing: border-box; }}
    .aic-player-wrap {{
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: flex-start;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}

    .aic-video-shell {{
        position: relative;

        width: min(100%, ${{CFG.playerMaxWidth}}px);

        aspect-ratio: ${{CFG.videoWidth}} / ${{CFG.videoHeight}};

        background: #000;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 26px rgba(0,0,0,.22);
    }}

    .aic-video-shell video {{
        display: block;

        width: 100%;
        height: 100%;

        object-fit: contain;
        object-position: center center;

        background: #000;
        cursor: pointer;
    }}
    .aic-frame-overlay {{
      position: absolute;
      top: 8px;
      left: 8px;
      z-index: 8;
      background: rgba(0,0,0,.84);
      color: #fff;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 15px;
      font-weight: 800;
      line-height: 1.15;
      padding: 6px 10px;
      border-radius: 3px;
      letter-spacing: .15px;
      user-select: none;
      pointer-events: none;
      text-shadow: 0 1px 2px rgba(0,0,0,.9);
    }}
    .aic-controls {{
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 12;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 30px 10px 9px;
      background: linear-gradient(to top, rgba(0,0,0,.82), rgba(0,0,0,0));
      transition: opacity .15s ease;
    }}
    .aic-icon-btn {{
      flex: 0 0 auto;
      width: 30px;
      height: 30px;
      border: 0;
      border-radius: 6px;
      background: rgba(0,0,0,.28);
      color: #fff;
      cursor: pointer;
      font-size: 15px;
      line-height: 30px;
      padding: 0;
    }}
    .aic-icon-btn:hover {{ background: rgba(255,255,255,.18); }}
    .aic-scrub-wrap {{
      position: relative;
      flex: 1 1 auto;
      min-width: 100px;
      display: flex;
      align-items: center;
    }}
    .aic-skip-btn {{
        flex: 0 0 auto;
        height: 30px;
        min-width: 46px;

        border: 0;
        border-radius: 6px;

        background: rgba(0,0,0,.35);
        color: #fff;

        padding: 0 7px;

        cursor: pointer;

        font-size: 11px;
        font-weight: 700;

        white-space: nowrap;
    }}

    .aic-skip-btn:hover {{
        background: rgba(255,255,255,.18);
    }}
    .aic-scrub-wrap input[type=range] {{
      width: 100%;
      margin: 0;
      cursor: pointer;
      accent-color: #fff;
    }}
    .aic-time {{
      flex: 0 0 auto;
      min-width: 92px;
      color: #fff;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
      font-weight: 650;
      white-space: nowrap;
      text-align: center;
      text-shadow: 0 1px 2px rgba(0,0,0,.9);
    }}
    .aic-frame-jump {{
      flex: 0 0 auto;
      display: flex;
      align-items: center;
      gap: 4px;
    }}
    .aic-frame-jump span {{
      color: #fff;
      font-size: 11px;
      font-weight: 650;
    }}
    .aic-frame-input {{
      width: 78px;
      height: 28px;
      border: 1px solid rgba(255,255,255,.45);
      border-radius: 5px;
      background: rgba(0,0,0,.58);
      color: #fff;
      padding: 2px 6px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      outline: none;
    }}
    .aic-go-btn {{
      height: 28px;
      border: 1px solid rgba(255,255,255,.42);
      border-radius: 5px;
      background: rgba(0,0,0,.48);
      color: #fff;
      padding: 0 7px;
      cursor: pointer;
      font-size: 11px;
      font-weight: 700;
    }}
    .aic-hover-preview {{
      display: none;
      position: absolute;
      z-index: 20;
      width: 210px;
      padding: 5px;
      background: rgba(0,0,0,.94);
      border-radius: 7px;
      box-shadow: 0 5px 20px rgba(0,0,0,.5);
      pointer-events: none;
    }}
    .aic-hover-preview img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 4px;
      background: #111;
    }}
    .aic-hover-meta {{
      margin-top: 4px;
      color: #fff;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
      font-weight: 700;
      line-height: 1.2;
      text-align: center;
      white-space: nowrap;
    }}
    @media (max-width: 800px) {{
      .aic-frame-jump span, .aic-time {{ display: none; }}
      .aic-frame-input {{ width: 68px; }}
      .aic-hover-preview {{ width: 170px; }}
    }}
  </style>

  <div class="aic-player-wrap">
    <div class="aic-video-shell" id="aic-video-shell">
      <video id="aic-video-tag" preload="metadata" src="${{videoUrl}}"></video>
      <div class="aic-frame-overlay" id="aic-frame-overlay"></div>

      <div class="aic-hover-preview" id="aic-hover-preview">
        <img id="aic-hover-img" alt="preview" />
        <div class="aic-hover-meta" id="aic-hover-meta"></div>
      </div>

      <div class="aic-controls" id="aic-controls">
        <button class="aic-icon-btn" id="aic-play" title="Play / Pause">▶</button>

        <button class="aic-skip-btn" id="aic-back-5" title="Lùi 5 giây">↶ 5s</button>
        <button class="aic-skip-btn" id="aic-forward-5" title="Tới 5 giây">5s ↷</button>

        <div class="aic-scrub-wrap">
          <input id="aic-scrub" type="range" min="0" max="${{CFG.maxFrame}}" step="1" value="${{CFG.initialFrame}}" />
        </div>
        <div class="aic-time" id="aic-time">00:00 / 00:00</div>
        <div class="aic-frame-jump">
          <span>Frame</span>
          <input class="aic-frame-input" id="aic-frame-input" type="number" min="0" max="${{CFG.maxFrame}}" step="1" value="${{CFG.initialFrame}}" />
          <button class="aic-go-btn" id="aic-go" title="Nhảy tới Frame ID">Go</button>
        </div>
        <button class="aic-icon-btn" id="aic-mute" title="Mute / Unmute">🔊</button>
        <button class="aic-icon-btn" id="aic-full" title="Fullscreen">⛶</button>
      </div>
    </div>
  </div>
`;

const shell = document.getElementById('aic-video-shell');
const video = document.getElementById('aic-video-tag');
const overlay = document.getElementById('aic-frame-overlay');
const controls = document.getElementById('aic-controls');
const playBtn = document.getElementById('aic-play');
const back5Btn = document.getElementById('aic-back-5');
const forward5Btn = document.getElementById('aic-forward-5');
const scrub = document.getElementById('aic-scrub');
const timeText = document.getElementById('aic-time');
const frameInput = document.getElementById('aic-frame-input');
const goBtn = document.getElementById('aic-go');
const muteBtn = document.getElementById('aic-mute');
const fullBtn = document.getElementById('aic-full');
const hoverPreview = document.getElementById('aic-hover-preview');
const hoverImg = document.getElementById('aic-hover-img');
const hoverMeta = document.getElementById('aic-hover-meta');

let isScrubbing = false;
let lastPreviewFrame = -1;
let previewTimer = null;

function updateUI(frameOverride = null) {{
  const frame = frameOverride === null
    ? frameFromTime(video.currentTime)
    : clamp(Math.round(frameOverride), 0, CFG.maxFrame);

  overlay.textContent = `${{CFG.videoId}} | FRAME ${{frame}}`;
  if (!isScrubbing) scrub.value = String(frame);
  if (document.activeElement !== frameInput) frameInput.value = String(frame);
  timeText.textContent = `${{formatTime(video.currentTime)}} / ${{formatTime(video.duration)}}`;
  playBtn.textContent = video.paused ? '▶' : '❚❚';
  muteBtn.textContent = video.muted || video.volume === 0 ? '🔇' : '🔊';
}}

function seekToFrame(rawFrame, keepPlaying = false) {{
  let frame = Number(rawFrame);
  if (!Number.isFinite(frame)) return;
  frame = clamp(Math.round(frame), 0, CFG.maxFrame);
  const wasPlaying = !video.paused;
  video.currentTime = (frame / CFG.fps) + 0.00001;
  scrub.value = String(frame);
  frameInput.value = String(frame);
  updateUI(frame);
  if (keepPlaying && wasPlaying) {{
    const p = video.play();
    if (p && p.catch) p.catch(() => {{}});
  }}
}}

function seekBySeconds(deltaSeconds) {{
    if (!Number.isFinite(video.duration)) return;

    const newTime = clamp(
        video.currentTime + deltaSeconds,
        0,
        video.duration
    );

    video.currentTime = newTime;

    const frame = frameFromTime(newTime);

    scrub.value = String(frame);
    frameInput.value = String(frame);

    updateUI(frame);
}}

function togglePlay() {{
  if (video.paused) {{
    const p = video.play();
    if (p && p.catch) p.catch(() => {{}});
  }} else {{
    video.pause();
  }}
}}

function requestPreview(frame) {{
  frame = clamp(Math.round(frame), 0, CFG.maxFrame);
  if (frame === lastPreviewFrame) return;
  lastPreviewFrame = frame;
  if (previewTimer !== null) clearTimeout(previewTimer);
  previewTimer = setTimeout(() => {{
    hoverImg.src = `${{thumbBaseUrl}}&frame=${{frame}}`;
  }}, 55);
}}

function showPreviewAt(frame, clientX) {{
  frame = clamp(Math.round(frame), 0, CFG.maxFrame);
  const shellRect = shell.getBoundingClientRect();
  const previewWidth = hoverPreview.offsetWidth || 210;
  let left = clientX - shellRect.left - (previewWidth / 2);
  left = clamp(left, 5, Math.max(5, shellRect.width - previewWidth - 5));

  hoverPreview.style.left = `${{left}}px`;
  hoverPreview.style.bottom = `${{Math.max(54, controls.offsetHeight + 8)}}px`;
  hoverPreview.style.display = 'block';
  hoverMeta.textContent = `${{CFG.videoId}} | FRAME ${{frame}} · ${{formatTime(frame / CFG.fps)}}`;
  requestPreview(frame);
}}

function hidePreview() {{
  if (!isScrubbing) hoverPreview.style.display = 'none';
}}

function previewFromPointer(event) {{
  const rect = scrub.getBoundingClientRect();
  const ratio = rect.width > 0
    ? clamp((event.clientX - rect.left) / rect.width, 0, 1)
    : 0;
  const frame = Math.round(ratio * CFG.maxFrame);
  showPreviewAt(frame, event.clientX);
}}

video.addEventListener('loadedmetadata', () => {{
  seekToFrame(CFG.initialFrame);
  updateUI(CFG.initialFrame);
}});
video.addEventListener('timeupdate', () => updateUI());
video.addEventListener('play', () => updateUI());
video.addEventListener('pause', () => updateUI());
video.addEventListener('seeked', () => updateUI());
video.addEventListener('click', togglePlay);

playBtn.addEventListener('click', (e) => {{ e.preventDefault(); togglePlay(); }});

back5Btn.addEventListener('click', (e) => {{
    e.preventDefault();
    seekBySeconds(-5);
}});

forward5Btn.addEventListener('click', (e) => {{
    e.preventDefault();
    seekBySeconds(5);
}});

scrub.addEventListener('input', (event) => {{
  isScrubbing = true;
  const frame = Number(event.target.value || 0);
  seekToFrame(frame);
  updateUI(frame);
}});
scrub.addEventListener('change', (event) => {{
  const frame = Number(event.target.value || 0);
  seekToFrame(frame);
  isScrubbing = false;
  setTimeout(() => {{ hoverPreview.style.display = 'none'; }}, 160);
}});
scrub.addEventListener('mousemove', previewFromPointer);
scrub.addEventListener('mouseenter', previewFromPointer);
scrub.addEventListener('mouseleave', hidePreview);
scrub.addEventListener('pointerdown', (event) => {{
  isScrubbing = true;
  previewFromPointer(event);
}});
window.addEventListener('pointerup', () => {{
  if (isScrubbing) {{
    isScrubbing = false;
    setTimeout(() => {{ hoverPreview.style.display = 'none'; }}, 180);
  }}
}});

function submitFrameInput() {{ seekToFrame(frameInput.value); }}
goBtn.addEventListener('click', submitFrameInput);
frameInput.addEventListener('keydown', (event) => {{
  if (event.key === 'Enter') {{
    event.preventDefault();
    submitFrameInput();
    frameInput.blur();
  }}
}});

muteBtn.addEventListener('click', () => {{
  video.muted = !video.muted;
  updateUI();
}});

fullBtn.addEventListener('click', () => {{
  if (document.fullscreenElement) {{
    document.exitFullscreen().catch(() => {{}});
  }} else if (shell.requestFullscreen) {{
    shell.requestFullscreen().catch(() => {{}});
  }}
}});

updateUI(CFG.initialFrame);
</script>
"""
    components.html(html_block, height=690, scrolling=False)


def format_seconds(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


# ============================================================
# CSV HELPERS
# ============================================================
def blank_answer_dataframe(query_type: str, event_count: int) -> pd.DataFrame:
    if query_type == "KIS":
        return pd.DataFrame([{"video_name": "", "frame_id": None}])
    if query_type == "Q&A":
        return pd.DataFrame([{"video_name": "", "frame_id": None, "answer": ""}])

    row: dict[str, object] = {"video_name": ""}
    for i in range(event_count):
        row[f"frame_{i + 1}"] = None
    return pd.DataFrame([row])


def generate_answer_dataframe(
    query_type: str,
    video_name: str,
    start_frame: int,
    end_frame: int,
    frame_step: int,
    default_answer: str,
    event_count: int,
) -> tuple[pd.DataFrame | None, str | None]:
    video_name = normalize_video_name(video_name)
    if not video_name:
        return None, "Hãy nhập tên video trước khi tạo bảng."
    if frame_step <= 0:
        return None, "Khoảng cách frame phải > 0."
    if end_frame < start_frame:
        return None, "Frame end phải >= frame start."

    frames = list(range(int(start_frame), int(end_frame) + 1, int(frame_step)))
    if not frames:
        return None, "Không tạo được frame nào từ khoảng đã chọn."

    if query_type in {"KIS", "Q&A"}:
        truncated = len(frames) > MAX_SUBMISSION_ROWS
        frames = frames[:MAX_SUBMISSION_ROWS]
        rows: list[dict] = []
        for frame in frames:
            row = {"video_name": video_name, "frame_id": int(frame)}
            if query_type == "Q&A":
                row["answer"] = default_answer
            rows.append(row)

        msg = None
        if truncated:
            msg = f"Khoảng đã tạo hơn {MAX_SUBMISSION_ROWS} dòng; app chỉ giữ {MAX_SUBMISSION_ROWS} dòng đầu."
        return pd.DataFrame(rows), msg

    # TRAKE: một dòng = một candidate video với N frame tương ứng N events.
    if len(frames) != event_count:
        return (
            None,
            f"TRAKE đang đặt {event_count} events nhưng start/end/step tạo ra {len(frames)} frame. "
            "Hãy chỉnh để số frame đúng bằng số events.",
        )

    row = {"video_name": video_name}
    for i, frame in enumerate(frames, start=1):
        row[f"frame_{i}"] = int(frame)
    return pd.DataFrame([row]), None


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
        text = str(value).strip()
        if not text:
            raise ValueError("frame rỗng")
        number = float(text)
        if not number.is_integer():
            raise ValueError(f"frame không phải số nguyên: {value}")
        frame = int(number)

    if frame < 0:
        raise ValueError(f"frame âm: {frame}")
    return frame


def build_submission_csv(
    df: pd.DataFrame,
    query_type: str,
    event_count: int,
) -> tuple[bytes | None, list[str], list[str], int]:
    errors: list[str] = []
    warnings: list[str] = []
    output_rows: list[list[object]] = []

    # Bỏ các dòng hoàn toàn trống do data_editor tạo thêm.
    records = df.to_dict("records")
    non_empty_records = [
        row for row in records if any(not _is_blank(v) for v in row.values())
    ]

    if len(non_empty_records) > MAX_SUBMISSION_ROWS:
        errors.append(f"File có {len(non_empty_records)} dòng, vượt giới hạn {MAX_SUBMISSION_ROWS} dòng.")
        return None, errors, warnings, len(non_empty_records)

    for row_idx, row in enumerate(non_empty_records, start=1):
        video_name = normalize_video_name(str(row.get("video_name", "")))
        if not video_name:
            errors.append(f"Dòng {row_idx}: thiếu video_name.")
            continue

        try:
            if query_type == "KIS":
                frame = _parse_frame_int(row.get("frame_id"))
                output_rows.append([video_name, frame])

            elif query_type == "Q&A":
                frame = _parse_frame_int(row.get("frame_id"))
                answer_raw = row.get("answer", "")
                answer = "" if answer_raw is None else str(answer_raw)
                if answer == "":
                    errors.append(f"Dòng {row_idx}: answer đang rỗng.")
                    continue
                if len(answer) > 100:
                    errors.append(
                        f"Dòng {row_idx}: answer dài {len(answer)} ký tự, vượt giới hạn 100 ký tự."
                    )
                    continue
                # Không strip answer để giữ nguyên khoảng trắng đầu/cuối theo quy định CSV.
                output_rows.append([video_name, frame, answer])

            else:  # TRAKE
                frames: list[int] = []
                row_has_error = False
                for i in range(event_count):
                    col = f"frame_{i + 1}"
                    try:
                        frames.append(_parse_frame_int(row.get(col)))
                    except Exception as exc:
                        errors.append(f"Dòng {row_idx}: {col} không hợp lệ ({exc}).")
                        row_has_error = True

                if row_has_error:
                    continue

                if frames != sorted(frames):
                    warnings.append(
                        f"Dòng {row_idx}: frame TRAKE chưa tăng theo thời gian ({frames}). Hãy kiểm tra lại thứ tự events."
                    )
                output_rows.append([video_name, *frames])

        except Exception as exc:
            errors.append(f"Dòng {row_idx}: {exc}.")

    if not non_empty_records:
        errors.append("Bảng đáp án đang trống.")

    if errors:
        return None, errors, warnings, len(non_empty_records)

    # CSV đúng chuẩn: UTF-8, dấu phẩy, không header, LF.
    # csv.writer tự quote answer khi có dấu phẩy / dấu ngoặc kép / xuống dòng.
    sio = io.StringIO(newline="")
    writer = csv.writer(sio, delimiter=",", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerows(output_rows)
    csv_bytes = sio.getvalue().encode("utf-8")
    return csv_bytes, errors, warnings, len(output_rows)


def normalize_csv_filename(filename: str, query_type: str) -> str:
    suffix = {"KIS": "kis", "Q&A": "qa", "TRAKE": "trake"}[query_type]
    name = str(filename or "").strip()
    if not name:
        return f"query-1-{suffix}.csv"
    if not name.lower().endswith(".csv"):
        name += ".csv"
    return name


# ============================================================
# TAB 1 - RETRIEVAL
# ============================================================
def render_retrieval_tab() -> None:
    st.subheader("Global SigLIP2 Retrieval")
    st.caption("Search trên toàn bộ file embedding trong workspace_vector_anchor/vector_embeddings")

    if not EMBEDDINGS_DIR.exists():
        st.error(f"Không tồn tại: {EMBEDDINGS_DIR}")
        return

    query = st.text_input(
        "Query",
        placeholder="Ví dụ: một người đứng trước màn hình",
        key="retrieval_query_input",
    )
    search = st.button("Search Top 100", type="primary", width="stretch", key="retrieval_search_btn")

    if search:
        query = query.strip()
        if not query:
            st.warning("Hãy nhập query.")
            return

        try:
            hf_token, hf_username = get_hf_auth()
        except Exception as exc:
            st.error(f"Hugging Face authentication lỗi: {exc}")
            st.code("hf auth login", language="bash")
            return

        try:
            with st.spinner("Đang load embeddings + embed query..."):
                vectors, metadata, file_summary = load_all_embeddings(str(EMBEDDINGS_DIR))
                normalized_vectors = l2_normalize_rows(vectors)
                embedder = load_embedder(MODEL_ID, USE_FP16, TEXT_MAX_LENGTH)
                query_vector = encode_query(embedder, query)
        except Exception as exc:
            st.error(f"Không chuẩn bị được retrieval index/query: {exc}")
            return

        if query_vector.shape[0] != normalized_vectors.shape[1]:
            st.error(
                f"Dimension mismatch: query={query_vector.shape[0]}, image={normalized_vectors.shape[1]}"
            )
            return

        scores = normalized_vectors @ query_vector
        top_k = min(TOP_K, len(scores))

        candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

        results: list[dict] = []
        for rank, idx in enumerate(top_indices, start=1):
            item = dict(metadata[int(idx)])
            item["_rank"] = rank
            item["_score"] = float(scores[idx])
            item["_image_url"] = build_hf_image_url(item)
            results.append(item)

        st.session_state["retrieval_result_bundle"] = {
            "query": query,
            "results": results,
            "vector_count": len(vectors),
            "dim": int(vectors.shape[1]),
            "file_summary": file_summary,
            "hf_username": hf_username,
            "hf_token": hf_token,
        }
        st.session_state["retrieval_visible_count"] = DEFAULT_VISIBLE_RESULTS

    bundle = st.session_state.get("retrieval_result_bundle")
    if not bundle:
        st.info("Nhập query rồi bấm Search. Index chỉ load khi cần, nên tab kiểm tra video không bị chậm bởi embeddings.")
        return

    results = bundle["results"]
    query = bundle["query"]
    file_summary = bundle["file_summary"]
    hf_token = bundle["hf_token"]

    visible_count = int(
        st.session_state.get(
            "retrieval_visible_count",
            DEFAULT_VISIBLE_RESULTS,
        )
    )

    visible_count = min(
        max(DEFAULT_VISIBLE_RESULTS, visible_count),
        len(results),
    )

    

    st.subheader(f'Top {len(results)} — "{query}"')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vectors", f"{bundle['vector_count']:,}")
    c2.metric("NPZ files", len(file_summary))
    c3.metric("Dim", bundle["dim"])
    c4.metric("Best score", f"{results[0]['_score']:.4f}")

    st.caption(
        f"HF: {bundle['hf_username']} · repo ảnh: {HF_IMAGE_REPO_ID} · "
        f"device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
    )

    table_rows = [
        {
            "rank": x["_rank"],
            "score": round(x["_score"], 6),
            "video_id": x.get("video_id", ""),
            "frame_id": x.get("frame_id", ""),
            "global_frame_id": x.get("global_frame_id", x.get("_frame_key", "")),
            "source_npz": x.get("_source_npz", ""),
        }
        for x in results
    ]

    with st.expander("Top 100 table"):
        st.dataframe(table_rows, width="stretch", hide_index=True)

    # with st.expander("Embedding files"):
    #     st.dataframe(file_summary, width="stretch", hide_index=True)

    # ============================================================
    # LOAD MORE CONTROLS
    # ============================================================
    visible_results = results[:visible_count]
    loaded_count = len(visible_results)
    total_count = len(results)
    remaining_count = total_count - loaded_count

    st.caption(
        f"Đang hiển thị **{loaded_count}/{total_count}** kết quả"
    )

    c10, c20, call = st.columns(3)

    with c10:
        if st.button(
            "+10",
            key="retrieval_load_10",
            width="stretch",
            disabled=remaining_count <= 0,
        ):
            st.session_state["retrieval_visible_count"] = min(
                loaded_count + 10,
                total_count,
            )
            st.rerun()

    with c20:
        if st.button(
            "+20",
            key="retrieval_load_20",
            width="stretch",
            disabled=remaining_count <= 0,
        ):
            st.session_state["retrieval_visible_count"] = min(
                loaded_count + 20,
                total_count,
            )
            st.rerun()

    with call:
        if st.button(
            "All",
            key="retrieval_load_all",
            width="stretch",
            disabled=remaining_count <= 0,
        ):
            st.session_state["retrieval_visible_count"] = total_count
            st.rerun()

    st.divider()

    # ============================================================
    # IMAGE GRID
    # ============================================================

    for start in range(0, len(visible_results), GRID_COLUMNS):
        cols = st.columns(GRID_COLUMNS)

        for col, item in zip(cols, visible_results[start : start + GRID_COLUMNS]):
            with col:
                image_url = item.get("_image_url") or build_hf_image_url(item)

                if image_url:
                    try:
                        image_bytes = fetch_private_hf_image(image_url=image_url, token=hf_token)
                        st.image(image_bytes, width="stretch")
                    except Exception as exc:
                        st.warning(f"Không đọc được ảnh từ private HF repo: {exc}")
                        st.code(image_url, language="text")
                else:
                    st.warning("Không xác định được HF image URL.")

                st.markdown(f"**#{item['_rank']} · {item['_score']:.4f}**")
                if item.get("video_id"):
                    st.caption(f"video: {item['video_id']}")
                if item.get("frame_id") is not None:
                    st.caption(f"frame: {item['frame_id']}")
                st.caption(f"source: {item.get('_source_npz', '')}")

# ============================================================
# TAB 2 - VIDEO INSPECTOR + CSV
# ============================================================
def render_video_csv_tab() -> None:
    st.subheader("Kiểm tra video & tạo CSV đáp án")
    st.caption(
        "Một màn hình video có overlay Frame ID, thumbnail preview khi tua và ô nhập Frame ID thủ công. Phần dưới dùng để tạo CSV tối đa 100 dòng."
    )

    # ------------------------
    # 1) VIDEO INSPECTOR
    # ------------------------
    st.markdown("### 1. Cửa sổ kiểm tra video")

    top1, top2 = st.columns([3, 1])
    with top1:
        default_video = st.session_state.get("inspected_video_id", "")
        video_input = st.text_input(
            "Tên video",
            value=default_video,
            placeholder="Ví dụ: L30_V001 hoặc L30_V001.mp4",
            key="video_inspector_input",
        )
    with top2:
        st.write("")
        st.write("")
        open_video = st.button("Mở video", type="primary", width="stretch", key="open_video_btn")

    with st.expander("Nâng cao: nếu app không tự tìm thấy video trên Hugging Face"):
        preferred_repo = st.selectbox(
            "Ưu tiên repo video",
            options=[repo_id for repo_id, _ in HF_VIDEO_REPOS],
            key="preferred_video_repo",
        )
        exact_hf_path = st.text_input(
            "HF path chính xác (để trống = tự suy luận)",
            placeholder="Ví dụ: Videos_L30/L30_V001.mp4",
            key="exact_hf_video_path",
        )
        st.caption("App luôn kiểm tra video local trước, sau đó mới tải từ HF và dùng cache của huggingface_hub.")

    if open_video:
        video_id = normalize_video_name(video_input)
        if not video_id:
            st.warning("Hãy nhập tên video.")
        else:
            try:
                hf_token, _ = get_hf_auth()
                with st.spinner(f"Đang tìm / tải {video_id}.mp4 ..."):
                    local_path, repo_id, path_in_repo = download_video_from_hf(
                        video_id=video_id,
                        token=hf_token,
                        exact_hf_path=exact_hf_path,
                        preferred_repo_id=preferred_repo,
                    )
                    info = get_video_info(local_path)

                st.session_state["inspected_video_id"] = video_id
                st.session_state["inspected_video_path"] = local_path
                st.session_state["inspected_video_repo"] = repo_id
                st.session_state["inspected_video_hf_path"] = path_in_repo
                st.session_state["inspected_video_info"] = info

                # Dùng video vừa mở làm mặc định cho khu vực tạo CSV.
                st.session_state["submission_video_name"] = video_id
            except Exception as exc:
                st.error(str(exc))

    video_path = st.session_state.get("inspected_video_path")
    video_id = st.session_state.get("inspected_video_id")
    info = st.session_state.get("inspected_video_info")

    if video_path and video_id and info and Path(video_path).exists():
        fps = float(info["fps"])
        frame_count = int(info["frame_count"])

        if frame_count <= 0:
            st.error("Không đọc được tổng số frame của video.")
        else:
            # Chỉ một màn hình video. Frame ID được overlay trực tiếp.
            # Ô Frame trong player cho phép nhập thủ công và nhấn Enter/Go để nhảy.
            render_custom_video_browser(
                video_path=str(video_path),
                video_id=str(video_id),
                info=info,
                initial_frame=0,
            )
            st.caption(
                f"{video_id} · {fps:.3f} FPS · {frame_count:,} frames · "
                f"{format_seconds(float(info['duration']))} · "
                "rê/kéo thanh tua để xem thumbnail preview; nhập Frame ID ngay trong player để nhảy thủ công"
            )
    else:
        st.info("Chưa mở video. Nhập video ID rồi bấm **Mở video**.")

    st.divider()

    # ------------------------
    # 2) CSV BUILDER
    # ------------------------
    st.markdown("### 2. Tạo file CSV đáp án")

    q1, q2, q3 = st.columns([1, 1.5, 1])
    with q1:
        query_type = st.selectbox(
            "Loại file",
            options=["KIS", "Q&A", "TRAKE"],
            key="submission_query_type",
        )
    with q2:
        video_name = st.text_input(
            "Video name dùng cho CSV",
            value=st.session_state.get("submission_video_name", video_id or ""),
            placeholder="L30_V001",
            key="submission_video_name_widget",
        )
        st.session_state["submission_video_name"] = video_name
    with q3:
        event_count = 1
        if query_type == "TRAKE":
            event_count = st.number_input(
                "Số events",
                min_value=1,
                max_value=20,
                value=int(st.session_state.get("trake_event_count", 4)),
                step=1,
                key="trake_event_count",
            )
        else:
            st.caption("KIS: video,frame\n\nQ&A: video,frame,answer")

    start_frame = 0
    end_frame = 0
    frame_step = 1

    if query_type != "TRAKE":
        r1, r2, r3 = st.columns(3)

        with r1:
            start_frame = st.number_input(
                "Frame start",
                min_value=0,
                value=int(st.session_state.get("answer_start_frame", 0)),
                step=1,
                key="answer_start_frame",
            )

        with r2:
            end_frame = st.number_input(
                "Frame end",
                min_value=0,
                value=int(st.session_state.get("answer_end_frame", start_frame)),
                step=1,
                key="answer_end_frame",
            )

        with r3:
            frame_step = st.number_input(
                "Khoảng cách frame",
                min_value=1,
                value=int(st.session_state.get("answer_frame_step", 30)),
                step=1,
                key="answer_frame_step",
            )

    default_answer = ""
    if query_type == "Q&A":
        default_answer = st.text_input(
            "Answer mặc định cho dải frame (có thể sửa từng dòng ở bảng dưới)",
            value=st.session_state.get("default_qa_answer", ""),
            max_chars=100,
            key="default_qa_answer",
        )

    mode_key = f"{query_type}:{int(event_count)}"
    if st.session_state.get("answer_editor_mode") != mode_key:
        st.session_state["answer_editor_mode"] = mode_key
        st.session_state["answer_editor_df"] = blank_answer_dataframe(query_type, int(event_count))
        st.session_state["answer_editor_version"] = int(st.session_state.get("answer_editor_version", 0)) + 1

    g1, g2 = st.columns([2, 1])
    with g1:
        generate = st.button(
            "Tạo bảng từ start → end → khoảng cách",
            type="primary",
            width="stretch",
            key="generate_answer_table",
        )
    with g2:
        clear = st.button("Xóa bảng", width="stretch", key="clear_answer_table")

    if generate:
        generated_df, message = generate_answer_dataframe(
            query_type=query_type,
            video_name=video_name,
            start_frame=int(start_frame),
            end_frame=int(end_frame),
            frame_step=int(frame_step),
            default_answer=default_answer,
            event_count=int(event_count),
        )
        if generated_df is None:
            st.error(message or "Không tạo được bảng.")
        else:
            st.session_state["answer_editor_df"] = generated_df
            st.session_state["answer_editor_version"] = int(st.session_state.get("answer_editor_version", 0)) + 1
            if message:
                st.warning(message)
            st.rerun()

    if clear:
        st.session_state["answer_editor_df"] = blank_answer_dataframe(query_type, int(event_count))
        st.session_state["answer_editor_version"] = int(st.session_state.get("answer_editor_version", 0)) + 1
        st.rerun()

    editor_df = st.session_state.get("answer_editor_df")
    if not isinstance(editor_df, pd.DataFrame):
        editor_df = blank_answer_dataframe(query_type, int(event_count))

    column_config: dict[str, object] = {
        "video_name": st.column_config.TextColumn("video_name", help="Tên video không có .mp4"),
    }
    if query_type in {"KIS", "Q&A"}:
        column_config["frame_id"] = st.column_config.NumberColumn(
            "frame_id", min_value=0, step=1, format="%d"
        )
    if query_type == "Q&A":
        column_config["answer"] = st.column_config.TextColumn(
            "answer", help="Tối đa 100 ký tự", max_chars=100
        )
    if query_type == "TRAKE":
        for i in range(int(event_count)):
            column_config[f"frame_{i + 1}"] = st.column_config.NumberColumn(
                f"frame_{i + 1}", min_value=0, step=1, format="%d"
            )

    editor_key = (
        f"answer_editor_{mode_key}_{int(st.session_state.get('answer_editor_version', 0))}"
    )
    edited_df = st.data_editor(
        editor_df,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config=column_config,
        key=editor_key,
    )
    st.session_state["answer_editor_df"] = edited_df

    st.caption(
        "Có thể sửa trực tiếp từng ô, thêm/xóa dòng. Khi export app sẽ bỏ header và kiểm tra giới hạn 100 dòng."
    )

    file_default = {
        "KIS": "default-kis.csv",
        "Q&A": "default-qa.csv",
        "TRAKE": "default-trake.csv",
    }[query_type]
    
    filename_key = f"csv_filename_{query_type}"

    if filename_key not in st.session_state:
        st.session_state[filename_key] = file_default

    filename = st.text_input(
        "Tên file CSV",
        key=filename_key,
    )

    filename = normalize_csv_filename(
        st.session_state[filename_key],
        query_type,
    )


    csv_bytes, errors, warnings, row_count = build_submission_csv(
        edited_df,
        query_type=query_type,
        event_count=int(event_count),
    )

    c1, c2 = st.columns([1, 3])
    c1.metric("Số dòng", f"{row_count}/{MAX_SUBMISSION_ROWS}")
    with c2:
        if query_type == "KIS":
            st.caption("Format export: video_name,frame_id")
        elif query_type == "Q&A":
            st.caption("Format export: video_name,frame_id,answer · answer tối đa 100 ký tự")
        else:
            st.caption(f"Format export: video_name + {int(event_count)} frame IDs theo thứ tự events")

    for warning in warnings:
        st.warning(warning)
    for error in errors:
        st.error(error)

    if csv_bytes is not None:
        csv_text = csv_bytes.decode("utf-8")
        with st.expander("Preview CSV thuần túy", expanded=True):
            st.code(csv_text, language="text")

        st.download_button(
            "Xuất file CSV",
            data=csv_bytes,
            file_name=filename,
            mime="text/csv; charset=utf-8",
            type="primary",
            width="stretch",
            key="download_submission_csv",
        )
        st.success("CSV hợp lệ về cấu trúc cơ bản: UTF-8, comma delimiter, không header, tối đa 100 dòng.")

    with st.expander("Quy tắc AIC26 đang được app kiểm tra"):
        st.markdown(
            """
- **KIS:** `video_name,frame_id`
- **Q&A:** `video_name,frame_id,answer`, answer tối đa **100 ký tự**
- **TRAKE:** `video_name,frame_1,frame_2,...`, số frame đúng bằng số events
- Tên video xuất ra **không có `.mp4`**
- Frame ID phải là **số nguyên**
- CSV dùng **UTF-8**, delimiter `,`, **không có header**
- Tối đa **100 dòng / file**
- Q&A dùng CSV quoting chuẩn khi answer có dấu phẩy, ngoặc kép hoặc xuống dòng
            """
        )


# ============================================================
# RENDER TABS
# ============================================================
# ============================================================
# MAIN NAVIGATION
# ============================================================

PAGE_RETRIEVAL = "🔎 Retrieval"
PAGE_CAPTION = "📝 Caption Search"
PAGE_VIDEO = "🎬 Kiểm tra video + CSV"

if "main_nav" not in st.session_state:
    st.session_state["main_nav"] = PAGE_RETRIEVAL

page = st.segmented_control(
    "Chức năng",
    options=[
        PAGE_RETRIEVAL,
        PAGE_CAPTION,
        PAGE_VIDEO,
    ],
    key="main_nav",
    width="stretch",
)

if page == PAGE_RETRIEVAL:
    render_retrieval_tab()

elif page == PAGE_CAPTION:
    render_caption_search_tab()

elif page == PAGE_VIDEO:
    render_video_csv_tab()