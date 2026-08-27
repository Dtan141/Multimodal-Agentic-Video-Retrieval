from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import torch

from siglip2_embedder import Siglip2Embedder
from huggingface_hub import HfApi, get_token, hf_hub_url


CURRENT_DIR = Path(__file__).resolve().parent
BASE_WORKSPACE = CURRENT_DIR / "workspace_vector_anchor"
EMBEDDINGS_DIR = BASE_WORKSPACE / "vector_embeddings"

HF_IMAGE_REPO_ID = "Chillguy2026/AIC_2026_data"
HF_IMAGE_REPO_TYPE = "dataset"
HF_IMAGE_ROOT = "keyframe_vector"

MODEL_ID = "google/siglip2-base-patch16-224"
TEXT_MAX_LENGTH = 64
USE_FP16 = True
TOP_K = 100
GRID_COLUMNS = 5


st.set_page_config(page_title="AIC Global Retrieval", page_icon="🔎", layout="wide")
st.title("AIC — Global SigLIP2 Retrieval")
st.caption("Search trên toàn bộ file embedding trong workspace_vector/vector_embeddings")


@st.cache_resource
def load_embedder(model_id: str, use_fp16: bool, text_max_length: int) -> Siglip2Embedder:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return Siglip2Embedder(model_id=model_id, device=device, use_fp16=use_fp16, text_max_length=text_max_length)

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

    # Xác nhận token có quyền đọc private dataset.
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
        raise RuntimeError(
            f"HF trả về content-type không phải ảnh: {content_type or 'unknown'}"
        )

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

    items = []
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

    vector_chunks = []
    all_metadata = []
    summary = []

    expected_dim = None
    seen_keys = set()

    for npz_path in npz_files:
        metadata_path = infer_metadata_path(npz_path)

        try:
            with np.load(npz_path, allow_pickle=False) as data:
                if "vectors" not in data:
                    summary.append({"file": str(npz_path.relative_to(root)), "status": "skip: missing vectors", "total": 0, "kept": 0})
                    continue

                vectors = np.asarray(data["vectors"], dtype=np.float32)

                if vectors.ndim != 2:
                    summary.append({"file": str(npz_path.relative_to(root)), "status": f"skip: shape {vectors.shape}", "total": len(vectors), "kept": 0})
                    continue

                if expected_dim is None:
                    expected_dim = vectors.shape[1]

                if vectors.shape[1] != expected_dim:
                    summary.append({"file": str(npz_path.relative_to(root)), "status": f"skip: dim {vectors.shape[1]} != {expected_dim}", "total": len(vectors), "kept": 0})
                    continue

                frame_ids = [str(x) for x in data["frame_ids"].tolist()] if "frame_ids" in data else [f"{npz_path.stem}_{i}" for i in range(len(vectors))]

            metadata = read_jsonl(metadata_path)

            if metadata and len(metadata) != len(vectors):
                summary.append({"file": str(npz_path.relative_to(root)), "status": f"skip: metadata {len(metadata)} != vectors {len(vectors)}", "total": len(vectors), "kept": 0})
                continue

            if not metadata:
                metadata = [{"global_frame_id": frame_id} for frame_id in frame_ids]

            kept_vectors = []
            kept_metadata = []

            for i, item in enumerate(metadata):
                fallback = frame_ids[i] if i < len(frame_ids) else f"{npz_path.stem}_{i}"
                key = get_frame_key(item, fallback)

                # Tránh duplicate nếu cùng keyframe nằm ở nhiều NPZ.
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

            summary.append({"file": str(npz_path.relative_to(root)), "status": "ok", "total": len(vectors), "kept": len(kept_vectors)})

        except Exception as exc:
            summary.append({"file": str(npz_path.relative_to(root)), "status": f"error: {exc}", "total": 0, "kept": 0})

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


try:
    hf_token, hf_username = get_hf_auth()
except Exception as exc:
    st.error(f"Hugging Face authentication lỗi: {exc}")
    st.code("hf auth login", language="bash")
    st.stop()


if not EMBEDDINGS_DIR.exists():
    st.error(f"Không tồn tại: {EMBEDDINGS_DIR}")
    st.stop()

try:
    with st.spinner("Đang load toàn bộ embeddings..."):
        vectors, metadata, file_summary = load_all_embeddings(str(EMBEDDINGS_DIR))
except Exception as exc:
    st.error(f"Không load được embeddings: {exc}")
    st.stop()

normalized_vectors = l2_normalize_rows(vectors)

st.sidebar.header("Global index")
st.sidebar.success(f"{len(vectors):,} unique vectors")
st.sidebar.write(f"Files: **{len(file_summary)}**")
st.sidebar.write(f"Dim: **{vectors.shape[1]}**")
st.sidebar.write(f"Device: **{'CUDA' if torch.cuda.is_available() else 'CPU'}**")
st.sidebar.success(f"HF authenticated: {hf_username}")
st.sidebar.write(f"Private image repo: `{HF_IMAGE_REPO_ID}`")

with st.sidebar.expander("Embedding files"):
    st.dataframe(file_summary, width='stretch', hide_index=True)

query = st.text_input("Query", placeholder="Ví dụ: một người đứng trước màn hình")
search = st.button("Search Top 100", type="primary", width='stretch')

if not search:
    st.stop()

query = query.strip()

if not query:
    st.warning("Hãy nhập query.")
    st.stop()

try:
    with st.spinner("Đang embed query..."):
        embedder = load_embedder(MODEL_ID, USE_FP16, TEXT_MAX_LENGTH)
        query_vector = encode_query(embedder, query)
except Exception as exc:
    st.error(f"Không embed được query: {exc}")
    st.stop()

if query_vector.shape[0] != normalized_vectors.shape[1]:
    st.error(f"Dimension mismatch: query={query_vector.shape[0]}, image={normalized_vectors.shape[1]}")
    st.stop()

scores = normalized_vectors @ query_vector
top_k = min(TOP_K, len(scores))

candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
top_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

results = []

for rank, idx in enumerate(top_indices, start=1):
    item = dict(metadata[int(idx)])
    item["_rank"] = rank
    item["_score"] = float(scores[idx])
    item["_image_url"] = build_hf_image_url(item)
    results.append(item)

st.subheader(f'Top {top_k} — "{query}"')

c1, c2, c3, c4 = st.columns(4)
c1.metric("Vectors", f"{len(vectors):,}")
c2.metric("NPZ files", len(file_summary))
c3.metric("Top K", top_k)
c4.metric("Best score", f"{results[0]['_score']:.4f}")

table_rows = [{
    "rank": x["_rank"],
    "score": round(x["_score"], 6),
    "video_id": x.get("video_id", ""),
    "frame_id": x.get("frame_id", ""),
    "global_frame_id": x.get("global_frame_id", x.get("_frame_key", "")),
    "source_npz": x.get("_source_npz", ""),
} for x in results]

with st.expander("Top 100 table"):
    st.dataframe(table_rows, width='stretch', hide_index=True)

for start in range(0, len(results), GRID_COLUMNS):
    cols = st.columns(GRID_COLUMNS)

    for col, item in zip(cols, results[start:start + GRID_COLUMNS]):
        with col:
            image_url = item.get("_image_url")

            if not image_url:
                image_url = build_hf_image_url(item)
                item["_image_url"] = image_url

            if image_url:
                try:
                    image_bytes = fetch_private_hf_image(
                        image_url=image_url,
                        token=hf_token,
                    )
                    st.image(
                        image_bytes,
                        width="stretch",
                    )
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