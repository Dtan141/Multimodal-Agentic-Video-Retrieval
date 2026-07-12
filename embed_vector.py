from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from siglip2_embedder import Siglip2Embedder
import torch.nn.functional as F
import gc

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_WORKSPACE = r"G:\.shortcut-targets-by-id\11I5_AMfAufb6crT2hzGrLEI3tMsTsKjX\AIC2026"
VECTOR_FRAMES_FOLDER = os.path.join(BASE_WORKSPACE, "keyframes_vector")
EMBEDDINGS_OUTPUT_FOLDER = os.path.join(BASE_WORKSPACE, "vector_embeddings")

os.makedirs(EMBEDDINGS_OUTPUT_FOLDER, exist_ok=True)

# --- CẤU HÌNH MODEL & PHẦN CỨNG ---
MODEL_ID = "google/siglip2-base-patch16-224" 
BATCH_SIZE = 32  # Tối ưu cho 4GB VRAM
TEXT_MAX_LENGTH = 64
USE_FP16 = True
SKIP_EXISTING = True

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def scan_image_paths(folder_path: str) -> list[str]:
    image_paths: list[str] = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if Path(file).suffix.lower() in ALLOWED_EXTS:
                image_paths.append(os.path.join(root, file))

    image_paths.sort()
    return image_paths

def parse_frame_metadata(path: str, source_folder: str) -> dict:
    """
    Ví dụ filename: L28_V0001_00020.jpg
    - folder_id: L28
    - video_id: L28_V0001
    - frame_id: 20
    """
    p = Path(path)
    stem = p.stem
    parts = stem.split("_")

    folder_id = parts[0] if len(parts) >= 1 else "unknown"
    video_id = "_".join(parts[:-1]) if len(parts) >= 2 else "unknown"

    try:
        frame_id = int(parts[-1]) if len(parts) >= 2 else 0
    except ValueError:
        frame_id = 0

    return {
        "filename": p.name,
        "global_frame_id": stem,
        "source_folder": source_folder,
        "folder_id": folder_id,
        "video_id": video_id,
        "frame_id": frame_id,
        "path": str(p),
    }

def load_image_batch(batch_paths: list[str]) -> tuple[list[Image.Image], list[str]]:
    images: list[Image.Image] = []
    valid_paths: list[str] = []

    for path in batch_paths:
        try:
            with Image.open(path) as img:
                images.append(img.convert("RGB").copy())
                valid_paths.append(path)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            print(f"⚠️ Bỏ qua ảnh lỗi: {path} | {exc}")

    return images, valid_paths


def save_metadata_jsonl(jsonl_path: str, metadata: list[dict]) -> None:
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def process_folder(embedder: Siglip2Embedder, folder: str) -> None:
    folder_path = os.path.join(VECTOR_FRAMES_FOLDER, folder)
    out_npz = os.path.join(EMBEDDINGS_OUTPUT_FOLDER, f"{folder}_embeddings.npz")
    out_jsonl = os.path.join(EMBEDDINGS_OUTPUT_FOLDER, f"{folder}_metadata.jsonl")

    if SKIP_EXISTING and os.path.exists(out_npz):
        print(f"⏭️ Đã tồn tại, bỏ qua: {out_npz}")
        return

    print(f"\n📂 Đang xử lý thư mục: {folder}")

    image_paths = scan_image_paths(folder_path)
    if not image_paths:
        print(f"  -> Thư mục không có ảnh hợp lệ, bỏ qua.")
        return

    all_vectors: list[np.ndarray] = []
    all_metadata: list[dict] = []

    for start in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=f"Embedding {folder}"):
        batch_paths = image_paths[start : start + BATCH_SIZE]
        batch_images, valid_paths = load_image_batch(batch_paths)

        if not batch_images:
            continue

        image_features = embedder.encode_images(batch_images)
        vectors_np = image_features.cpu().numpy().astype(np.float32)

        for idx, path in enumerate(valid_paths):
            all_vectors.append(vectors_np[idx])
            all_metadata.append(parse_frame_metadata(path, source_folder=folder))

        del batch_images, image_features, vectors_np

    if not all_vectors:
        print(f"⚠️ Không tạo được vector nào cho thư mục: {folder}")
        return

    vectors = np.asarray(all_vectors, dtype=np.float32)
    metadata_arr = np.asarray(all_metadata, dtype=object)

    np.savez_compressed(
        out_npz,
        vectors=vectors,
        metadata=metadata_arr,
        model_id=np.asarray(MODEL_ID),
        embedding_source=np.asarray("siglip2_get_image_features_pooler_output"),
        embedding_norm=np.asarray("l2"),
        vector_dim=np.asarray(vectors.shape[1], dtype=np.int32),
    )

    save_metadata_jsonl(out_jsonl, all_metadata)

    print(f"✅ Đã lưu {vectors.shape[0]} vector, dim={vectors.shape[1]}")
    print(f"   NPZ   : {out_npz}")
    print(f"   JSONL : {out_jsonl}")

    del all_vectors, all_metadata, vectors, metadata_arr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_embeddings() -> None:
    if not os.path.isdir(VECTOR_FRAMES_FOLDER):
        raise FileNotFoundError(f"Không tìm thấy VECTOR_FRAMES_FOLDER: {VECTOR_FRAMES_FOLDER}")

    folders = [
        f for f in os.listdir(VECTOR_FRAMES_FOLDER)
        if os.path.isdir(os.path.join(VECTOR_FRAMES_FOLDER, f))
    ]
    folders.sort()

    if not folders:
        print(f"⚠️ Không tìm thấy thư mục ảnh nào trong {VECTOR_FRAMES_FOLDER}")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    embedder = Siglip2Embedder(
        model_id=MODEL_ID,
        device=device,
        use_fp16=USE_FP16,
        text_max_length=TEXT_MAX_LENGTH,
    )

    for folder in folders:
        process_folder(embedder, folder)

    print("\n🎉 Hoàn tất embedding toàn bộ folder.")


if __name__ == "__main__":
    process_embeddings()
