from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

from siglip2_embedder import Siglip2Embedder

# ============================================================
# HUGGING FACE INPUT
# ============================================================

HF_REPO_ID = "Chillguy2026/dataset_video"
TARGET_FOLDERS = ["Videos_L26"]
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv")

# ============================================================
# WORKSPACE
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_WORKSPACE = os.path.join(CURRENT_DIR, "workspace_vector_anchor")

LOCAL_DATASET_DIR = os.path.join(BASE_WORKSPACE, "hf_downloaded_videos")
LOCAL_TEMP_FOLDER = os.path.join(BASE_WORKSPACE, "temp_vector_processing")
KEYFRAMES_VECTOR_FOLDER = os.path.join(BASE_WORKSPACE, "keyframe_vector")
EMBEDDINGS_OUTPUT_FOLDER = os.path.join(BASE_WORKSPACE, "vector_embeddings")

for directory in (
    LOCAL_DATASET_DIR,
    LOCAL_TEMP_FOLDER,
    KEYFRAMES_VECTOR_FOLDER,
    EMBEDDINGS_OUTPUT_FOLDER,
):
    os.makedirs(directory, exist_ok=True)

# ============================================================
# FILTER / STORAGE CONFIG
# ============================================================

FRAME_STRIDE = 30
SHARPNESS_RADIUS = 14

JPEG_QUALITY = 85

# ============================================================
# SIGLIP2 CONFIG
# ============================================================

MODEL_ID = "google/siglip2-base-patch16-224"
BATCH_SIZE = 16
TEXT_MAX_LENGTH = 64
USE_FP16 = True

# ============================================================
# RUN CONFIG
# ============================================================

SKIP_EXISTING_FOLDER = True
COPY_VIDEO_TO_TEMP = True

# ============================================================
# HELPERS
# ============================================================
from huggingface_hub import HfApi, snapshot_download


MAX_VIDEOS_PER_DOWNLOAD = 100
VIDEO_START_INDEX = 400

SKIP_EXISTING_BATCH = True

def get_batch_tag() -> str:
    batch_end = VIDEO_START_INDEX + MAX_VIDEOS_PER_DOWNLOAD - 1
    return f"batch_{VIDEO_START_INDEX:04d}_{batch_end:04d}"

def download_target_folder(folder_name: str) -> str:
    print("\n" + "=" * 72)
    print(f"📥 HF folder: {folder_name}")
    print("=" * 72)

    api = HfApi()

    # Chỉ lấy danh sách file trên server, chưa download video.
    repo_files = api.list_repo_files(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )

    # Chỉ lấy video thuộc folder cần xử lý.
    video_files = [
        path
        for path in repo_files
        if path.startswith(f"{folder_name}/")
        and path.lower().endswith(VIDEO_EXTENSIONS)
    ]

    video_files.sort()

    print(f"📦 Remote videos trong {folder_name}: {len(video_files)}")

    # Chọn đúng batch cần download.
    selected_files = video_files[
        VIDEO_START_INDEX:
        VIDEO_START_INDEX + MAX_VIDEOS_PER_DOWNLOAD
    ]

    if not selected_files:
        raise RuntimeError(
            f"Không có video trong batch "
            f"{VIDEO_START_INDEX}:"
            f"{VIDEO_START_INDEX + MAX_VIDEOS_PER_DOWNLOAD}"
        )

    print(
        f"⬇️ Chỉ download {len(selected_files)} videos "
        f"[{VIDEO_START_INDEX}:"
        f"{VIDEO_START_INDEX + len(selected_files)}]"
    )

    print(f"   First: {selected_files[0]}")
    print(f"   Last : {selected_files[-1]}")

    snapshot_path = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",

        # QUAN TRỌNG:
        # chỉ download đúng các video đã chọn.
        allow_patterns=selected_files,

        local_dir=LOCAL_DATASET_DIR,
    )

    target_path = os.path.join(
        snapshot_path,
        folder_name,
    )

    if not os.path.isdir(target_path):
        raise FileNotFoundError(
            f"Không tìm thấy folder HF: {target_path}"
        )

    return target_path

def get_folder_code(folder_name: str) -> str:
    return folder_name[len("Videos_"):] if folder_name.startswith("Videos_") else folder_name

def build_video_identity(video_path: str, target_folder: str) -> tuple[str, str]:
    video_file = os.path.basename(video_path)
    raw_video_id = os.path.splitext(video_file)[0]
    folder_code = get_folder_code(target_folder)

    if raw_video_id.startswith(f"{folder_code}_"):
        video_id = raw_video_id
    else:
        video_id = f"{folder_code}_{raw_video_id}"

    return video_file, video_id


# def download_target_folder(folder_name: str) -> str:
#     print("\n" + "=" * 72)
#     print(f"📥 \\HF folder: {folder_name}")
#     print("=" * 72)

#     snapshot_path = snapshot_download(
#         repo_id=HF_REPO_ID,
#         repo_type="dataset",
#         allow_patterns=f"{folder_name}/*",
#         local_dir=LOCAL_DATASET_DIR,
#     )

#     target_path = os.path.join(snapshot_path, folder_name)
#     if not os.path.isdir(target_path):
#         raise FileNotFoundError(f"Không tìm thấy folder HF: {target_path}")
#     return target_path

def find_videos(folder_path: str) -> list[str]:
    videos: list[str] = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(VIDEO_EXTENSIONS):
                videos.append(os.path.join(root, filename))
    videos.sort()
    return videos

# ============================================================
# PASS 1: SAMPLE + PHASH, KHÔNG GHI ẢNH TẠM
# ============================================================
def calculate_sharpness(frame):
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

def collect_anchor_frames(video_path: str,) -> list[tuple[int, float]]:
    """
    Returns:
        [
            (
                selected_frame_id,
                sharpness_score,
            ),
            ...
        ]

    Ví dụ:
        anchor = 30
        search window = 25..35
        frame nét nhất = 33

        => (33, sharpness)
    """

    if FRAME_STRIDE <= 2 * SHARPNESS_RADIUS:
        raise ValueError(
            "FRAME_STRIDE nên lớn hơn "
            "2 * SHARPNESS_RADIUS để các window "
            "không overlap."
        )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Không mở được video: {video_path}"
        )

    selected_frames: list[tuple[int, float]] = []
    current_frame = 0
    # Anchor đầu tiên
    anchor_frame = 0

    window_start = max(0, anchor_frame - SHARPNESS_RADIUS,)

    window_end = (anchor_frame + SHARPNESS_RADIUS)

    best_frame_id = None
    best_sharpness = -1.0

    def commit_anchor():
        nonlocal best_frame_id
        nonlocal best_sharpness

        if best_frame_id is None:
            return

        selected_frames.append(
            (
                int(best_frame_id),
                float(best_sharpness),
            )
        )

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # ----------------------------------------
            # Nếu đã đi qua window hiện tại
            # → commit anchor và chuyển anchor mới
            # ----------------------------------------

            while current_frame > window_end:
                commit_anchor()

                anchor_frame += FRAME_STRIDE

                window_start = max(0,anchor_frame - SHARPNESS_RADIUS,)

                window_end = (anchor_frame + SHARPNESS_RADIUS)

                best_frame_id = None
                best_sharpness = -1.0

            # ----------------------------------------
            # Chỉ đánh giá frame trong vùng ±radius
            # ----------------------------------------

            if ( window_start <= current_frame <= window_end):
                sharpness = calculate_sharpness(frame)

                if sharpness > best_sharpness:
                    best_sharpness = sharpness
                    best_frame_id = current_frame

            current_frame += 1

        # Anchor cuối video
        commit_anchor()

    finally:
        cap.release()

    # selected_frame_id sẽ tăng dần vì window không overlap
    return selected_frames

# ============================================================
# EMBED RAW FRAME -> SAU ĐÓ MỚI SAVE JPEG85
# ============================================================

def make_frame_metadata(
    final_path: str,
    target_folder: str,
    video_id: str,
    frame_id: int,
) -> dict:
    path = Path(final_path)
    relative_path = os.path.relpath(final_path, BASE_WORKSPACE).replace("\\", "/")

    return {
        "filename": path.name,
        "global_frame_id": path.stem,
        "source_folder": target_folder,
        "folder_id": get_folder_code(target_folder),
        "video_id": video_id,
        "frame_id": int(frame_id),
        "path": str(path),
        "relative_path": relative_path,
    }


def flush_winner_batch(embedder: Siglip2Embedder, frame_batch: list[tuple[int, float, np.ndarray]], target_folder: str, video_id: str, output_dir: str) -> tuple[list[np.ndarray], list[dict]]:
    if not frame_batch:
        return [], []

    pil_images: list[Image.Image] = []

    try:
        for _, _, frame_bgr in frame_batch:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(frame_rgb))

        image_features = embedder.encode_images(pil_images)

        vectors_np = image_features.cpu().numpy().astype(np.float32, copy=False)

        batch_vectors: list[np.ndarray] = []
        batch_metadata: list[dict] = []

        for index, (frame_id, _, frame_bgr) in enumerate(frame_batch):
            final_name = f"{video_id}_{frame_id:05d}.jpg"
            final_path = os.path.join(output_dir, final_name)

            success = cv2.imwrite(
                final_path,
                frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

            if not success:
                raise RuntimeError(f"Không lưu được JPEG: {final_path}")

            batch_vectors.append(vectors_np[index].copy())

            batch_metadata.append(
                make_frame_metadata(
                    final_path=final_path,
                    target_folder=target_folder,
                    video_id=video_id,
                    frame_id=frame_id,
                )
            )

        return batch_vectors, batch_metadata

    finally:
        for image in pil_images:
            try:
                image.close()
            except Exception:
                pass

        if "image_features" in locals():
            del image_features

        if "vectors_np" in locals():
            del vectors_np

def embed_and_save_anchor_frames(embedder: Siglip2Embedder, video_path: str, selected_frames: list[tuple[int, float]], target_folder: str, video_id: str, output_dir: str) -> tuple[list[np.ndarray], list[dict]]:
    if not selected_frames:
        return [], []

    # Đảm bảo frame ID tăng dần theo thời gian
    selected_frames = sorted(selected_frames, key=lambda x: x[0])

    # frame_id -> sharpness
    selected_sharpness = {
        frame_id: sharpness
        for frame_id, sharpness in selected_frames
    }

    selected_frame_ids = [
        frame_id
        for frame_id, _ in selected_frames
    ]

    selected_set = set(selected_frame_ids)
    last_selected = selected_frame_ids[-1]

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video pass 2: {video_path}")

    frame_batch: list[tuple[int, float, np.ndarray]] = []
    video_vectors: list[np.ndarray] = []
    video_metadata: list[dict] = []
    processed_ids: set[int] = set()

    current_frame = 0

    progress = tqdm(total=len(selected_frame_ids), desc=f"Embed {video_id}", leave=False)

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if current_frame in selected_set:
                sharpness = float(selected_sharpness[current_frame])

                frame_batch.append(
                    (
                        current_frame,
                        sharpness,
                        frame.copy(),
                    )
                )

                processed_ids.add(current_frame)

                if len(frame_batch) >= BATCH_SIZE:
                    vectors, metadata = flush_winner_batch(embedder=embedder, frame_batch=frame_batch, target_folder=target_folder, video_id=video_id, output_dir=output_dir)

                    video_vectors.extend(vectors)
                    video_metadata.extend(metadata)

                    progress.update(len(frame_batch))
                    frame_batch.clear()

            if current_frame >= last_selected:
                break

            current_frame += 1

        # Flush batch cuối
        if frame_batch:
            vectors, metadata = flush_winner_batch(embedder=embedder, frame_batch=frame_batch, target_folder=target_folder, video_id=video_id, output_dir=output_dir)

            video_vectors.extend(vectors)
            video_metadata.extend(metadata)

            progress.update(len(frame_batch))
            frame_batch.clear()

    finally:
        progress.close()
        cap.release()

    missing = set(selected_frame_ids) - processed_ids

    if missing:
        print(f"⚠️ {video_id}: không decode được {len(missing)} selected frame.")
        print(f"   Missing frame IDs: {sorted(missing)}")

    return video_vectors, video_metadata

# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_one_video(
    embedder: Siglip2Embedder,
    source_video_path: str,
    target_folder: str,
) -> tuple[list[np.ndarray], list[dict]]:
    
    video_file, video_id = build_video_identity(source_video_path, target_folder)
    print(f"\n🎬 [{target_folder} / {video_file}]")

    if COPY_VIDEO_TO_TEMP:
        local_video_path = os.path.join(LOCAL_TEMP_FOLDER, video_file)
        print("  ⏳ Copy video vào local temp...")
        shutil.copy2(source_video_path, local_video_path)
    else:
        local_video_path = source_video_path

    output_dir = os.path.join(
        KEYFRAMES_VECTOR_FOLDER,
        target_folder,
        f"{video_id}_keyframes",
    )
    os.makedirs(output_dir, exist_ok=True)

    try:
        print(
        f"  1/2 Temporal anchors "
        f"(stride={FRAME_STRIDE}, "
        f"radius=±{SHARPNESS_RADIUS})..."
    )

        selected_frames = collect_anchor_frames(
            local_video_path
        )

        print(
            f"      Anchors selected: "
            f"{len(selected_frames)}"
        )

        if not selected_frames:
            return [], []


        print(
            f"  2/2 Raw anchor frame "
            f"→ SigLIP2 "
            f"→ JPEG Q{JPEG_QUALITY}..."
        )

        vectors, metadata = (
            embed_and_save_anchor_frames(
                embedder=embedder,
                video_path=local_video_path,
                selected_frames=selected_frames,
                target_folder=target_folder,
                video_id=video_id,
                output_dir=output_dir,
            )
        )

        print(
            f"      ✅ "
            f"{len(vectors)} vectors "
            f"+ {len(metadata)} keyframes"
        )

        return vectors, metadata

    finally:
        if COPY_VIDEO_TO_TEMP and os.path.exists(local_video_path):
            os.remove(local_video_path)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

def save_metadata_jsonl(path: str, metadata: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for item in metadata:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


# def save_folder_embeddings(
#     target_folder: str,
#     vectors_list: list[np.ndarray],
#     metadata: list[dict],
# ) -> None:
#     if not vectors_list:
#         print(f"⚠️ {target_folder}: không có vector để lưu.")
#         return

#     vectors = np.stack(vectors_list, axis=0).astype(np.float32, copy=False)
#     frame_ids = np.asarray(
#         [item["global_frame_id"] for item in metadata],
#         dtype=str,
#     )

#     out_npz = os.path.join(
#         EMBEDDINGS_OUTPUT_FOLDER,
#         f"{target_folder}_embeddings.npz",
#     )
#     out_jsonl = os.path.join(
#         EMBEDDINGS_OUTPUT_FOLDER,
#         f"{target_folder}_metadata.jsonl",
#     )

#     # Không dùng dtype=object metadata trong NPZ => không cần allow_pickle=True khi load.
#     np.savez_compressed(
#         out_npz,
#         vectors=vectors,
#         frame_ids=frame_ids,
#         model_id=np.asarray(MODEL_ID),
#         embedding_source=np.asarray("siglip2_get_image_features"),
#         embedding_norm=np.asarray("l2"),
#         vector_dim=np.asarray(vectors.shape[1], dtype=np.int32),
#     )
#     save_metadata_jsonl(out_jsonl, metadata)

#     print(f"\n✅ {target_folder}: {vectors.shape[0]} vectors, dim={vectors.shape[1]}")
#     print(f"   NPZ   : {out_npz}")
#     print(f"   JSONL : {out_jsonl}")
def save_folder_embeddings(target_folder: str, batch_tag: str, vectors_list: list[np.ndarray], metadata: list[dict]) -> None:
    if not vectors_list:
        print(
            f"⚠️ {target_folder} {batch_tag}: "
            "không có vector."
        )
        return

    vectors = np.stack(
        vectors_list,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    frame_ids = np.asarray(
        [
            item["global_frame_id"]
            for item in metadata
        ],
        dtype=str,
    )

    output_name = (
        f"{target_folder}_{batch_tag}"
    )

    out_npz = os.path.join(
        EMBEDDINGS_OUTPUT_FOLDER,
        f"{output_name}_embeddings.npz",
    )

    out_jsonl = os.path.join(
        EMBEDDINGS_OUTPUT_FOLDER,
        f"{output_name}_metadata.jsonl",
    )

    np.savez_compressed(
        out_npz,
        vectors=vectors,
        frame_ids=frame_ids,
        model_id=np.asarray(MODEL_ID),
        embedding_source=np.asarray(
            "siglip2_get_image_features"
        ),
        embedding_norm=np.asarray("l2"),
        vector_dim=np.asarray(
            vectors.shape[1],
            dtype=np.int32,
        ),
    )

    save_metadata_jsonl(
        out_jsonl,
        metadata,
    )

    print(
        f"\n✅ {target_folder} "
        f"{batch_tag}: "
        f"{vectors.shape[0]} vectors"
    )

    print(f"   NPZ   : {out_npz}")
    print(f"   JSONL : {out_jsonl}")

# ============================================================
# PROCESS ONE Videos_Lxx FOLDER
# ============================================================

def process_target_folder(embedder: Siglip2Embedder, target_folder: str) -> None:
    batch_tag = get_batch_tag()

    output_name = (
        f"{target_folder}_{batch_tag}"
    )

    out_npz = os.path.join(
        EMBEDDINGS_OUTPUT_FOLDER,
        f"{output_name}_embeddings.npz",
    )

    out_jsonl = os.path.join(
        EMBEDDINGS_OUTPUT_FOLDER,
        f"{output_name}_metadata.jsonl",
    )

    # if (
    #     SKIP_EXISTING_FOLDER
    #     and os.path.exists(out_npz)
    #     and os.path.exists(out_jsonl)
    # ):
    #     print(f"⏭️ Skip {target_folder}: embedding output đã tồn tại.")
    #     return
    if (
        SKIP_EXISTING_BATCH
        and os.path.exists(out_npz)
        and os.path.exists(out_jsonl)
    ):
        print(
            f"⏭️ Skip {target_folder} "
            f"{batch_tag}: output đã tồn tại."
        )
        return

    input_folder = download_target_folder(target_folder)
    video_paths = find_videos(input_folder)

    if not video_paths:
        print(f"⚠️ {target_folder}: không có video.")
        return

    print(f"🎞️ {target_folder}: {len(video_paths)} videos")
    # print("TEST")
    video_paths = video_paths[:100]  # Giới hạn 100 video đầu tiên  
    folder_vectors: list[np.ndarray] = []
    folder_metadata: list[dict] = []

    for index, video_path in enumerate(video_paths, start=1):
        print(f"\n[{index}/{len(video_paths)}]")
        try:
            vectors, metadata = process_one_video(
                embedder=embedder,
                source_video_path=video_path,
                target_folder=target_folder,
            )
            folder_vectors.extend(vectors)
            folder_metadata.extend(metadata)
        except Exception as exc:
            print(f"❌ Lỗi video {os.path.basename(video_path)}: {exc}")

    save_folder_embeddings(
        target_folder=target_folder,
        batch_tag=batch_tag,
        vectors_list=folder_vectors,
        metadata=folder_metadata,
    )

    del folder_vectors, folder_metadata
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")

    embedder = Siglip2Embedder(
        model_id=MODEL_ID,
        device=device,
        use_fp16=USE_FP16,
        text_max_length=TEXT_MAX_LENGTH,
    )

    for target_folder in TARGET_FOLDERS:
        process_target_folder(embedder, target_folder)

    print("\n🎉 Hoàn tất toàn bộ pipeline.")


if __name__ == "__main__":
    main()

