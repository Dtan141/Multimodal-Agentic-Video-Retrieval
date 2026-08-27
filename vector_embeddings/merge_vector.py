from pathlib import Path
import json

import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("workspace_vector_anchor/vector_embeddings")

NPZ_PATTERN = "Videos_L26_batch_*_embeddings.npz"
JSONL_PATTERN = "Videos_L26_batch_*_metadata.jsonl"

OUTPUT_NPZ = INPUT_DIR / "Videos_L26_embeddings.npz"
OUTPUT_JSONL = INPUT_DIR / "Videos_L26_metadata.jsonl"


# ============================================================
# MERGE NPZ
# ============================================================

def merge_npz_files(input_files: list[Path], output_file: Path) -> None:
    if not input_files:
        raise RuntimeError("Không tìm thấy file NPZ để merge.")

    all_vectors = []
    all_frame_ids = []

    expected_dim = None
    expected_model_id = None
    expected_embedding_source = None
    expected_embedding_norm = None

    print("📦 Merge NPZ:")

    for path in input_files:
        print(f"  → {path.name}")

        with np.load(path, allow_pickle=False) as data:
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            frame_ids = np.asarray(data["frame_ids"], dtype=str)

            if vectors.ndim != 2:
                raise ValueError(
                    f"{path.name}: vectors phải có shape [N, D], "
                    f"hiện tại {vectors.shape}"
                )

            if len(vectors) != len(frame_ids):
                raise ValueError(
                    f"{path.name}: vectors={len(vectors)} "
                    f"nhưng frame_ids={len(frame_ids)}"
                )

            vector_dim = vectors.shape[1]

            model_id = (
                str(data["model_id"].item())
                if "model_id" in data
                else None
            )

            embedding_source = (
                str(data["embedding_source"].item())
                if "embedding_source" in data
                else None
            )

            embedding_norm = (
                str(data["embedding_norm"].item())
                if "embedding_norm" in data
                else None
            )

            if expected_dim is None:
                expected_dim = vector_dim
                expected_model_id = model_id
                expected_embedding_source = embedding_source
                expected_embedding_norm = embedding_norm

            if vector_dim != expected_dim:
                raise ValueError(
                    f"{path.name}: vector_dim={vector_dim}, "
                    f"expected={expected_dim}"
                )

            if model_id != expected_model_id:
                raise ValueError(
                    f"{path.name}: model_id khác batch trước."
                )

            if embedding_source != expected_embedding_source:
                raise ValueError(
                    f"{path.name}: embedding_source khác batch trước."
                )

            if embedding_norm != expected_embedding_norm:
                raise ValueError(
                    f"{path.name}: embedding_norm khác batch trước."
                )

            all_vectors.append(vectors)
            all_frame_ids.append(frame_ids)

            print(
                f"     {len(vectors):,} vectors "
                f"| dim={vector_dim}"
            )

    merged_vectors = np.concatenate(
        all_vectors,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    merged_frame_ids = np.concatenate(
        all_frame_ids,
        axis=0,
    )

    # Kiểm tra duplicate frame ID.
    unique_count = len(
        np.unique(merged_frame_ids)
    )

    if unique_count != len(merged_frame_ids):
        print(
            f"⚠️ Có "
            f"{len(merged_frame_ids) - unique_count:,} "
            f"frame_id bị trùng."
        )

    np.savez_compressed(
        output_file,
        vectors=merged_vectors,
        frame_ids=merged_frame_ids,
        model_id=np.asarray(expected_model_id),
        embedding_source=np.asarray(
            expected_embedding_source
        ),
        embedding_norm=np.asarray(
            expected_embedding_norm
        ),
        vector_dim=np.asarray(
            expected_dim,
            dtype=np.int32,
        ),
    )

    print()
    print("✅ NPZ merged")
    print(f"   Output : {output_file}")
    print(
        f"   Shape  : "
        f"{merged_vectors.shape}"
    )


# ============================================================
# MERGE JSONL
# ============================================================

def merge_jsonl_files(input_files: list[Path], output_file: Path) -> None:
    if not input_files:
        raise RuntimeError(
            "Không tìm thấy file JSONL để merge."
        )

    total_lines = 0
    seen_frame_ids = set()
    duplicate_count = 0

    print()
    print("📄 Merge JSONL:")

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for path in input_files:
            file_count = 0

            print(f"  → {path.name}")

            with path.open(
                "r",
                encoding="utf-8",
            ) as fin:

                for line in fin:
                    line = line.strip()

                    if not line:
                        continue

                    item = json.loads(line)

                    frame_key = item.get(
                        "global_frame_id"
                    )

                    if frame_key:
                        if frame_key in seen_frame_ids:
                            duplicate_count += 1
                        else:
                            seen_frame_ids.add(
                                frame_key
                            )

                    fout.write(
                        json.dumps(
                            item,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    total_lines += 1
                    file_count += 1

            print(
                f"     {file_count:,} records"
            )

    print()
    print("✅ JSONL merged")
    print(f"   Output : {output_file}")
    print(
        f"   Records: {total_lines:,}"
    )

    if duplicate_count:
        print(
            f"⚠️ Duplicate global_frame_id: "
            f"{duplicate_count:,}"
        )


# ============================================================
# VERIFY NPZ ↔ JSONL
# ============================================================

def verify_outputs(npz_path: Path, jsonl_path: Path) -> None:
    with np.load(
        npz_path,
        allow_pickle=False,
    ) as data:
        vector_count = len(
            data["vectors"]
        )

        frame_id_count = len(
            data["frame_ids"]
        )

    metadata_count = 0

    with jsonl_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                metadata_count += 1

    print()
    print("=" * 60)
    print("🔍 VERIFY")
    print("=" * 60)

    print(
        f"Vectors   : {vector_count:,}"
    )

    print(
        f"Frame IDs : {frame_id_count:,}"
    )

    print(
        f"Metadata  : {metadata_count:,}"
    )

    if (
        vector_count
        == frame_id_count
        == metadata_count
    ):
        print(
            "✅ NPZ và JSONL khớp hoàn toàn."
        )

    else:
        print(
            "❌ Số lượng không khớp!"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    npz_files = sorted(
        INPUT_DIR.glob(NPZ_PATTERN)
    )

    jsonl_files = sorted(
        INPUT_DIR.glob(JSONL_PATTERN)
    )

    print("NPZ files:")
    for path in npz_files:
        print(f"  {path.name}")

    print()
    print("JSONL files:")
    for path in jsonl_files:
        print(f"  {path.name}")

    if len(npz_files) != len(jsonl_files):
        raise RuntimeError(
            f"Số batch NPZ ({len(npz_files)}) "
            f"khác JSONL ({len(jsonl_files)})."
        )

    merge_npz_files(
        npz_files,
        OUTPUT_NPZ,
    )

    merge_jsonl_files(
        jsonl_files,
        OUTPUT_JSONL,
    )

    verify_outputs(
        OUTPUT_NPZ,
        OUTPUT_JSONL,
    )


if __name__ == "__main__":
    main()