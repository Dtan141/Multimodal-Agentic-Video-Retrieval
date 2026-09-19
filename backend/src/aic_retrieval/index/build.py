"""Build the FAISS index + metadata store from the ``*_embeddings.npz`` shards.

Mirrors the dedup/alignment logic of the old ``load_all_embeddings``
(``AIC2026_retrieval_pipeline.py:154-281``) but persists the result instead of
rebuilding it in RAM on every session. Vectors are added to FAISS per shard to
keep peak memory low.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aic_retrieval.index.faiss_index import FaissIndex
from aic_retrieval.index.metadata_store import COLUMNS, MetadataStore
from aic_retrieval.logging_conf import get_logger

logger = get_logger(__name__)


def _metadata_path(npz_path: Path) -> Path:
    if npz_path.name.endswith("_embeddings.npz"):
        return npz_path.with_name(npz_path.name.replace("_embeddings.npz", "_metadata.jsonl"))
    return npz_path.with_suffix(".jsonl")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _frame_key(item: dict, fallback: str) -> str:
    if item.get("global_frame_id"):
        return str(item["global_frame_id"])
    if item.get("video_id") is not None and item.get("frame_id") is not None:
        return f"{item['video_id']}_{int(item['frame_id']):05d}"
    return fallback


def build_index(
    embeddings_dir: Path,
    index_dir: Path,
) -> dict:
    """Build and persist the FAISS index + metadata store.

    Returns a summary dict (per-file counts + totals).
    """
    npz_files = sorted(embeddings_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {embeddings_dir}")

    faiss_index: FaissIndex | None = None
    expected_dim: int | None = None
    records: list[dict] = []
    seen_keys: set[str] = set()
    summary: list[dict] = []

    for npz_path in npz_files:
        rel = npz_path.name
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                if "vectors" not in data:
                    summary.append({"file": rel, "status": "skip: no vectors", "kept": 0})
                    continue
                vectors = np.asarray(data["vectors"], dtype=np.float32)
                if vectors.ndim != 2:
                    summary.append({"file": rel, "status": f"skip: shape {vectors.shape}", "kept": 0})
                    continue
                frame_ids = (
                    [str(x) for x in data["frame_ids"].tolist()]
                    if "frame_ids" in data
                    else [f"{npz_path.stem}_{i}" for i in range(len(vectors))]
                )

            if expected_dim is None:
                expected_dim = vectors.shape[1]
                faiss_index = FaissIndex.empty(expected_dim)
            if vectors.shape[1] != expected_dim:
                summary.append({"file": rel, "status": f"skip: dim {vectors.shape[1]}!={expected_dim}", "kept": 0})
                continue

            metadata = _read_jsonl(_metadata_path(npz_path))
            if metadata and len(metadata) != len(vectors):
                summary.append({"file": rel, "status": f"skip: meta {len(metadata)}!=vec {len(vectors)}", "kept": 0})
                continue
            if not metadata:
                metadata = [{"global_frame_id": fid} for fid in frame_ids]

            keep_idx: list[int] = []
            for i, item in enumerate(metadata):
                fallback = frame_ids[i] if i < len(frame_ids) else f"{npz_path.stem}_{i}"
                key = _frame_key(item, fallback)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                keep_idx.append(i)
                records.append(
                    {
                        "global_frame_id": key,
                        "video_id": item.get("video_id"),
                        "frame_id": item.get("frame_id"),
                        "filename": item.get("filename"),
                        "source_folder": item.get("source_folder"),
                        "folder_id": item.get("folder_id"),
                    }
                )

            if keep_idx:
                assert faiss_index is not None
                faiss_index.add(vectors[keep_idx])

            summary.append({"file": rel, "status": "ok", "total": int(len(vectors)), "kept": len(keep_idx)})
            logger.info("%s: kept %d / %d", rel, len(keep_idx), len(vectors))

        except Exception as exc:  # noqa: BLE001 - report and continue
            summary.append({"file": rel, "status": f"error: {exc}", "kept": 0})
            logger.error("Failed on %s: %s", rel, exc)

    if faiss_index is None or faiss_index.ntotal == 0:
        raise RuntimeError("No valid vectors were indexed.")

    if faiss_index.ntotal != len(records):
        raise RuntimeError(
            f"Index/metadata mismatch: {faiss_index.ntotal} vectors vs {len(records)} records"
        )

    faiss_index.save(index_dir)
    MetadataStore.from_records(records).save(index_dir)

    return {
        "total_vectors": faiss_index.ntotal,
        "dim": faiss_index.dim,
        "files": summary,
        "columns": COLUMNS,
    }
