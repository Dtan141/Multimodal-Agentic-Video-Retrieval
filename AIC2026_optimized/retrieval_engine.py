from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def infer_metadata_path(npz_path: Path) -> Path:
    if npz_path.name.endswith("_embeddings.npz"):
        return npz_path.with_name(npz_path.name.replace("_embeddings.npz", "_metadata.jsonl"))
    return npz_path.with_suffix(".jsonl")


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


def l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, 1e-12)
    return vectors


@dataclass(slots=True)
class RetrievalEngine:
    vectors: np.ndarray
    metadata: list[dict]
    summary: list[dict]
    source: str

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    def search(self, query_vector: np.ndarray, top_k: int = 100) -> list[tuple[int, float]]:
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query_vector.shape[0] != self.dim:
            raise ValueError(f"Dimension mismatch: query={query_vector.shape[0]}, index={self.dim}")

        # vectors are normalized once at build/load time; query is normalized by SigLIP2.
        scores = self.vectors @ query_vector
        k = min(max(1, int(top_k)), self.size)

        if k == self.size:
            top_indices = np.argsort(scores)[::-1]
        else:
            candidate_indices = np.argpartition(scores, -k)[-k:]
            top_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

        return [(int(idx), float(scores[idx])) for idx in top_indices[:k]]


def build_engine_from_raw(root: Path) -> RetrievalEngine:
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
                    summary.append({"file": str(npz_path.relative_to(root)), "status": "skip: missing vectors", "total": 0, "kept": 0})
                    continue

                vectors = np.asarray(data["vectors"], dtype=np.float32)
                if vectors.ndim != 2:
                    summary.append({"file": str(npz_path.relative_to(root)), "status": f"skip: shape {vectors.shape}", "total": len(vectors), "kept": 0})
                    continue

                if expected_dim is None:
                    expected_dim = int(vectors.shape[1])
                if vectors.shape[1] != expected_dim:
                    summary.append({"file": str(npz_path.relative_to(root)), "status": f"skip: dim {vectors.shape[1]} != {expected_dim}", "total": len(vectors), "kept": 0})
                    continue

                frame_ids = (
                    [str(x) for x in data["frame_ids"].tolist()]
                    if "frame_ids" in data
                    else [f"{npz_path.stem}_{i}" for i in range(len(vectors))]
                )

            metadata = read_jsonl(metadata_path)
            if metadata and len(metadata) != len(vectors):
                summary.append({"file": str(npz_path.relative_to(root)), "status": f"skip: metadata {len(metadata)} != vectors {len(vectors)}", "total": len(vectors), "kept": 0})
                continue
            if not metadata:
                metadata = [{"global_frame_id": frame_id} for frame_id in frame_ids]

            keep_indices: list[int] = []
            kept_metadata: list[dict] = []
            rel_npz = str(npz_path.relative_to(root))

            for i, item in enumerate(metadata):
                fallback = frame_ids[i] if i < len(frame_ids) else f"{npz_path.stem}_{i}"
                key = get_frame_key(item, fallback)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                enriched = dict(item)
                enriched["_source_npz"] = rel_npz
                enriched["_frame_key"] = key
                keep_indices.append(i)
                kept_metadata.append(enriched)

            if keep_indices:
                chunk = np.ascontiguousarray(vectors[keep_indices], dtype=np.float32)
                l2_normalize_rows(chunk)  # exactly once, not on every query
                vector_chunks.append(chunk)
                all_metadata.extend(kept_metadata)

            summary.append({"file": rel_npz, "status": "ok", "total": len(vectors), "kept": len(keep_indices)})
        except Exception as exc:
            summary.append({"file": str(npz_path.relative_to(root)), "status": f"error: {exc}", "total": 0, "kept": 0})

    if not vector_chunks:
        raise RuntimeError("Không có vector hợp lệ để retrieval.")

    matrix = np.ascontiguousarray(np.concatenate(vector_chunks, axis=0), dtype=np.float32)
    return RetrievalEngine(matrix, all_metadata, summary, source="raw-npz")


def save_compiled_index(engine: RetrievalEngine, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "vectors.npy", engine.vectors, allow_pickle=False)

    with (index_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for item in engine.metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "version": 1,
        "vector_count": engine.size,
        "dim": engine.dim,
        "normalized": True,
        "source": engine.source,
        "files": engine.summary,
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_compiled_index(index_dir: Path) -> RetrievalEngine:
    vectors_path = index_dir / "vectors.npy"
    metadata_path = index_dir / "metadata.jsonl"
    manifest_path = index_dir / "manifest.json"

    if not vectors_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Compiled search index chưa tồn tại.")

    vectors = np.load(vectors_path, allow_pickle=False)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    metadata = read_jsonl(metadata_path)
    if len(metadata) != len(vectors):
        raise RuntimeError(f"Compiled index lỗi: metadata={len(metadata)} != vectors={len(vectors)}")

    summary: list[dict] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = list(manifest.get("files", []))

    return RetrievalEngine(vectors=vectors, metadata=metadata, summary=summary, source="compiled-index")
