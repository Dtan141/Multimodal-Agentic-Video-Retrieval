"""WS2 verification: FAISS index == brute-force, and npz<->faiss<->metadata alignment.

Skips automatically if the index has not been built yet.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from aic_retrieval.config import get_settings
from aic_retrieval.index.faiss_index import FaissIndex
from aic_retrieval.index.metadata_store import MetadataStore

settings = get_settings()
INDEX_DIR = settings.index_dir
pytestmark = pytest.mark.skipif(
    not (INDEX_DIR / "vectors.faiss").exists(),
    reason="index not built (run `aic build-index`)",
)


@pytest.fixture(scope="module")
def index() -> FaissIndex:
    return FaissIndex.load(INDEX_DIR)


@pytest.fixture(scope="module")
def store() -> MetadataStore:
    return MetadataStore.load(INDEX_DIR)


def test_index_and_store_same_length(index, store):
    assert index.ntotal == len(store)
    assert index.dim == 768


def test_self_query_returns_self(index):
    for row in (0, 1234, index.ntotal // 2, index.ntotal - 1):
        vec = index.index.reconstruct(row)
        scores, ids = index.search(vec, top_k=1)
        assert int(ids[0][0]) == row
        assert scores[0][0] == pytest.approx(1.0, abs=1e-3)


def test_faiss_matches_bruteforce(index):
    # Reconstruct the full matrix once; compare faiss top-k to numpy dot top-k.
    mat = index.index.reconstruct_n(0, index.ntotal)  # [N, D], already normalized
    top_k = 20
    for q_row in (0, 5000, index.ntotal - 3):
        q = mat[q_row]
        scores, ids = index.search(q, top_k=top_k)
        faiss_ids = ids[0].tolist()

        bf = mat @ q
        bf_ids = np.argpartition(bf, -top_k)[-top_k:]
        bf_ids = bf_ids[np.argsort(bf[bf_ids])[::-1]].tolist()

        assert faiss_ids == bf_ids, f"mismatch for query row {q_row}"


def test_alignment_npz_faiss_metadata(index, store):
    # Take a source vector from an npz, find its metadata row, reconstruct that
    # row from faiss, and confirm the vectors match => end-to-end alignment.
    npz_path = settings.embeddings_dir / "Videos_L21_embeddings.npz"
    if not npz_path.exists():
        pytest.skip("L21 npz missing")

    with np.load(npz_path, allow_pickle=False) as data:
        src_vec = np.asarray(data["vectors"][0], dtype=np.float32)

    jsonl = settings.embeddings_dir / "Videos_L21_metadata.jsonl"
    with jsonl.open(encoding="utf-8") as f:
        first = json.loads(f.readline())
    key = first["global_frame_id"]

    row_id = int(store.df.index[store.df["global_frame_id"] == key][0])
    meta = store.get(row_id)
    assert meta["global_frame_id"] == key
    assert meta["video_id"] == first["video_id"]

    recon = index.index.reconstruct(row_id)
    # src is normalized already; cosine with reconstructed row must be ~1.
    cos = float(np.dot(recon, src_vec) / (np.linalg.norm(recon) * np.linalg.norm(src_vec)))
    assert cos == pytest.approx(1.0, abs=1e-3)
