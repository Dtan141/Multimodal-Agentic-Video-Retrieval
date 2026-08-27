from __future__ import annotations

import time

from config import EMBEDDINGS_DIR, SEARCH_INDEX_DIR
from retrieval_engine import build_engine_from_raw, save_compiled_index


def main() -> None:
    start = time.perf_counter()
    print(f"Embedding source : {EMBEDDINGS_DIR}")
    print(f"Search index out : {SEARCH_INDEX_DIR}")
    engine = build_engine_from_raw(EMBEDDINGS_DIR)
    save_compiled_index(engine, SEARCH_INDEX_DIR)
    elapsed = time.perf_counter() - start
    print(f"Done: {engine.size:,} vectors x {engine.dim} dims")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
