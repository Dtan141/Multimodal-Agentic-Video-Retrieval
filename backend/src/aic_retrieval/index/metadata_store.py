"""Frame metadata store aligned row-for-row with the FAISS index.

Row ``i`` in :class:`~aic_retrieval.index.faiss_index.FaissIndex` corresponds to
row ``i`` here. Backed by a single parquet file so ~400K rows load fast and use
little memory. Timestamp (frame_id / fps) is added later in WS3/WS4 once the
per-video fps from the metadata JSON is available.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aic_retrieval.logging_conf import get_logger

logger = get_logger(__name__)

# Columns kept from the per-frame *_metadata.jsonl sidecars.
COLUMNS = [
    "global_frame_id",
    "video_id",
    "frame_id",
    "filename",
    "source_folder",
    "folder_id",
]

_PARQUET_NAME = "frames.parquet"


class MetadataStore:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    @classmethod
    def from_records(cls, records: list[dict]) -> MetadataStore:
        df = pd.DataFrame.from_records(records, columns=COLUMNS)
        # frame_id as nullable integer (some rows may miss it).
        df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce").astype("Int64")
        return cls(df)

    def save(self, index_dir: Path) -> Path:
        index_dir.mkdir(parents=True, exist_ok=True)
        path = index_dir / _PARQUET_NAME
        self.df.to_parquet(path, index=False)
        logger.info("Saved metadata store: %s (%d rows)", path, len(self.df))
        return path

    @classmethod
    def load(cls, index_dir: Path) -> MetadataStore:
        path = index_dir / _PARQUET_NAME
        if not path.exists():
            raise FileNotFoundError(f"Metadata store not found: {path}")
        return cls(pd.read_parquet(path))

    def get(self, row_id: int) -> dict:
        return self._row_to_dict(row_id, self.df.iloc[row_id])

    def get_many(self, row_ids: list[int]) -> list[dict]:
        sub = self.df.iloc[row_ids]
        return [
            self._row_to_dict(int(rid), row)
            for rid, (_, row) in zip(row_ids, sub.iterrows(), strict=False)
        ]

    @staticmethod
    def _row_to_dict(row_id: int, row: pd.Series) -> dict:
        item = row.to_dict()
        frame_id = item.get("frame_id")
        item["frame_id"] = None if pd.isna(frame_id) else int(frame_id)
        item["row_id"] = row_id
        return item
