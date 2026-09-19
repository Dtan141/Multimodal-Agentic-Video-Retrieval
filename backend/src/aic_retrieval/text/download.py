"""Download the per-video metadata JSON (OCR/ASR/Object/Caption) from Hugging Face.

Uses the *full* ``Videos_*/*.json`` tree (has speech/ASR, unlike
``metadata_optimized/``). Token comes from ``hf auth login``.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from aic_retrieval.logging_conf import get_logger

logger = get_logger(__name__)


def download_metadata(
    repo_id: str,
    repo_type: str,
    dest_dir: Path,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Download metadata JSON into ``dest_dir`` and return the local path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    patterns = allow_patterns or ["Videos_*/*.json"]
    logger.info("Downloading %s (%s) patterns=%s -> %s", repo_id, repo_type, patterns, dest_dir)
    local = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=patterns,
        local_dir=str(dest_dir),
    )
    count = sum(1 for _ in Path(local).rglob("Videos_*/*.json"))
    logger.info("Downloaded %d metadata JSON files into %s", count, local)
    return Path(local)
