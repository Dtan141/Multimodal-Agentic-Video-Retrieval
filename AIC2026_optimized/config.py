import os
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
BASE_WORKSPACE = Path(os.getenv("AIC_WORKSPACE", str(CURRENT_DIR / "workspace"))).expanduser().resolve()
EMBEDDINGS_DIR = BASE_WORKSPACE / "vector_embeddings"
SEARCH_INDEX_DIR = BASE_WORKSPACE / "search_index"

HF_IMAGE_REPO_ID = "Chillguy2026/AIC_2026_data"
HF_IMAGE_REPO_TYPE = "dataset"
HF_IMAGE_ROOT = "keyframe_vector"

HF_VIDEO_REPOS: tuple[tuple[str, str], ...] = (
    ("Chillguy2026/dataset_video", "dataset"),
)

LOCAL_VIDEO_ROOTS: tuple[Path, ...] = (
    CURRENT_DIR / "videos",
    CURRENT_DIR / "hf_download_videos",
    BASE_WORKSPACE / "videos",
)

MODEL_ID = "google/siglip2-base-patch16-224"
TEXT_MAX_LENGTH = 64
USE_FP16 = True

TOP_K = 100
GRID_COLUMNS = 5
DEFAULT_VISIBLE_RESULTS = 10
IMAGE_FETCH_WORKERS = 8
IMAGE_CACHE_SIZE = 512
MAX_SUBMISSION_ROWS = 100
