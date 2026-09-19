"""Hugging Face asset access: keyframe images + source videos.

Ported from ``AIC2026_retrieval_pipeline.py`` (build_hf_image_url:298,
fetch_private_hf_image:103, download_video_from_hf:369) into a reusable module
that reads its config from :class:`Settings` and uses the ambient HF token.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import httpx
from huggingface_hub import get_token, hf_hub_download, hf_hub_url

from aic_retrieval.config import Settings
from aic_retrieval.logging_conf import get_logger

logger = get_logger(__name__)


def source_folder_for(video_id: str) -> str:
    """L21_V001 -> Videos_L21."""
    prefix = video_id.split("_", 1)[0]
    return f"Videos_{prefix}"


def keyframe_filename(video_id: str, frame_id: int) -> str:
    return f"{video_id}_{int(frame_id):05d}.jpg"


def build_image_url(settings: Settings, video_id: str, filename: str, source_folder: str | None = None) -> str:
    source_folder = source_folder or source_folder_for(video_id)
    path_in_repo = f"{settings.hf_image_root}/{source_folder}/{video_id}_keyframes/{filename}"
    return hf_hub_url(
        repo_id=settings.hf_image_repo_id,
        filename=path_in_repo,
        repo_type=settings.hf_image_repo_type,
    )


def fetch_image_bytes(url: str, token: str | None = None) -> bytes:
    """Legacy uncached HTTP fetch (kept for reference; routes use the cached path)."""
    token = token or get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = httpx.get(url, headers=headers, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    if not ctype.startswith("image/"):
        raise ValueError(f"HF returned non-image content-type: {ctype or 'unknown'}")
    return resp.content


# ---- cached keyframe access (disk via HF cache + in-memory thumbnail LRU) ----
def _path_in_repo(settings: Settings, video_id: str, filename: str, source_folder: str | None, root: str) -> str:
    source_folder = source_folder or source_folder_for(video_id)
    return f"{root}/{source_folder}/{video_id}_keyframes/{filename}"


@lru_cache(maxsize=8192)
def _download_keyframe(repo_id: str, repo_type: str, path_in_repo: str) -> str:
    """Local path to the keyframe, downloaded once and cached on disk by HF."""
    return hf_hub_download(
        repo_id=repo_id, repo_type=repo_type, filename=path_in_repo, token=get_token()
    )


@lru_cache(maxsize=4096)
def _thumb_bytes(local_path: str, width: int, quality: int) -> bytes:
    import io

    from PIL import Image

    img = Image.open(local_path).convert("RGB")
    w, h = img.size
    if w > width:
        img = img.resize((width, max(1, round(h * width / w))))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def get_keyframe_bytes(
    settings: Settings,
    video_id: str,
    filename: str,
    source_folder: str | None = None,
    *,
    root: str | None = None,
    full: bool = False,
    width: int | None = None,
) -> bytes:
    """Return keyframe bytes: a cached downscaled thumbnail (default) or the original."""
    root = root or settings.hf_image_root
    path_in_repo = _path_in_repo(settings, video_id, filename, source_folder, root)
    local = _download_keyframe(settings.hf_image_repo_id, settings.hf_image_repo_type, path_in_repo)
    if full:
        return Path(local).read_bytes()
    return _thumb_bytes(local, width or settings.thumb_width, settings.thumb_quality)


def warm_keyframe(settings: Settings, video_id: str, frame_id: int, filename: str | None = None) -> None:
    """Best-effort cache warm (used by /prefetch); swallows errors.

    Warms whichever uploaded keyframe root has the image (vector or metadata) — both
    are cheap thumbnails. Frames in neither (needing video extraction) are skipped.
    """
    fn = filename or keyframe_filename(video_id, frame_id)
    for root in (settings.hf_image_root, settings.hf_image_root_metadata):
        try:
            get_keyframe_bytes(settings, video_id, fn, root=root)
            return
        except Exception:  # noqa: BLE001
            continue
    logger.debug("prefetch warm skipped for %s#%s (no uploaded keyframe)", video_id, frame_id)


# ---- exact-frame extraction from source video (for metadata-set frames) ----
def _frame_id_from_filename(filename: str) -> int | None:
    try:
        return int(Path(filename).stem.split("_")[-1])
    except (ValueError, IndexError):
        return None


@lru_cache(maxsize=2048)
def _extract_frame_bytes(video_path: str, frame_id: int, width: int, quality: int, full: bool) -> bytes:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not read frame {frame_id}")
    if not full:
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, max(1, round(h * width / w))))
    ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to encode frame")
    return enc.tobytes()


def get_frame_bytes(
    settings: Settings,
    video_id: str,
    frame_id: int | None = None,
    filename: str | None = None,
    source_folder: str | None = None,
    *,
    prefer: str | None = None,
    full: bool = False,
    width: int | None = None,
) -> bytes:
    """Image bytes for a result frame.

    Both keyframe sets have uploaded thumbnails on HF: vector frames under
    ``keyframe_vector/`` and metadata frames under ``keyframe_metadata/``. We try the
    preferred root first (from the hit's source), then the other. Only if a frame is in
    neither (e.g. an arbitrary frame from the inspector) do we extract it from the source
    video via OpenCV (cached).
    """
    filename = filename or (keyframe_filename(video_id, frame_id) if frame_id is not None else None)

    roots = [settings.hf_image_root, settings.hf_image_root_metadata]
    if prefer == "metadata":
        roots = [settings.hf_image_root_metadata, settings.hf_image_root]

    if filename:
        for root in roots:
            try:
                return get_keyframe_bytes(
                    settings, video_id, filename, source_folder, root=root, full=full, width=width
                )
            except Exception:  # noqa: BLE001 - try next root, then video
                continue

    fid = frame_id if frame_id is not None else _frame_id_from_filename(filename or "")
    if fid is None:
        raise FileNotFoundError("no uploaded keyframe and no frame_id to extract")
    video_path = get_video_path(settings, video_id)
    return _extract_frame_bytes(str(video_path), int(fid), width or settings.thumb_width, settings.thumb_quality, full)


def _candidate_video_paths(video_id: str) -> list[str]:
    folder = source_folder_for(video_id)
    prefix = video_id.split("_", 1)[0]
    candidates = [
        f"{folder}/{video_id}.mp4",
        f"videos/{folder}/{video_id}.mp4",
        f"Videos/{folder}/{video_id}.mp4",
        f"videos/{prefix}/{video_id}.mp4",
        f"{video_id}.mp4",
    ]
    return list(dict.fromkeys(candidates))


@lru_cache(maxsize=256)
def resolve_video_path(settings_key: str, video_id: str) -> str:
    """Return a local path to the video, downloading (cached) from HF if needed.

    ``settings_key`` carries the repo id/type so lru_cache stays correct across
    config changes while keeping the signature hashable.
    """
    repo_id, repo_type = settings_key.split("|", 1)
    token = get_token()
    errors: list[str] = []
    for path_in_repo in _candidate_video_paths(video_id):
        try:
            return hf_hub_download(
                repo_id=repo_id, filename=path_in_repo, repo_type=repo_type, token=token
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path_in_repo}: {type(exc).__name__}")
    raise FileNotFoundError(f"Video {video_id} not found. Tried: {'; '.join(errors)}")


def get_video_path(settings: Settings, video_id: str) -> Path:
    key = f"{settings.hf_video_repo_id}|{settings.hf_video_repo_type}"
    return Path(resolve_video_path(key, video_id))


@lru_cache(maxsize=256)
def get_video_info(path: str) -> dict:
    """fps / frame_count / duration / width / height (ported from the old app)."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0:
        fps = 25.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": frame_count / fps if frame_count else 0.0,
        "width": width,
        "height": height,
    }
