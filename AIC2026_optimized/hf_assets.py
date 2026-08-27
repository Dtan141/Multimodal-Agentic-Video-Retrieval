from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import requests
from huggingface_hub import HfApi, get_token, hf_hub_url

from config import (
    HF_IMAGE_REPO_ID,
    HF_IMAGE_REPO_TYPE,
    HF_IMAGE_ROOT,
    IMAGE_CACHE_SIZE,
    IMAGE_FETCH_WORKERS,
)

_thread_local = threading.local()


def get_hf_auth_uncached() -> tuple[str, str]:
    token = get_token()
    if not token:
        raise RuntimeError("Chưa đăng nhập Hugging Face. Hãy chạy `hf auth login` rồi khởi động lại Streamlit.")

    api = HfApi(token=token)
    user = api.whoami()
    username = user.get("name", "authenticated") if isinstance(user, dict) else "authenticated"
    api.dataset_info(repo_id=HF_IMAGE_REPO_ID)
    return token, username


def build_hf_image_url(item: dict) -> str | None:
    source_folder = item.get("source_folder")
    video_id = item.get("video_id")
    filename = item.get("filename")
    if not source_folder or not video_id or not filename:
        return None

    path_in_repo = f"{HF_IMAGE_ROOT}/{source_folder}/{video_id}_keyframes/{filename}"
    return hf_hub_url(
        repo_id=HF_IMAGE_REPO_ID,
        filename=path_in_repo,
        repo_type=HF_IMAGE_REPO_TYPE,
    )


def _session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=1)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _thread_local.session = session
    return session


@lru_cache(maxsize=IMAGE_CACHE_SIZE)
def fetch_private_hf_image(image_url: str, token: str) -> bytes:
    response = _session().get(
        image_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=(5, 20),
        allow_redirects=True,
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise RuntimeError(f"HF trả về content-type không phải ảnh: {content_type or 'unknown'}")
    return response.content


def fetch_images_parallel(urls: list[str], token: str, max_workers: int = IMAGE_FETCH_WORKERS) -> dict[str, bytes | Exception]:
    unique_urls = list(dict.fromkeys(url for url in urls if url))
    if not unique_urls:
        return {}

    results: dict[str, bytes | Exception] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(unique_urls))) as executor:
        future_map = {
            executor.submit(fetch_private_hf_image, url, token): url
            for url in unique_urls
        }
        for future in as_completed(future_map):
            url = future_map[future]
            try:
                results[url] = future.result()
            except Exception as exc:
                results[url] = exc
    return results
