"""WS6 API smoke tests (TestClient triggers lifespan -> loads the service).

Skips if the index/text stores are not built. Network endpoints (/image,
/video) are exercised manually, not here.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aic_retrieval.config import get_settings

settings = get_settings()
pytestmark = pytest.mark.skipif(
    not (settings.index_dir / "vectors.faiss").exists()
    or not (settings.index_dir / "text" / "caption.pkl").exists(),
    reason="index/text not built",
)


@pytest.fixture(scope="module")
def client():
    from aic_retrieval.api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["vectors"] > 0
    assert "object" in body["modules"]


def test_search(client):
    r = client.post("/search", json={"query": "a man riding a horse", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 5
    first = body["results"][0]
    assert {"video_id", "frame_id", "image_path", "source"} <= set(first)
    assert first["image_path"].startswith("/image?")


def test_hybrid_no_rerank(client):
    r = client.post(
        "/search/hybrid",
        json={"query": "a man riding a horse", "modules": {"object": "person horse"},
              "top_k": 10, "rerank": False},
    )
    assert r.status_code == 200
    assert r.json()["count"] > 0


def test_hybrid_unknown_module(client):
    r = client.post("/search/hybrid", json={"query": "x", "modules": {"nope": "y"}})
    assert r.status_code == 400


def test_submission_kis(client):
    r = client.post(
        "/submission/validate",
        json={"query_type": "KIS", "rows": [{"video_name": "L21_V001.mp4", "frame_id": 100}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["errors"] == []
    assert body["csv"].strip() == "L21_V001,100"


def test_submission_qa_answer_too_long(client):
    r = client.post(
        "/submission/validate",
        json={"query_type": "Q&A", "rows": [{"video_name": "V", "frame_id": 1, "answer": "x" * 101}]},
    )
    assert r.json()["errors"]
