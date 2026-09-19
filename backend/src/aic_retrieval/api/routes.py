"""API routes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response

from aic_retrieval.api.schemas import (
    AgentSearchRequest,
    AgentSearchResponse,
    HealthResponse,
    HybridRequest,
    PrefetchRequest,
    SearchRequest,
    SearchResponse,
    SubmissionRequest,
    SubmissionResponse,
)
from aic_retrieval.hf_assets import (
    get_frame_bytes,
    get_video_info,
    get_video_path,
    keyframe_filename,
    source_folder_for,
    warm_keyframe,
)
from aic_retrieval.logging_conf import get_logger
from aic_retrieval.submission import build_submission_csv

logger = get_logger(__name__)
router = APIRouter()


def _service(request: Request):
    svc = getattr(request.app.state, "service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Search service not loaded")
    return svc


def _prefer_root(hit: dict) -> str | None:
    """Pick the keyframe root to try first from the hit's origin."""
    sources = hit.get("sources") or ([hit["source"]] if hit.get("source") else [])
    if "semantic" in sources:
        return "vector"
    if any(str(s).startswith("meta:") for s in sources):
        return "metadata"
    return None


def _enrich(settings, hits: list[dict]) -> list[dict]:
    """Ensure each hit has filename/source_folder + a proxy image path."""
    out = []
    for h in hits:
        h = dict(h)
        vid = h.get("video_id")
        fid = h.get("frame_id")
        if vid and fid is not None:
            h.setdefault("filename", keyframe_filename(vid, fid))
            h.setdefault("source_folder", source_folder_for(vid))
            prefer = _prefer_root(h)
            h["image_path"] = f"/image?video_id={vid}&frame_id={fid}" + (f"&set={prefer}" if prefer else "")
        out.append(h)
    return out


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    svc = _service(request)
    return HealthResponse(status="ok", vectors=svc.semantic.index.ntotal, modules=svc.modules)


@router.get("/modules")
def modules(request: Request) -> dict:
    return {"modules": _service(request).modules}


@router.post("/search", response_model=SearchResponse)
def search(request: Request, body: SearchRequest) -> SearchResponse:
    svc = _service(request)
    hits = svc.semantic_search(
        body.query, top_k=body.top_k, include=body.include_videos, exclude=body.exclude_videos
    )
    return SearchResponse(query=body.query, count=len(hits), results=_enrich(svc.settings, hits))


@router.post("/search/hybrid", response_model=SearchResponse)
def search_hybrid(request: Request, body: HybridRequest) -> SearchResponse:
    svc = _service(request)
    unknown = set(body.modules) - set(svc.modules)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown modules: {sorted(unknown)}")
    if not body.query.strip() and not any((v or "").strip() for v in body.modules.values()):
        raise HTTPException(status_code=400, detail="Provide a query or at least one module keyword")
    hits = svc.hybrid_search(
        body.query,
        module_queries=body.modules,
        semantic_query=body.semantic_query,
        top_k=body.top_k,
        rerank=body.rerank,
        include=body.include_videos,
        exclude=body.exclude_videos,
    )
    return SearchResponse(query=body.query, count=len(hits), results=_enrich(svc.settings, hits))


@router.post("/agent/search", response_model=AgentSearchResponse)
def agent_search(request: Request, body: AgentSearchRequest) -> AgentSearchResponse:
    svc = _service(request)
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent unavailable (GROQ_API_KEY not set)")
    result = agent.run(
        body.query,
        top_k=body.top_k,
        rerank=body.rerank,
        include=body.include_videos,
        exclude=body.exclude_videos,
    )
    result["results"] = _enrich(svc.settings, result["results"])
    return AgentSearchResponse(**result)


@router.get("/image")
def image(
    request: Request,
    video_id: str = Query(...),
    frame_id: int | None = Query(None),
    filename: str | None = Query(None),
    source_folder: str | None = Query(None),
    full: bool = Query(False),
    set_: str | None = Query(None, alias="set"),
) -> Response:
    svc = _service(request)
    if not filename and frame_id is None:
        raise HTTPException(status_code=400, detail="Provide filename or frame_id")
    try:
        data = get_frame_bytes(
            svc.settings, video_id, frame_id, filename, source_folder, prefer=set_, full=full
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"Image not available: {exc}") from exc
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@router.post("/prefetch")
def prefetch(request: Request, body: PrefetchRequest, background: BackgroundTasks) -> dict:
    """Warm the thumbnail cache for the given keyframes in the background."""
    svc = _service(request)
    items = body.items[: svc.settings.top_k * 2]

    def _warm() -> None:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for it in items:
                ex.submit(warm_keyframe, svc.settings, it.video_id, it.frame_id, it.filename)

    background.add_task(_warm)
    return {"warming": len(items)}


@router.get("/video/{video_id}/info")
def video_info(request: Request, video_id: str) -> dict:
    svc = _service(request)
    vid = video_id[:-4] if video_id.lower().endswith(".mp4") else video_id
    try:
        path = get_video_path(svc.settings, vid)
        info = get_video_info(str(path))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"video_id": vid, **info}


@router.get("/video/{video_id}")
def video(request: Request, video_id: str) -> FileResponse:
    svc = _service(request)
    vid = video_id[:-4] if video_id.lower().endswith(".mp4") else video_id
    try:
        path = get_video_path(svc.settings, vid)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # starlette FileResponse handles HTTP Range requests (seeking) automatically.
    return FileResponse(path, media_type="video/mp4", filename=f"{vid}.mp4")


@router.post("/submission/validate", response_model=SubmissionResponse)
def submission_validate(body: SubmissionRequest) -> SubmissionResponse:
    result = build_submission_csv(body.query_type, body.rows, body.event_count)
    return SubmissionResponse(**result)
