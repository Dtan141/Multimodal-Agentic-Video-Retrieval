"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=1000)
    include_videos: list[str] = Field(default_factory=list)
    exclude_videos: list[str] = Field(default_factory=list)


class HybridRequest(BaseModel):
    query: str = ""  # semantic query is optional (can search by modules alone)
    modules: dict[str, str] = Field(default_factory=dict, description="module -> keywords")
    semantic_query: str | None = None
    top_k: int | None = Field(None, ge=1, le=1000)
    rerank: bool = False  # optional cross-encoder rerank; off by default
    include_videos: list[str] = Field(default_factory=list)
    exclude_videos: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[dict[str, Any]]


class AgentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=1000)
    rerank: bool = False  # optional cross-encoder rerank; off by default
    include_videos: list[str] = Field(default_factory=list)
    exclude_videos: list[str] = Field(default_factory=list)


class AgentSearchResponse(BaseModel):
    query: str
    plan: dict[str, Any]
    count: int
    results: list[dict[str, Any]]


class SubmissionRequest(BaseModel):
    query_type: str = Field(..., description="KIS | Q&A | TRAKE")
    rows: list[dict[str, Any]]
    event_count: int = 1


class SubmissionResponse(BaseModel):
    csv: str | None
    errors: list[str]
    warnings: list[str]
    row_count: int


class HealthResponse(BaseModel):
    status: str
    vectors: int
    modules: list[str]


class PrefetchItem(BaseModel):
    video_id: str
    frame_id: int
    filename: str | None = None


class PrefetchRequest(BaseModel):
    items: list[PrefetchItem] = Field(default_factory=list)
