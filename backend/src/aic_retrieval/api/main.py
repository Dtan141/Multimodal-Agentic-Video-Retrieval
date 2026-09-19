"""FastAPI application. Loads the SearchService once at startup (lifespan)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aic_retrieval.api.routes import router
from aic_retrieval.config import get_settings
from aic_retrieval.logging_conf import get_logger, setup_logging
from aic_retrieval.search.service import SearchService

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Starting AIC retrieval API ...")
    app.state.settings = settings
    app.state.service = SearchService.load(settings)  # FAISS + BM25 + embedder, once

    app.state.agent = None
    if settings.groq_api_key:
        from aic_retrieval.agent.llm_client import LLMClient
        from aic_retrieval.agent.orchestrator import AgentSearcher

        app.state.agent = AgentSearcher(app.state.service, LLMClient.from_settings(settings))
        logger.info("Agent enabled (model=%s)", settings.groq_model)
    else:
        logger.warning("GROQ_API_KEY not set -> /agent/search disabled")

    logger.info("Ready: %d vectors, modules=%s", app.state.service.semantic.index.ntotal, app.state.service.modules)
    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="AIC 2026 Retrieval API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
