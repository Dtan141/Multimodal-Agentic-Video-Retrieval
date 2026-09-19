"""SearchService: the single orchestration entry point.

Used by both the FastAPI routes (WS6) and the agent (WS5). Provides:
  * semantic_search      - vector only (always available)
  * module_search        - one metadata module (agent tool / manual branch)
  * hybrid_search        - semantic + chosen modules -> RRF -> hydrate -> rerank
"""

from __future__ import annotations

from aic_retrieval.config import Settings, get_settings
from aic_retrieval.logging_conf import get_logger
from aic_retrieval.search.fusion import rrf_fuse
from aic_retrieval.search.hydration import Hydrator
from aic_retrieval.search.lexical import LexicalSearcher
from aic_retrieval.search.semantic import SemanticSearcher
from aic_retrieval.search.video_filter import VideoScope

logger = get_logger(__name__)


class SearchService:
    def __init__(
        self,
        settings: Settings,
        semantic: SemanticSearcher,
        lexical: LexicalSearcher,
        hydrator: Hydrator,
    ) -> None:
        self.settings = settings
        self.semantic = semantic
        self.lexical = lexical
        self.hydrator = hydrator
        self.scope = VideoScope.from_store(semantic.store)
        self._reranker = None  # lazy (downloads a model on first use)

    @classmethod
    def load(cls, settings: Settings | None = None) -> SearchService:
        settings = settings or get_settings()
        logger.info("Loading SearchService ...")
        return cls(
            settings,
            SemanticSearcher.load(settings),
            LexicalSearcher.load(settings),
            Hydrator.load(settings.index_dir),
        )

    @property
    def reranker(self):
        if self._reranker is None:
            from aic_retrieval.search.rerank import CrossEncoderReranker

            self._reranker = CrossEncoderReranker.load(self.settings)
        return self._reranker

    @property
    def modules(self) -> list[str]:
        return self.lexical.available

    # ---- primitives (also exposed as agent tools) -------------------------
    def semantic_search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[dict]:
        allowed = self.scope.allowed_row_ids(include, exclude)
        return self.semantic.search(query, top_k or self.settings.top_k, allowed_ids=allowed)

    def module_search(
        self,
        module: str,
        query: str,
        top_k: int | None = None,
        *,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[dict]:
        keep = VideoScope.keep_fn(include, exclude)
        return self.lexical.search(module, query, top_k or self.settings.top_k, keep=keep)

    # ---- hybrid -----------------------------------------------------------
    def hybrid_search(
        self,
        query: str,
        module_queries: dict[str, str] | None = None,
        *,
        semantic_query: str | None = None,
        rerank_query: str | None = None,
        top_k: int | None = None,
        rerank: bool = False,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[dict]:
        top_k = top_k or self.settings.top_k
        module_queries = module_queries or {}

        allowed = self.scope.allowed_row_ids(include, exclude)
        keep = VideoScope.keep_fn(include, exclude)

        # Semantic is optional: run it only when there is query text. This lets the
        # manual mode search by metadata modules alone (no semantic query).
        semantic_text = (semantic_query or query or "").strip()
        sources: dict[str, list[dict]] = {}
        if semantic_text:
            sources["semantic"] = self.semantic.search(semantic_text, top_k, allowed_ids=allowed)
        for module, kw in module_queries.items():
            if kw and kw.strip():
                sources[f"meta:{module}"] = self.lexical.search(module, kw, top_k, keep=keep)

        if not sources:
            return []

        fused = rrf_fuse(
            sources,
            rrf_k=self.settings.rrf_k,
            temporal_margin=self.settings.temporal_margin,
            top_k=top_k,
        )
        hydrated = [self.hydrator.hydrate(h) for h in fused]

        if rerank and len(sources) > 1:
            # captions are English + the cross-encoder is English-only, so rerank
            # with the English query (semantic_query) when available.
            rq = rerank_query or semantic_query or query
            hydrated = self.reranker.rerank(rq, hydrated, top_n=top_k)
        return hydrated
