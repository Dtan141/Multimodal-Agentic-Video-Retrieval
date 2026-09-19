"""Agent orchestrator: route the query, then run hybrid search with the plan.

Reuses an already-loaded :class:`SearchService` (so the API doesn't double-load
models). Returns the ranked results plus a transparent reasoning trace of which
tools/keywords the agent chose.
"""

from __future__ import annotations

from aic_retrieval.agent.llm_client import LLMClient
from aic_retrieval.agent.router import route
from aic_retrieval.config import Settings, get_settings
from aic_retrieval.logging_conf import get_logger
from aic_retrieval.search.service import SearchService

logger = get_logger(__name__)


class AgentSearcher:
    def __init__(self, service: SearchService, llm: LLMClient) -> None:
        self.service = service
        self.llm = llm

    @classmethod
    def load(cls, settings: Settings | None = None, service: SearchService | None = None) -> AgentSearcher:
        settings = settings or get_settings()
        service = service or SearchService.load(settings)
        return cls(service, LLMClient.from_settings(settings))

    def run(
        self,
        query: str,
        *,
        top_k: int | None = None,
        rerank: bool = False,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> dict:
        plan = route(self.llm, query, self.service.modules)
        logger.info(
            "Agent plan: semantic=%r modules=%s", plan["semantic_query"], list(plan["module_queries"])
        )
        results = self.service.hybrid_search(
            query,
            module_queries=plan["module_queries"],
            semantic_query=plan["semantic_query"],
            rerank_query=plan["semantic_query"],  # English -> matches English captions
            top_k=top_k,
            rerank=rerank,
            include=include,
            exclude=exclude,
        )
        return {"query": query, "plan": plan, "count": len(results), "results": results}
