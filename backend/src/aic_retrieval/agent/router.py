"""Prompt-based query router.

The LLM analyzes the (possibly Vietnamese) query and decides which metadata
tools to use plus the keywords for each. Semantic search is ALWAYS run, so the
router's job is (a) produce a clean English visual description for SigLIP2, and
(b) fill only the relevant metadata modules. No native tool-calling required —
we ask for a strict JSON object and parse it defensively.
"""

from __future__ import annotations

from aic_retrieval.agent.llm_client import LLMClient
from aic_retrieval.logging_conf import get_logger

logger = get_logger(__name__)

# Each metadata module = one tool the agent may choose.
TOOL_DOCS = {
    "object": "Detected object labels in the frame (English, e.g. person, car, horse, dog, boat).",
    "ocr": "Text that appears ON SCREEN (captions, signs, banners). Keep original language (often Vietnamese).",
    "asr": "Words SPOKEN in the audio (speech transcript). Keep original language (often Vietnamese).",
    "caption": "An English sentence describing the whole scene.",
}

SYSTEM_PROMPT = (
    "You are a query planner for a Vietnamese video retrieval system. "
    "Semantic image search (SigLIP2) ALWAYS runs, so always give a concise ENGLISH "
    "visual description of the scene in `semantic_query`. Then decide which of the "
    "available metadata tools help and give keywords for each; leave a tool out if "
    "it does not help.\n"
    "Rules:\n"
    "- object/caption keywords: ENGLISH.\n"
    "- ocr/asr keywords: keep the ORIGINAL language of the query (usually Vietnamese) "
    "  since on-screen text and speech are Vietnamese.\n"
    "- Only include a tool when it clearly helps (e.g. use `ocr` when the query mentions "
    "  visible text/a name/a number on screen; `asr` when it mentions something said).\n"
    "Return ONLY a JSON object with this shape:\n"
    '{"semantic_query": str, "modules": {"object": [str], "ocr": [str], "asr": [str], '
    '"caption": [str]}, "reasoning": str}'
)


def route(llm: LLMClient, query: str, available_modules: list[str]) -> dict:
    """Return a plan: {semantic_query, module_queries: {module: str}, reasoning}."""
    tools = "\n".join(f"- {m}: {TOOL_DOCS.get(m, '')}" for m in available_modules)
    user = f"Available metadata tools:\n{tools}\n\nUser query: {query!r}"

    try:
        raw = llm.chat_json(SYSTEM_PROMPT, user)
    except Exception as exc:  # noqa: BLE001 - fall back to semantic-only
        logger.warning("Router failed, falling back to semantic-only: %s", exc)
        return {"semantic_query": query, "module_queries": {}, "reasoning": f"fallback: {exc}"}

    semantic_query = (raw.get("semantic_query") or query).strip() or query
    module_queries: dict[str, str] = {}
    for module, kws in (raw.get("modules") or {}).items():
        if module not in available_modules:
            continue
        if isinstance(kws, str):
            kws = [kws]
        text = " ".join(str(k).strip() for k in (kws or []) if str(k).strip())
        if text:
            module_queries[module] = text

    return {
        "semantic_query": semantic_query,
        "module_queries": module_queries,
        "reasoning": raw.get("reasoning", ""),
    }
