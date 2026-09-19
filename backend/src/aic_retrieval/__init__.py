"""AIC 2026 multimedia retrieval backend.

Package layout:
    config           - pydantic-settings configuration (yaml + env)
    logging_conf     - logging setup
    embedding        - SigLIP2 text/image embedder (single source of truth)
    index            - FAISS index + frame metadata store        (WS2)
    text             - metadata ingestion + per-module BM25       (WS3)
    search           - semantic / lexical / fusion / rerank       (WS4)
    agent            - Groq LLM client, query router, tools       (WS5)
    api              - FastAPI app                                 (WS6)
    cli              - command line entry point
"""

__version__ = "0.1.0"
