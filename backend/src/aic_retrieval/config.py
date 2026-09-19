"""Central configuration for the AIC retrieval backend.

Values are resolved with this precedence (highest first):
    1. explicit init kwargs
    2. environment variables (prefixed ``AIC_``; secrets like ``GROQ_API_KEY`` also
       accepted without the prefix)
    3. a local ``.env`` file
    4. ``backend/configs/default.yaml``
    5. field defaults below

This replaces the hard-coded module-level constants and Windows paths scattered
across the old scripts (e.g. ``AIC2026_retrieval_pipeline.py:26-61``,
``upload_hf_vector.py:12``, ``vector_embeddings/embed_vector.py:18``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# backend/src/aic_retrieval/config.py -> parents[3] == repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = BACKEND_ROOT / "configs" / "default.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AIC_",
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        yaml_file=str(DEFAULT_CONFIG_FILE),
        yaml_file_encoding="utf-8",
    )

    # ---- paths -------------------------------------------------------------
    repo_root: Path = REPO_ROOT
    embeddings_dir: Path = REPO_ROOT / "database" / "vector_embeddings"
    index_dir: Path = REPO_ROOT / "database" / "index"
    metadata_dir: Path = REPO_ROOT / "database" / "metadata"
    clean_metadata_dir: Path = REPO_ROOT / "database" / "metadata_clean"

    # ---- embedding model ---------------------------------------------------
    model_id: str = "google/siglip2-base-patch16-224"
    text_max_length: int = 64
    use_fp16: bool = True
    device: str | None = None  # None -> auto (cuda if available else cpu)

    # ---- retrieval / fusion ------------------------------------------------
    top_k: int = 100
    rrf_k: int = 60
    temporal_margin: int = 3
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---- Hugging Face assets ----------------------------------------------
    hf_image_repo_id: str = "Chillguy2026/AIC_2026_data"
    hf_image_repo_type: str = "dataset"
    hf_image_root: str = "keyframe_vector"            # semantic/vector keyframes
    hf_image_root_metadata: str = "keyframe_metadata"  # metadata-pipeline keyframes
    hf_metadata_repo_id: str = "Chillguy2026/AIC_2026_metadata"
    hf_metadata_repo_type: str = "dataset"
    hf_video_repo_id: str = "Chillguy2026/dataset_video"
    hf_video_repo_type: str = "dataset"

    # ---- keyframe thumbnails ----------------------------------------------
    thumb_width: int = 384
    thumb_quality: int = 80

    # ---- API server --------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # ---- agent LLM (Groq free tier, OpenAI-compatible) --------------------
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "qwen/qwen3.8-27b"
    groq_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AIC_GROQ_API_KEY", "GROQ_API_KEY"),
    )
    llm_timeout: float = 30.0

    # ---- misc --------------------------------------------------------------
    log_level: str = "INFO"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # init > env > .env > default.yaml > secrets
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
