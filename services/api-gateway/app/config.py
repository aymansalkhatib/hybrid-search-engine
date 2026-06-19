"""API Gateway configuration (env-driven, 12-factor)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.ir_common.config import find_repo_env
from shared.ir_common.datasets import parse_datasets


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=find_repo_env(), extra="ignore")

    service_name: str = "api-gateway"
    version: str = "0.1.0"
    log_level: str = "INFO"

    # The dataset catalog (same DATASETS list the services use) — drives /catalog.
    # No default: an unset DATASETS yields an empty catalog (no dataset baked in).
    datasets_raw: str = Field(default="", validation_alias="DATASETS")

    # Downstream services — must be set in .env
    preprocessing_url: str
    indexing_url: str
    representation_url: str
    retrieval_url: str
    query_refinement_url: str
    evaluation_url: str
    doc_store_url: str
    # Extra features (§11) — defaults so the gateway still boots if they're not configured.
    clustering_url: str = "http://clustering-service:8008"
    topic_url: str = "http://topic-service:8009"

    @property
    def datasets(self) -> list[str]:
        return parse_datasets(self.datasets_raw)


settings = Settings()
