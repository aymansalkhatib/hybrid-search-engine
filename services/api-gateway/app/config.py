"""API Gateway configuration (env-driven, 12-factor)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_name: str = "api-gateway"
    version: str = "0.1.0"
    log_level: str

    # Downstream services — must be set in .env
    preprocessing_url: str
    indexing_url: str
    representation_url: str
    retrieval_url: str
    query_refinement_url: str
    evaluation_url: str


settings = Settings()
