"""Preprocessing-service configuration (env-driven, 12-factor)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.ir_common.config import find_repo_env


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=find_repo_env(), extra="ignore")

    service_name: str = "preprocessing-service"
    version: str = "0.1.0"
    log_level: str = "INFO"

    # Cap batch size so a single request can't exhaust memory; callers chunk
    # large corpora. Env name is service-prefixed because the shared .env is
    # passed to every service — unprefixed names would collide across services.
    max_batch_size: int = Field(
        default=5000, validation_alias="PREPROCESSING_MAX_BATCH_SIZE"
    )


settings = Settings()
