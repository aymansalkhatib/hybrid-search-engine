"""Query-refinement-service configuration (env-driven, 12-factor)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.ir_common.config import find_repo_env


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=find_repo_env(), extra="ignore")

    service_name: str = "query-refinement-service"
    version: str = "0.1.0"
    log_level: str = "INFO"

    # Cap how many tokens we'll refine in one query so a pathological request can't
    # fan out into thousands of WordNet lookups. Service-prefixed because the shared
    # .env is passed to every container — unprefixed names would collide.
    max_query_terms: int = Field(
        default=64, validation_alias="QUERY_REFINEMENT_MAX_QUERY_TERMS"
    )


settings = Settings()
