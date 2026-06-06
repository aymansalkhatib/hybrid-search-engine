"""API Gateway configuration (env-driven, 12-factor)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Locate the repo-root .env by walking up from this file, so the service runs
# standalone from any working directory. In Docker the layout is flattened
# (/app/app/config.py) and no .env is shipped — config comes from compose as
# real env vars — so this resolves to None and pydantic reads the environment.
_HERE = Path(__file__).resolve()
_ENV_FILE = next((p / ".env" for p in _HERE.parents if (p / ".env").exists()), None)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

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
