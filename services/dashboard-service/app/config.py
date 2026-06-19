"""dashboard-service configuration (env-driven, 12-factor).

The console proxies to each service **server-side** over the compose network using
the internal ``*_URL`` values, and exposes the host-published ``*_PORT`` values to
the browser so it can deep-link to each service's Swagger UI / admin UI. Both sets
already live in the repo ``.env`` — nothing here is hard-coded.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.ir_common.config import find_repo_env
from shared.ir_common.datasets import parse_datasets


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=find_repo_env(), extra="ignore")

    service_name: str = "dashboard-service"
    version: str = "0.1.0"
    log_level: str = "INFO"

    # The configured dataset catalog (same DATASETS list every service uses).
    datasets_raw: str = Field(default="", validation_alias="DATASETS")

    # ---- internal proxy targets (compose-network service URLs) ----
    api_gateway_url: str = Field(default="http://api-gateway:8000", validation_alias="API_GATEWAY_URL")
    preprocessing_url: str
    indexing_url: str
    representation_url: str
    retrieval_url: str
    query_refinement_url: str
    evaluation_url: str
    doc_store_url: str
    clustering_url: str = Field(default="http://clustering-service:8008", validation_alias="CLUSTERING_URL")
    topic_url: str = Field(default="http://topic-service:8009", validation_alias="TOPIC_URL")

    # ---- host-published ports (browser-facing deep links) ----
    api_gateway_port: int = Field(default=8000, validation_alias="API_GATEWAY_PORT")
    preprocessing_port: int = Field(default=8001, validation_alias="PREPROCESSING_PORT")
    indexing_port: int = Field(default=8002, validation_alias="INDEXING_PORT")
    representation_port: int = Field(default=8003, validation_alias="REPRESENTATION_PORT")
    retrieval_port: int = Field(default=8004, validation_alias="RETRIEVAL_PORT")
    query_refinement_port: int = Field(default=8005, validation_alias="QUERY_REFINEMENT_PORT")
    evaluation_port: int = Field(default=8006, validation_alias="EVALUATION_PORT")
    doc_store_port: int = Field(default=8007, validation_alias="DOC_STORE_PORT")
    clustering_port: int = Field(default=8008, validation_alias="CLUSTERING_PORT")
    topic_port: int = Field(default=8009, validation_alias="TOPIC_PORT")
    mongo_express_port: int = Field(default=8081, validation_alias="MONGO_EXPRESS_PORT")
    dashboard_port: int = Field(default=8090, validation_alias="DASHBOARD_PORT")

    @property
    def datasets(self) -> list[str]:
        return parse_datasets(self.datasets_raw)


settings = Settings()
