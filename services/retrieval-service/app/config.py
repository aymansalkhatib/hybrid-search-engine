"""Retrieval-service configuration (env-driven, 12-factor).

This service owns no heavy artifacts: it asks the representation-service for per-model
scores and reads the original docs by id from the doc-store. So its only dependencies
are those two service URLs plus the dataset catalog (an allow-list for ``dataset``).
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.ir_common.config import find_repo_env
from shared.ir_common.datasets import parse_datasets, resolve_dataset


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=find_repo_env(), extra="ignore")

    service_name: str = "retrieval-service"
    version: str = "0.1.0"
    log_level: str = "INFO"

    # The dataset catalog (same DATASETS list the other services use). No default:
    # an unset DATASETS yields an empty catalog and search reports "no datasets".
    datasets_raw: str = Field(default="", validation_alias="DATASETS")

    # Default to localhost so the service still runs standalone; compose overrides
    # these to the docker-network hostnames.
    representation_url: str = Field(
        default="http://localhost:8003", validation_alias="REPRESENTATION_URL"
    )
    # The inverted index — source for Boolean (index-only) search.
    indexing_url: str = Field(
        default="http://localhost:8002", validation_alias="INDEXING_URL"
    )
    doc_store_url: str = Field(
        default="http://localhost:8007", validation_alias="DOC_STORE_URL"
    )
    # Extra features (§11): cluster- and topic-based **pruning** (restrict the search to
    # the query's nearest groups). Optional — only called when a search sets cluster_prune
    # / topic_prune; defaults so it runs standalone.
    clustering_url: str = Field(
        default="http://localhost:8008", validation_alias="CLUSTERING_URL"
    )
    topic_url: str = Field(
        default="http://localhost:8009", validation_alias="TOPIC_URL"
    )
    # Cap the pruned candidate pool so a search stays within the latency budget even when a
    # cluster/topic is huge (the model still re-scores every candidate). None = no cap.
    prune_max_members: int = Field(
        default=50000, ge=1, validation_alias="PRUNE_MAX_MEMBERS"
    )

    @property
    def datasets(self) -> list[str]:
        """The configured catalog as an ordered list of ir-datasets ids."""
        return parse_datasets(self.datasets_raw)

    def resolve_dataset(self, value: str) -> str | None:
        """Return the dataset id if it's in the catalog, or None if unknown."""
        return resolve_dataset(value, self.datasets)


settings = Settings()
