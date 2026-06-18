"""doc-store-service configuration (env-driven, 12-factor)."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.ir_common.config import find_repo_env
from shared.ir_common.datasets import parse_datasets, resolve_dataset
from shared.ir_common.paths import resolve_dataset_home


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=find_repo_env(), extra="ignore")

    service_name: str = "doc-store-service"
    version: str = "0.1.0"
    log_level: str = "INFO"

    # The dataset catalog: one comma-separated list of ir-datasets ids (the first is
    # the primary/required dataset; the rest are bonus). No default on purpose — if
    # DATASETS is unset the catalog is empty and dataset ops report "no datasets".
    datasets_raw: str = Field(default="", validation_alias="DATASETS")

    # MongoDB (the raw-doc store). Defaults to localhost so the service still runs
    # standalone; compose overrides MONGO_URL to the docker-network hostname.
    mongo_url: str = Field(
        default="mongodb://localhost:27017", validation_alias="MONGO_URL"
    )
    mongo_db: str = Field(default="ir_project", validation_alias="MONGO_DB")
    mongo_docs_collection: str = Field(
        default="documents", validation_alias="MONGO_DOCS_COLLECTION"
    )
    mongo_queries_collection: str = Field(
        default="queries", validation_alias="MONGO_QUERIES_COLLECTION"
    )
    mongo_qrels_collection: str = Field(
        default="qrels", validation_alias="MONGO_QRELS_COLLECTION"
    )
    ingest_batch_size: int = Field(
        default=5000, validation_alias="DOC_STORE_INGEST_BATCH_SIZE"
    )
    # Where datasets are downloaded & kept permanently (also ir_datasets' cache).
    ir_datasets_home_env: str | None = Field(default=None, validation_alias="IR_DATASETS_HOME")

    @property
    def datasets(self) -> list[str]:
        """The configured catalog as an ordered list of ir-datasets ids."""
        return parse_datasets(self.datasets_raw)

    def resolve_dataset(self, value: str) -> str | None:
        """Return the dataset id if it's in the catalog, or None if unknown."""
        return resolve_dataset(value, self.datasets)

    @property
    def dataset_home(self) -> Path:
        return resolve_dataset_home(self.ir_datasets_home_env)


settings = Settings()

# Pin ir_datasets' cache to our dataset home so downloads, the local corpus and our
# download markers all live in the same place (in Docker and when run from source).
os.environ["IR_DATASETS_HOME"] = str(settings.dataset_home)
