"""Indexing-service configuration (env-driven, 12-factor)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.ir_common.config import find_repo_env
from shared.ir_common.datasets import parse_datasets, resolve_dataset
from shared.ir_common.paths import resolve_artifacts_dir


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=find_repo_env(), extra="ignore")

    service_name: str = "indexing-service"
    version: str = "0.1.0"
    log_level: str = "INFO"

    # The dataset catalog: one comma-separated list of ir-datasets ids (the first is
    # the primary/required dataset; the rest are bonus). No default on purpose — if
    # DATASETS is unset the catalog is empty and dataset ops report "no datasets".
    datasets_raw: str = Field(default="", validation_alias="DATASETS")

    # Default to localhost so the service still runs standalone; compose overrides
    # these to the docker-network hostnames.
    preprocessing_url: str = Field(
        default="http://localhost:8001", validation_alias="PREPROCESSING_URL"
    )
    # The index is built from the docs stored in the doc-store (DB-centric pipeline).
    doc_store_url: str = Field(
        default="http://localhost:8007", validation_alias="DOC_STORE_URL"
    )
    # Docs per /preprocess/batch call when building (must stay ≤ preprocessing's cap).
    preprocess_batch_size: int = Field(
        default=1000, validation_alias="INDEXING_PREPROCESS_BATCH_SIZE"
    )
    # Container artifacts path (ignored when running from source — see resolve_artifacts_dir).
    artifacts_dir_env: str | None = Field(default=None, validation_alias="ARTIFACTS_DIR")

    @property
    def datasets(self) -> list[str]:
        """The configured catalog as an ordered list of ir-datasets ids."""
        return parse_datasets(self.datasets_raw)

    def resolve_dataset(self, value: str) -> str | None:
        """Return the dataset id if it's in the catalog, or None if unknown."""
        return resolve_dataset(value, self.datasets)

    @property
    def artifacts_dir(self) -> Path:
        return resolve_artifacts_dir(self.artifacts_dir_env)


# The indexing-service never reads the dataset cache directly — it builds the index
# from the doc-store (MongoDB), so it needs no IR_DATASETS_HOME pin.
settings = Settings()
