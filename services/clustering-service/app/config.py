"""Clustering-service configuration (env-driven, 12-factor).

Like the representation-service, clustering reads the corpus from the **doc-store** and
normalises it via the **preprocessing-service** — never touching another service's
internals. The fitted clustering is persisted under the artifacts volume
(``data/artifacts/clustering``) so it survives restarts and feeds the dashboard plots.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.ir_common.config import find_repo_env
from shared.ir_common.datasets import parse_datasets, resolve_dataset
from shared.ir_common.paths import resolve_artifacts_dir


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=find_repo_env(), extra="ignore")

    service_name: str = "clustering-service"
    version: str = "0.1.0"
    log_level: str = "INFO"

    # Dataset catalog (same DATASETS list every service uses) — the allow-list for ``dataset``.
    datasets_raw: str = Field(default="", validation_alias="DATASETS")

    # Build source. Defaults to localhost so the service runs standalone; compose
    # overrides to the docker-network hostname. Clustering vectorises raw text itself
    # (TF-IDF with English stop-words), so it needs no preprocessing-service.
    doc_store_url: str = Field(default="http://localhost:8007", validation_alias="DOC_STORE_URL")

    # Docs streamed per progress tick while reading the corpus.
    preprocess_batch_size: int = Field(default=1000, ge=1, validation_alias="CLUSTERING_STREAM_BATCH_SIZE")
    # Points returned for the 2-D scatter plot (a sample keeps the payload light).
    plot_sample_size: int = Field(default=1500, ge=50, le=20000, validation_alias="CLUSTERING_PLOT_SAMPLE")

    artifacts_dir_env: str | None = Field(default=None, validation_alias="ARTIFACTS_DIR")

    @property
    def datasets(self) -> list[str]:
        return parse_datasets(self.datasets_raw)

    def resolve_dataset(self, value: str) -> str | None:
        return resolve_dataset(value, self.datasets)

    @property
    def clustering_dir(self) -> Path:
        path = resolve_artifacts_dir(self.artifacts_dir_env) / "clustering"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
