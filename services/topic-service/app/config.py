"""Topic-service configuration (env-driven, 12-factor).

Like the clustering-service, topic detection reads the corpus from the **doc-store** and
vectorises raw text itself (no preprocessing-service dependency). The fitted LDA model is
persisted under the artifacts volume (``data/artifacts/topic``) so it survives restarts
and feeds the dashboard topic charts.
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

    service_name: str = "topic-service"
    version: str = "0.1.0"
    log_level: str = "INFO"

    # Dataset catalog (same DATASETS list every service uses) — the allow-list for ``dataset``.
    datasets_raw: str = Field(default="", validation_alias="DATASETS")

    # Build source. Defaults to localhost so the service runs standalone; compose
    # overrides to the docker-network hostname.
    doc_store_url: str = Field(default="http://localhost:8007", validation_alias="DOC_STORE_URL")

    # Docs streamed per progress tick while reading the corpus.
    stream_batch_size: int = Field(default=1000, ge=1, validation_alias="TOPIC_STREAM_BATCH_SIZE")

    artifacts_dir_env: str | None = Field(default=None, validation_alias="ARTIFACTS_DIR")

    @property
    def datasets(self) -> list[str]:
        return parse_datasets(self.datasets_raw)

    def resolve_dataset(self, value: str) -> str | None:
        return resolve_dataset(value, self.datasets)

    @property
    def topic_dir(self) -> Path:
        path = resolve_artifacts_dir(self.artifacts_dir_env) / "topic"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
