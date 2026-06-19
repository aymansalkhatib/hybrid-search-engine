"""Evaluation-service configuration (env-driven, 12-factor).

This service owns no model artifacts: it drives the **retrieval-service** with the
stored test queries and scores the results against the **qrels** read from the
**doc-store**. So its dependencies are those two service URLs plus the dataset catalog
(an allow-list for ``dataset``). Finished reports are persisted under the artifacts
volume (``data/artifacts/eval``) so they survive restarts and feed the report's charts.
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

    service_name: str = "evaluation-service"
    version: str = "0.1.0"
    log_level: str = "INFO"

    # The dataset catalog (same DATASETS list every service uses) — the allow-list for
    # ``dataset``. No default: an unset DATASETS yields an empty catalog.
    datasets_raw: str = Field(default="", validation_alias="DATASETS")

    # Default to localhost so the service still runs standalone; compose overrides these
    # to the docker-network hostnames. Evaluation calls retrieval directly (service-to-
    # service, not via the gateway) and reads queries/qrels from the doc-store.
    retrieval_url: str = Field(default="http://localhost:8004", validation_alias="RETRIEVAL_URL")
    doc_store_url: str = Field(default="http://localhost:8007", validation_alias="DOC_STORE_URL")
    # Optional — only used when an evaluation requests query refinement (with/without
    # comparison). Defaults to localhost so the service still runs standalone.
    query_refinement_url: str = Field(default="http://localhost:8005", validation_alias="QUERY_REFINEMENT_URL")
    # Only used to preflight with-clustering / with-topics runs; retrieval does the re-ranking.
    clustering_url: str = Field(default="http://localhost:8008", validation_alias="CLUSTERING_URL")
    topic_url: str = Field(default="http://localhost:8009", validation_alias="TOPIC_URL")

    # Offline-eval throughput: how many search requests to issue concurrently. Bounded —
    # the representation-service does the per-query scoring, so this trades its CPU for
    # wall-clock. Safe to lower to 1 (sequential) on a constrained box.
    concurrency: int = Field(default=8, ge=1, le=64, validation_alias="EVAL_CONCURRENCY")
    # Page size when streaming the stored test queries out of the doc-store.
    queries_page_size: int = Field(default=5000, ge=1, validation_alias="EVAL_QUERIES_PAGE_SIZE")
    # Per-search HTTP timeout to retrieval — a single search scores the whole corpus, and
    # a cold model load on the first query can take a while.
    search_timeout: float = Field(default=120.0, gt=0, validation_alias="EVAL_SEARCH_TIMEOUT")

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
    def reports_dir(self) -> Path:
        """Where evaluation reports (JSON + CSV) are persisted — under the data volume."""
        path = resolve_artifacts_dir(self.artifacts_dir_env) / "eval"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
