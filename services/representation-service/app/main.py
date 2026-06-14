"""Representation service — builds and serves document representations.

Offline: ``POST /build`` streams the corpus from the doc-store through the
preprocessing-service and fits a model (TF-IDF today), persisting it to the data
volume. Online: ``POST /encode`` vectorizes a query with a loaded model (no fitting
at query time). Raw document text is owned by the doc-store, not
this service; a model only keeps doc ids aligned with its matrix.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

import app.domain  # noqa: F401  — importing the package registers the models
from app.adapters.artifact_store import RepresentationStore
from app.adapters.doc_store_client import DocStoreClient
from app.adapters.preprocessing_client import PreprocessingClient
from app.api.routes import router
from app.config import settings
from app.domain.base import available_models
from shared.contracts import HealthResponse, ServiceInfo
from shared.ir_common.errors import install_error_handlers
from shared.ir_common.jobs import JobRegistry

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(settings.service_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = RepresentationStore(settings.artifacts_dir)
    app.state.preprocessing = PreprocessingClient(settings.preprocessing_url)
    app.state.doc_store = DocStoreClient(settings.doc_store_url)  # corpus source for builds
    app.state.models = {}  # (dataset_id, model) -> BaseRepresentation (lazy-loaded)
    # Runs builds as background jobs (one active per dataset+model) with live progress.
    app.state.jobs = JobRegistry()
    logger.info(
        "%s v%s ready (artifacts=%s, models=%s)",
        settings.service_name, settings.version, settings.artifacts_dir,
        ",".join(available_models()),
    )
    yield
    app.state.preprocessing.close()
    app.state.doc_store.close()
    logger.info("%s shutting down", settings.service_name)


app = FastAPI(
    title="IR Project — Representation Service",
    version=settings.version,
    description="Builds & serves document representations (TF-IDF / VSM; BM25 & embeddings next).",
    lifespan=lifespan,
)

install_error_handlers(app)
app.include_router(router)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(service=settings.service_name, version=settings.version)


@app.get("/", response_model=ServiceInfo, tags=["meta"])
def info() -> ServiceInfo:
    return ServiceInfo(
        service=settings.service_name,
        version=settings.version,
        description="Representation service — build & serve TF-IDF (and future BM25/embeddings).",
    )
