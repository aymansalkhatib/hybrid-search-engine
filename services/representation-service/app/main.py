"""Representation service — builds and serves document representations.

Offline: ``POST /build`` streams the corpus from the doc-store through the
preprocessing-service and fits a model (TF-IDF, BM25, Word2Vec or BERT), persisting
it to the data volume. Online scoring primitives: ``POST /rank`` ranks the corpus
with a single model and ``POST /score`` scores a given set of docs — the
retrieval-service orchestrates these into a hybrid. ``POST /encode`` shows how a
lexical model weighs a query — no fitting at query time. Raw
document text is owned by the doc-store, not this service; a model only keeps doc ids
aligned with its matrix.
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
from app.domain.base import REPRESENTATION_VERSION, available_models
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
    app.state.models = {}  # (dataset_id, model) -> BaseRepresentation
    # Warm the cache: load any already-built models now so the first query doesn't pay
    # the (potentially multi-hundred-MB) pickle-load latency on the request path — keeps
    # the online query within the ≤20 s budget. ``store.load`` is non-fatal (returns None
    # on a corrupt/incompatible artifact), so a bad model simply stays lazy-loaded.
    preloaded = 0
    for built in app.state.store.list_built(REPRESENTATION_VERSION):
        rep = app.state.store.load(built.dataset_id, built.model, REPRESENTATION_VERSION)
        if rep is not None:
            app.state.models[(built.dataset_id, built.model)] = rep
            preloaded += 1
    # Runs builds as background jobs (one active per dataset+model) with live progress.
    app.state.jobs = JobRegistry()
    logger.info(
        "%s v%s ready (artifacts=%s, models=%s, preloaded=%d)",
        settings.service_name, settings.version, settings.artifacts_dir,
        ",".join(available_models()), preloaded,
    )
    yield
    app.state.preprocessing.close()
    app.state.doc_store.close()
    logger.info("%s shutting down", settings.service_name)


app = FastAPI(
    title="IR Project — Representation Service",
    version=settings.version,
    description="Builds & serves document representations (TF-IDF/VSM, BM25, Word2Vec, BERT) + per-model scoring.",
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
        description="Representation service — build & serve TF-IDF, BM25, Word2Vec, BERT; rank/score primitives.",
    )
