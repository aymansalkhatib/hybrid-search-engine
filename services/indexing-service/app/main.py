"""Indexing service — builds and serves the inverted index over a corpus.

Offline: ``POST /build`` streams the dataset through the preprocessing-service and
accumulates postings + doc lengths, persisting the result to the data volume.
Online: the term/doc query endpoints read the loaded index, and ``POST /match`` does
**Boolean retrieval** (AND/OR over the postings, no scoring model) — the inverted-index
-only search the retrieval-service exposes. Raw document text is owned by the doc-store,
not this service.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapters.artifact_store import ArtifactStore
from app.adapters.doc_store_client import DocStoreClient
from app.adapters.preprocessing_client import PreprocessingClient
from app.api.routes import router
from app.config import settings
from shared.contracts import HealthResponse, ServiceInfo
from shared.ir_common.errors import install_error_handlers
from shared.ir_common.jobs import JobRegistry

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(settings.service_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = ArtifactStore(settings.artifacts_dir)
    app.state.preprocessing = PreprocessingClient(settings.preprocessing_url)
    app.state.doc_store = DocStoreClient(settings.doc_store_url)  # corpus source for builds
    app.state.indexes = {}  # dataset_id -> InvertedIndex (lazy-loaded on first use)
    # Runs builds as background jobs (one active per dataset) with live progress;
    # a second build for an in-flight dataset gets 409.
    app.state.jobs = JobRegistry()
    logger.info(
        "%s v%s ready (artifacts=%s, preprocessing=%s)",
        settings.service_name, settings.version, settings.artifacts_dir,
        settings.preprocessing_url,
    )
    yield
    app.state.preprocessing.close()
    app.state.doc_store.close()
    logger.info("%s shutting down", settings.service_name)


app = FastAPI(
    title="IR Project — Indexing Service",
    version=settings.version,
    description="Builds & serves the inverted index (postings, df, doc lengths, avgdl) + Boolean match.",
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
        description="Indexing service — inverted index build & queries.",
    )
