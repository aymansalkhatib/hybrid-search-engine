"""API Gateway — the single external entry point for the IR system.

The product UI talks **only** to the gateway; the functional services
are not published to the host — they live on the compose network and are reachable
only through here. Responsibilities are split one-file-per-concern under ``routers/``:

* ``catalog``        — dataset options + live status the UI renders.
* ``lifecycle``      — the offline flow (download → ingest → index) + job polling.
* ``preprocessing`` / ``indexing`` / ``representation`` / ``retrieval`` / ``evaluation`` /
  ``docstore`` — passthrough to each service's own API (under a ``/<service>`` prefix),
  reusing ``shared.contracts`` so the gateway's Swagger documents and validates the whole
  system. ``retrieval`` is the online query path (``/retrieval/search``); ``evaluation``
  measures MAP/nDCG/Recall/P@10 per model.

This module just wires the app: open keep-alive clients to each downstream service,
mount the routers, and expose ``/health`` + ``/``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.clients import ServiceClient
from app.config import settings
from app.routers import (
    catalog,
    clustering,
    docstore,
    evaluation,
    indexing,
    lifecycle,
    preprocessing,
    refinement,
    representation,
    retrieval,
    topic,
)
from shared.contracts import HealthResponse, ServiceInfo
from shared.ir_common.errors import install_error_handlers

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(settings.service_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One keep-alive client per downstream service, shared by all routers via app.state.
    app.state.preprocessing = ServiceClient(settings.preprocessing_url)
    app.state.indexing = ServiceClient(settings.indexing_url)
    app.state.representation = ServiceClient(settings.representation_url)
    app.state.retrieval = ServiceClient(settings.retrieval_url)
    app.state.query_refinement = ServiceClient(settings.query_refinement_url)
    app.state.evaluation = ServiceClient(settings.evaluation_url)
    app.state.doc_store = ServiceClient(settings.doc_store_url)
    app.state.clustering = ServiceClient(settings.clustering_url)
    app.state.topic = ServiceClient(settings.topic_url)
    logger.info("%s v%s started", settings.service_name, settings.version)
    yield
    for client in (app.state.preprocessing, app.state.indexing,
                   app.state.representation, app.state.retrieval,
                   app.state.query_refinement,
                   app.state.evaluation, app.state.doc_store,
                   app.state.clustering, app.state.topic):
        client.close()
    logger.info("%s shutting down", settings.service_name)


app = FastAPI(
    title="IR Project — API Gateway",
    version=settings.version,
    description="Single external entry point for the IR system. Routes to internal services.",
    lifespan=lifespan,
)

install_error_handlers(app)

app.include_router(catalog.router)
app.include_router(lifecycle.router)
app.include_router(preprocessing.router)
app.include_router(indexing.router)
app.include_router(representation.router)
app.include_router(retrieval.router)
app.include_router(refinement.router)
app.include_router(evaluation.router)
app.include_router(clustering.router)
app.include_router(topic.router)
app.include_router(docstore.router)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(service=settings.service_name, version=settings.version)


@app.get("/", response_model=ServiceInfo, tags=["meta"])
def info() -> ServiceInfo:
    return ServiceInfo(
        service=settings.service_name,
        version=settings.version,
        description="API Gateway — single entry point for the IR system.",
    )
