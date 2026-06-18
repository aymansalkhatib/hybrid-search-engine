"""Retrieval service — the online query path (matching & ranking).

This is the SOA seam for *retrieval*: it takes a query and returns
ranked documents. Two families of search:

* **model-based** — single representation or a **hybrid** (serial re-rank / parallel
  fusion), scored by the representation-service (``/rank`` / ``/score``).
* **Boolean** — inverted-index-only match (AND/OR), delegated to the indexing-service's
  ``/match`` primitive, with **no** relevance scoring.

It owns no heavy artifacts — the model owners (representation, indexing) do the work
where the data lives; this service orchestrates the strategy and fetches the top-k
**original** docs by id from the doc-store for display. Being artifact-free keeps the
image light (no scikit-learn / torch) and avoids loading the corpus twice.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapters.doc_store_client import DocStoreClient
from app.adapters.indexing_client import IndexingClient
from app.adapters.representation_client import RepresentationClient
from app.api.routes import router
from app.config import settings
from shared.contracts import HealthResponse, ServiceInfo
from shared.ir_common.errors import install_error_handlers

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(settings.service_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Keep-alive clients to the services retrieval composes: the model scorer
    # (representation), the inverted index (indexing, for Boolean search), and the
    # original-text store. No artifacts are loaded here.
    app.state.representation = RepresentationClient(settings.representation_url)
    app.state.indexing = IndexingClient(settings.indexing_url)
    app.state.doc_store = DocStoreClient(settings.doc_store_url)
    logger.info(
        "%s v%s ready (representation=%s, indexing=%s, doc_store=%s)",
        settings.service_name, settings.version,
        settings.representation_url, settings.indexing_url, settings.doc_store_url,
    )
    yield
    app.state.representation.close()
    app.state.indexing.close()
    app.state.doc_store.close()
    logger.info("%s shutting down", settings.service_name)


app = FastAPI(
    title="IR Project — Retrieval Service",
    version=settings.version,
    description="Match & rank: model search (single/hybrid + fusion) or Boolean index search; original docs by id.",
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
        description="Retrieval service — model search (single/hybrid + fusion) & Boolean index search; original docs by id.",
    )
