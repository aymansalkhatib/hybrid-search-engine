"""doc-store-service — owns the RAW/original document store in MongoDB.

Offline: download + ingest raw docs by id (``/dataset/prepare``). Online: serve the
original text **by id** (``/doc``, ``/docs``) for display of the top-k results. The
service stays up even if Mongo is briefly unavailable; data endpoints return 503.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapters.mongo_store import MongoDocStore
from app.api.routes import router
from app.config import settings
from shared.contracts import HealthResponse, ServiceInfo
from shared.ir_common.errors import install_error_handlers
from shared.ir_common.jobs import JobRegistry

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(settings.service_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = MongoDocStore(
        settings.mongo_url,
        settings.mongo_db,
        settings.mongo_docs_collection,
        settings.mongo_queries_collection,
        settings.mongo_qrels_collection,
    )
    # Best-effort index creation; don't crash startup if Mongo isn't ready yet
    # (compose may start both at once — endpoints handle Mongo-down with 503).
    try:
        app.state.store.ensure_indexes()
        logger.info("connected to Mongo at %s (db=%s)", settings.mongo_url, settings.mongo_db)
    except Exception as exc:  # noqa: BLE001 - log and continue, retried lazily
        logger.warning("Mongo not ready at startup (%s): %s", settings.mongo_url, exc)
    # Runs download/ingest as background jobs (one active per dataset) with progress.
    app.state.jobs = JobRegistry()
    logger.info("%s v%s ready", settings.service_name, settings.version)
    yield
    app.state.store.close()
    logger.info("%s shutting down", settings.service_name)


app = FastAPI(
    title="IR Project — Doc Store Service",
    version=settings.version,
    description="Raw/original document store (MongoDB): offline ingest + by-id reads.",
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
        description="Doc-store service — raw docs in MongoDB, read by id at query time.",
    )
