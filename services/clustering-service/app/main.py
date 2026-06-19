"""Clustering service — unsupervised document clustering (extra feature, §11).

Offline: ``POST /build`` streams the corpus from the doc-store through the
preprocessing-service, vectorises it (TF-IDF) and partitions it with KMeans, persisting
the fitted model. Online endpoints are cheap reads off that artifact: the per-cluster
table (sizes + top terms), a 2-D projection for the scatter plot, a silhouette quality
score, and assigning a new query to its nearest cluster. Independent of the trained
representations and toggleable on its own, so its before/after value (cluster plots) is
evaluable in isolation — exactly what the assignment's extra-features section asks.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapters.artifact_store import ClusteringStore
from app.adapters.doc_store_client import DocStoreClient
from app.api.routes import router
from app.config import settings
from shared.contracts import HealthResponse, ServiceInfo
from shared.ir_common.errors import install_error_handlers
from shared.ir_common.jobs import JobRegistry

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(settings.service_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = ClusteringStore(settings.clustering_dir)
    app.state.doc_store = DocStoreClient(settings.doc_store_url)
    app.state.models = {}  # dataset_id -> ClusterModel (lazy-loaded from disk)
    app.state.jobs = JobRegistry()
    logger.info("%s v%s ready (clustering_dir=%s)", settings.service_name, settings.version, settings.clustering_dir)
    yield
    app.state.doc_store.close()
    logger.info("%s shutting down", settings.service_name)


app = FastAPI(
    title="IR Project — Clustering Service",
    version=settings.version,
    description="Unsupervised document clustering (TF-IDF + KMeans): sizes, top terms, silhouette, 2-D plot.",
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
        description="Clustering service — TF-IDF + KMeans document clustering with plots & assignment.",
    )
