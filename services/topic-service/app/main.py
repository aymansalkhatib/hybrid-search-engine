"""Topic service — LDA topic modelling (extra feature, §11).

Offline: ``POST /build`` streams the corpus from the doc-store, vectorises raw text and
fits Latent Dirichlet Allocation, persisting the model. Online endpoints are cheap reads
off that artifact: the per-topic table (top terms + weights) for the topic charts, the
corpus's dominant-topic distribution, perplexity (fit quality), and inferring a query's
topic mixture. Independent of the trained representations and toggleable on its own, so
its appropriate evaluation — the topic charts — is produced in isolation.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapters.artifact_store import TopicStore
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
    app.state.store = TopicStore(settings.topic_dir)
    app.state.doc_store = DocStoreClient(settings.doc_store_url)
    app.state.models = {}  # dataset_id -> TopicModel (lazy-loaded from disk)
    app.state.jobs = JobRegistry()
    logger.info("%s v%s ready (topic_dir=%s)", settings.service_name, settings.version, settings.topic_dir)
    yield
    app.state.doc_store.close()
    logger.info("%s shutting down", settings.service_name)


app = FastAPI(
    title="IR Project — Topic Service",
    version=settings.version,
    description="LDA topic modelling: per-topic top terms & weights, dominant-topic sizes, perplexity, infer.",
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
        description="Topic service — LDA topic modelling with topic charts & query topic inference.",
    )
