"""Evaluation service — offline IR effectiveness measurement.

The most-graded question in the assignment: *how good is each representation?* This
service runs a dataset's stored **test queries** through the retrieval-service, compares
each ranking against the **qrels** (both read from the doc-store), and reports
**MAP, nDCG, Recall, P@10** per model — MAP and nDCG being the primary metrics. Runs are
**offline** background jobs; finished reports are persisted (JSON + CSV) and **labeled**
so the required **before/after** comparison is just two labels.

It owns no model artifacts — retrieval (and, behind it, representation) does the scoring
where the data lives — so the image stays light. The only persistent state is the
reports directory under the data volume.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapters.doc_store_client import DocStoreClient
from app.adapters.report_store import ReportStore
from app.adapters.retrieval_client import RetrievalClient
from app.api.routes import router
from app.config import settings
from app.domain import metrics as metric_engine
from shared.contracts import HealthResponse, ServiceInfo
from shared.ir_common.errors import install_error_handlers
from shared.ir_common.jobs import JobRegistry

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(settings.service_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Keep-alive clients to the services evaluation drives: the query runner (retrieval)
    # and the ground-truth source (doc-store). A generous search timeout covers a cold
    # model load on the first query of a run.
    app.state.retrieval = RetrievalClient(settings.retrieval_url, timeout=settings.search_timeout)
    app.state.doc_store = DocStoreClient(settings.doc_store_url)
    app.state.store = ReportStore(settings.reports_dir)
    # Long offline evaluations run as background jobs (one active per dataset+label).
    app.state.jobs = JobRegistry()
    logger.info(
        "%s v%s ready (retrieval=%s, doc_store=%s, reports=%s, metric_engine=%s, concurrency=%d)",
        settings.service_name, settings.version, settings.retrieval_url, settings.doc_store_url,
        settings.reports_dir, metric_engine.engine_name(), settings.concurrency,
    )
    yield
    app.state.retrieval.close()
    app.state.doc_store.close()
    logger.info("%s shutting down", settings.service_name)


app = FastAPI(
    title="IR Project — Evaluation Service",
    version=settings.version,
    description="Evaluate representations: MAP · nDCG · Recall · P@10 per model, before/after, with persisted reports.",
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
        description="Evaluation service — MAP/nDCG/Recall/P@10 per representation; labeled before/after reports.",
    )
