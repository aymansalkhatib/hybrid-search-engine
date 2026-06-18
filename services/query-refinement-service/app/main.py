"""Query-refinement service — refines a raw query before retrieval.

Stages (each independently toggleable for before/after evaluation, §5 req 5):
spell-correction (SymSpell) and synonym expansion (WordNet). It runs on the raw
query and returns a refined query string; the retrieval path then preprocesses that
string like any other query. All resources are loaded once at startup — no online
fitting, well inside the ≤20 s query budget.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapters.resources import build_symspell, load_stopwords, warm_wordnet
from app.api.routes import router
from app.config import settings
from app.domain.refiner import QueryRefiner
from shared.contracts import HealthResponse, ServiceInfo
from shared.ir_common.errors import install_error_handlers

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(settings.service_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the refiner once at startup: load the SymSpell dictionary, the stopword
    # set, and warm WordNet so the first online query pays no lazy-load latency.
    logger.info("Loading refinement resources (SymSpell + WordNet)...")
    sym = build_symspell()
    stop = load_stopwords()
    warm_wordnet()
    app.state.refiner = QueryRefiner(sym, stop, max_query_terms=settings.max_query_terms)
    logger.info("%s v%s ready", settings.service_name, settings.version)
    yield
    logger.info("%s shutting down", settings.service_name)


app = FastAPI(
    title="IR Project — Query Refinement Service",
    version=settings.version,
    description="Refines raw queries: spell-correction (SymSpell) + synonym expansion (WordNet).",
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
        description="Query refinement — spell-correction + synonym expansion before retrieval.",
    )
