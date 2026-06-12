"""Preprocessing service — normalization, tokenization, stopwords, stemming,
lemmatization. The same logic is applied to documents and queries.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapters.nltk_resources import build_preprocessor
from app.api.routes import router
from app.config import settings
from shared.contracts import HealthResponse, ServiceInfo
from shared.ir_common.errors import install_error_handlers

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(settings.service_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the preprocessor once at startup (loads NLTK corpora into memory).
    logger.info("Loading NLTK resources...")
    app.state.preprocessor = build_preprocessor()
    logger.info("%s v%s ready", settings.service_name, settings.version)
    yield
    logger.info("%s shutting down", settings.service_name)


app = FastAPI(
    title="IR Project — Preprocessing Service",
    version=settings.version,
    description="Normalizes documents and queries (same pipeline for both).",
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
        description="Preprocessing service — normalization for docs and queries.",
    )
