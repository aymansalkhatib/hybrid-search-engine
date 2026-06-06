"""API Gateway — single entry point for the IR system.

Phase 0: only meta endpoints (/health, /). Routing to internal services is added
in later phases as those services come online.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from shared.contracts import HealthResponse, ServiceInfo

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(settings.service_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("%s v%s started", settings.service_name, settings.version)
    yield
    logger.info("%s shutting down", settings.service_name)


app = FastAPI(
    title="IR Project — API Gateway",
    version=settings.version,
    description="Single entry point for the IR system. Routes requests to internal services.",
    lifespan=lifespan,
)


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
