"""dashboard-service — the Verification Console (dev/QA observability UI).

A control panel to *see and exercise* every service we've built through one page. It
serves a static single-page app and reverse-proxies ``/api/<service>/…`` to each
service over the compose network (so the browser has one origin → no CORS).

This is an observability/admin tool — the same category as ``mongo-express`` — and is
deliberately allowed to reach each service independently to verify it in isolation.
It is **not** the graded product UI (Phase 9, Streamlit, gateway-only).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from app.api.routes import router
from app.config import settings
from app.domain.registry import build_registry
from shared.contracts import HealthResponse
from shared.ir_common.errors import install_error_handlers

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(settings.service_name)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class NoCacheStaticFiles(StaticFiles):
    """Serve the SPA with ``Cache-Control: no-cache`` so the browser revalidates on
    every load. The frontend is a bind-mounted, live-edited folder (see compose) —
    without this, browsers serve a stale CSS/JS after an edit and the UI looks broken.
    """

    async def get_response(self, path: str, scope) -> Response:
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=180.0)
    app.state.registry = {e.key: e for e in build_registry(settings)}
    logger.info("%s v%s ready — proxying %d services",
                settings.service_name, settings.version, len(app.state.registry))
    yield
    await app.state.http.aclose()
    logger.info("%s shutting down", settings.service_name)


app = FastAPI(
    title="IR Project — Verification Console",
    version=settings.version,
    description="Observability console: serves the dashboard SPA and reverse-proxies every service.",
    lifespan=lifespan,
)

install_error_handlers(app)
app.include_router(router)


@app.get("/healthz", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(service=settings.service_name, version=settings.version)


# Serve the SPA. Mounted LAST so /api/* and /healthz take precedence over static.
app.mount("/", NoCacheStaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
