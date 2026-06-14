"""HTTP layer (thin) for the dashboard-service.

Two responsibilities only:

* ``GET /api/_meta`` — the service registry + configured datasets, so the front-end
  knows what to render and where each service's Swagger/admin UI lives.
* ``/api/{service}/{path}`` — a transparent **reverse proxy** to one service. The
  browser only ever talks to this origin (no CORS), and each service is reachable
  *independently* — exactly the SOA trait this console exists to verify.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from app.config import settings

logger = logging.getLogger("dashboard-service")
router = APIRouter(tags=["console"])

PROXY_METHODS = ["GET", "POST", "PUT", "DELETE"]


@router.get("/api/_meta")
def meta(request: Request) -> JSONResponse:
    """Registry + dataset catalog the front-end renders the whole UI from."""
    registry = request.app.state.registry
    return JSONResponse({
        "dashboard": {"name": "IR Verification Console", "version": settings.version},
        "datasets": settings.datasets,
        "services": [entry.public() for entry in registry.values()],
    })


@router.api_route("/api/{service}/{path:path}", methods=PROXY_METHODS)
async def proxy(service: str, path: str, request: Request) -> Response:
    """Forward the request to ``<service>/<path>`` and return its response verbatim."""
    entry = request.app.state.registry.get(service)
    if entry is None or entry.url is None:
        return JSONResponse(
            status_code=502,
            content={"error": {"code": "not_proxyable",
                               "message": f"service '{service}' is not proxyable from the console"}},
        )

    target = entry.url.rstrip("/") + "/" + path
    client: httpx.AsyncClient = request.app.state.http
    body = await request.body()
    headers = {}
    ctype = request.headers.get("content-type")
    if ctype:
        headers["content-type"] = ctype

    try:
        upstream = await client.request(
            request.method,
            target,
            params=request.url.query or None,
            content=body or None,
            headers=headers,
        )
    except httpx.HTTPError as exc:  # connection refused ⇒ service down (not built / stopped)
        return JSONResponse(
            status_code=503,
            content={"error": {"code": "service_down", "service": service,
                               "message": f"{entry.label} unreachable at {entry.url} — {exc}"}},
        )

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )
