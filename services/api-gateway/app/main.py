"""API Gateway — single entry point for the IR system.

The UI talks only to the gateway. It exposes:
- ``GET /catalog`` — the dataset "options" with live status (download/ingest/index)
  aggregated from the doc-store and indexing services; what the UI renders.
- thin proxies for the front-end-facing offline actions (``download``/``ingest``/
  ``index``) that return a ``JobRef``, plus ``GET /jobs/{service}/{id}`` to poll
  progress. Routing for query-time services is added in later phases.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi import FastAPI

from app.clients import ServiceClient
from app.config import settings
from shared.contracts import HealthResponse, JobRef, ServiceInfo
from shared.ir_common.errors import install_error_handlers

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(settings.service_name)


# ---- catalog response models (gateway-only; not a cross-service contract) ----

class CatalogDownload(BaseModel):
    downloaded: bool = False
    doc_count: Optional[int] = None


class CatalogIngest(BaseModel):
    ingested_count: int = 0
    fully_ingested: Optional[bool] = None


class CatalogIndex(BaseModel):
    built: bool = False
    num_docs: Optional[int] = None
    # The preprocessing options the index was actually built with, so the UI can
    # reflect them (and normalize query terms the same way) instead of guessing.
    options: Optional[dict] = None


class CatalogEntry(BaseModel):
    dataset_id: str
    download: CatalogDownload = Field(default_factory=CatalogDownload)
    ingest: CatalogIngest = Field(default_factory=CatalogIngest)
    index: CatalogIndex = Field(default_factory=CatalogIndex)
    active_jobs: list[JobRef] = Field(default_factory=list)


class CatalogResponse(BaseModel):
    datasets: list[CatalogEntry]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.doc_store = ServiceClient(settings.doc_store_url)
    app.state.indexing = ServiceClient(settings.indexing_url)
    logger.info("%s v%s started", settings.service_name, settings.version)
    yield
    app.state.doc_store.close()
    app.state.indexing.close()
    logger.info("%s shutting down", settings.service_name)


app = FastAPI(
    title="IR Project — API Gateway",
    version=settings.version,
    description="Single entry point for the IR system. Routes requests to internal services.",
    lifespan=lifespan,
)

install_error_handlers(app)


# ---- helpers -------------------------------------------------------------

def _forward(resp: httpx.Response) -> JSONResponse:
    """Pass a downstream response through unchanged (status + JSON body)."""
    try:
        content = resp.json()
    except ValueError:
        content = {"error": {"code": "bad_gateway", "message": resp.text}}
    return JSONResponse(status_code=resp.status_code, content=content)


def _proxy(call) -> JSONResponse:
    """Run a downstream call, mapping connection failures to a clean 503."""
    try:
        return _forward(call())
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"downstream service unavailable: {exc}") from exc


def _safe_get_json(client: ServiceClient, path: str, params: dict) -> Optional[dict]:
    """GET that returns parsed JSON on 200, else None (a down service ⇒ blank status)."""
    try:
        resp = client.get(path, params=params)
        if resp.status_code == 200:
            return resp.json()
    except httpx.HTTPError:
        pass
    return None


# ---- catalog -------------------------------------------------------------

@app.get("/catalog", response_model=CatalogResponse, tags=["catalog"])
def catalog(request: Request) -> CatalogResponse:
    """The dataset options + live status for the UI (aggregated from the services)."""
    doc_store = request.app.state.doc_store
    indexing = request.app.state.indexing
    entries: list[CatalogEntry] = []
    for dataset_id in settings.datasets:
        ds = _safe_get_json(doc_store, "/dataset/status", {"dataset": dataset_id}) or {}
        ix = _safe_get_json(indexing, "/index/status", {"dataset": dataset_id}) or {}
        active = [j for j in (ds.get("active_job"), ix.get("active_job")) if j]
        stats = ix.get("stats") or {}
        entries.append(
            CatalogEntry(
                dataset_id=dataset_id,
                download=CatalogDownload(
                    downloaded=ds.get("downloaded", False), doc_count=ds.get("doc_count")
                ),
                ingest=CatalogIngest(
                    ingested_count=ds.get("ingested_count", 0),
                    fully_ingested=ds.get("fully_ingested"),
                ),
                index=CatalogIndex(
                    built=ix.get("built", False),
                    num_docs=stats.get("num_docs"),
                    options=stats.get("options"),
                ),
                active_jobs=active,
            )
        )
    return CatalogResponse(datasets=entries)


@app.get("/datasets/info", tags=["catalog"])
def proxy_info(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
) -> JSONResponse:
    """Dataset details (size, qrels) for preview **before download** — network-free."""
    return _proxy(lambda: request.app.state.doc_store.get("/dataset/info", params={"dataset": dataset}))


@app.get("/datasets/docs", tags=["catalog"])
def proxy_docs(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    offset: int = Query(0, ge=0, description="Starting seq (0-based position)"),
    limit: int = Query(50, ge=1, le=5000, description="Page size"),
) -> JSONResponse:
    """Browse the stored documents by page (for the UI's database viewer)."""
    return _proxy(
        lambda: request.app.state.doc_store.get(
            "/docs/list", params={"dataset": dataset, "offset": offset, "limit": limit}
        )
    )


# ---- offline-action proxies (return a JobRef/JobStatus) ------------------

@app.post("/datasets/download", tags=["catalog"])
def proxy_download(request: Request, body: dict = Body(...)) -> JSONResponse:
    """Start a corpus download (the only internet step). Body: `{dataset, force?}`."""
    return _proxy(lambda: request.app.state.doc_store.post("/dataset/download", json=body))


@app.post("/datasets/ingest", tags=["catalog"])
def proxy_ingest(request: Request, body: dict = Body(...)) -> JSONResponse:
    """Start ingesting ALL raw docs into Mongo (offline). Body: `{dataset, force?}`.

    A forced re-ingest **replaces** the corpus, which makes any existing index stale
    (it was built from the previous docs). So we clear that index first (best-effort)
    to keep ingest and index consistent — otherwise an old index lingers and shows up
    after the re-ingest. A down indexing service won't block the ingest.
    """
    dataset = body.get("dataset")
    if dataset and body.get("force"):
        try:
            request.app.state.indexing.request("DELETE", "/index", params={"dataset": dataset})
        except httpx.HTTPError:
            pass  # best-effort — don't block the ingest if indexing is unreachable
    return _proxy(lambda: request.app.state.doc_store.post("/dataset/prepare", json=body))


@app.post("/datasets/index", tags=["catalog"])
def proxy_index(request: Request, body: dict = Body(...)) -> JSONResponse:
    """Start building the inverted index (offline, whole or range).

    Body: `{dataset, start?, stop?, limit?, force?, options?}`.
    """
    return _proxy(lambda: request.app.state.indexing.post("/build", json=body))


@app.delete("/datasets", tags=["catalog"])
def proxy_delete(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    files: bool = Query(True),
    docs: bool = Query(True),
    index: bool = Query(True, description="Also remove the built index (artifact + cache)"),
) -> JSONResponse:
    """Remove a dataset's local data: corpus files, Mongo docs, and the built index.

    The doc-store owns files/docs; the indexing service owns the index. Index removal
    is best-effort — a down indexing service won't fail the docs/files delete — and the
    merged result adds ``index_deleted`` (``null`` if the indexing service was unreachable).
    """
    doc_store = request.app.state.doc_store
    indexing = request.app.state.indexing

    try:
        ds_resp = doc_store.request(
            "DELETE", "/dataset", params={"dataset": dataset, "files": files, "docs": docs}
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"downstream service unavailable: {exc}") from exc

    try:
        body = ds_resp.json()
    except ValueError:
        body = {"error": {"code": "bad_gateway", "message": ds_resp.text}}
    if ds_resp.status_code >= 400:
        return JSONResponse(status_code=ds_resp.status_code, content=body)

    index_deleted = None  # null ⇒ indexing service unreachable / not attempted
    if index:
        try:
            ix_resp = request.app.state.indexing.request(
                "DELETE", "/index", params={"dataset": dataset}
            )
            index_deleted = ix_resp.json().get("index_deleted") if ix_resp.status_code < 400 else False
        except (httpx.HTTPError, ValueError):
            index_deleted = None

    body["index_deleted"] = index_deleted
    return JSONResponse(status_code=ds_resp.status_code, content=body)


@app.get("/jobs/{service}/{job_id}", tags=["catalog"])
def proxy_job(service: str, job_id: str, request: Request) -> JSONResponse:
    """Poll a background job's progress. ``service`` is 'docstore' or 'indexing'."""
    client = {
        "docstore": request.app.state.doc_store,
        "indexing": request.app.state.indexing,
    }.get(service)
    if client is None:
        raise HTTPException(status_code=404, detail=f"unknown service '{service}' — use 'docstore' or 'indexing'")
    return _proxy(lambda: client.get(f"/jobs/{job_id}"))


# ---- meta ----------------------------------------------------------------

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
