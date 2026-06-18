"""Dataset lifecycle — the offline flow the UI drives: download → ingest → index.

These are cross-service **orchestrations** (not raw passthrough): a forced re-ingest
clears the now-stale index, and a dataset delete removes its index too. Each action
returns a ``JobRef``/``JobStatus``; poll progress via ``GET /jobs/{service}/{id}``.
The raw per-service equivalents also exist under ``/indexing`` and ``/docstore`` —
these endpoints are the recommended product flow that keeps the two services in sync.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.proxy import proxy

router = APIRouter(tags=["lifecycle"])


@router.post("/datasets/download")
def download(request: Request, body: dict = Body(...)) -> JSONResponse:
    """Start a corpus download (the only internet step). Body: `{dataset, force?}`."""
    return proxy(lambda: request.app.state.doc_store.post("/dataset/download", json=body))


@router.post("/datasets/ingest")
def ingest(request: Request, body: dict = Body(...)) -> JSONResponse:
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
    return proxy(lambda: request.app.state.doc_store.post("/dataset/prepare", json=body))


@router.post("/datasets/index")
def index(request: Request, body: dict = Body(...)) -> JSONResponse:
    """Start building the inverted index (offline, whole or range).

    Body: `{dataset, start?, stop?, limit?, force?, options?}`.
    """
    return proxy(lambda: request.app.state.indexing.post("/build", json=body))


@router.delete("/datasets")
def delete(
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
        merged = ds_resp.json()
    except ValueError:
        merged = {"error": {"code": "bad_gateway", "message": ds_resp.text}}
    if ds_resp.status_code >= 400:
        return JSONResponse(status_code=ds_resp.status_code, content=merged)

    index_deleted = None  # null ⇒ indexing service unreachable / not attempted
    if index:
        try:
            ix_resp = indexing.request("DELETE", "/index", params={"dataset": dataset})
            index_deleted = ix_resp.json().get("index_deleted") if ix_resp.status_code < 400 else False
        except (httpx.HTTPError, ValueError):
            index_deleted = None

    merged["index_deleted"] = index_deleted
    return JSONResponse(status_code=ds_resp.status_code, content=merged)


@router.get("/jobs/{service}/{job_id}")
def job(service: str, job_id: str, request: Request) -> JSONResponse:
    """Poll a background job's progress. ``service`` is 'docstore' or 'indexing'."""
    client = {
        "docstore": request.app.state.doc_store,
        "indexing": request.app.state.indexing,
    }.get(service)
    if client is None:
        raise HTTPException(status_code=404, detail=f"unknown service '{service}' — use 'docstore' or 'indexing'")
    return proxy(lambda: client.get(f"/jobs/{job_id}"))
