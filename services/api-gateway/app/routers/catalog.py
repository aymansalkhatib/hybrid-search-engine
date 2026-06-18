"""Catalog — the dataset options + live status the UI renders.

Aggregates read-only status from the doc-store and indexing services into one shape
the front-end can paint directly, plus the network-free dataset preview and the
stored-docs browser. These are gateway-only response models (not cross-service
contracts), so they live here next to the endpoint that builds them.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.proxy import proxy, safe_get_json
from shared.contracts import JobRef

router = APIRouter(tags=["catalog"])


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


# ---- endpoints -----------------------------------------------------------

@router.get("/catalog", response_model=CatalogResponse)
def catalog(request: Request) -> CatalogResponse:
    """The dataset options + live status for the UI (aggregated from the services)."""
    doc_store = request.app.state.doc_store
    indexing = request.app.state.indexing
    entries: list[CatalogEntry] = []
    for dataset_id in settings.datasets:
        ds = safe_get_json(doc_store, "/dataset/status", {"dataset": dataset_id}) or {}
        ix = safe_get_json(indexing, "/index/status", {"dataset": dataset_id}) or {}
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


@router.get("/datasets/info")
def dataset_info(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
) -> JSONResponse:
    """Dataset details (size, qrels) for preview **before download** — network-free."""
    return proxy(lambda: request.app.state.doc_store.get("/dataset/info", params={"dataset": dataset}))


@router.get("/datasets/status")
def dataset_status(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    with_total: bool = Query(
        False, description="Re-confirm the total doc_count from ir-datasets (verifies a full ingest)"
    ),
) -> JSONResponse:
    """One dataset's readiness (download/ingest counts); ``with_total=true`` adds ``fully_ingested``."""
    return proxy(
        lambda: request.app.state.doc_store.get(
            "/dataset/status", params={"dataset": dataset, "with_total": with_total}
        )
    )


@router.get("/datasets/docs")
def dataset_docs(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    offset: int = Query(0, ge=0, description="Starting seq (0-based position)"),
    limit: int = Query(50, ge=1, le=5000, description="Page size"),
) -> JSONResponse:
    """Browse the stored documents by page (for the UI's database viewer)."""
    return proxy(
        lambda: request.app.state.doc_store.get(
            "/docs/list", params={"dataset": dataset, "offset": offset, "limit": limit}
        )
    )
