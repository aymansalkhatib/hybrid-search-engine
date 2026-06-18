"""HTTP layer (thin) for the doc-store-service.

Offline (async jobs): ``POST /dataset/download`` fetches the corpus locally and
``POST /dataset/prepare`` ingests the **raw** docs into Mongo — both return a
``JobRef`` and run in the background with live progress (poll ``GET /jobs/{id}``).
Online: ``GET /doc`` / ``POST /docs`` read the **original** text **by id** — what
the UI shows for the top-k results (TA-graded).
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, status

from app.config import settings
from app.domain.download_progress import DownloadMonitor
from app.domain.ingest import ingest_documents, ingest_qrels, ingest_queries
from shared.contracts import (
    AllQrelsResponse,
    DatasetInfo,
    DatasetStatus,
    DeleteResult,
    DocListItem,
    DocListResponse,
    DocsRequest,
    DocsResponse,
    DownloadDatasetRequest,
    JobStatus,
    PrepareDatasetRequest,
    Qrel,
    QrelListItem,
    QrelsForQueryResponse,
    QrelsListResponse,
    QueryListItem,
    QueryListResponse,
    RawDoc,
    RawQuery,
    ref_from_job,
    status_from_job,
)
from shared.ir_common.dataset_loader import DatasetLoader
from shared.ir_common.datasets import (
    delete_dataset_files,
    is_downloaded,
    manifest_doc_count,
    write_manifest,
)
from shared.ir_common.jobs import JobAlreadyActive, Progress

logger = logging.getLogger("doc-store-service")
router = APIRouter(tags=["doc-store"])

PROGRESS_EVERY = 1000  # update the job counter every N docs (keep lock churn low)


# ---- helpers -------------------------------------------------------------

def _resolve_or_400(dataset: str) -> str:
    dataset_id = settings.resolve_dataset(dataset)
    if dataset_id is None:
        catalog = settings.datasets
        detail = (
            "no datasets configured — set DATASETS in .env"
            if not catalog
            else f"unknown dataset '{dataset}' — not in the configured catalog ({', '.join(catalog)})"
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
    return dataset_id


def _require_mongo(request: Request):
    store = request.app.state.store
    if not store.ping():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"document database unavailable at {settings.mongo_url}",
        )
    return store


def _submit_or_409(request: Request, job_type: str, dataset_id: str, fn):
    """Start a background job, mapping a duplicate (in-flight) job to HTTP 409."""
    try:
        return request.app.state.jobs.submit(job_type, dataset_id, fn)
    except JobAlreadyActive as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


def _active_job(request: Request, dataset_id: str):
    """The in-flight download or ingest for this dataset, if any."""
    jobs = request.app.state.jobs
    return jobs.active_for("download", dataset_id) or jobs.active_for("ingest", dataset_id)


# ---- endpoints -----------------------------------------------------------

@router.get("/dataset/info", response_model=DatasetInfo)
def dataset_info(
    dataset: str = Query(..., description="Dataset id from the catalog"),
) -> DatasetInfo:
    """Dataset details for **preview before download** — network-free (ir-datasets metadata).

    Returns size (docs/queries/qrels) and ``has_qrels`` without fetching the corpus, so a UI
    can show what a dataset is before the user downloads it. Needs no database.
    """
    dataset_id = _resolve_or_400(dataset)
    loader = DatasetLoader(dataset_id)
    md = loader.metadata()
    docs = md.get("docs") or {}
    queries = md.get("queries") or {}
    qrels = md.get("qrels") or {}
    return DatasetInfo(
        dataset_id=dataset_id,
        doc_count=docs.get("count"),
        num_queries=queries.get("count"),
        num_qrels=qrels.get("count"),
        has_qrels=loader.has_qrels(),
        downloaded=is_downloaded(settings.dataset_home, dataset_id),
    )


@router.get("/dataset/status", response_model=DatasetStatus)
def dataset_status(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    with_total: bool = Query(
        False,
        description="Re-confirm the dataset's total doc_count from ir-datasets (verifies a full ingest)",
    ),
) -> DatasetStatus:
    """Readiness check. ``downloaded`` reflects a complete local download; ``doc_count``
    comes from the download manifest (cheap). ``with_total=true`` re-confirms the total
    so you can verify a **full** ingest (``fully_ingested``)."""
    dataset_id = _resolve_or_400(dataset)
    store = _require_mongo(request)
    n = store.count(dataset_id)
    doc_count = manifest_doc_count(settings.dataset_home, dataset_id)
    fully = None
    if with_total:
        doc_count = DatasetLoader(dataset_id).doc_count()
        fully = n >= doc_count
    active = _active_job(request, dataset_id)
    return DatasetStatus(
        dataset_id=dataset_id,
        downloaded=is_downloaded(settings.dataset_home, dataset_id),
        ingested=n > 0,
        ingested_count=n,
        queries_count=store.count_queries(dataset_id),
        qrels_count=store.count_qrels(dataset_id),
        doc_count=doc_count,
        fully_ingested=fully,
        active_job=ref_from_job(active) if active else None,
    )


@router.post("/dataset/download", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
def download_dataset(req: DownloadDatasetRequest, request: Request) -> JobStatus:
    """**The only endpoint that fetches from the internet.** Starts a background download.

    Materializes the dataset's corpus into the local store (``data/dataset/<id>``) so
    every later step runs fully offline, reporting progress as ``processed/total``.
    Idempotent: if already local and not ``force`` the job ends instantly as ``skipped``.
    A second download of the same dataset while one is in flight returns **409**.
    """
    dataset_id = _resolve_or_400(req.dataset)
    home = settings.dataset_home
    force = req.force

    def fn(progress: Progress) -> dict:
        loader = DatasetLoader(dataset_id)
        if is_downloaded(home, dataset_id) and not force:
            return {"skipped": True, "dataset_id": dataset_id,
                    "doc_count": manifest_doc_count(home, dataset_id)}
        # docs_count() is instant (from metadata) — the materialize phase has a real %.
        progress.update(total=loader.doc_count(), message="downloading corpus archive…")
        count = 0
        docs = loader.iter_docs()
        # Phase 1 — archive download/extract. ir_datasets fetches the whole archive
        # before yielding any doc, so report bytes landing in its temp dir (live MB)
        # instead of a frozen 0%. The first next() is what triggers the download.
        with DownloadMonitor(progress, label="downloading archive"):
            first = next(docs, None)
        # Phase 2 — materialize. Docs now stream, so switch to a doc-counted percent.
        if first is not None:
            count = 1
            progress.update(processed=count, message="materializing corpus")
            for _ in docs:
                count += 1
                if count % PROGRESS_EVERY == 0:
                    progress.update(processed=count, message="materializing corpus")
        progress.update(processed=count, message="finalizing")
        write_manifest(home, dataset_id, doc_count=count, complete=True)
        return {"skipped": False, "dataset_id": dataset_id, "doc_count": count}

    return status_from_job(_submit_or_409(request, "download", dataset_id, fn))


@router.post("/dataset/prepare", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
def prepare_dataset(req: PrepareDatasetRequest, request: Request) -> JobStatus:
    """Ingest the dataset's facts into Mongo: **docs + queries + qrels** (background, offline).

    Requires the corpus already downloaded (else **409** — call ``/dataset/download``
    first). Docs dominate progress (``ingested/total``); the small queries/qrels are a
    quick final phase. Idempotent **per collection**: each is (re)ingested only if empty
    or ``force`` is set — so a store that already has docs (from before queries/qrels
    existed) gets backfilled on the next prepare. A concurrent prepare returns **409**.
    """
    dataset_id = _resolve_or_400(req.dataset)
    store = _require_mongo(request)
    home = settings.dataset_home
    if not is_downloaded(home, dataset_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"dataset '{req.dataset}' ({dataset_id}) is not downloaded locally; "
                "call POST /dataset/download first (the only endpoint that uses the internet)"
            ),
        )
    force = req.force
    batch_size = settings.ingest_batch_size

    def fn(progress: Progress) -> dict:
        docs_n = store.count(dataset_id)
        queries_n = store.count_queries(dataset_id)
        qrels_n = store.count_qrels(dataset_id)
        if not force and docs_n > 0 and queries_n > 0 and qrels_n > 0:
            return {
                "skipped": True,
                "dataset_id": dataset_id,
                "ingested_count": docs_n,
                "queries_count": queries_n,
                "qrels_count": qrels_n,
            }
        total = manifest_doc_count(home, dataset_id) or DatasetLoader(dataset_id).doc_count()
        progress.update(total=total, message="ingesting raw docs")
        store.ensure_indexes()          # idempotent — also covers a Mongo-not-ready startup
        loader = DatasetLoader(dataset_id)

        # Documents — the heavy phase that drives the progress bar.
        if force or docs_n == 0:
            store.delete_dataset(dataset_id)  # clean slate (idempotent re-ingest)
            docs_count = ingest_documents(
                docs=loader.iter_docs(),
                write_batch=lambda batch: store.write_batch(dataset_id, batch),
                batch_size=batch_size,
                on_progress=lambda n: progress.update(processed=n),
            )
        else:
            docs_count = docs_n

        # Queries — small; keep the bar full and just switch the message.
        if force or queries_n == 0:
            store.delete_queries(dataset_id)
            progress.update(message="ingesting queries")
            queries_count = ingest_queries(
                queries=loader.iter_queries(),
                write_batch=lambda batch: store.write_queries_batch(dataset_id, batch),
                batch_size=batch_size,
            )
        else:
            queries_count = queries_n

        # Qrels — small; one row per (query_id, doc_id) judgment.
        if force or qrels_n == 0:
            store.delete_qrels(dataset_id)
            progress.update(message="ingesting qrels")
            qrels_count = ingest_qrels(
                qrels=loader.iter_qrels(),
                write_batch=lambda batch: store.write_qrels_batch(dataset_id, batch),
                batch_size=batch_size,
            )
        else:
            qrels_count = qrels_n

        return {
            "skipped": False,
            "dataset_id": dataset_id,
            "ingested_count": docs_count,
            "queries_count": queries_count,
            "qrels_count": qrels_count,
        }

    return status_from_job(_submit_or_409(request, "ingest", dataset_id, fn))


@router.delete("/dataset", response_model=DeleteResult)
def delete_dataset(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    files: bool = Query(True, description="Delete the downloaded corpus folder"),
    docs: bool = Query(True, description="Delete the ingested docs/queries/qrels from Mongo"),
) -> DeleteResult:
    """Independently remove a dataset's local data — other datasets are unaffected.

    ``docs=true`` clears all three Mongo collections for this dataset (documents,
    queries, qrels) since they are the same dataset's facts.
    """
    dataset_id = _resolve_or_400(dataset)
    if _active_job(request, dataset_id) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"a job is in progress for dataset '{dataset}'; wait for it to finish",
        )
    docs_deleted = queries_deleted = qrels_deleted = 0
    if docs:
        store = _require_mongo(request)
        docs_deleted = store.delete_dataset(dataset_id)
        queries_deleted = store.delete_queries(dataset_id)
        qrels_deleted = store.delete_qrels(dataset_id)
    files_deleted = delete_dataset_files(settings.dataset_home, dataset_id) if files else False
    return DeleteResult(
        dataset_id=dataset_id,
        files_deleted=files_deleted,
        docs_deleted=docs_deleted,
        queries_deleted=queries_deleted,
        qrels_deleted=qrels_deleted,
    )


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(request: Request, job_id: str) -> JobStatus:
    """Poll a background job's live progress (download/ingest)."""
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"job '{job_id}' not found")
    return status_from_job(job)


@router.get("/jobs", response_model=list[JobStatus])
def list_jobs(
    request: Request,
    type: Optional[str] = Query(None, description="Filter by job type: download|ingest"),
    dataset: Optional[str] = Query(None, description="Filter by dataset id"),
) -> list[JobStatus]:
    key = _resolve_or_400(dataset) if dataset else None
    jobs = request.app.state.jobs.list(type=type, key=key)
    return [status_from_job(j) for j in jobs]


@router.get("/doc", response_model=RawDoc)
def get_doc(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    doc_id: str = Query(...),
) -> RawDoc:
    """Return the ORIGINAL document text by id (query-time display path)."""
    dataset_id = _resolve_or_400(dataset)
    store = _require_mongo(request)
    text = store.get_one(dataset_id, doc_id)
    if text is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"doc '{doc_id}' not in store for dataset '{dataset}'",
        )
    return RawDoc(doc_id=doc_id, text=text)


@router.post("/docs", response_model=DocsResponse)
def get_docs(req: DocsRequest, request: Request) -> DocsResponse:
    """Batch fetch originals by id — used by retrieval to show top-k results."""
    dataset_id = _resolve_or_400(req.dataset)
    store = _require_mongo(request)
    found = store.get_many(dataset_id, req.doc_ids)
    docs = [RawDoc(doc_id=i, text=found[i]) for i in req.doc_ids if i in found]
    missing = [i for i in req.doc_ids if i not in found]
    return DocsResponse(docs=docs, missing=missing)


@router.get("/docs/list", response_model=DocListResponse)
def list_docs(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    offset: int = Query(0, ge=0, description="Starting seq (0-based position) of the page"),
    limit: int = Query(50, ge=1, le=5000, description="Page size"),
) -> DocListResponse:
    """Browse stored documents by page, ordered by ingest position (``seq``).

    Powers both the UI's "browse the database" view and the indexer's corpus paging,
    so the index is built from the **same** stored docs the UI displays.
    """
    dataset_id = _resolve_or_400(dataset)
    store = _require_mongo(request)
    total = store.count(dataset_id)
    rows = store.list_docs(dataset_id, offset=offset, limit=limit)
    return DocListResponse(
        dataset_id=dataset_id,
        total=total,
        offset=offset,
        limit=limit,
        docs=[DocListItem(**r) for r in rows],
    )


# ---- queries -------------------------------------------------------------

@router.get("/queries", response_model=QueryListResponse)
def list_queries(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    offset: int = Query(0, ge=0, description="Starting seq (0-based position) of the page"),
    limit: int = Query(50, ge=1, le=5000, description="Page size"),
) -> QueryListResponse:
    """Browse the stored test queries by page, ordered by ingest position (``seq``)."""
    dataset_id = _resolve_or_400(dataset)
    store = _require_mongo(request)
    total = store.count_queries(dataset_id)
    rows = store.list_queries(dataset_id, offset=offset, limit=limit)
    return QueryListResponse(
        dataset_id=dataset_id,
        total=total,
        offset=offset,
        limit=limit,
        queries=[QueryListItem(**r) for r in rows],
    )


@router.get("/query", response_model=RawQuery)
def get_query(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    query_id: str = Query(...),
) -> RawQuery:
    """Return a single test query's text by id."""
    dataset_id = _resolve_or_400(dataset)
    store = _require_mongo(request)
    text = store.get_query(dataset_id, query_id)
    if text is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"query '{query_id}' not in store for dataset '{dataset}'",
        )
    return RawQuery(query_id=query_id, text=text)


# ---- qrels (relevance judgments) -----------------------------------------

@router.get("/qrels", response_model=QrelsForQueryResponse)
def get_qrels_for_query(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    query_id: str = Query(...),
) -> QrelsForQueryResponse:
    """All judged documents (the gold set) for a single query."""
    dataset_id = _resolve_or_400(dataset)
    store = _require_mongo(request)
    rows = store.get_qrels_for_query(dataset_id, query_id)
    return QrelsForQueryResponse(
        dataset_id=dataset_id,
        query_id=query_id,
        judgments=[Qrel(**r) for r in rows],
    )


@router.get("/qrels/list", response_model=QrelsListResponse)
def list_qrels(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    offset: int = Query(0, ge=0, description="Starting seq (0-based position) of the page"),
    limit: int = Query(50, ge=1, le=5000, description="Page size"),
) -> QrelsListResponse:
    """Browse all stored relevance judgments by page, ordered by ``seq``."""
    dataset_id = _resolve_or_400(dataset)
    store = _require_mongo(request)
    total = store.count_qrels(dataset_id)
    rows = store.list_qrels(dataset_id, offset=offset, limit=limit)
    return QrelsListResponse(
        dataset_id=dataset_id,
        total=total,
        offset=offset,
        limit=limit,
        qrels=[QrelListItem(**r) for r in rows],
    )


@router.get("/qrels/all", response_model=AllQrelsResponse)
def all_qrels(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
) -> AllQrelsResponse:
    """The full qrels as ``{query_id: {doc_id: relevance}}`` — the eval/fusion input shape."""
    dataset_id = _resolve_or_400(dataset)
    store = _require_mongo(request)
    return AllQrelsResponse(dataset_id=dataset_id, qrels=store.all_qrels(dataset_id))
