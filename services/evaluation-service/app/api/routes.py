"""HTTP layer (thin) for the evaluation-service.

* ``POST /evaluate`` — start a background evaluation of one or more model configs on a
  dataset's test queries; returns a ``JobStatus`` (poll ``GET /jobs/{id}``). The finished
  report is persisted and also returned in the job's ``result``.
* ``GET /reports`` / ``GET /report/{id}`` / ``DELETE /report/{id}`` — manage saved reports
  (the labeled, persisted tables the report's charts come from).
* ``GET /jobs`` / ``GET /jobs/{id}`` — poll evaluation progress.

The heavy lifting (querying retrieval, scoring against qrels) lives in the domain
(:mod:`app.domain.evaluator`); this module is only request → domain → response.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, status

from app.config import settings
from app.domain.evaluator import NoQrelsError, run_evaluation
from shared.contracts import (
    DEFAULT_METRICS,
    EvalRunSpec,
    EvaluateRequest,
    EvaluationReport,
    JobStatus,
    PerQueryReport,
    ReportDeleteResult,
    ReportsList,
    status_from_job,
)
from shared.ir_common.datasets import safe_id
from shared.ir_common.jobs import JobAlreadyActive, Progress

logger = logging.getLogger("evaluation-service")
router = APIRouter(tags=["evaluation"])

JOB_TYPE = "evaluate"

# A no-arg evaluation measures the four required base representations (§2.4) so the
# default report is the full baseline table. Hybrids are added by passing explicit runs.
_DEFAULT_RUNS = ("tfidf", "bm25", "embedding", "bert")


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


def _sanitize_label(label: str) -> str:
    """Filesystem-safe label for the report id (keep word chars / . _ -)."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-._")
    return cleaned or "report"


def _report_id(dataset_id: str, label: str) -> str:
    return f"{safe_id(dataset_id)}__{_sanitize_label(label)}"


def _job_key(report_id: str) -> str:
    """One in-flight evaluation per report id (dataset+label)."""
    return report_id


def _default_runs() -> list[EvalRunSpec]:
    return [EvalRunSpec(label=m, model=m) for m in _DEFAULT_RUNS]


def _submit_or_409(request: Request, key: str, fn):
    try:
        return request.app.state.jobs.submit(JOB_TYPE, key, fn)
    except JobAlreadyActive as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


# ---- evaluate ------------------------------------------------------------

@router.post("/evaluate", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
def evaluate(req: EvaluateRequest, request: Request) -> JobStatus:
    """Start a background evaluation of the given model configs on a dataset's test queries.

    Pre-flight: **400** unknown dataset; **503** if retrieval or the doc-store is down;
    **409** if the dataset has no test queries / qrels stored (ingest it first), or if an
    evaluation with the same label is already running. Idempotent: an existing report with
    this label is returned as ``skipped`` unless ``force``.
    """
    dataset_id = _resolve_or_400(req.dataset)
    state = request.app.state

    if not state.retrieval.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"retrieval-service unavailable at {settings.retrieval_url}; it runs the queries being evaluated",
        )
    if not state.doc_store.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"doc-store unavailable at {settings.doc_store_url}; evaluation reads its queries & qrels",
        )
    try:
        ds_status = state.doc_store.dataset_status(dataset_id)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"doc-store error while checking dataset readiness: {exc}",
        ) from exc
    if ds_status.qrels_count <= 0 or ds_status.queries_count <= 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"dataset '{dataset_id}' has no test queries/qrels stored "
                f"(queries={ds_status.queries_count}, qrels={ds_status.qrels_count}); "
                "ingest it first via the doc-store POST /dataset/prepare"
            ),
        )

    runs = req.runs or _default_runs()
    metrics = req.metrics or list(DEFAULT_METRICS)
    report_id = _report_id(dataset_id, req.label)
    store = state.store
    force = req.force
    top_k = req.top_k
    max_queries = req.max_queries
    per_query = req.per_query
    retrieval = state.retrieval
    doc_store = state.doc_store
    concurrency = settings.concurrency
    label = req.label

    def fn(progress: Progress) -> dict:
        if not force and store.exists(report_id):
            existing = store.load(report_id)
            if existing is not None:
                return {"skipped": True, "report_id": report_id, "report": existing.model_dump()}
        report, sidecar = run_evaluation(
            report_id=report_id,
            dataset_id=dataset_id,
            label=label,
            runs=runs,
            metrics=metrics,
            top_k=top_k,
            max_queries=max_queries,
            per_query=per_query,
            retrieval=retrieval,
            doc_store=doc_store,
            concurrency=concurrency,
            advance=progress.advance,
            set_total=lambda n: progress.update(total=n),
            set_message=lambda m: progress.update(message=m),
        )
        store.save(report)
        if sidecar is not None:
            store.save_per_query(report_id, sidecar)
        return {"skipped": False, "report_id": report_id, "report": report.model_dump()}

    # NoQrelsError inside the job is surfaced via the failed job; pre-flight already
    # catches the common "not ingested" case with a clean 409.
    return status_from_job(_submit_or_409(request, _job_key(report_id), fn))


# ---- reports -------------------------------------------------------------

@router.get("/reports", response_model=ReportsList)
def list_reports(request: Request) -> ReportsList:
    """List every saved report (newest first) — the before/after comparison set."""
    return ReportsList(items=request.app.state.store.list())


@router.get("/report/{report_id}", response_model=EvaluationReport)
def get_report(request: Request, report_id: str) -> EvaluationReport:
    """Fetch one saved report's full metrics table by id."""
    report = request.app.state.store.load(report_id)
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"no report '{report_id}'")
    return report


@router.get("/report/{report_id}/per-query", response_model=PerQueryReport)
def get_per_query(
    request: Request,
    report_id: str,
    run: Optional[str] = Query(None, description="Comma-separated run labels to include (default: all)"),
) -> PerQueryReport:
    """Per-query scores for the drill-down (best/worst & win-loss). Optionally subset to
    specific run labels to keep the payload small. **404** if no sidecar was stored
    (the report was run with ``per_query=false``)."""
    sidecar = request.app.state.store.load_per_query(report_id)
    if sidecar is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no per-query data for report '{report_id}' (re-run with per_query enabled)",
        )
    runs = sidecar.get("runs", {})
    if run:
        wanted = {r.strip() for r in run.split(",") if r.strip()}
        runs = {label: scores for label, scores in runs.items() if label in wanted}
    return PerQueryReport(
        report_id=sidecar.get("report_id", report_id),
        dataset_id=sidecar.get("dataset_id", ""),
        metrics=sidecar.get("metrics", []),
        queries=sidecar.get("queries", {}),
        runs=runs,
    )


@router.delete("/report/{report_id}", response_model=ReportDeleteResult)
def delete_report(request: Request, report_id: str) -> ReportDeleteResult:
    """Delete a saved report (its JSON + CSV)."""
    if request.app.state.jobs.active_for(JOB_TYPE, _job_key(report_id)) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"an evaluation is in progress for '{report_id}'; wait for it to finish",
        )
    deleted = request.app.state.store.delete(report_id)
    return ReportDeleteResult(report_id=report_id, deleted=deleted)


# ---- jobs ----------------------------------------------------------------

@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(request: Request, job_id: str) -> JobStatus:
    """Poll an evaluation job's live progress (and its finished report in ``result``)."""
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"job '{job_id}' not found")
    return status_from_job(job)


@router.get("/jobs", response_model=list[JobStatus])
def list_jobs(
    request: Request,
    dataset: Optional[str] = Query(None, description="Filter by dataset id"),
) -> list[JobStatus]:
    jobs = request.app.state.jobs.list(type=JOB_TYPE)
    if dataset is not None:
        dataset_id = _resolve_or_400(dataset)
        prefix = f"{safe_id(dataset_id)}__"
        jobs = [j for j in jobs if j.key.startswith(prefix)]
    return [status_from_job(j) for j in jobs]
