"""Passthrough to the evaluation-service — IR effectiveness measurement.

Mirrors the service's API under ``/evaluation``:

* ``POST /evaluation/evaluate`` — start a background evaluation of one or more model
  configs on a dataset's test queries (returns a ``JobStatus``; poll ``/evaluation/jobs/{id}``).
* ``GET /evaluation/reports`` / ``GET /evaluation/report/{id}`` / ``DELETE /evaluation/report/{id}`` —
  the labeled, persisted MAP/nDCG/Recall/P@10 tables the report's charts come from.
* ``GET /evaluation/jobs`` / ``GET /evaluation/jobs/{id}`` — poll evaluation progress.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, Request, status
from fastapi.responses import JSONResponse

from app.proxy import clean, proxy
from shared.contracts import (
    EvaluateRequest,
    EvaluationReport,
    JobStatus,
    PerQueryReport,
    ReportDeleteResult,
    ReportsList,
)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/evaluate", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
def evaluate(req: EvaluateRequest, request: Request) -> JSONResponse:
    """Start a background evaluation (MAP/nDCG/Recall/P@10) of the given model configs."""
    return proxy(lambda: request.app.state.evaluation.post("/evaluate", json=req.model_dump(mode="json")))


@router.get("/reports", response_model=ReportsList)
def list_reports(request: Request) -> JSONResponse:
    """List every saved evaluation report (the before/after comparison set)."""
    return proxy(lambda: request.app.state.evaluation.get("/reports"))


@router.get("/report/{report_id}", response_model=EvaluationReport)
def get_report(request: Request, report_id: str) -> JSONResponse:
    """Fetch one saved report's full metrics table by id."""
    return proxy(lambda: request.app.state.evaluation.get(f"/report/{report_id}"))


@router.get("/report/{report_id}/per-query", response_model=PerQueryReport)
def get_per_query(
    request: Request,
    report_id: str,
    run: Optional[str] = Query(None, description="Comma-separated run labels to include (default: all)"),
) -> JSONResponse:
    """Per-query scores for the drill-down (best/worst & win-loss)."""
    return proxy(lambda: request.app.state.evaluation.get(f"/report/{report_id}/per-query", params=clean({"run": run})))


@router.delete("/report/{report_id}", response_model=ReportDeleteResult)
def delete_report(request: Request, report_id: str) -> JSONResponse:
    """Delete a saved report (its JSON + CSV)."""
    return proxy(lambda: request.app.state.evaluation.request("DELETE", f"/report/{report_id}"))


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(request: Request, job_id: str) -> JSONResponse:
    return proxy(lambda: request.app.state.evaluation.get(f"/jobs/{job_id}"))


@router.get("/jobs", response_model=list[JobStatus])
def list_jobs(
    request: Request,
    dataset: Optional[str] = Query(None, description="Filter by dataset id"),
) -> JSONResponse:
    return proxy(lambda: request.app.state.evaluation.get("/jobs", params=clean({"dataset": dataset})))
