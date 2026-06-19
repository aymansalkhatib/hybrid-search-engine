"""Passthrough to the clustering-service — unsupervised document clustering (extra, §11).

Mirrors the service API under ``/clustering``: an offline ``/build`` job, the cluster
table & 2-D plot data the UI charts, and assigning a query to its nearest cluster. The
UI toggles "search with vs without" this feature independently of the rest.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, Request, status
from fastapi.responses import JSONResponse

from app.proxy import clean, proxy
from shared.contracts import (
    AssignRequest,
    AssignResponse,
    ClusterBuildRequest,
    ClusterDeleteResult,
    ClusterInfo,
    ClusterPlotResponse,
    ClusterStatusResponse,
    JobStatus,
)

router = APIRouter(prefix="/clustering", tags=["clustering"])


@router.post("/build", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
def build(req: ClusterBuildRequest, request: Request) -> JSONResponse:
    """Start an offline clustering build (TF-IDF + KMeans) for a dataset."""
    return proxy(lambda: request.app.state.clustering.post("/build", json=req.model_dump(mode="json")))


@router.get("/status", response_model=ClusterStatusResponse)
def cluster_status(request: Request, dataset: str = Query(...)) -> JSONResponse:
    return proxy(lambda: request.app.state.clustering.get("/status", params={"dataset": dataset}))


@router.get("/clusters", response_model=list[ClusterInfo])
def clusters(request: Request, dataset: str = Query(...)) -> JSONResponse:
    return proxy(lambda: request.app.state.clustering.get("/clusters", params={"dataset": dataset}))


@router.get("/plot", response_model=ClusterPlotResponse)
def plot(request: Request, dataset: str = Query(...)) -> JSONResponse:
    return proxy(lambda: request.app.state.clustering.get("/plot", params={"dataset": dataset}))


@router.post("/assign", response_model=AssignResponse)
def assign(req: AssignRequest, request: Request) -> JSONResponse:
    return proxy(lambda: request.app.state.clustering.post("/assign", json=req.model_dump(mode="json")))


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(request: Request, job_id: str) -> JSONResponse:
    return proxy(lambda: request.app.state.clustering.get(f"/jobs/{job_id}"))


@router.get("/jobs", response_model=list[JobStatus])
def list_jobs(request: Request, dataset: Optional[str] = Query(None)) -> JSONResponse:
    return proxy(lambda: request.app.state.clustering.get("/jobs", params=clean({"dataset": dataset})))


@router.delete("/clustering", response_model=ClusterDeleteResult)
def delete_clustering(request: Request, dataset: str = Query(...)) -> JSONResponse:
    return proxy(lambda: request.app.state.clustering.request("DELETE", "/clustering", params={"dataset": dataset}))
