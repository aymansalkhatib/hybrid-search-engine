"""Passthrough to the topic-service — LDA topic modelling (extra feature, §11).

Mirrors the service API under ``/topic``: an offline ``/build`` job, the topic table the
UI charts (top terms + weights), and inferring a query's topic mixture. Toggleable and
evaluated on its own via the topic charts.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, Request, status
from fastapi.responses import JSONResponse

from app.proxy import clean, proxy
from shared.contracts import (
    InferRequest,
    InferResponse,
    JobStatus,
    TopicBuildRequest,
    TopicDeleteResult,
    TopicInfo,
    TopicStatusResponse,
)

router = APIRouter(prefix="/topic", tags=["topic"])


@router.post("/build", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
def build(req: TopicBuildRequest, request: Request) -> JSONResponse:
    """Start an offline LDA topic-model build for a dataset."""
    return proxy(lambda: request.app.state.topic.post("/build", json=req.model_dump(mode="json")))


@router.get("/status", response_model=TopicStatusResponse)
def topic_status(request: Request, dataset: str = Query(...)) -> JSONResponse:
    return proxy(lambda: request.app.state.topic.get("/status", params={"dataset": dataset}))


@router.get("/topics", response_model=list[TopicInfo])
def topics(request: Request, dataset: str = Query(...)) -> JSONResponse:
    return proxy(lambda: request.app.state.topic.get("/topics", params={"dataset": dataset}))


@router.post("/infer", response_model=InferResponse)
def infer(req: InferRequest, request: Request) -> JSONResponse:
    return proxy(lambda: request.app.state.topic.post("/infer", json=req.model_dump(mode="json")))


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(request: Request, job_id: str) -> JSONResponse:
    return proxy(lambda: request.app.state.topic.get(f"/jobs/{job_id}"))


@router.get("/jobs", response_model=list[JobStatus])
def list_jobs(request: Request, dataset: Optional[str] = Query(None)) -> JSONResponse:
    return proxy(lambda: request.app.state.topic.get("/jobs", params=clean({"dataset": dataset})))


@router.delete("/topic", response_model=TopicDeleteResult)
def delete_topic(request: Request, dataset: str = Query(...)) -> JSONResponse:
    return proxy(lambda: request.app.state.topic.request("DELETE", "/topic", params={"dataset": dataset}))
