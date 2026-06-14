"""Contracts for background jobs (download / ingest / build progress).

Long offline operations are started asynchronously: the endpoint returns a
``JobRef`` and the client polls ``GET /jobs/{id}`` for a ``JobStatus`` (state +
percent). The two helpers below build these models from a ``Job`` by duck-typing
its attributes, so this module stays free of any dependency on ``ir_common``.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class JobRef(BaseModel):
    """Returned by endpoints that start a background job."""

    job_id: str
    type: str               # "download" | "ingest" | "build"
    key: str                # the dataset id the job operates on
    state: str              # pending | running | succeeded | failed | skipped


class JobStatus(BaseModel):
    """Full progress snapshot of a background job (polled by the UI)."""

    job_id: str
    type: str
    key: str
    state: str
    processed: int = 0
    total: Optional[int] = None
    percent: Optional[float] = None     # 0–100, or None while total is unknown
    message: str = ""
    error: Optional[str] = None
    started_at: str
    finished_at: Optional[str] = None
    result: dict = Field(default_factory=dict)


def ref_from_job(job) -> JobRef:
    """Build a JobRef from a ``Job`` (duck-typed)."""
    return JobRef(job_id=job.id, type=job.type, key=job.key, state=job.state)


def status_from_job(job) -> JobStatus:
    """Build a JobStatus from a ``Job`` (duck-typed)."""
    return JobStatus(
        job_id=job.id,
        type=job.type,
        key=job.key,
        state=job.state,
        processed=job.processed,
        total=job.total,
        percent=job.percent,
        message=job.message,
        error=job.error,
        started_at=job.started_at,
        finished_at=job.finished_at,
        result=job.result,
    )
