"""In-process background jobs with progress tracking.

A tiny, thread-based job runner shared by services that expose long **offline**
operations (dataset download, ingest, index build) as **async** endpoints: the
HTTP call returns a ``JobRef`` immediately and the work runs in a daemon thread
that reports progress, which clients poll via ``GET /jobs/{id}``.

Deliberately in-memory and dependency-free (stdlib ``threading``): job history is
lost on restart, but the *real* state (manifest on disk, docs in Mongo, index
artifact) is always re-derivable from the status endpoints — jobs only track the
**live** progress of an operation in flight. One active job is allowed per
``(type, key)`` so repeated triggers (e.g. double-clicks) can't pile up.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger("ir.jobs")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobState:
    """The states a job moves through (terminal: succeeded/failed/skipped)."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


_TERMINAL = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.SKIPPED})


@dataclass
class Job:
    """A unit of background work and its live progress."""

    id: str
    type: str                       # "download" | "ingest" | "build"
    key: str                        # the dataset id the job operates on
    state: str = JobState.PENDING
    processed: int = 0
    total: Optional[int] = None
    message: str = ""
    error: Optional[str] = None
    started_at: str = field(default_factory=_now)
    finished_at: Optional[str] = None
    result: dict = field(default_factory=dict)

    @property
    def percent(self) -> Optional[float]:
        """0–100, or ``None`` while the total is still unknown."""
        if not self.total:
            return None
        return round(min(self.processed / self.total, 1.0) * 100, 2)

    @property
    def active(self) -> bool:
        return self.state not in _TERMINAL


class Progress:
    """Thread-safe handle a job function uses to report progress."""

    def __init__(self, job: Job, lock: threading.Lock) -> None:
        self._job = job
        self._lock = lock

    def set_total(self, total: Optional[int]) -> None:
        with self._lock:
            self._job.total = total

    def advance(self, n: int = 1) -> None:
        with self._lock:
            self._job.processed += n

    def update(
        self,
        *,
        processed: Optional[int] = None,
        total: Optional[int] = None,
        message: Optional[str] = None,
    ) -> None:
        with self._lock:
            if processed is not None:
                self._job.processed = processed
            if total is not None:
                self._job.total = total
            if message is not None:
                self._job.message = message


# A job body: receives a Progress handle and returns a result dict. Returning
# ``{"skipped": True, ...}`` marks the job SKIPPED instead of SUCCEEDED.
JobFn = Callable[[Progress], Optional[dict]]


class JobAlreadyActive(Exception):
    """Raised by ``submit`` when a job for the same (type, key) is in flight."""

    def __init__(self, job: Job) -> None:
        self.job = job
        super().__init__(f"a '{job.type}' job for '{job.key}' is already in progress")


class JobRegistry:
    """Thread-safe registry that runs jobs in daemon threads and tracks them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def submit(self, type: str, key: str, fn: JobFn) -> Job:
        """Start ``fn`` in the background and return its Job.

        Raises :class:`JobAlreadyActive` if a job for the same ``(type, key)`` is
        still running — callers map that to HTTP 409.
        """
        with self._lock:
            active = self._active_for(type, key)
            if active is not None:
                raise JobAlreadyActive(active)
            job = Job(id=uuid.uuid4().hex, type=type, key=key, state=JobState.RUNNING)
            self._jobs[job.id] = job
        progress = Progress(job, self._lock)
        threading.Thread(
            target=self._run,
            args=(job, fn, progress),
            daemon=True,
            name=f"job-{type}-{job.id[:8]}",
        ).start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, *, type: Optional[str] = None, key: Optional[str] = None) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [
            j for j in jobs
            if (type is None or j.type == type) and (key is None or j.key == key)
        ]

    def active_for(self, type: str, key: str) -> Optional[Job]:
        with self._lock:
            return self._active_for(type, key)

    # ---- internals ----
    def _active_for(self, type: str, key: str) -> Optional[Job]:
        """Find an in-flight job for (type, key). Caller must hold the lock."""
        for job in self._jobs.values():
            if job.type == type and job.key == key and job.active:
                return job
        return None

    def _run(self, job: Job, fn: JobFn, progress: Progress) -> None:
        try:
            result = fn(progress) or {}
            with self._lock:
                job.result = result
                job.state = JobState.SKIPPED if result.get("skipped") else JobState.SUCCEEDED
                job.finished_at = _now()
        except Exception as exc:  # noqa: BLE001 - surfaced to the client via the Job
            with self._lock:
                job.state = JobState.FAILED
                job.error = str(exc)
                job.finished_at = _now()
            logger.exception("job %s (%s/%s) failed", job.id, job.type, job.key)
