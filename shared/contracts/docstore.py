"""Contracts for the doc-store-service (raw/original documents in MongoDB).

The doc-store owns the **original** document text, keyed by ``doc_id`` per dataset.
It is populated **offline** (download + ingest) and read **by ID at query time** to
display the original document — the TA-graded requirement.

A dataset is referenced by its **id** (a value from the configured ``DATASETS``
catalog), so callers stay dataset-agnostic. The download and ingest are long
offline steps, so they run as **background jobs** (see ``shared.contracts.jobs``):
the endpoint returns a ``JobRef`` and the client polls ``GET /jobs/{id}`` for progress.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from shared.contracts.jobs import JobRef


class DatasetStatus(BaseModel):
    """Readiness check — how much of this dataset's raw text is stored in Mongo.

    The Mongo counts are cheap. ``doc_count`` comes from the download manifest when
    present; with ``with_total=true`` it is (re)confirmed from ir-datasets so you can
    verify a **full** ingest (``fully_ingested``). ``active_job`` surfaces any
    download/ingest currently in flight for this dataset (with its progress).
    """

    dataset_id: str         # concrete ir-datasets id
    downloaded: bool        # True if the corpus is fully fetched locally (offline-ready)
    ingested: bool          # True if any raw docs are stored for this dataset
    ingested_count: int     # number of raw docs stored
    doc_count: Optional[int] = None        # total docs in the dataset (manifest/with_total)
    fully_ingested: Optional[bool] = None  # ingested_count >= doc_count (only with_total)
    active_job: Optional[JobRef] = None    # in-flight download/ingest, if any


class DatasetInfo(BaseModel):
    """Dataset details available **before** downloading (network-free metadata).

    Sourced from ir-datasets' shipped metadata, so a UI can preview a dataset's size
    and whether it has qrels (required by the assignment) before fetching anything.
    """

    dataset_id: str
    doc_count: Optional[int] = None     # total documents (from ir-datasets metadata)
    num_queries: Optional[int] = None   # test queries
    num_qrels: Optional[int] = None     # relevance judgments
    has_qrels: bool = False             # required by the assignment (no qrels → rejected)
    downloaded: bool = False            # already fetched to the local store?


class DownloadDatasetRequest(BaseModel):
    dataset: str = Field(description="Dataset id from the configured catalog (DATASETS)")
    force: bool = Field(default=False, description="Re-fetch even if already local")


class PrepareDatasetRequest(BaseModel):
    dataset: str = Field(description="Dataset id from the configured catalog (DATASETS)")
    force: bool = Field(default=False, description="Re-ingest even if already present")


class DeleteDatasetRequest(BaseModel):
    """Independent removal of a dataset's local data (others are unaffected)."""

    dataset: str = Field(description="Dataset id from the configured catalog (DATASETS)")
    files: bool = Field(default=True, description="Delete the downloaded corpus folder")
    docs: bool = Field(default=True, description="Delete the ingested raw docs from Mongo")


class DeleteResult(BaseModel):
    dataset_id: str
    files_deleted: bool     # the on-disk corpus folder was removed
    docs_deleted: int       # number of raw docs removed from Mongo


class RawDoc(BaseModel):
    doc_id: str
    text: str               # the ORIGINAL document text (never preprocessed)


class DocListItem(BaseModel):
    """One row of a paginated document listing (browse the stored corpus)."""

    seq: int                # 0-based position in the dataset's stored order
    doc_id: str
    text: str               # the ORIGINAL document text


class DocListResponse(BaseModel):
    """A page of stored documents — for browsing the DB and as the index build source.

    Pages are ordered by ``seq`` (the ingest position), so ``offset`` is the starting
    ``seq`` and successive pages walk the corpus deterministically.
    """

    dataset_id: str
    total: int              # total docs stored for this dataset
    offset: int             # starting seq of this page
    limit: int              # page size requested
    docs: list[DocListItem]


class DocsRequest(BaseModel):
    """Batch fetch of original docs by id — used by retrieval to show top-k results."""

    dataset: str
    doc_ids: list[str]


class DocsResponse(BaseModel):
    docs: list[RawDoc]
    missing: list[str]      # ids not found in the store
