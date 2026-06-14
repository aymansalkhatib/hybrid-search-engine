"""MongoDB adapter for the raw-document store.

One collection holds every dataset's raw docs as ``{dataset, seq, doc_id, text}``:

* a **unique index on (dataset, doc_id)** — the by-ID lookup used at query time, and
  it also blocks duplicate ingests;
* an index on **(dataset, seq)** — ``seq`` is the 0-based ingest position, giving a
  stable order for **paginated browsing** and for the indexer to **page the corpus**.

The original text is stored verbatim (never preprocessed); it is what the UI displays
for the top-k results and the single source the inverted index is built from.
"""

from __future__ import annotations

import logging
from typing import Iterable

from pymongo import ASCENDING, MongoClient
from pymongo.errors import PyMongoError

logger = logging.getLogger("doc-store-service")


class MongoDocStore:
    def __init__(self, url: str, db_name: str, collection: str) -> None:
        # Fail fast (3s) when Mongo is unreachable so endpoints return 503 quickly
        # instead of hanging the whole request.
        self._client: MongoClient = MongoClient(url, serverSelectionTimeoutMS=3000)
        self._coll = self._client[db_name][collection]

    def ping(self) -> bool:
        try:
            self._client.admin.command("ping")
            return True
        except PyMongoError:
            return False

    def ensure_indexes(self) -> None:
        self._coll.create_index(
            [("dataset", ASCENDING), ("doc_id", ASCENDING)],
            unique=True,
            name="dataset_docid",
        )
        # Non-unique on purpose: a pre-existing collection may hold seq-less docs
        # (null) from before this field existed; a unique index would reject the
        # duplicate nulls. seq uniqueness is guaranteed by the ingest counter.
        self._coll.create_index(
            [("dataset", ASCENDING), ("seq", ASCENDING)],
            name="dataset_seq",
        )

    def count(self, dataset_id: str) -> int:
        return self._coll.count_documents({"dataset": dataset_id})

    def delete_dataset(self, dataset_id: str) -> int:
        return self._coll.delete_many({"dataset": dataset_id}).deleted_count

    def write_batch(self, dataset_id: str, docs: list[tuple[int, str, str]]) -> None:
        """Insert a batch of ``(seq, doc_id, text)`` rows."""
        if not docs:
            return
        self._coll.insert_many(
            [{"dataset": dataset_id, "seq": s, "doc_id": d, "text": t} for s, d, t in docs],
            ordered=False,
        )

    def list_docs(self, dataset_id: str, offset: int, limit: int) -> list[dict]:
        """A page of stored docs ordered by ``seq`` in ``[offset, offset+limit)``.

        Uses the ``(dataset, seq)`` index as a range scan (no costly ``skip``), so it
        stays fast even when paging deep into a 500K-doc corpus for an index build.
        """
        cursor = (
            self._coll.find(
                {"dataset": dataset_id, "seq": {"$gte": offset, "$lt": offset + limit}},
                projection={"_id": 0, "seq": 1, "doc_id": 1, "text": 1},
            )
            .sort("seq", ASCENDING)
        )
        return [{"seq": d["seq"], "doc_id": d["doc_id"], "text": d["text"]} for d in cursor]

    def get_one(self, dataset_id: str, doc_id: str) -> str | None:
        doc = self._coll.find_one(
            {"dataset": dataset_id, "doc_id": doc_id},
            projection={"_id": 0, "text": 1},
        )
        return None if doc is None else doc["text"]

    def get_many(self, dataset_id: str, doc_ids: Iterable[str]) -> dict[str, str]:
        cursor = self._coll.find(
            {"dataset": dataset_id, "doc_id": {"$in": list(doc_ids)}},
            projection={"_id": 0, "doc_id": 1, "text": 1},
        )
        return {d["doc_id"]: d["text"] for d in cursor}

    def close(self) -> None:
        self._client.close()
