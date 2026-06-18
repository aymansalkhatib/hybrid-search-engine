"""MongoDB adapter for the dataset store (documents + queries + qrels).

Each dataset fact type lives in its **own collection**, keyed by ``dataset`` so several
datasets share the store without colliding:

* ``documents`` — ``{dataset, seq, doc_id, text}``; unique ``(dataset, doc_id)`` is the
  by-ID lookup used at query time (and blocks duplicate ingests); ``(dataset, seq)``
  gives a stable order for paginated browsing and for the indexer to page the corpus.
* ``queries`` — ``{dataset, seq, query_id, text}``; unique ``(dataset, query_id)``.
* ``qrels`` — ``{dataset, seq, query_id, doc_id, relevance}`` (the TREC qrels shape, one
  document per judgment); unique ``(dataset, query_id, doc_id)`` dedupes / makes ingest
  idempotent, and ``(dataset, query_id)`` groups a query's gold set in one indexed scan.

``seq`` is the 0-based ingest position in every collection, so each supports stable
paginated browsing. The original document text is stored verbatim (never preprocessed);
it is what the UI displays for the top-k results and the single source the inverted index
is built from.
"""

from __future__ import annotations

import logging
from typing import Iterable

from pymongo import ASCENDING, MongoClient
from pymongo.errors import PyMongoError

logger = logging.getLogger("doc-store-service")


class MongoDocStore:
    def __init__(
        self,
        url: str,
        db_name: str,
        collection: str,
        queries_collection: str = "queries",
        qrels_collection: str = "qrels",
    ) -> None:
        # Fail fast (3s) when Mongo is unreachable so endpoints return 503 quickly
        # instead of hanging the whole request.
        self._client: MongoClient = MongoClient(url, serverSelectionTimeoutMS=3000)
        db = self._client[db_name]
        self._coll = db[collection]
        self._queries = db[queries_collection]
        self._qrels = db[qrels_collection]

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
        # queries: one row per (dataset, query_id); seq for stable browsing.
        self._queries.create_index(
            [("dataset", ASCENDING), ("query_id", ASCENDING)],
            unique=True,
            name="dataset_queryid",
        )
        self._queries.create_index(
            [("dataset", ASCENDING), ("seq", ASCENDING)],
            name="queries_dataset_seq",
        )
        # qrels: one row per judgment; unique (dataset, query_id, doc_id) dedupes,
        # (dataset, query_id) groups a query's gold set, seq for stable browsing.
        self._qrels.create_index(
            [("dataset", ASCENDING), ("query_id", ASCENDING), ("doc_id", ASCENDING)],
            unique=True,
            name="dataset_query_doc",
        )
        self._qrels.create_index(
            [("dataset", ASCENDING), ("query_id", ASCENDING)],
            name="qrels_dataset_query",
        )
        self._qrels.create_index(
            [("dataset", ASCENDING), ("seq", ASCENDING)],
            name="qrels_dataset_seq",
        )

    def count(self, dataset_id: str) -> int:
        return self._coll.count_documents({"dataset": dataset_id})

    def count_queries(self, dataset_id: str) -> int:
        return self._queries.count_documents({"dataset": dataset_id})

    def count_qrels(self, dataset_id: str) -> int:
        return self._qrels.count_documents({"dataset": dataset_id})

    def delete_dataset(self, dataset_id: str) -> int:
        return self._coll.delete_many({"dataset": dataset_id}).deleted_count

    def delete_queries(self, dataset_id: str) -> int:
        return self._queries.delete_many({"dataset": dataset_id}).deleted_count

    def delete_qrels(self, dataset_id: str) -> int:
        return self._qrels.delete_many({"dataset": dataset_id}).deleted_count

    def write_batch(self, dataset_id: str, docs: list[tuple[int, str, str]]) -> None:
        """Insert a batch of ``(seq, doc_id, text)`` rows."""
        if not docs:
            return
        self._coll.insert_many(
            [{"dataset": dataset_id, "seq": s, "doc_id": d, "text": t} for s, d, t in docs],
            ordered=False,
        )

    def write_queries_batch(self, dataset_id: str, rows: list[tuple[int, str, str]]) -> None:
        """Insert a batch of ``(seq, query_id, text)`` query rows."""
        if not rows:
            return
        self._queries.insert_many(
            [{"dataset": dataset_id, "seq": s, "query_id": q, "text": t} for s, q, t in rows],
            ordered=False,
        )

    def write_qrels_batch(
        self, dataset_id: str, rows: list[tuple[int, str, str, int]]
    ) -> None:
        """Insert a batch of ``(seq, query_id, doc_id, relevance)`` judgment rows."""
        if not rows:
            return
        self._qrels.insert_many(
            [
                {"dataset": dataset_id, "seq": s, "query_id": q, "doc_id": d, "relevance": r}
                for s, q, d, r in rows
            ],
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

    # ---- queries -------------------------------------------------------

    def get_query(self, dataset_id: str, query_id: str) -> str | None:
        q = self._queries.find_one(
            {"dataset": dataset_id, "query_id": query_id},
            projection={"_id": 0, "text": 1},
        )
        return None if q is None else q["text"]

    def list_queries(self, dataset_id: str, offset: int, limit: int) -> list[dict]:
        """A page of stored queries ordered by ``seq`` in ``[offset, offset+limit)``."""
        cursor = self._queries.find(
            {"dataset": dataset_id, "seq": {"$gte": offset, "$lt": offset + limit}},
            projection={"_id": 0, "seq": 1, "query_id": 1, "text": 1},
        ).sort("seq", ASCENDING)
        return [
            {"seq": q["seq"], "query_id": q["query_id"], "text": q["text"]} for q in cursor
        ]

    # ---- qrels ---------------------------------------------------------

    def get_qrels_for_query(self, dataset_id: str, query_id: str) -> list[dict]:
        """All judged ``(doc_id, relevance)`` rows for one query (a query's gold set)."""
        cursor = self._qrels.find(
            {"dataset": dataset_id, "query_id": query_id},
            projection={"_id": 0, "query_id": 1, "doc_id": 1, "relevance": 1},
        )
        return [
            {"query_id": j["query_id"], "doc_id": j["doc_id"], "relevance": j["relevance"]}
            for j in cursor
        ]

    def list_qrels(self, dataset_id: str, offset: int, limit: int) -> list[dict]:
        """A page of stored judgments ordered by ``seq`` in ``[offset, offset+limit)``."""
        cursor = self._qrels.find(
            {"dataset": dataset_id, "seq": {"$gte": offset, "$lt": offset + limit}},
            projection={"_id": 0, "seq": 1, "query_id": 1, "doc_id": 1, "relevance": 1},
        ).sort("seq", ASCENDING)
        return [
            {
                "seq": j["seq"],
                "query_id": j["query_id"],
                "doc_id": j["doc_id"],
                "relevance": j["relevance"],
            }
            for j in cursor
        ]

    def all_qrels(self, dataset_id: str) -> dict[str, dict[str, int]]:
        """Reconstruct the nested ``{query_id: {doc_id: relevance}}`` map (eval shape).

        Safe to materialize whole: qrels are small (≈15K judgments for Quora).
        """
        cursor = self._qrels.find(
            {"dataset": dataset_id},
            projection={"_id": 0, "query_id": 1, "doc_id": 1, "relevance": 1},
        )
        qrels: dict[str, dict[str, int]] = {}
        for j in cursor:
            qrels.setdefault(j["query_id"], {})[j["doc_id"]] = j["relevance"]
        return qrels

    def close(self) -> None:
        self._client.close()
