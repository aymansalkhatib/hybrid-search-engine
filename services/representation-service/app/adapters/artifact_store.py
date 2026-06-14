"""Persist/load representation models as pickle artifacts under the data volume.

Idempotent builds: a model is cached on disk keyed by
``{dataset}~{model}~v{version}`` and re-served without rebuilding. A pickle holds the
whole model (fitted vectorizer + sparse matrix + doc_ids) — the same approach the
indexing-service uses for its index. Writes are atomic (tmp + ``replace``) so a crash
mid-write can't leave a corrupt artifact.

One dataset can hold several models side by side (tfidf, bm25, …); ``~`` separates the
parts because a dataset id may itself contain ``-`` (e.g. ``msmarco-passage``).
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

from app.domain.base import BaseRepresentation
from shared.contracts import BuiltRepresentation
from shared.ir_common.datasets import safe_id as _safe

logger = logging.getLogger("representation-service")

_SEP = "~"


class RepresentationStore:
    def __init__(self, base_dir: Path) -> None:
        self.dir = Path(base_dir) / "representation"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, dataset_id: str, model: str, version: str) -> Path:
        return self.dir / f"{_safe(dataset_id)}{_SEP}{model}{_SEP}v{version}.pkl"

    def exists(self, dataset_id: str, model: str, version: str) -> bool:
        return self.path_for(dataset_id, model, version).exists()

    def save(self, rep: BaseRepresentation) -> Path:
        path = self.path_for(rep.dataset_id, rep.model, rep.version)
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(rep, fh, protocol=pickle.HIGHEST_PROTOCOL)
            # Flush to disk before the rename so an unclean crash can't leave a
            # "saved" model that never actually hit the platters.
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
        logger.info("saved representation artifact: %s", path)
        return path

    def load(self, dataset_id: str, model: str, version: str) -> BaseRepresentation | None:
        path = self.path_for(dataset_id, model, version)
        if not path.exists():
            return None
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def delete(self, dataset_id: str, model: str, version: str) -> bool:
        """Remove the persisted artifact for a (dataset, model). True if one existed.

        Cleans up the temp file too, in case a build crashed mid-write.
        """
        path = self.path_for(dataset_id, model, version)
        removed = False
        if path.exists():
            path.unlink()
            removed = True
            logger.info("deleted representation artifact: %s", path)
        tmp = path.with_suffix(".pkl.tmp")
        if tmp.exists():
            tmp.unlink()
        return removed

    def list_built(self, version: str) -> list[BuiltRepresentation]:
        """All persisted (dataset_id, model) pairs for the current artifact version."""
        suffix = f"{_SEP}v{version}.pkl"
        out: list[BuiltRepresentation] = []
        for p in self.dir.glob(f"*{suffix}"):
            stem = p.name[: -len(suffix)]
            safe, _, model = stem.rpartition(_SEP)
            if not model:
                continue
            out.append(
                BuiltRepresentation(dataset_id=safe.replace("__", "/"), model=model)
            )
        return sorted(out, key=lambda b: (b.dataset_id, b.model))
