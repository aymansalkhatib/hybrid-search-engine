"""Persist/load inverted indexes as pickle artifacts under the mounted data volume.

Idempotent builds: an index is cached on disk keyed by
``{dataset}-{version}`` and re-served without rebuilding. Writes are atomic
(tmp + ``replace``) so a crash mid-write can't leave a corrupt artifact.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

from app.domain.inverted_index import InvertedIndex
from shared.ir_common.datasets import safe_id as _safe

logger = logging.getLogger("indexing-service")


class ArtifactStore:
    def __init__(self, base_dir: Path) -> None:
        self.dir = Path(base_dir) / "index"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, dataset_id: str, version: str) -> Path:
        return self.dir / f"{_safe(dataset_id)}-v{version}.pkl"

    def exists(self, dataset_id: str, version: str) -> bool:
        return self.path_for(dataset_id, version).exists()

    def save(self, index: InvertedIndex) -> Path:
        path = self.path_for(index.dataset_id, index.version)
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
            # Flush to disk before the rename so an unclean host/daemon crash can't
            # leave a "saved" index that never actually hit the platters.
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
        logger.info("saved index artifact: %s", path)
        return path

    def delete(self, dataset_id: str, version: str) -> bool:
        """Remove the persisted index artifact for a dataset. True if one was removed.

        Cleans up the temp file too, in case a build crashed mid-write. Deleting an
        index is independent of the corpus/docs — callers wire it into a dataset
        delete so removing a dataset also clears its built index from the cache.
        """
        path = self.path_for(dataset_id, version)
        removed = False
        if path.exists():
            path.unlink()
            removed = True
            logger.info("deleted index artifact: %s", path)
        tmp = path.with_suffix(".pkl.tmp")
        if tmp.exists():
            tmp.unlink()
        return removed

    def load(self, dataset_id: str, version: str) -> InvertedIndex | None:
        path = self.path_for(dataset_id, version)
        if not path.exists():
            return None
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def list_datasets(self, version: str) -> list[str]:
        suffix = f"-v{version}.pkl"
        return sorted(
            p.name[: -len(suffix)].replace("__", "/")
            for p in self.dir.glob(f"*{suffix}")
        )
