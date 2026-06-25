"""Persist a fitted topic model under the data volume (one pickle per dataset).

The :class:`~app.domain.topic_model.TopicModel` holds the fitted CountVectorizer + LDA
alongside the summary, so a single pickle serves every online endpoint after a restart
without re-fitting. Writes are atomic (tmp + replace) so a crash mid-write can't leave a
corrupt artifact.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

from app.domain.topic_model import SCHEMA_VERSION, TopicModel
from shared.ir_common.datasets import safe_id

logger = logging.getLogger("topic-service")


class TopicStore:
    def __init__(self, base_dir: Path) -> None:
        self.dir = Path(base_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, dataset_id: str) -> Path:
        return self.dir / f"{safe_id(dataset_id)}.pkl"

    def exists(self, dataset_id: str) -> bool:
        return self._path(dataset_id).exists()

    def save(self, model: TopicModel) -> Path:
        path = self._path(model.dataset_id)
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
        logger.info("saved topic model: %s", path)
        return path

    def load(self, dataset_id: str) -> TopicModel | None:
        path = self._path(dataset_id)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as fh:
                model = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001 — a stale/incompatible artifact shouldn't crash reads
            logger.warning("could not load topic model %s (%s); treating as absent", path.name, exc)
            return None
        if getattr(model, "schema_version", 0) != SCHEMA_VERSION:
            logger.warning("topic model %s is schema v%s (current v%s) — treating as absent so it rebuilds",
                           path.name, getattr(model, "schema_version", 0), SCHEMA_VERSION)
            return None
        return model

    def delete(self, dataset_id: str) -> bool:
        path = self._path(dataset_id)
        existed = path.exists()
        path.unlink(missing_ok=True)
        if existed:
            logger.info("deleted topic model: %s", dataset_id)
        return existed
