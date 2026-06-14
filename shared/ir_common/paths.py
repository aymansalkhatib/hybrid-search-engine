"""Filesystem path helpers for services that persist artifacts.

Mirrors the `find_repo_env` trick (see `config.py`): when running from source we
resolve paths relative to the repo root; inside Docker — where no repo markers are
shipped into the image — we fall back to the container path from env. So the same
code is correct in both places with **no Docker-detection flag**.
"""

from __future__ import annotations

from pathlib import Path

# Markers that only exist in a source checkout, never inside a service image.
_REPO_MARKERS = (".git", "docker-compose.yml", ".env.example")


def repo_root() -> Path | None:
    """Return the repo root when running from source, else ``None`` (e.g. in Docker)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if any((parent / marker).exists() for marker in _REPO_MARKERS):
            return parent
    return None


def resolve_artifacts_dir(env_value: str | None) -> Path:
    """Directory for built artifacts (indexes, tfidf matrices, embeddings…).

    From source → ``<repo>/data/artifacts`` (gitignored, ignoring the container
    path in ``.env``). In Docker → the ``ARTIFACTS_DIR`` env path (the mounted
    ``./data`` volume). The directory is created if missing.
    """
    root = repo_root()
    if root is not None:
        path = root / "data" / "artifacts"
    else:
        path = Path(env_value or "/app/data/artifacts")
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_dataset_home(env_value: str | None) -> Path:
    """Directory where datasets are downloaded and kept **permanently** (the
    ir_datasets cache, plus our download markers).

    From source → ``<repo>/data/dataset``; in Docker → the ``IR_DATASETS_HOME`` env
    path (the mounted ``./data/dataset``). Created if missing. Services export this
    as ``IR_DATASETS_HOME`` so ir_datasets and our markers share one location.
    """
    root = repo_root()
    if root is not None:
        path = root / "data" / "dataset"
    else:
        path = Path(env_value or "/app/data/dataset")
    path.mkdir(parents=True, exist_ok=True)
    return path
