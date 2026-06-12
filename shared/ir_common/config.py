"""Shared configuration helpers for all services."""

from __future__ import annotations

from pathlib import Path


def find_repo_env() -> Path | None:
    """Locate the repo-root ``.env`` by walking up from this file, so any
    service runs standalone from any working directory.

    In Docker no ``.env`` is shipped into the image — config comes from compose
    as real env vars — so this returns ``None`` and pydantic-settings falls
    back to reading the environment.
    """
    here = Path(__file__).resolve()
    return next((p / ".env" for p in here.parents if (p / ".env").exists()), None)
