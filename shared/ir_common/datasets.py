"""Dataset catalog resolution + per-dataset local-download bookkeeping.

Two concerns live here, both about keeping datasets **local, separated and
dataset-agnostic**:

1. **Catalog** — the supported datasets are a single comma-separated env value
   (``DATASETS``), e.g. ``beir/quora/test,wikir/en1k/test``. ``parse_datasets``
   turns it into an ordered, de-duplicated list (the first entry is the primary /
   required dataset by convention; the rest are bonus). Callers refer to a dataset
   by its **concrete id**, and ``resolve_dataset`` treats the configured list as an
   allow-list — so the system stays dataset-agnostic (ids live only in env) and
   safe (no arbitrary, unvetted downloads). Adding a dataset = append to ``DATASETS``.
2. **Per-dataset manifest** — a dataset is fetched from the internet only through
   the dedicated *download* endpoint. ir_datasets already stores each dataset in
   its **own folder** under the dataset home (e.g. ``<home>/beir/quora/test``), so
   datasets are independently deletable. After a full download we drop a
   ``.manifest.json`` **inside that same folder** (not a shared marker dir), so
   the readiness flag travels with the data: delete the folder and the marker goes
   with it. Every other operation (ingest, index) checks ``is_downloaded`` and
   refuses to run (rather than silently downloading) when it's missing.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

MANIFEST_NAME = ".manifest.json"


def parse_datasets(raw: Optional[str]) -> list[str]:
    """Parse the comma-separated ``DATASETS`` value into a clean, ordered list.

    Whitespace is trimmed, blanks dropped, and duplicates removed while preserving
    first-seen order (so the primary dataset stays first).
    """
    if not raw:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for part in raw.split(","):
        ds = part.strip()
        if ds and ds not in seen:
            seen.add(ds)
            out.append(ds)
    return out


def resolve_dataset(value: str, datasets: list[str]) -> Optional[str]:
    """Return ``value`` if it is in the configured catalog, else ``None``.

    Membership in the configured list is the allow-list — callers can only act on
    datasets an operator put in ``DATASETS``.
    """
    v = value.strip()
    return v if v in datasets else None


def safe_id(dataset_id: str) -> str:
    """Filesystem-safe form of a dataset id (``beir/quora/test`` -> ``beir__quora__test``).

    Used for *flat* artifact filenames (e.g. the index pickle). The dataset corpus
    itself keeps ir_datasets' natural nested layout — see :func:`dataset_dir`.
    """
    return dataset_id.replace("/", "__").replace(":", "__")


def dataset_dir(home: Path, dataset_id: str) -> Path:
    """The dataset's own folder under the home — where ir_datasets stores it.

    ``<home>/beir/quora/test`` for ``beir/quora/test``. This colocates our manifest
    with the corpus files so the two are deleted together.
    """
    return Path(home) / dataset_id


def manifest_path(home: Path, dataset_id: str) -> Path:
    return dataset_dir(home, dataset_id) / MANIFEST_NAME


def read_manifest(home: Path, dataset_id: str) -> Optional[dict]:
    """Return the download manifest for ``dataset_id``, or None if absent/unreadable."""
    path = manifest_path(home, dataset_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_manifest(home: Path, dataset_id: str, *, doc_count: int, complete: bool = True) -> None:
    """Record that ``dataset_id`` is locally available (written after a download)."""
    path = manifest_path(home, dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "doc_count": doc_count,
                "complete": complete,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    )


def is_downloaded(home: Path, dataset_id: str) -> bool:
    """True iff ``dataset_id`` was **fully** fetched locally (complete manifest present)."""
    manifest = read_manifest(home, dataset_id)
    return bool(manifest and manifest.get("complete"))


def manifest_doc_count(home: Path, dataset_id: str) -> Optional[int]:
    """The doc_count recorded at download time, or None if not downloaded."""
    manifest = read_manifest(home, dataset_id)
    return manifest.get("doc_count") if manifest else None


def delete_dataset_files(home: Path, dataset_id: str) -> bool:
    """Remove a dataset's local folder (corpus + manifest). Independent of others.

    Returns True if a folder was removed. Empty parent dirs (e.g. ``<home>/beir``)
    are pruned afterwards so no stale shells remain.
    """
    folder = dataset_dir(home, dataset_id)
    if not folder.exists():
        return False
    shutil.rmtree(folder)
    # Prune now-empty parents up to (but not including) the home root.
    home = Path(home).resolve()
    parent = folder.parent.resolve()
    while parent != home and home in parent.parents:
        try:
            next(parent.iterdir())
            break  # parent still has contents — stop pruning
        except StopIteration:
            parent.rmdir()
            parent = parent.parent
        except OSError:
            break
    return True
