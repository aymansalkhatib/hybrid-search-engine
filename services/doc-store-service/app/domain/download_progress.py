"""Live progress for the archive-download phase.

``ir_datasets`` downloads a dataset's whole archive into a temp dir and only starts
yielding documents *after* it is fetched and extracted — so during the (often long)
download there is no per-document signal and the job would otherwise sit at 0%.

This monitor runs in a background thread and reports the **bytes landing in the
ir_datasets temp dir** as a live "XX.X MB" message, so the UI shows real movement
while the archive downloads. It measures growth *since it started*, so leftover temp
files from a previous run don't inflate the number. ir_datasets gives no reliable
total size up-front (and no byte callback), hence MB rather than a percentage here;
the percentage kicks in for the materialize phase, which is doc-counted.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger("doc-store-service")


def ir_datasets_tmp_dir() -> Path:
    """Where ir_datasets writes in-flight downloads (``<tmp>/ir_datasets``)."""
    import os

    base = os.environ.get("IR_DATASETS_TMP") or tempfile.gettempdir()
    return Path(base) / "ir_datasets"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:  # file vanished mid-walk (temp churn) — ignore
            pass
    return total


class DownloadMonitor:
    """Background thread that reports MB flowing into the ir_datasets temp dir.

    Use as a context manager around the blocking first read of ``docs_iter`` (which
    triggers the download): it updates the job ``message`` ~once a second and stops
    automatically on exit.
    """

    def __init__(self, progress, label: str = "downloading archive", interval: float = 1.0) -> None:
        self._progress = progress
        self._label = label
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "DownloadMonitor":
        self._thread = threading.Thread(target=self._run, daemon=True, name="download-monitor")
        self._thread.start()
        return self

    def _run(self) -> None:
        tmp = ir_datasets_tmp_dir()
        baseline = _dir_size(tmp)
        while not self._stop.is_set():
            grown = max(0, _dir_size(tmp) - baseline)
            mb = grown / (1024 * 1024)
            self._progress.update(message=f"{self._label} — {mb:,.1f} MB")
            self._stop.wait(self._interval)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
