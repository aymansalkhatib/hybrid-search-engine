"""Persist evaluation reports under the data volume — JSON (full) + CSV (flat, for charts).

Each report is keyed by a stable ``report_id`` (``<dataset>__<label>``) so re-running the
same label overwrites in place, and two labels (e.g. ``baseline`` / ``with-refinement``)
sit side by side for the required before/after comparison. Alongside the JSON we write a
tidy CSV (one row per ``run × metric``) that pandas/matplotlib can chart directly for the
Arabic report. Writes are atomic (tmp + ``replace``) so a crash mid-write
can't leave a corrupt report.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
from pathlib import Path

from shared.contracts import EvaluationReport, ReportSummary

logger = logging.getLogger("evaluation-service")


class ReportStore:
    def __init__(self, base_dir: Path) -> None:
        self.dir = Path(base_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def json_path(self, report_id: str) -> Path:
        return self.dir / f"{report_id}.json"

    def csv_path(self, report_id: str) -> Path:
        return self.dir / f"{report_id}.csv"

    def per_query_path(self, report_id: str) -> Path:
        return self.dir / f"{report_id}.perquery.json"

    def exists(self, report_id: str) -> bool:
        return self.json_path(report_id).exists()

    def save(self, report: EvaluationReport) -> Path:
        path = self.json_path(report.report_id)
        self._atomic_write(path, report.model_dump_json(indent=2))
        self._atomic_write(self.csv_path(report.report_id), _to_csv(report))
        logger.info("saved evaluation report: %s", path)
        return path

    def save_per_query(self, report_id: str, sidecar: dict) -> None:
        """Persist the per-query sidecar (best/worst & win-loss drill-down data)."""
        self._atomic_write(self.per_query_path(report_id), json.dumps(sidecar))

    def load_per_query(self, report_id: str) -> dict | None:
        path = self.per_query_path(report_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — a stale sidecar shouldn't crash the drill-down
            logger.warning("could not load per-query sidecar %s (%s)", path.name, exc)
            return None

    def load(self, report_id: str) -> EvaluationReport | None:
        path = self.json_path(report_id)
        if not path.exists():
            return None
        try:
            return EvaluationReport(**json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001 — a stale/incompatible file shouldn't crash reads
            logger.warning("could not load report %s (%s); treating as absent", path.name, exc)
            return None

    def delete(self, report_id: str) -> bool:
        """Remove a report's JSON (and its CSV sidecar). True if the JSON existed."""
        path = self.json_path(report_id)
        removed = path.exists()
        path.unlink(missing_ok=True)
        self.csv_path(report_id).unlink(missing_ok=True)
        self.per_query_path(report_id).unlink(missing_ok=True)
        if removed:
            logger.info("deleted evaluation report: %s", report_id)
        return removed

    def list(self) -> list[ReportSummary]:
        """Cheap metadata for every persisted report (newest first)."""
        out: list[ReportSummary] = []
        for path in self.dir.glob("*.json"):
            # Skip the per-query sidecars (``*.perquery.json``) — they aren't reports, and
            # path.stem only strips the final ``.json`` so they'd be mis-loaded and warn.
            if path.name.endswith(".perquery.json"):
                continue
            report = self.load(path.stem)
            if report is None:
                continue
            out.append(
                ReportSummary(
                    report_id=report.report_id,
                    dataset_id=report.dataset_id,
                    label=report.label,
                    created_at=report.created_at,
                    models=[r.label for r in report.runs],
                    num_queries=report.num_queries,
                )
            )
        return sorted(out, key=lambda s: s.created_at, reverse=True)

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)


def _to_csv(report: EvaluationReport) -> str:
    """Flatten a report to ``run_label, model, mode, metric, value`` rows (failed runs skipped)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["run_label", "model", "mode", "metric", "value"])
    for run in report.runs:
        if run.error:
            continue
        for metric in report.metrics:
            if metric in run.metrics:
                writer.writerow([run.label, run.model, run.mode or "", metric, run.metrics[metric]])
    return buf.getvalue()
