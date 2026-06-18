"""Compute IR effectiveness metrics (MAP, nDCG, Recall, P@k) from qrels + runs.

The **primary** engine is **ranx**, which consumes the exact shapes the
rest of the system already produces — qrels as ``{query_id: {doc_id: relevance}}`` (the
doc-store's ``/qrels/all``) and a run as ``{query_id: {doc_id: score}}`` (built from the
retrieval hits). Using a trusted, standard library keeps the most-graded numbers
unimpeachable.

If ranx can't be imported (e.g. a standalone run without the heavy numba stack), we fall
back to a small, dependency-free implementation that follows the **same trec_eval
conventions** — gain ``2**rel - 1`` with a ``log2(rank + 1)`` discount, average precision
over the retrieved depth, and per-query metrics averaged over the gold (judged) query
set. Only one engine runs per call, so the two never disagree on a given report; the
fallback simply guarantees the service still works.

Metric names use the ``metric@cutoff`` convention (``map``, ``ndcg@10``, ``recall@100``,
``precision@10``); a bare name (``map``, ``ndcg``) uses the full retrieved depth.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger("evaluation-service")

# A relevance map for one query: {doc_id: relevance_grade}. >0 means relevant.
QrelsMap = dict[str, dict[str, int]]
# A run for one query: {doc_id: score}. Higher score = ranked higher.
RunMap = dict[str, dict[str, float]]

try:  # Primary engine — see module docstring.
    from ranx import Qrels as _RanxQrels
    from ranx import Run as _RanxRun
    from ranx import evaluate as _ranx_evaluate

    _HAS_RANX = True
except Exception as exc:  # noqa: BLE001 — any import failure means "use the fallback"
    _HAS_RANX = False
    logger.warning("ranx unavailable (%s); using the built-in metric implementation", exc)


def engine_name() -> str:
    """Which metric engine will be used — recorded in the report for transparency."""
    return "ranx" if _HAS_RANX else "builtin"


def compute(qrels: QrelsMap, run: RunMap, metrics: list[str]) -> dict[str, float]:
    """Mean of each metric over the queries present in ``qrels``.

    ``qrels`` and ``run`` are expected to share the same query-id set (the caller filters
    to the judged, searched queries). A query with an empty hit list scores 0 on every
    metric — which is exactly right.
    """
    if not metrics:
        return {}
    if not qrels or not run:
        return {m: 0.0 for m in metrics}
    if _HAS_RANX:
        try:
            return _compute_ranx(qrels, run, metrics)
        except Exception as exc:  # noqa: BLE001 — never let a metric-engine hiccup fail the job
            logger.warning("ranx evaluate failed (%s); falling back to the built-in", exc)
    return _compute_builtin(qrels, run, metrics)


# --------------------------------------------------------------------------- #
#  ranx engine
# --------------------------------------------------------------------------- #

# A doc id that can't appear in any qrels — used to keep a query with no retrieved
# results in the Run (ranx dislikes an empty per-query dict) while scoring it as 0.
_EMPTY_PLACEHOLDER = {"\x00__no_results__": 0.0}


def _compute_ranx(qrels: QrelsMap, run: RunMap, metrics: list[str]) -> dict[str, float]:
    qrels_obj = _RanxQrels.from_dict(
        {q: {d: int(r) for d, r in rels.items()} for q, rels in qrels.items()}
    )
    run_obj = _RanxRun.from_dict(
        {q: ({d: float(s) for d, s in hits.items()} or dict(_EMPTY_PLACEHOLDER)) for q, hits in run.items()}
    )
    res = _ranx_evaluate(qrels_obj, run_obj, metrics)
    # evaluate returns a {metric: value} dict for several metrics, or a bare float for one.
    if isinstance(res, dict):
        return {m: float(res[m]) for m in metrics}
    return {metrics[0]: float(res)}


# --------------------------------------------------------------------------- #
#  built-in engine (trec_eval conventions, no dependencies)
# --------------------------------------------------------------------------- #

def _parse_metric(name: str) -> tuple[str, Optional[int]]:
    """Split ``'ndcg@10'`` -> ``('ndcg', 10)``; a bare name -> ``(name, None)``."""
    base, _, cut = name.partition("@")
    base = base.strip().lower()
    if base == "p":
        base = "precision"
    cutoff = int(cut) if cut.strip() else None
    return base, cutoff


def _compute_builtin(qrels: QrelsMap, run: RunMap, metrics: list[str]) -> dict[str, float]:
    per_query = per_query_builtin(qrels, run, metrics)
    n = len(per_query)
    if not n:
        return {m: 0.0 for m in metrics}
    return {m: sum(scores[m] for scores in per_query.values()) / n for m in metrics}


def per_query_builtin(qrels: QrelsMap, run: RunMap, metrics: list[str]) -> dict[str, dict[str, float]]:
    """Per-query metric scores ``{query_id: {metric: value}}`` (built-in, exact alignment).

    Used for the drill-down (best/worst, win/loss). Always the built-in implementation —
    it lets us map each score to its query id deterministically (ranx returns aggregate
    arrays). The values follow the same conventions as :func:`compute`, so a per-query
    mean matches the headline metric to within rounding.
    """
    parsed = [(m, *_parse_metric(m)) for m in metrics]  # (orig, base, cutoff)
    out: dict[str, dict[str, float]] = {}
    for qid, rels in qrels.items():
        # Rank the run's docs by score desc (ties broken by doc id for determinism).
        ranked = sorted(run.get(qid, {}).items(), key=lambda kv: (-kv[1], kv[0]))
        ranked_ids = [doc_id for doc_id, _ in ranked]
        out[qid] = {orig: _metric_for_query(base, cutoff, ranked_ids, rels) for orig, base, cutoff in parsed}
    return out


def compute_per_query(qrels: QrelsMap, run: RunMap, metrics: list[str]) -> dict[str, dict[str, float]]:
    """Public per-query scorer ``{query_id: {metric: value}}`` — see :func:`per_query_builtin`."""
    if not metrics or not qrels:
        return {}
    return per_query_builtin(qrels, run, metrics)


def _metric_for_query(base: str, cutoff: Optional[int], ranked_ids: list[str], rels: dict[str, int]) -> float:
    relevant = {doc_id for doc_id, grade in rels.items() if grade > 0}
    num_rel = len(relevant)

    if base == "precision":
        k = cutoff or 10
        top = ranked_ids[:k]
        return sum(1 for d in top if d in relevant) / k

    if base == "recall":
        if num_rel == 0:
            return 0.0
        k = cutoff or len(ranked_ids)
        top = ranked_ids[:k]
        return sum(1 for d in top if d in relevant) / num_rel

    if base == "map":
        if num_rel == 0:
            return 0.0
        depth = cutoff or len(ranked_ids)
        hits = 0
        precision_sum = 0.0
        for rank, doc_id in enumerate(ranked_ids[:depth], start=1):
            if doc_id in relevant:
                hits += 1
                precision_sum += hits / rank
        return precision_sum / num_rel

    if base == "ndcg":
        k = cutoff or len(ranked_ids)
        dcg = 0.0
        for rank, doc_id in enumerate(ranked_ids[:k], start=1):
            grade = rels.get(doc_id, 0)
            if grade > 0:
                dcg += (2 ** grade - 1) / math.log2(rank + 1)
        ideal = sorted((g for g in rels.values() if g > 0), reverse=True)[:k]
        idcg = sum((2 ** g - 1) / math.log2(rank + 1) for rank, g in enumerate(ideal, start=1))
        return dcg / idcg if idcg > 0 else 0.0

    raise ValueError(f"unknown metric '{base}' (supported: map, ndcg, recall, precision)")
