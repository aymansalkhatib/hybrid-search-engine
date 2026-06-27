"""Orchestrate an evaluation run — the core of the service.

Given a dataset and a list of model configs (:class:`EvalRunSpec`), for each config:

1. load the dataset's **qrels** and **test queries** from the doc-store;
2. restrict to the **judged** queries (those with qrels — the gold set), optionally
   sampling the first ``max_queries`` for a quick run;
3. run every judged query through the **retrieval-service** (concurrently) to build the
   run ``{query_id: {doc_id: score}}``;
4. score that run against the qrels with the metric engine (:mod:`metrics`) — both the
   **mean** (the headline table/charts) and the **per-query** breakdown (the drill-down).

The result is an :class:`EvaluationReport` comparing all configs on the same query set,
plus an optional **per-query sidecar** (``{queries, runs}``) for best/worst & win-loss
analysis. A config whose model isn't built (retrieval 404) fails **just that run** (with
an ``error``); the others still produce numbers. Each run also records ``avg_query_ms``
— the retrieval-service's own measured latency, for the quality-vs-speed view.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from time import perf_counter
from typing import Callable, Optional

import httpx

from app.adapters.doc_store_client import DocStoreClient
from app.adapters.refinement_client import RefinementClient
from app.adapters.retrieval_client import RetrievalClient
from app.domain import metrics as metric_engine
from shared.contracts import EvalRunSpec, EvaluationReport, RefineOptions, RunEvaluation

logger = logging.getLogger("evaluation-service")

# A per-query score map for one run: {query_id: {metric: value}}.
PerQueryScores = dict[str, dict[str, float]]


class NoQrelsError(Exception):
    """Raised when a dataset has no qrels stored — evaluation is impossible without them."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _downstream_message(exc: httpx.HTTPStatusError) -> str:
    """A clean reason from a downstream error (forwards retrieval's 'build it first')."""
    try:
        body = exc.response.json()
        return body.get("error", {}).get("message") or body.get("detail") or exc.response.text
    except ValueError:
        return exc.response.text


def _run_params(spec: EvalRunSpec) -> dict:
    """What was evaluated, recorded in the report for reproducibility."""
    params: dict = {"k1": spec.k1, "b": spec.b}
    if spec.model == "hybrid" and spec.hybrid is not None:
        params["hybrid"] = spec.hybrid.model_dump()
    return params


def run_evaluation(
    *,
    report_id: str,
    dataset_id: str,
    label: str,
    runs: list[EvalRunSpec],
    metrics: list[str],
    top_k: int,
    max_queries: int | None,
    per_query: bool,
    retrieval: RetrievalClient,
    doc_store: DocStoreClient,
    concurrency: int,
    advance: Callable[[int], None],
    set_total: Callable[[int], None],
    set_message: Callable[[str], None],
    refine_options: Optional[RefineOptions] = None,
    refinement: Optional[RefinementClient] = None,
    cluster_prune: bool = False,
    topic_prune: bool = False,
    prune_top_n: int = 3,
) -> tuple[EvaluationReport, Optional[dict]]:
    """Evaluate every run config on the dataset's judged queries.

    Returns ``(report, per_query_sidecar)``. The sidecar is ``None`` when ``per_query`` is
    off, else ``{"queries": {qid: text}, "runs": {label: {qid: {metric: value}}}}``.
    """
    set_message("loading qrels & queries")
    qrels = doc_store.all_qrels(dataset_id)
    if not qrels:
        raise NoQrelsError(
            f"dataset '{dataset_id}' has no qrels stored; ingest it first via the "
            "doc-store POST /dataset/prepare (qrels are required to evaluate)"
        )
    queries = doc_store.all_queries(dataset_id)

    # The gold set = judged queries we also have text for (so we can search them).
    judged = sorted(qid for qid in qrels if qid in queries)
    missing_text = len(qrels) - len(judged)
    if missing_text:
        logger.warning("%d judged queries have no stored text and are skipped", missing_text)
    if max_queries is not None:
        judged = judged[:max_queries]
    qrels_eval = {qid: qrels[qid] for qid in judged}

    # Optional query refinement (the with/without comparison): refine each judged query
    # ONCE here, then every run searches the refined text — so a 'with-refinement' report
    # is the same models on a refined query set, directly comparable to the baseline. The
    # sidecar keeps the ORIGINAL query text for display so before/after drill-downs align.
    search_queries = queries
    refine_params: Optional[dict] = None
    if refine_options is not None and refinement is not None:
        set_message(f"refining {len(judged)} queries")
        search_queries = _refine_queries(judged, queries, refine_options, refinement, concurrency)
        refine_params = refine_options.model_dump()

    set_total(max(len(judged) * len(runs), 0))
    set_message(f"evaluating {len(runs)} run(s) over {len(judged)} queries")

    run_results: list[RunEvaluation] = []
    per_query_runs: dict[str, PerQueryScores] = {}
    for spec in runs:
        result, pq = _evaluate_one(
            spec=spec, judged=judged, queries=search_queries, qrels_eval=qrels_eval,
            metrics=metrics, top_k=top_k, dataset_id=dataset_id, retrieval=retrieval,
            concurrency=concurrency, advance=advance, per_query=per_query,
            refine=refine_params, cluster_prune=cluster_prune, topic_prune=topic_prune,
            prune_top_n=prune_top_n,
        )
        run_results.append(result)
        if pq is not None:
            per_query_runs[spec.label] = pq

    report = EvaluationReport(
        report_id=report_id,
        dataset_id=dataset_id,
        label=label,
        created_at=_now(),
        metrics=metrics,
        top_k=top_k,
        num_queries=len(judged),
        runs=run_results,
        engine=metric_engine.engine_name(),
    )
    sidecar = None
    if per_query and per_query_runs:
        sidecar = {
            "report_id": report_id,
            "dataset_id": dataset_id,
            "metrics": metrics,
            "queries": {qid: queries[qid] for qid in judged},
            "runs": per_query_runs,
        }
    return report, sidecar


def _refine_queries(
    judged: list[str],
    queries: dict[str, str],
    options: RefineOptions,
    refinement: RefinementClient,
    concurrency: int,
) -> dict[str, str]:
    """Refine every judged query once (concurrently). A blip on one query falls back to
    its raw text, so refinement never fails the whole evaluation."""
    def work(qid: str) -> tuple[str, str]:
        try:
            return qid, refinement.refine(text=queries[qid], options=options)
        except httpx.HTTPError:
            return qid, queries[qid]

    out = dict(queries)
    if judged:
        workers = max(1, min(concurrency, len(judged)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for qid, text in pool.map(work, judged):
                out[qid] = text
    return out


def _evaluate_one(
    *,
    spec: EvalRunSpec,
    judged: list[str],
    queries: dict[str, str],
    qrels_eval: dict[str, dict[str, int]],
    metrics: list[str],
    top_k: int,
    dataset_id: str,
    retrieval: RetrievalClient,
    concurrency: int,
    advance: Callable[[int], None],
    per_query: bool,
    refine: Optional[dict] = None,
    cluster_prune: bool = False,
    topic_prune: bool = False,
    prune_top_n: int = 3,
) -> tuple[RunEvaluation, Optional[PerQueryScores]]:
    """Run one model config over all judged queries and score it. Never raises for a model
    that simply isn't built — that's recorded as the run's ``error`` instead."""
    t0 = perf_counter()
    params = _run_params(spec)
    if refine is not None:
        params = {**params, "refine": refine}
    if cluster_prune:
        params = {**params, "cluster_prune": True, "prune_top_n": prune_top_n}
    if topic_prune:
        params = {**params, "topic_prune": True, "prune_top_n": prune_top_n}

    if not judged:
        return RunEvaluation(label=spec.label, model=spec.model, params=params), None

    # Validate the config with the first query synchronously: a not-built model (404) or a
    # bad request fails this run immediately, without hammering retrieval N times.
    first = judged[0]
    try:
        hits, mode, took = retrieval.search(dataset=dataset_id, spec=spec, query=queries[first], top_k=top_k, cluster_prune=cluster_prune, topic_prune=topic_prune, prune_top_n=prune_top_n)
    except httpx.HTTPStatusError as exc:
        advance(len(judged))  # keep the overall progress bar honest
        return RunEvaluation(
            label=spec.label, model=spec.model, params=params, num_queries=len(judged),
            error=_downstream_message(exc), took_ms=round((perf_counter() - t0) * 1000, 2),
        ), None
    except httpx.HTTPError as exc:
        advance(len(judged))
        return RunEvaluation(
            label=spec.label, model=spec.model, params=params, num_queries=len(judged),
            error=f"retrieval unavailable: {exc}", took_ms=round((perf_counter() - t0) * 1000, 2),
        ), None

    run_dict: dict[str, dict[str, float]] = {first: {d: s for d, s in hits}}
    latency_sum = took
    advance(1)

    def work(qid: str) -> tuple[str, dict[str, float], float]:
        try:
            h, _, ms = retrieval.search(dataset=dataset_id, spec=spec, query=queries[qid], top_k=top_k, cluster_prune=cluster_prune, topic_prune=topic_prune, prune_top_n=prune_top_n)
            return qid, {d: s for d, s in h}, ms
        except httpx.HTTPError:
            # A transient blip on one query degrades to an empty result (scored 0) rather
            # than failing the whole run — the validated config is known to work.
            return qid, {}, 0.0

    rest = judged[1:]
    if rest:
        workers = max(1, min(concurrency, len(rest)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for qid, scores, ms in pool.map(work, rest):
                run_dict[qid] = scores
                latency_sum += ms
                advance(1)

    values = metric_engine.compute(qrels_eval, run_dict, metrics)
    pq = metric_engine.compute_per_query(qrels_eval, run_dict, metrics) if per_query else None
    result = RunEvaluation(
        label=spec.label,
        model=spec.model,
        mode=mode,
        params=params,
        num_queries=len(judged),
        metrics=values,
        took_ms=round((perf_counter() - t0) * 1000, 2),
        avg_query_ms=round(latency_sum / len(judged), 2) if judged else None,
    )
    return result, pq
