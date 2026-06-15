"""HTTP layer (thin) for the representation-service — maps requests to the domain.

* ``POST /build`` — offline build of one model (TF-IDF / BM25 / Embedding) as a
  background job (idempotent; live progress via ``GET /jobs/{id}`` or ``GET /status``).
* ``POST /search`` — the online query path: rank documents with a single model **or**
  a **hybrid** (serial re-rank / parallel fusion), and fetch the **original** top-k
  docs by id from the doc-store for display.
* ``POST /encode`` — inspect how a lexical model weighs a query (TF-IDF / BM25).
"""

from __future__ import annotations

import logging
import time
from typing import Iterable, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, status

from app.config import settings
from app.domain.base import (
    REPRESENTATION_VERSION,
    BaseRepresentation,
    Normalizer,
    available_models,
    get_model_class,
)
from app.domain.builder import build_representation
from app.domain.hybrid import parallel_search, serial_search
from shared.contracts import (
    BuildRepresentationRequest,
    BuiltRepresentations,
    EncodeRequest,
    EncodeResponse,
    EncodedVector,
    JobStatus,
    PreprocessOptions,
    RepresentationDeleteResult,
    RepresentationStats,
    RepresentationStatusResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    WeightedTerm,
    ref_from_job,
    status_from_job,
)
from shared.ir_common.jobs import JobAlreadyActive, Progress

logger = logging.getLogger("representation-service")
router = APIRouter(tags=["representation"])

JOB_TYPE = "represent"


# ---- helpers -------------------------------------------------------------

def _resolve_or_400(dataset: str) -> str:
    dataset_id = settings.resolve_dataset(dataset)
    if dataset_id is None:
        catalog = settings.datasets
        detail = (
            "no datasets configured — set DATASETS in .env"
            if not catalog
            else f"unknown dataset '{dataset}' — not in the configured catalog ({', '.join(catalog)})"
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
    return dataset_id


def _validate_model_or_400(model: str) -> str:
    if get_model_class(model) is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown model '{model}' — available: {', '.join(available_models())}",
        )
    return model


def _job_key(dataset_id: str, model: str) -> str:
    """One in-flight build per dataset+model (different models may build concurrently)."""
    return f"{dataset_id}::{model}"


def _get_model(request: Request, dataset_id: str, model: str) -> Optional[BaseRepresentation]:
    """Return the model from the in-memory cache, lazily loading it from disk."""
    cache = request.app.state.models
    key = (dataset_id, model)
    if key not in cache:
        loaded = request.app.state.store.load(dataset_id, model, REPRESENTATION_VERSION)
        if loaded is not None:
            cache[key] = loaded
    return cache.get(key)


def _load_required(request: Request, dataset_id: str, model: str) -> BaseRepresentation:
    """Load a built model or raise 404 (model name already validated by the contract)."""
    rep = _get_model(request, dataset_id, model)
    if rep is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no '{model}' representation for dataset '{dataset_id}'; build it first via POST /build",
        )
    return rep


def _require_model(request: Request, dataset: str, model: str) -> BaseRepresentation:
    dataset_id = _resolve_or_400(dataset)
    _validate_model_or_400(model)
    return _load_required(request, dataset_id, model)


def _make_normalizer(state) -> Normalizer:
    """A query normalizer bound to the preprocessing-service: each lexical model uses
    it with **its own** build-time options, so the query lands in the model's space."""
    def normalize(text: str, options: dict) -> str:
        opts = PreprocessOptions(**options) if options else PreprocessOptions()
        return state.preprocessing.normalize_batch([text], opts)[0]
    return normalize


def _require_preprocessing(state, reps: Iterable[BaseRepresentation]) -> None:
    """503 if any involved model needs preprocessing and the service is down."""
    if any(r.requires_preprocessing for r in reps) and not state.preprocessing.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"preprocessing-service unavailable at {settings.preprocessing_url}; "
                "lexical models normalize the query the same way as the corpus"
            ),
        )


def _stats(rep: BaseRepresentation, *, cached: bool) -> RepresentationStats:
    extra = rep.stats_extra()
    return RepresentationStats(
        dataset_id=rep.dataset_id,
        model=rep.model,
        num_docs=rep.num_docs,
        vocab_size=extra.get("vocab_size"),
        nnz=extra.get("nnz"),
        density=extra.get("density"),
        dim=extra.get("dim"),
        avgdl=extra.get("avgdl"),
        version=rep.version,
        options=PreprocessOptions(**rep.options) if rep.options else PreprocessOptions(),
        params=rep.params or {},
        built_at=rep.built_at,
        cached=cached,
    )


def _submit_or_409(request: Request, key: str, fn):
    try:
        return request.app.state.jobs.submit(JOB_TYPE, key, fn)
    except JobAlreadyActive as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


def _build_params(req: BuildRepresentationRequest) -> dict:
    """Pick the params block matching the requested model."""
    return {
        "tfidf": req.params,
        "bm25": req.bm25,
        "embedding": req.embedding,
        "bert": req.bert,
    }[req.model].model_dump()


# ---- build ---------------------------------------------------------------

@router.post("/build", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
def build(req: BuildRepresentationRequest, request: Request) -> JobStatus:
    """Start a background build of a representation (tfidf / bm25 / embedding).

    The corpus is read from the **doc-store**, so the dataset must be **ingested**
    first (download → ingest → represent). **409** if not ingested; **503** if a
    needed dependency (doc-store, or preprocessing for lexical models) is down.
    Idempotent: a cached model with identical options/params ends as ``skipped``
    unless ``force``. Concurrent build of the same dataset+model → **409**.
    """
    dataset_id = _resolve_or_400(req.dataset)
    model = _validate_model_or_400(req.model)
    state = request.app.state
    doc_store = state.doc_store

    if not doc_store.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"doc-store unavailable at {settings.doc_store_url}; a build reads the corpus from it",
        )
    try:
        ingested = doc_store.ingested_count(dataset_id)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"doc-store error while checking ingest status: {exc}",
        ) from exc
    if ingested <= 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"dataset '{dataset_id}' has no documents in the store; ingest it first "
                "via the doc-store POST /dataset/prepare (download → ingest → represent)"
            ),
        )
    needs_pp = get_model_class(model).requires_preprocessing
    if needs_pp and not state.preprocessing.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"preprocessing-service unavailable at {settings.preprocessing_url}; "
                f"the '{model}' build needs it to tokenize the corpus"
            ),
        )

    start, stop = req.start, req.stop
    options_dump = req.options.model_dump()
    params_dump = _build_params(req)
    force = req.force

    def fn(progress: Progress) -> dict:
        if not force:
            existing = _get_model(request, dataset_id, model)
            if existing is not None and existing.options == options_dump and existing.params == params_dump:
                return {"skipped": True, "stats": _stats(existing, cached=True).model_dump()}
        total = max(stop - (start or 0), 0) if stop is not None else ingested
        progress.update(total=total, message=f"building {model}")
        rep = build_representation(
            model=model,
            dataset_id=dataset_id,
            docs=doc_store.iter_docs(dataset_id, start=start, stop=stop),
            normalize_batch=lambda texts: state.preprocessing.normalize_batch(texts, req.options),
            options=options_dump,
            params=params_dump,
            batch_size=settings.preprocess_batch_size,
            on_progress=lambda n: progress.update(processed=n),
        )
        state.store.save(rep)
        state.models[(dataset_id, model)] = rep
        return {"skipped": False, "stats": _stats(rep, cached=False).model_dump()}

    return status_from_job(_submit_or_409(request, _job_key(dataset_id, model), fn))


# ---- status / catalog ----------------------------------------------------

@router.get("/status", response_model=RepresentationStatusResponse)
def representation_status(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    model: str = Query("tfidf", description="Representation model"),
) -> RepresentationStatusResponse:
    dataset_id = _resolve_or_400(dataset)
    _validate_model_or_400(model)
    rep = _get_model(request, dataset_id, model)
    active = request.app.state.jobs.active_for(JOB_TYPE, _job_key(dataset_id, model))
    return RepresentationStatusResponse(
        dataset_id=dataset_id,
        model=model,
        built=rep is not None,
        stats=_stats(rep, cached=True) if rep is not None else None,
        active_job=ref_from_job(active) if active else None,
    )


@router.get("/built", response_model=BuiltRepresentations)
def built(request: Request) -> BuiltRepresentations:
    return BuiltRepresentations(items=request.app.state.store.list_built(REPRESENTATION_VERSION))


@router.get("/stats", response_model=RepresentationStats)
def stats(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    model: str = Query("tfidf", description="Representation model"),
) -> RepresentationStats:
    return _stats(_require_model(request, dataset, model), cached=True)


# ---- search (the online query path) --------------------------------------

@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, request: Request) -> SearchResponse:
    """Rank documents for a query with a single model or a hybrid, and return the
    top-k with their **original** text (fetched by id from the doc-store).

    Hybrid (``model='hybrid'``): ``serial`` re-ranks one model's candidates with
    another; ``parallel`` fuses several models' lists (RRF / weighted). BM25's
    ``k1``/``b`` are taken from the request (per-query tuning). **404** if a needed
    model isn't built; **503** if preprocessing is down for a lexical model.
    """
    dataset_id = _resolve_or_400(req.dataset)
    state = request.app.state
    normalize = _make_normalizer(state)
    knobs = {"k1": req.k1, "b": req.b}  # consumed by BM25; other models ignore them
    t0 = time.perf_counter()

    if req.model == "hybrid":
        spec = req.hybrid
        names = [spec.first, spec.rerank] if spec.mode == "serial" else list(spec.components)
        models = {n: _load_required(request, dataset_id, n) for n in dict.fromkeys(names)}
        _require_preprocessing(state, models.values())
        if spec.mode == "serial":
            results = serial_search(
                first=models[spec.first], rerank=models[spec.rerank],
                raw_query=req.query, normalize=normalize,
                candidates=spec.candidates, top_k=req.top_k, **knobs,
            )
        else:
            results = parallel_search(
                models=models, components=spec.components,
                raw_query=req.query, normalize=normalize,
                fusion=spec.fusion, weights=spec.weights, rrf_k=spec.rrf_k,
                top_k=req.top_k, **knobs,
            )
        mode = spec.mode
    else:
        _validate_model_or_400(req.model)
        rep = _load_required(request, dataset_id, req.model)
        _require_preprocessing(state, [rep])
        results = rep.search(raw_query=req.query, top_k=req.top_k, normalize=normalize, **knobs)
        mode = None

    # Show the ORIGINAL document text — read by id from the doc-store (graded path).
    texts: dict[str, str] = {}
    if req.with_text and results:
        try:
            texts = state.doc_store.fetch_originals(dataset_id, [d for d, _ in results])
        except httpx.HTTPError:
            texts = {}  # degrade gracefully: still return ranking if the doc-store blips

    hits = [
        SearchHit(rank=i + 1, doc_id=d, score=round(float(s), 6), text=texts.get(d))
        for i, (d, s) in enumerate(results)
    ]
    return SearchResponse(
        dataset_id=dataset_id, model=req.model, mode=mode, query=req.query,
        took_ms=round((time.perf_counter() - t0) * 1000, 2), total=len(hits), hits=hits,
    )


# ---- encode (inspect a query's weighting — lexical models) ---------------

@router.post("/encode", response_model=EncodeResponse)
def encode(req: EncodeRequest, request: Request) -> EncodeResponse:
    """Show the top-weighted terms a lexical model assigns to a query (TF-IDF weight
    or BM25 idf). The query is normalized with the model's own build options."""
    rep = _require_model(request, req.dataset, req.model)
    state = request.app.state
    _require_preprocessing(state, [rep])
    normalize = _make_normalizer(state)
    options = PreprocessOptions(**rep.options) if rep.options else PreprocessOptions()

    vectors: list[EncodedVector] = []
    if rep.model == "tfidf":
        normalized = [normalize(t, rep.options) for t in req.texts]
        matrix = rep.encode(normalized)
        names = rep.feature_names()
        for i in range(matrix.shape[0]):
            row = matrix.getrow(i)
            idx, data = row.indices, row.data
            order = data.argsort()[::-1][: req.top_terms]
            terms = [WeightedTerm(term=str(names[idx[j]]), weight=round(float(data[j]), 6)) for j in order]
            vectors.append(EncodedVector(nnz=int(row.nnz), terms=terms))
        dim = len(rep.feature_names())
    elif rep.model == "bm25":
        for text in req.texts:
            ranked = rep.encode_terms(text, normalize, req.top_terms)
            terms = [WeightedTerm(term=t, weight=round(float(w), 6)) for t, w in ranked]
            vectors.append(EncodedVector(nnz=len(terms), terms=terms))
        dim = rep.stats_extra().get("vocab_size", 0)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"/encode is for lexical models (tfidf, bm25); use /search for '{rep.model}'",
        )

    return EncodeResponse(model=rep.model, dim=dim, options=options, vectors=vectors)


# ---- jobs ----------------------------------------------------------------

@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(request: Request, job_id: str) -> JobStatus:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"job '{job_id}' not found")
    return status_from_job(job)


@router.get("/jobs", response_model=list[JobStatus])
def list_jobs(
    request: Request,
    dataset: Optional[str] = Query(None, description="Filter by dataset id"),
    model: Optional[str] = Query(None, description="Filter by model (requires dataset)"),
) -> list[JobStatus]:
    if dataset is not None:
        dataset_id = _resolve_or_400(dataset)
        if model:
            key = _job_key(dataset_id, model)
            return [status_from_job(j) for j in request.app.state.jobs.list(type=JOB_TYPE, key=key)]
        prefix = f"{dataset_id}::"
        jobs = request.app.state.jobs.list(type=JOB_TYPE)
        return [status_from_job(j) for j in jobs if j.key.startswith(prefix)]
    return [status_from_job(j) for j in request.app.state.jobs.list(type=JOB_TYPE)]


# ---- delete --------------------------------------------------------------

@router.delete("/representation", response_model=RepresentationDeleteResult)
def delete_representation(
    request: Request,
    dataset: str = Query(..., description="Dataset id from the catalog"),
    model: str = Query("tfidf", description="Representation model"),
) -> RepresentationDeleteResult:
    dataset_id = _resolve_or_400(dataset)
    _validate_model_or_400(model)
    if request.app.state.jobs.active_for(JOB_TYPE, _job_key(dataset_id, model)) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"a build is in progress for '{dataset_id}' / '{model}'; wait for it to finish",
        )
    removed = request.app.state.store.delete(dataset_id, model, REPRESENTATION_VERSION)
    was_cached = request.app.state.models.pop((dataset_id, model), None) is not None
    return RepresentationDeleteResult(
        dataset_id=dataset_id, model=model, deleted=removed, was_cached=was_cached
    )
