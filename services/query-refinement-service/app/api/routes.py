"""HTTP layer (thin) — maps requests to the domain QueryRefiner."""

from __future__ import annotations

from fastapi import APIRouter, Request

from shared.contracts import RefineRequest, RefineResponse, TermCorrection

router = APIRouter(tags=["query-refinement"])


@router.post("/refine", response_model=RefineResponse)
def refine(req: RefineRequest, request: Request) -> RefineResponse:
    """Refine a raw query (spell-correction + optional synonym expansion).

    Returns the refined query string plus an explainable breakdown (corrections,
    added terms, suggestions) so the UI can show "did you mean" and the report can
    measure refinement's before/after effect.
    """
    opts = req.options
    result = request.app.state.refiner.refine(
        req.text,
        correct_spelling=opts.correct_spelling,
        expand_synonyms=opts.expand_synonyms,
        max_synonyms_per_term=opts.max_synonyms_per_term,
        max_edit_distance=opts.max_edit_distance,
        protect_proper_nouns=opts.protect_proper_nouns,
    )
    return RefineResponse(
        original=result.original,
        corrected=result.corrected,
        refined=result.refined,
        corrections=[TermCorrection(original=c.original, corrected=c.corrected) for c in result.corrections],
        added_terms=result.added_terms,
        suggestions=result.suggestions,
        applied=result.applied,
    )
