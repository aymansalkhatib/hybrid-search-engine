"""Contracts for the query-refinement-service.

Query refinement runs on the **raw** query (before the preprocessing pipeline that
docs and queries share), because correcting ``infromation`` → ``information`` needs
the original word, not its stemmed/stopword-stripped form. The service returns a
refined query string; the retrieval path then preprocesses it exactly as usual, so
refinement stays an independent, toggleable stage (assignment §5 req 5).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RefineOptions(BaseModel):
    """Per-request toggles for each refinement stage (mirrors PreprocessOptions)."""

    # Spell-correction is safe and almost always helps recall, so it's on by default.
    correct_spelling: bool = True
    # Synonym expansion broadens recall but can hurt precision, so it's OFF by
    # default and the UI exposes it as an opt-in toggle (before/after evaluable).
    expand_synonyms: bool = False
    # How many WordNet synonyms to append per content term when expanding.
    max_synonyms_per_term: int = Field(default=2, ge=0, le=10)
    # SymSpell edit-distance budget per token (≤ the dictionary's built max of 2).
    max_edit_distance: int = Field(default=2, ge=1, le=2)
    # Don't "correct" or expand likely entities — proper nouns (capitalised mid-query),
    # acronyms (ALL-CAPS) and short tokens. On clean, entity-rich corpora (e.g. Quora:
    # "Bloomberg"→"bloomer", "MIT"→"it") naive correction corrupts queries and HURTS
    # accuracy; this keeps correction to genuine lowercase misspellings. On by default.
    protect_proper_nouns: bool = True


class RefineRequest(BaseModel):
    text: str
    options: RefineOptions = Field(default_factory=RefineOptions)


class TermCorrection(BaseModel):
    """One token the spell-corrector changed (drives a 'did you mean' hint)."""

    original: str
    corrected: str


class RefineResponse(BaseModel):
    """Full, explainable refinement output for one query."""

    original: str
    # Spelling-corrected query (before any synonym expansion).
    corrected: str
    # Final query to feed retrieval (corrected + any appended synonyms).
    refined: str
    # Which tokens the corrector changed.
    corrections: list[TermCorrection] = Field(default_factory=list)
    # Synonyms appended by expansion (empty when expansion is off).
    added_terms: list[str] = Field(default_factory=list)
    # Alternative full-query suggestions the UI can offer (e.g. the corrected query).
    suggestions: list[str] = Field(default_factory=list)
    # Names of the stages that actually changed something, for the report/UI.
    applied: list[str] = Field(default_factory=list)
