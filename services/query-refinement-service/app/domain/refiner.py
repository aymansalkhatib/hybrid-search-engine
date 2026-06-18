"""Core query-refinement logic — framework-independent (no FastAPI here).

Two stages, each independently toggleable so the assignment's "with vs without"
evaluation is a single flag flip:

* **Spell correction** — each token is looked up in SymSpell; if the closest
  in-dictionary word differs, the token is replaced. Word-by-word (not
  ``lookup_compound``) so token alignment is preserved and every change is
  reportable as a "did you mean" hint.
* **Synonym expansion** — each *content* token gets up to N single-word WordNet
  synonyms appended, broadening recall. Off by default because it can dilute
  precision; the UI exposes it as an opt-in toggle.

**Entity protection (``protect_proper_nouns``, on by default).** A frequency-dictionary
corrector has no entry for proper nouns / acronyms, so it "corrects" them to the nearest
common word — on a clean, entity-rich corpus (Quora: ``Bloomberg→bloomer``, ``MIT→it``,
``Quora→quota``) this corrupts the query and *lowers* accuracy. So we only correct tokens
that look like ordinary lowercase words, and never touch:

* **ALL-CAPS** tokens (acronyms: MIT, AAP, NASA),
* tokens with **internal/trailing capitals** (brands: iPhone, McKinsey),
* **capitalised** tokens **mid-query** (proper nouns: Bloomberg, Quora) — sentence-initial
  capitalisation (the first token) is *not* a proper-noun signal, so a leading typo can
  still be fixed,
* **short** tokens (< 3 chars: Mr, US, AI) and tokens with an **apostrophe**
  (contractions/possessives: don't, bachelor's).

The refiner works on the **raw** query and returns a refined query *string*; the
retrieval path preprocesses that string like any other query, so refinement stays a
clean, separable stage upstream of representation/matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from symspellpy import SymSpell, Verbosity

# Split into word tokens, keeping apostrophes inside words (don't, it's). Curly quotes
# are normalised to straight ones first so possessives tokenise consistently.
_TOKEN_RE = re.compile(r"[A-Za-z']+")
# Don't try to correct/expand very short tokens (initials, "of", "AI") — noisy.
_MIN_WORD_LEN = 3


@dataclass
class Correction:
    original: str
    corrected: str


@dataclass
class RefineResult:
    original: str
    corrected: str
    refined: str
    corrections: list[Correction] = field(default_factory=list)
    added_terms: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)


def _looks_like_entity(token: str, *, is_first: bool) -> bool:
    """True for tokens we should not correct/expand: acronyms, brands, proper nouns,
    contractions/possessives. ``is_first`` relaxes the proper-noun rule for the first
    token, whose capital is just sentence case (so a leading misspelling stays fixable)."""
    if "'" in token:                                  # don't, it's, bachelor's
        return True
    if len(token) >= 2 and token.isupper():           # MIT, AAP, NASA
        return True
    if any(c.isupper() for c in token[1:]):           # iPhone, McKinsey, eBay
        return True
    if token[:1].isupper() and not is_first:          # Bloomberg, Quora (mid-query)
        return True
    return False


def _match_case(original: str, corrected: str) -> str:
    """Re-apply the original token's capitalisation to the correction (display nicety;
    retrieval lowercases anyway). Only sentence-initial Titlecase reaches here."""
    if original[:1].isupper():
        return corrected[:1].upper() + corrected[1:]
    return corrected


class QueryRefiner:
    def __init__(self, sym_spell: SymSpell, stopwords: set[str], max_query_terms: int = 64) -> None:
        self._sym = sym_spell
        self._stop = stopwords
        self._max_terms = max_query_terms

    def refine(
        self,
        text: str,
        *,
        correct_spelling: bool = True,
        expand_synonyms: bool = False,
        max_synonyms_per_term: int = 2,
        max_edit_distance: int = 2,
        protect_proper_nouns: bool = True,
    ) -> RefineResult:
        tokens = _TOKEN_RE.findall(text.replace("’", "'"))[: self._max_terms]
        applied: list[str] = []

        # ---- stage 1: spell correction (token-aligned, entity-aware) ----
        corrected_tokens: list[str] = list(tokens)
        corrections: list[Correction] = []
        if correct_spelling:
            for i, tok in enumerate(tokens):
                fixed = self._correct_token(
                    tok, max_edit_distance, is_first=(i == 0), protect=protect_proper_nouns
                )
                if fixed != tok:
                    corrected_tokens[i] = fixed
                    corrections.append(Correction(original=tok, corrected=fixed))
            if corrections:
                applied.append("spell_correction")
        corrected_text = " ".join(corrected_tokens)

        # ---- stage 2: synonym expansion (appended, originals kept) ----
        added: list[str] = []
        if expand_synonyms and max_synonyms_per_term > 0:
            present = {t.lower() for t in corrected_tokens}
            for i, tok in enumerate(corrected_tokens):
                if protect_proper_nouns and _looks_like_entity(tok, is_first=(i == 0)):
                    continue
                for syn in self._synonyms(tok, max_synonyms_per_term):
                    if syn not in present:
                        present.add(syn)
                        added.append(syn)
            if added:
                applied.append("synonym_expansion")

        refined_text = " ".join(corrected_tokens + added) if added else corrected_text

        # The corrected query is the most useful single suggestion to surface.
        suggestions: list[str] = []
        if corrected_text and corrected_text.lower() != text.strip().lower():
            suggestions.append(corrected_text)

        return RefineResult(
            original=text,
            corrected=corrected_text,
            refined=refined_text,
            corrections=corrections,
            added_terms=added,
            suggestions=suggestions,
            applied=applied,
        )

    def _correct_token(self, token: str, max_edit_distance: int, *, is_first: bool, protect: bool) -> str:
        """Closest in-dictionary word, or the token unchanged if it's too short, an
        entity we protect, already valid, or has no suggestion."""
        if len(token) < _MIN_WORD_LEN:
            return token
        if protect and _looks_like_entity(token, is_first=is_first):
            return token
        edit = min(max_edit_distance, 2)
        suggestions = self._sym.lookup(token.lower(), Verbosity.CLOSEST, max_edit_distance=edit)
        if not suggestions:
            return token
        best = suggestions[0].term
        # SymSpell lowercases; only treat it as a correction when the word actually
        # changed (ignoring case), then re-apply the original capitalisation.
        if best == token.lower():
            return token
        return _match_case(token, best)

    def _synonyms(self, word: str, max_n: int) -> list[str]:
        """Up to ``max_n`` single-word WordNet synonyms for a content word."""
        lower = word.lower()
        if len(lower) < _MIN_WORD_LEN or lower in self._stop:
            return []
        # Imported lazily so the domain module stays import-light and testable without
        # NLTK data present; resources.warm_wordnet() loads the data at startup.
        from nltk.corpus import wordnet as wn

        out: list[str] = []
        seen = {lower}
        for synset in wn.synsets(lower):
            for name in synset.lemma_names():
                term = name.replace("_", " ").lower()
                if " " in term or term in seen:  # single words only, no duplicates
                    continue
                seen.add(term)
                out.append(term)
                if len(out) >= max_n:
                    return out
        return out
