"""HTTP client for the representation-service scoring primitives.

The retrieval-service does **not** own the model artifacts — the representation-service
is their single owner (it holds the fitted matrices in memory). Retrieval asks for
per-model scores over the compose network and applies the retrieval strategy (single /
serial / parallel) on top. Two primitives mirror ``BaseRepresentation``:

* :meth:`rank`  → ``POST /rank``   — rank the corpus with one model (= ``rep.search``).
* :meth:`score` → ``POST /score``  — score a given set of ids (= ``rep.score_docs``),
  used by serial re-ranking.

Keeping the models behind a service boundary (rather than importing their code and
loading the artifacts twice) is what §4/§8 of the constitution require.
"""

from __future__ import annotations

import httpx

from shared.contracts import RankRequest, RankResponse, ScoreRequest, ScoreResponse

# A ranked result: (external doc_id, score), higher = more relevant.
Scored = tuple[str, float]


class RepresentationClient:
    """Thin wrapper over the representation-service scoring API."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        # Generous timeout: a single /rank scores the whole corpus for one model.
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def is_healthy(self) -> bool:
        """True if the service answers /health — used to fail a search fast & clearly."""
        try:
            return self._client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def rank(
        self, *, dataset: str, model: str, query: str, top_k: int, k1: float, b: float
    ) -> list[Scored]:
        """Rank the corpus for ``query`` with one model → ``(doc_id, score)`` desc."""
        req = RankRequest(dataset=dataset, model=model, query=query, top_k=top_k, k1=k1, b=b)
        resp = self._client.post("/rank", json=req.model_dump())
        resp.raise_for_status()
        body = RankResponse(**resp.json())
        return [(h.doc_id, h.score) for h in body.hits]

    def score(
        self, *, dataset: str, model: str, query: str, doc_ids: list[str], k1: float, b: float
    ) -> dict[str, float]:
        """Score a specific set of ids with one model → ``{doc_id: score}``."""
        req = ScoreRequest(dataset=dataset, model=model, query=query, doc_ids=doc_ids, k1=k1, b=b)
        resp = self._client.post("/score", json=req.model_dump())
        resp.raise_for_status()
        body = ScoreResponse(**resp.json())
        return dict(body.scores)

    def close(self) -> None:
        self._client.close()
