"""HTTP client the evaluation-service uses to refine test queries before retrieval.

Used only when an evaluation requests refinement (the with/without comparison): each
judged query is refined **once** here, then every model run searches the refined text —
so a 'with-refinement' report is the same models on a refined query set, directly
comparable to the 'baseline' report. The underlying ``httpx.Client`` is shared across
the evaluator's worker threads (httpx clients are thread-safe).
"""

from __future__ import annotations

import httpx

from shared.contracts import RefineOptions, RefineResponse


class RefinementClient:
    """Thin wrapper over the query-refinement-service ``/refine`` endpoint."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def is_healthy(self) -> bool:
        """True if the service answers /health — used to fail an evaluation fast & clearly."""
        try:
            return self._client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def refine(self, *, text: str, options: RefineOptions) -> str:
        """Return the refined query text. Raises ``httpx.HTTPStatusError`` on a non-2xx."""
        resp = self._client.post(
            "/refine", json={"text": text, "options": options.model_dump(mode="json")}
        )
        resp.raise_for_status()
        return RefineResponse(**resp.json()).refined

    def close(self) -> None:
        self._client.close()
