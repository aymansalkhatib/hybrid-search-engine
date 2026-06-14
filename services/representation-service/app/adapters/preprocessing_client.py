"""HTTP client for the preprocessing-service — the only way this service gets tokens.

Both the corpus (at build time) and queries (at encode time) go through the same
endpoint with the **same options**, which is what keeps documents and queries in one
comparable space. Keeping preprocessing behind a service boundary (rather than
importing its code) is what §8 of the constitution requires.
"""

from __future__ import annotations

import httpx

from shared.contracts import (
    PreprocessBatchRequest,
    PreprocessBatchResponse,
    PreprocessOptions,
)


class PreprocessingClient:
    """Thin wrapper over the preprocessing-service REST API."""

    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        # Generous timeout: a batch of ~1000 docs is POS-tagged + lemmatized server-side.
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def is_healthy(self) -> bool:
        """True if the service answers /health — used to fail builds/encodes fast."""
        try:
            return self._client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def normalize_batch(
        self, texts: list[str], options: PreprocessOptions
    ) -> list[str]:
        """Return one normalized (space-joined tokens) string per input text."""
        req = PreprocessBatchRequest(texts=texts, options=options)
        resp = self._client.post("/preprocess/batch", json=req.model_dump())
        resp.raise_for_status()
        body = PreprocessBatchResponse(**resp.json())
        return [item.normalized_text for item in body.items]

    def close(self) -> None:
        self._client.close()
