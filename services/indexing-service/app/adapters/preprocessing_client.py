"""HTTP client for the preprocessing-service — the only way the indexer gets tokens.

Keeping preprocessing behind a service boundary (rather than importing its code)
is what §8 of the constitution requires: no service imports another's internals.
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
        """True if the service answers /health — used to fail builds fast & clearly."""
        try:
            return self._client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def preprocess_batch(
        self, texts: list[str], options: PreprocessOptions
    ) -> list[list[str]]:
        req = PreprocessBatchRequest(texts=texts, options=options)
        resp = self._client.post("/preprocess/batch", json=req.model_dump())
        resp.raise_for_status()
        body = PreprocessBatchResponse(**resp.json())
        return [item.tokens for item in body.items]

    def close(self) -> None:
        self._client.close()
