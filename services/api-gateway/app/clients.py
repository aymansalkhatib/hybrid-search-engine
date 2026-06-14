"""Thin HTTP clients the gateway uses to reach internal services.

The gateway is the single entry point: the UI talks only to it and
it forwards to the doc-store and indexing services. These wrappers keep the
forwarding in one place and intentionally do little beyond issuing the request and
handing back the raw ``httpx.Response`` for the caller to translate.
"""

from __future__ import annotations

import httpx


class ServiceClient:
    """A keep-alive HTTP client bound to one downstream service base URL."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def get(self, path: str, params: dict | None = None) -> httpx.Response:
        return self._client.get(path, params=params)

    def post(self, path: str, json: dict | None = None) -> httpx.Response:
        return self._client.post(path, json=json)

    def request(self, method: str, path: str, *, params: dict | None = None,
                json: dict | None = None) -> httpx.Response:
        return self._client.request(method, path, params=params, json=json)

    def close(self) -> None:
        self._client.close()
