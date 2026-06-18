"""Forwarding helpers shared by every gateway router.

The gateway is the single external entry point: the product UI talks
only to it and it forwards to the internal services over the compose network. These
helpers keep the forwarding identical everywhere — issue the downstream request and
hand its response back unchanged (status + JSON body), mapping a connection failure
to a clean 503 instead of a stack trace.
"""

from __future__ import annotations

from typing import Callable, Optional

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from app.clients import ServiceClient


def forward(resp: httpx.Response) -> JSONResponse:
    """Pass a downstream response through unchanged (status + JSON body)."""
    try:
        content = resp.json()
    except ValueError:
        content = {"error": {"code": "bad_gateway", "message": resp.text}}
    return JSONResponse(status_code=resp.status_code, content=content)


def proxy(call: Callable[[], httpx.Response]) -> JSONResponse:
    """Run a downstream call, mapping connection failures to a clean 503."""
    try:
        return forward(call())
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"downstream service unavailable: {exc}") from exc


def safe_get_json(client: ServiceClient, path: str, params: dict) -> Optional[dict]:
    """GET that returns parsed JSON on 200, else None (a down service ⇒ blank status)."""
    try:
        resp = client.get(path, params=params)
        if resp.status_code == 200:
            return resp.json()
    except httpx.HTTPError:
        pass
    return None


def clean(params: dict) -> dict:
    """Drop ``None`` query params so they aren't forwarded as empty strings."""
    return {k: v for k, v in params.items() if v is not None}
