"""Consistent error handling for every service.

Installs exception handlers that render the standard error envelope:
``{"error": {"code", "message", "details"}}``. Reused by all services
so error shapes never diverge across the system.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from shared.contracts import ErrorDetail, ErrorEnvelope

logger = logging.getLogger("ir_common.errors")


def _envelope(code: str, message: str, details: dict | None = None) -> dict:
    body = ErrorEnvelope(error=ErrorDetail(code=code, message=message, details=details))
    return jsonable_encoder(body.model_dump())


def install_error_handlers(app: FastAPI) -> None:
    """Register handlers so all errors leave the service in the standard envelope."""

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_envelope(
                "validation_error",
                "Request validation failed",
                {"errors": jsonable_encoder(exc.errors())},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _on_http_error(_: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope("http_error", str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def _on_unhandled(_: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope("internal_error", "Internal server error"),
        )
