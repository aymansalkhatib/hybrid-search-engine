"""Common contracts shared by every service (meta endpoints, error envelope)."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Returned by every service's `GET /health`."""

    status: str = "ok"
    service: str
    version: str = "0.1.0"


class ServiceInfo(BaseModel):
    """Returned by every service's `GET /` (human/diagnostic info)."""

    service: str
    version: str = "0.1.0"
    description: str = ""


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[dict[str, Any]] = None


class ErrorEnvelope(BaseModel):
    """Consistent error shape across all services: {"error": {...}}."""

    error: ErrorDetail
