"""Shared request/response contracts used across all services.

Every cross-service payload is a pydantic model defined here, imported by both
the caller and the callee. No ad-hoc dicts cross a service boundary.
"""

from shared.contracts.common import (
    ErrorDetail,
    ErrorEnvelope,
    HealthResponse,
    ServiceInfo,
)

__all__ = [
    "ErrorDetail",
    "ErrorEnvelope",
    "HealthResponse",
    "ServiceInfo",
]
