"""
Audit Logging Middleware
=========================
Logs all API requests to the database for security audit trail.
"""

from __future__ import annotations

import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

SKIP_AUDIT_PATHS = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Records every API request in the audit_logs table.
    Non-blocking: logging failures don't affect the response.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path in SKIP_AUDIT_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000

        # Log asynchronously to avoid blocking response
        try:
            await self._log_request(request, response.status_code, latency_ms)
        except Exception as e:
            logger.debug(f"Audit logging failed (non-critical): {e}")

        return response

    async def _log_request(
        self, request: Request, status_code: int, latency_ms: float
    ) -> None:
        """Write audit log entry."""
        # Structured log (picked up by OpenTelemetry / Sentry)
        logger.info(
            "api_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 2),
                "ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "")[:200],
                "request_id": getattr(request.state, "request_id", ""),
            },
        )

        # Database audit log (best-effort)
        try:
            from backend.db.session import AsyncSessionLocal
            from backend.db.models import AuditLog
            async with AsyncSessionLocal() as session:
                log = AuditLog(
                    request_id=getattr(request.state, "request_id", ""),
                    method=request.method,
                    path=request.url.path[:512],
                    status_code=status_code,
                    ip_address=(request.client.host if request.client else "")[:50],
                    user_agent=request.headers.get("user-agent", "")[:255],
                    latency_ms=latency_ms,
                )
                session.add(log)
                await session.commit()
        except Exception:
            pass  # Non-critical
