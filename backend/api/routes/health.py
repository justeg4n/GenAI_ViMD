"""
Health & Metrics Routes
========================
GET /health — Liveness/readiness check
GET /metrics — Prometheus metrics endpoint
"""

from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()

START_TIME = time.time()


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_s: float
    environment: str
    models: dict
    database: str
    redis: str


class ComponentStatus(BaseModel):
    status: str
    latency_ms: Optional[float] = None
    error: Optional[str] = None


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service health including model availability, database, and Redis connectivity.",
    tags=["Health"],
)
async def health_check(request: Request):
    """Liveness + readiness health check."""
    uptime = time.time() - START_TIME
    registry = getattr(request.app.state, "model_registry", None)

    # Check database
    db_status = "unknown"
    try:
        from backend.db.session import check_db_connection
        db_ok = await check_db_connection()
        db_status = "connected" if db_ok else "error"
    except Exception as e:
        db_status = f"error: {str(e)[:50]}"

    # Check Redis
    redis_status = "unknown"
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        await r.ping()
        redis_status = "connected"
        await r.close()
    except Exception as e:
        redis_status = f"error: {str(e)[:50]}"

    # Model availability
    models = {}
    if registry:
        models = registry.get_model_status()

    overall_status = "healthy"
    if db_status != "connected":
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version="2.0.0",
        uptime_s=round(uptime, 2),
        environment=os.getenv("APP_ENV", "development"),
        models=models,
        database=db_status,
        redis=redis_status,
    )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Exposes Prometheus-formatted metrics for scraping.",
    tags=["Health"],
)
async def prometheus_metrics():
    """Expose Prometheus metrics."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        from fastapi.responses import PlainTextResponse
        uptime = time.time() - START_TIME
        return PlainTextResponse(
            content=f"# VietSpeech AI Metrics\nvietspeech_uptime_seconds {uptime:.2f}\n",
            media_type="text/plain",
        )


@router.get("/", tags=["Health"])
async def root():
    return {
        "service": "VietSpeech AI Platform",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }
