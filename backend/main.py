"""
VietSpeech AI — FastAPI Backend
=================================
Main application entry point.

Preserved routes (DO NOT MODIFY):
  /login, /logout, /register, /profile

Added routes:
  POST /upload-audio
  POST /transcribe
  POST /detect-accent
  POST /generate-speech
  POST /voice-clone
  POST /translate
  GET  /health
  GET  /metrics
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from backend.api.auth_adapter import auth_router
from backend.api.routes.audio import router as audio_router
from backend.api.routes.clone import router as clone_router
from backend.api.routes.dialect import router as dialect_router
from backend.api.routes.health import router as health_router
from backend.api.routes.transcribe import router as transcribe_router
from backend.api.routes.translate import router as translate_router
from backend.api.routes.tts import router as tts_router
from backend.middleware.audit import AuditMiddleware
from backend.middleware.rate_limit import RateLimitMiddleware
from backend.services.model_registry import ModelRegistry
from backend.db.session import init_db

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / Shutdown lifecycle."""
    logger.info("Starting VietSpeech AI backend...")

    # Initialize database
    await init_db()
    logger.info("✓ Database initialized")

    # Initialize model registry (lazy-loads models on first request)
    registry = ModelRegistry()
    await registry.initialize()
    app.state.model_registry = registry
    logger.info("✓ Model registry initialized")

    # Configure Sentry if DSN is set
    import os
    sentry_dsn = os.getenv("SENTRY_DSN", "")
    if sentry_dsn:
        import sentry_sdk
        sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=0.1)
        logger.info("✓ Sentry initialized")

    logger.info("✓ VietSpeech AI backend started successfully")
    yield

    # Shutdown
    logger.info("Shutting down VietSpeech AI backend...")
    await registry.cleanup()


# ── Application Factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    import os

    app = FastAPI(
        title="VietSpeech AI Platform",
        description=(
            "Production-grade Vietnamese Multi-Dialect Speech AI platform. "
            "Supports ASR, dialect identification, TTS, voice cloning, "
            "and Vietnamese↔Ede/Bahnar translation."
        ),
        version="2.0.0",
        docs_url="/docs" if os.getenv("APP_ENV") != "production" else None,
        redoc_url="/redoc" if os.getenv("APP_ENV") != "production" else None,
        lifespan=lifespan,
    )

    # ── CORS ──
    allowed_origins = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:8080"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # ── GZip compression ──
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Rate limiting ──
    app.add_middleware(RateLimitMiddleware)

    # ── Audit logging ──
    app.add_middleware(AuditMiddleware)

    # ── Request ID + Timing ──
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        import uuid
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return response

    # ── Global Exception Handler ──
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": getattr(request.state, "request_id", ""),
            },
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    # Preserved auth routes (adapter — do not modify existing auth logic)
    app.include_router(auth_router, tags=["Authentication"])

    # New AI routes
    app.include_router(audio_router, prefix="/api/v1", tags=["Audio"])
    app.include_router(transcribe_router, prefix="/api/v1", tags=["ASR"])
    app.include_router(dialect_router, prefix="/api/v1", tags=["Dialect"])
    app.include_router(tts_router, prefix="/api/v1", tags=["TTS"])
    app.include_router(clone_router, prefix="/api/v1", tags=["Voice Cloning"])
    app.include_router(translate_router, prefix="/api/v1", tags=["Translation"])
    app.include_router(health_router, tags=["Health"])

    return app


# ── Application Instance ──────────────────────────────────────────────────────
app = create_app()


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=os.getenv("APP_ENV") == "development",
        workers=1,  # Use 1 worker for GPU model sharing
        log_level="info",
    )
