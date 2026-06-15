"""
Database Session Manager
=========================
SQLAlchemy async session management for PostgreSQL.
Supports connection pooling, health checks, and migration integration.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)

# ── Engine ────────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}".format(
        user=os.getenv("POSTGRES_USER", "vietspeech_user"),
        password=os.getenv("POSTGRES_PASSWORD", "change-me"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        db=os.getenv("POSTGRES_DB", "vietspeech"),
    )
)

engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("APP_ENV") == "development",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=300,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ── Base ──────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""
    pass


# ── Dependency ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db_dependency() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    async with get_db() as session:
        yield session


# ── Initialization ────────────────────────────────────────────────────────────
async def init_db() -> None:
    """Initialize database tables (for development; use Alembic in production)."""
    try:
        async with engine.begin() as conn:
            # Import all models to register them with Base
            from backend.db import models as _models  # noqa: F401
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized")
    except Exception as e:
        logger.warning(f"Database initialization failed (might not be needed): {e}")


async def check_db_connection() -> bool:
    """Check database connectivity."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


async def close_db() -> None:
    """Dispose database connection pool."""
    await engine.dispose()
