"""
Rate Limiting Middleware
=========================
Redis-backed sliding window rate limiter.
Applies per-IP and per-user limits.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 60))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Paths exempt from rate limiting
EXEMPT_PATHS = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter using Redis.
    Falls back to in-memory limiter if Redis is unavailable.
    """

    def __init__(self, app, rate_limit: int = RATE_LIMIT_PER_MINUTE):
        super().__init__(app)
        self.rate_limit = rate_limit
        self._redis = None
        self._memory_store: dict[str, list[float]] = {}
        self._use_redis = True

    async def _get_redis(self):
        if not self._use_redis:
            return None
        try:
            if self._redis is None:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
                await self._redis.ping()
            return self._redis
        except Exception as e:
            logger.warning(f"Redis unavailable for rate limiting, using memory: {e}")
            self._use_redis = False
            return None

    async def dispatch(self, request: Request, call_next):
        # Skip exempt paths
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"
        user_id = getattr(request.state, "user_id", None)
        key = f"rl:{user_id or client_ip}"

        # Check rate limit
        allowed, remaining = await self._check_rate_limit(key)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests. Please slow down.",
                    "retry_after_seconds": 60,
                },
                headers={
                    "X-RateLimit-Limit": str(self.rate_limit),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": "60",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    async def _check_rate_limit(self, key: str) -> tuple[bool, int]:
        """Check sliding window rate limit. Returns (allowed, remaining)."""
        now = time.time()
        window_start = now - 60.0

        redis = await self._get_redis()
        if redis:
            try:
                pipe = redis.pipeline()
                pipe.zremrangebyscore(key, 0, window_start)
                pipe.zadd(key, {str(now): now})
                pipe.zcard(key)
                pipe.expire(key, 120)
                results = await pipe.execute()
                count = results[2]
                remaining = max(0, self.rate_limit - count)
                return count <= self.rate_limit, remaining
            except Exception as e:
                logger.warning(f"Redis rate limit check failed: {e}")

        # In-memory fallback
        if key not in self._memory_store:
            self._memory_store[key] = []
        timestamps = self._memory_store[key]
        timestamps = [t for t in timestamps if t > window_start]
        timestamps.append(now)
        self._memory_store[key] = timestamps
        count = len(timestamps)
        remaining = max(0, self.rate_limit - count)
        return count <= self.rate_limit, remaining
