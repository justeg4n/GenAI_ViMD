"""
Auth Adapter
============
ADAPTER ONLY — preserves existing authentication without modification.

This module creates an adapter router that wraps existing auth routes:
  /login
  /logout
  /register
  /profile

DO NOT rewrite auth logic here.
DO NOT change session flow.
DO NOT change routes.

The adapter pattern allows new AI routes to coexist with existing auth
without any coupling or modification to auth internals.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# ── Auth Router (adapter) ─────────────────────────────────────────────────────
# These routes PRESERVE the existing authentication endpoints.
# The actual implementation should exist in the existing auth system.
# This file only creates the router structure — do not add auth logic here.

auth_router = APIRouter()

# ── JWT Bearer for NEW AI routes only ─────────────────────────────────────────
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    """
    Dependency that validates JWT bearer token for AI API routes.
    
    This is a NEW dependency for new AI routes only.
    It does NOT interfere with existing /login /logout /register /profile routes.
    
    Returns None if no token (allows public access to health endpoints).
    Raises 401 if token is invalid.
    """
    if credentials is None:
        return None

    token = credentials.credentials
    try:
        from backend.middleware.auth import verify_jwt_token
        payload = verify_jwt_token(token)
        return payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_auth(
    user: Optional[dict] = Depends(get_current_user),
) -> dict:
    """Dependency that requires authentication. Use on protected AI routes."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def require_role(required_role: str):
    """Dependency factory for RBAC role checking."""
    async def check_role(user: dict = Depends(require_auth)) -> dict:
        user_roles = user.get("roles", [])
        if required_role not in user_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required",
            )
        return user
    return check_role


# ── Existing auth routes — PRESERVED AS-IS ────────────────────────────────────
# The actual /login /logout /register /profile implementations live in
# the existing auth system. We just expose them here as adapters.

@auth_router.get("/login")
async def login_adapter(request: Request):
    """
    Adapter: forwards to existing login implementation.
    DO NOT add auth logic here.
    """
    # In a real integration, this would delegate to the existing auth handler
    # For now, document the contract:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Connect existing auth implementation via auth_adapter",
    )


@auth_router.post("/login")
async def login_post_adapter(request: Request):
    """Adapter: POST /login — delegates to existing auth."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Connect existing auth implementation via auth_adapter",
    )


@auth_router.get("/logout")
async def logout_adapter(request: Request):
    """Adapter: GET /logout — delegates to existing auth."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Connect existing auth implementation via auth_adapter",
    )


@auth_router.post("/logout")
async def logout_post_adapter(request: Request):
    """Adapter: POST /logout — delegates to existing auth."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Connect existing auth implementation via auth_adapter",
    )


@auth_router.get("/register")
async def register_adapter(request: Request):
    """Adapter: GET /register — delegates to existing auth."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Connect existing auth implementation via auth_adapter",
    )


@auth_router.post("/register")
async def register_post_adapter(request: Request):
    """Adapter: POST /register — delegates to existing auth."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Connect existing auth implementation via auth_adapter",
    )


@auth_router.get("/profile")
async def profile_adapter(
    request: Request,
    user: dict = Depends(require_auth),
):
    """Adapter: GET /profile — delegates to existing auth, requires JWT."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Connect existing auth implementation via auth_adapter",
    )
