"""
Database Models
===============
SQLAlchemy ORM models for the VietSpeech AI platform.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, Float, ForeignKey,
    Integer, String, Text, func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from backend.db.session import Base


def gen_uuid():
    return str(uuid.uuid4())


# ── User ──────────────────────────────────────────────────────────────────────
class User(Base):
    """User account — linked to existing auth system."""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    roles = Column(String(255), default="user")   # CSV roles
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    audio_uploads = relationship("AudioUpload", back_populates="user")
    transcriptions = relationship("Transcription", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")


# ── AudioUpload ────────────────────────────────────────────────────────────────
class AudioUpload(Base):
    """Uploaded audio file record."""
    __tablename__ = "audio_uploads"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    original_filename = Column(String(255))
    storage_path = Column(String(512), nullable=False)
    duration_s = Column(Float)
    sample_rate = Column(Integer, default=16000)
    file_size_bytes = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())
    expires_at = Column(DateTime, nullable=True)   # Auto-cleanup

    user = relationship("User", back_populates="audio_uploads")
    transcriptions = relationship("Transcription", back_populates="audio_upload")


# ── Transcription ──────────────────────────────────────────────────────────────
class Transcription(Base):
    """ASR transcription result."""
    __tablename__ = "transcriptions"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    audio_id = Column(String(36), ForeignKey("audio_uploads.id"), nullable=True)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    text = Column(Text, nullable=False)
    language = Column(String(10), default="vi")
    model_used = Column(String(100))
    confidence = Column(Float, default=1.0)
    wer = Column(Float, nullable=True)
    cer = Column(Float, nullable=True)
    latency_ms = Column(Float)
    province = Column(String(100), nullable=True)  # Detected province
    created_at = Column(DateTime, server_default=func.now())

    audio_upload = relationship("AudioUpload", back_populates="transcriptions")
    user = relationship("User", back_populates="transcriptions")


# ── DialectDetection ──────────────────────────────────────────────────────────
class DialectDetection(Base):
    """Dialect identification result."""
    __tablename__ = "dialect_detections"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    audio_id = Column(String(36), ForeignKey("audio_uploads.id"), nullable=True)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    province = Column(String(100))
    region = Column(String(50))
    confidence = Column(Float)
    ethnic_viet = Column(Float, default=1.0)
    ethnic_ede = Column(Float, default=0.0)
    ethnic_bahnar = Column(Float, default=0.0)
    code_switching_score = Column(Float, default=0.0)
    latency_ms = Column(Float)
    created_at = Column(DateTime, server_default=func.now())


# ── Translation ───────────────────────────────────────────────────────────────
class Translation(Base):
    """Translation result record."""
    __tablename__ = "translations"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    source_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=False)
    source_lang = Column(String(10), nullable=False)
    target_lang = Column(String(10), nullable=False)
    confidence = Column(Float, default=1.0)
    latency_ms = Column(Float)
    created_at = Column(DateTime, server_default=func.now())


# ── AuditLog ──────────────────────────────────────────────────────────────────
class AuditLog(Base):
    """Security audit log for all API requests."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    request_id = Column(String(36))
    method = Column(String(10))
    path = Column(String(512))
    status_code = Column(Integer)
    ip_address = Column(String(50))
    user_agent = Column(String(255))
    latency_ms = Column(Float)
    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="audit_logs")
