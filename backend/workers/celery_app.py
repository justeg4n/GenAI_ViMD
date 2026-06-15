"""
Celery Worker Application
=========================
Async background tasks for:
  - Long audio transcription (> 30s)
  - Batch dialect analysis
  - Training job scheduling
  - Model checkpoint sync to GDrive
  - Audio file cleanup (TTL expiry)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from celery import Celery

logger = logging.getLogger(__name__)

# ── Celery App ────────────────────────────────────────────────────────────────
celery_app = Celery(
    "vietspeech_ai",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2"),
    include=[
        "backend.workers.tasks.audio_tasks",
        "backend.workers.tasks.cleanup_tasks",
    ],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Time limits
    task_soft_time_limit=600,       # 10 minutes soft limit
    task_time_limit=900,            # 15 minutes hard limit
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Result
    result_expires=3600,            # 1 hour TTL for results

    # Queues
    task_routes={
        "backend.workers.tasks.audio_tasks.*": {"queue": "audio"},
        "backend.workers.tasks.cleanup_tasks.*": {"queue": "default"},
    },

    # Retry policy
    task_max_retries=3,
    task_default_retry_delay=60,

    # Beat schedule (periodic tasks)
    beat_schedule={
        "cleanup-expired-audio": {
            "task": "backend.workers.tasks.cleanup_tasks.cleanup_expired_audio",
            "schedule": 3600.0,    # Every hour
        },
        "sync-checkpoints-to-gdrive": {
            "task": "backend.workers.tasks.cleanup_tasks.sync_checkpoints",
            "schedule": 300.0,     # Every 5 minutes
        },
    },
)
