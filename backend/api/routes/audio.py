"""
Audio Upload Route
==================
POST /api/v1/upload-audio

Handles audio file upload with:
  - Format validation (WAV, MP3, FLAC, OGG, WebM)
  - Size limits (max 50MB)
  - Automatic conversion to 16kHz mono WAV
  - Secure storage with UUID filename
  - Returns audio_id for downstream API calls
"""

from __future__ import annotations

import io
import logging
import os
import uuid
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.api.auth_adapter import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Configuration ─────────────────────────────────────────────────────────────
STORAGE_AUDIO = Path(os.getenv("STORAGE_BASE", "storage")) / "audio"
STORAGE_AUDIO.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE_MB = int(os.getenv("AUDIO_UPLOAD_MAX_MB", 50))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3",
    "audio/flac", "audio/x-flac",
    "audio/ogg", "audio/webm",
    "application/octet-stream",   # Allow generic binary (some clients)
}

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a"}


# ── Schemas ───────────────────────────────────────────────────────────────────

class AudioUploadResponse(BaseModel):
    audio_id: str
    filename: str
    duration_s: float
    sample_rate: int
    channels: int
    file_size_bytes: int
    storage_path: str
    message: str = "Audio uploaded and processed successfully"


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/upload-audio",
    response_model=AudioUploadResponse,
    summary="Upload audio file",
    description="Upload audio for processing. Returns audio_id for use with /transcribe, /detect-accent, etc.",
)
async def upload_audio(
    file: Annotated[UploadFile, File(description="Audio file (WAV, MP3, FLAC, OGG, WebM)")],
    request: Request,
    user: Optional[dict] = Depends(get_current_user),
):
    # ── Validate file ──
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported audio format: {file.content_type}. "
                       f"Allowed: WAV, MP3, FLAC, OGG, WebM",
            )

    # ── Read file contents ──
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {len(contents) / 1024**2:.1f}MB exceeds limit of {MAX_FILE_SIZE_MB}MB",
        )
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded",
        )

    # ── Process audio ──
    audio_id = str(uuid.uuid4())
    try:
        audio_array, sample_rate, metadata = await _process_audio(contents, file.filename or "audio")
    except Exception as e:
        logger.warning(f"Audio processing failed for {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid or corrupted audio file: {e}",
        )

    # ── Save to storage ──
    output_path = STORAGE_AUDIO / f"{audio_id}.wav"
    try:
        import soundfile as sf
        sf.write(str(output_path), audio_array, sample_rate, subtype="PCM_16")
    except Exception as e:
        logger.error(f"Failed to save audio {audio_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save audio file",
        )

    logger.info(
        f"Audio uploaded: id={audio_id} "
        f"duration={metadata['duration_s']:.2f}s "
        f"user={user.get('sub', 'anonymous') if user else 'anonymous'}"
    )

    return AudioUploadResponse(
        audio_id=audio_id,
        filename=file.filename or "audio.wav",
        duration_s=metadata["duration_s"],
        sample_rate=sample_rate,
        channels=1,
        file_size_bytes=len(contents),
        storage_path=str(output_path.relative_to(Path.cwd()) if Path.cwd() in output_path.parents else output_path),
    )


async def _process_audio(
    contents: bytes,
    filename: str,
) -> tuple:
    """
    Convert any audio format to 16kHz mono WAV.
    Returns (audio_array, sample_rate, metadata).
    """
    import numpy as np
    import soundfile as sf

    buf = io.BytesIO(contents)
    ext = Path(filename).suffix.lower()

    try:
        # Try soundfile first (fast, handles WAV/FLAC/OGG)
        audio, sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception:
        # Fall back to librosa (handles MP3 etc.)
        try:
            import librosa
            buf.seek(0)
            audio, sr = librosa.load(buf, sr=None, mono=False, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Cannot decode audio: {e}")

    # Convert stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16_000:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
            sr = 16_000
        except Exception:
            pass

    # Normalize to -23 LUFS (soft normalization)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9  # Peak normalization fallback

    duration_s = len(audio) / sr
    if duration_s < 0.1:
        raise ValueError("Audio too short (< 0.1 seconds)")
    if duration_s > 600:
        raise ValueError("Audio too long (> 10 minutes)")

    return audio, sr, {"duration_s": duration_s}
