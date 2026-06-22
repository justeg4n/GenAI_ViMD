"""
_env.py — Environment detection + path configuration
Dùng chung cho tất cả notebooks (Kaggle / Colab / Local).

Usage (Colab):
    import sys, os
    # Nếu đã clone repo về /content/project:
    sys.path.insert(0, '/content/project/training/notebooks')
    from _env import *

Usage (Kaggle):
    import sys, os
    sys.path.insert(0, '/kaggle/working/project/training/notebooks')
    from _env import *
"""
import os, sys
from pathlib import Path

# ─── Detect environment ───────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists('/kaggle/working')

# Colab: kiểm tra cả 2 cách (import trực tiếp và kiểm tra /content)
try:
    import google.colab  # noqa: F401
    IS_COLAB = True
except ImportError:
    IS_COLAB = os.path.exists('/content') and not IS_KAGGLE

IS_LOCAL  = not IS_KAGGLE and not IS_COLAB
ENV_NAME  = 'Kaggle' if IS_KAGGLE else ('Colab' if IS_COLAB else 'Local')

# ─── Kaggle ───────────────────────────────────────────────────────────────────
if IS_KAGGLE:
    KAGGLE_INPUT   = '/kaggle/input'
    VIMD_SLUG      = os.environ.get('VIMD_SLUG', 'genai-vimd')
    ARTIFACT_SLUG  = os.environ.get('ARTIFACT_SLUG', 'disa-model-artifacts')

    VIMD_PATH      = os.path.join(KAGGLE_INPUT, VIMD_SLUG, 'ViMD', 'data')
    PREV_CKPT_DIR  = os.path.join(KAGGLE_INPUT, ARTIFACT_SLUG)
    CODE_DIR       = '/kaggle/working/project'

    PROJECT_ROOT   = '/kaggle/working'
    PROCESSED_PATH = '/kaggle/working/storage/processed'
    DATASET_EXPORT = '/kaggle/working/storage/datasets/hf_dataset'
    WEIGHTS_DIR    = '/kaggle/working/storage/weights'
    CHECKPOINT_DIR = '/kaggle/working/storage/checkpoints'
    LOGS_DIR       = '/kaggle/working/storage/logs'
    EXPORTS_DIR    = '/kaggle/working/storage/exports'

    # Audio: dùng /kaggle/working/ (đủ nhanh, persist sau Save Version)
    _AUDIO_LOCAL   = None

# ─── Colab ────────────────────────────────────────────────────────────────────
elif IS_COLAB:
    DRIVE_ROOT     = '/content/drive/MyDrive/GenAI_ViMD'
    PROJECT_ROOT   = DRIVE_ROOT
    VIMD_PATH      = os.path.join(DRIVE_ROOT, 'ViMD', 'data')
    VIMD_SLUG      = 'genai-vimd'
    PREV_CKPT_DIR  = os.path.join(DRIVE_ROOT, 'storage', 'checkpoints')

    # Code dir: ưu tiên clone về /content/project, fallback về Drive
    if os.path.exists('/content/project/training/notebooks'):
        CODE_DIR = '/content/project'
    else:
        CODE_DIR = DRIVE_ROOT

    PROCESSED_PATH = os.path.join(DRIVE_ROOT, 'storage', 'processed')
    DATASET_EXPORT = os.path.join(DRIVE_ROOT, 'storage', 'datasets', 'hf_dataset')
    WEIGHTS_DIR    = os.path.join(DRIVE_ROOT, 'storage', 'weights')
    CHECKPOINT_DIR = os.path.join(DRIVE_ROOT, 'storage', 'checkpoints')
    LOGS_DIR       = os.path.join(DRIVE_ROOT, 'storage', 'logs')
    EXPORTS_DIR    = os.path.join(DRIVE_ROOT, 'storage', 'exports')

    # ⚡ Audio cache: copy từ Drive sang /content/audio_cache/ trước khi train
    # Drive I/O ~5-10x chậm hơn local disk. Xem 00_setup_colab Cell "Audio Cache".
    _AUDIO_LOCAL   = '/content/audio_cache'

# ─── Local ────────────────────────────────────────────────────────────────────
else:
    _nb = Path(os.path.abspath('.')).resolve()
    if _nb.name == 'notebooks':
        PROJECT_ROOT = str(_nb.parent.parent)
    elif _nb.name == 'training':
        PROJECT_ROOT = str(_nb.parent)
    else:
        PROJECT_ROOT = str(_nb)
    VIMD_PATH      = os.path.join(PROJECT_ROOT, 'ViMD', 'data')
    VIMD_SLUG      = 'genai-vimd'
    PREV_CKPT_DIR  = os.path.join(PROJECT_ROOT, 'storage', 'checkpoints')
    CODE_DIR       = PROJECT_ROOT
    PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'storage', 'processed')
    DATASET_EXPORT = os.path.join(PROJECT_ROOT, 'storage', 'datasets', 'hf_dataset')
    WEIGHTS_DIR    = os.path.join(PROJECT_ROOT, 'storage', 'weights')
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'storage', 'checkpoints')
    LOGS_DIR       = os.path.join(PROJECT_ROOT, 'storage', 'logs')
    EXPORTS_DIR    = os.path.join(PROJECT_ROOT, 'storage', 'exports')
    _AUDIO_LOCAL   = None

# ─── Derived paths ────────────────────────────────────────────────────────────
FINAL_DATASET  = os.path.join(DATASET_EXPORT, 'vietspeech-vimd')
FEATURES_DIR   = os.path.join(PROCESSED_PATH, 'features')
ASR_CKPT_DIR   = os.path.join(CHECKPOINT_DIR, 'asr_wav2vec2')
ASR_WEIGHT_DIR = os.path.join(WEIGHTS_DIR, 'asr_wav2vec2')

# ⚡ AUDIO_DIR:
#   - Colab: /content/audio_cache/ (local disk, ~10x nhanh hơn Drive)
#             Chạy cell "Audio Cache" trong 00_setup_colab.ipynb trước khi train!
#   - Kaggle/Local: {FINAL_DATASET}/audio/
if IS_COLAB and _AUDIO_LOCAL:
    AUDIO_DIR = _AUDIO_LOCAL
else:
    AUDIO_DIR = os.path.join(FINAL_DATASET, 'audio')

# ─── Auto-create dirs ─────────────────────────────────────────────────────────
_dirs = [PROCESSED_PATH, FEATURES_DIR, DATASET_EXPORT, WEIGHTS_DIR,
         CHECKPOINT_DIR, LOGS_DIR, EXPORTS_DIR,
         ASR_CKPT_DIR, ASR_WEIGHT_DIR,
         os.path.join(WEIGHTS_DIR, 'dialect'),
         os.path.join(WEIGHTS_DIR, 'translation'),
         os.path.join(WEIGHTS_DIR, 'xtts-v2'),
         os.path.join(CHECKPOINT_DIR, 'translation'),
         os.path.join(CHECKPOINT_DIR, 'tts'),
         os.path.join(LOGS_DIR, 'mlflow'),
         os.path.join(EXPORTS_DIR, 'onnx'),
         os.path.join(EXPORTS_DIR, 'torchscript')]
for _d in _dirs:
    Path(_d).mkdir(parents=True, exist_ok=True)

# ─── Batch size / num_workers auto-config ─────────────────────────────────────
try:
    import torch
    _vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    # T4 (Colab Free / Kaggle) = 15GB → batch 8; P100 = 16GB → batch 16; <8GB → batch 4
    BATCH_SIZE  = 16 if _vram >= 15.5 else (8 if _vram >= 13 else 4)
    NUM_WORKERS = 0   # Kaggle/Colab: multiprocessing deadlock risk
except Exception:
    BATCH_SIZE  = 4
    NUM_WORKERS = 0

__all__ = [
    'IS_KAGGLE', 'IS_COLAB', 'IS_LOCAL', 'ENV_NAME',
    'VIMD_PATH', 'VIMD_SLUG', 'PREV_CKPT_DIR', 'CODE_DIR',
    'PROJECT_ROOT', 'PROCESSED_PATH', 'DATASET_EXPORT',
    'WEIGHTS_DIR', 'CHECKPOINT_DIR', 'LOGS_DIR', 'EXPORTS_DIR',
    'FINAL_DATASET', 'AUDIO_DIR', 'FEATURES_DIR',
    'ASR_CKPT_DIR', 'ASR_WEIGHT_DIR',
    'BATCH_SIZE', 'NUM_WORKERS',
]
