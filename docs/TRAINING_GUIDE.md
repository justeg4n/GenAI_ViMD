# VietSpeech AI — Training Guide

## Overview

All model training runs on Google Colab (T4 GPU, 16GB VRAM) via Jupyter notebooks.
After training, models are saved to Google Drive and can be downloaded for local inference.

---

## Prerequisites

1. **Google Account** with Google Drive (~50GB free space recommended)
2. **Google Colab** (free or Pro for A100/better GPU)
3. **ViMD Dataset** uploaded to Google Drive at `MyDrive/VietSpeech/ViMD/`

### Upload ViMD to Google Drive

```bash
# From your local machine (assuming ViMD is at the project path):
# Option A: Use rclone (recommended for large datasets)
rclone copy "Project/ViMD/" gdrive:"VietSpeech/ViMD/" --progress

# Option B: Google Drive web interface (drag & drop)
# Navigate to drive.google.com → New Folder "VietSpeech/ViMD" → Upload
```

---

## Training Sequence

Run notebooks **in order**. Each builds on the previous.

### Step 0: Environment Setup

**Notebook**: `training/notebooks/00_setup_colab.ipynb`

1. Upload `00_setup_colab.ipynb` to Colab
2. Run all cells — installs packages, clones project, mounts Drive
3. Verify: `✓ Environment ready` in last cell output

### Step 1: Dataset Preparation (~2-4 hours)

**Notebook**: `training/notebooks/01_dataset_prep.ipynb`

- Loads ViMD parquet shards
- Validates audio quality
- Extracts Mel/MFCC/F0 features
- Applies augmentation (2x dataset)
- Exports HuggingFace Dataset to `VietSpeech/hf_dataset/`

**Expected output**:
```
Loaded 19,000 utterances from ViMD
Valid samples: 18,750 / 19,000
Features extracted for 18,750 samples
Combined dataset: 37,500 samples (orig + augmented)
Train: 30,000 | Val: 3,750 | Test: 3,750
```

### Step 2: Dialect Identification (~6-10 hours on T4)

**Notebook**: `training/notebooks/02_dialect_identification.ipynb`

- Fine-tunes WavLM-Large for 63-province classification
- Target: Province Acc ≥ 70%, Regional Acc ≥ 95%
- Saves to `VietSpeech/weights/dialect/dialect_model.pt`

**Use locally after training**:
```bash
cp "VietSpeech/weights/dialect/dialect_model.pt" storage/weights/dialect/
cp "VietSpeech/weights/dialect/label_maps.json" storage/weights/dialect/
```

### Step 3: ASR Whisper Fine-tuning (~8-12 hours on T4)

**Notebook**: `training/notebooks/03_asr_whisper.ipynb`

- Fine-tunes Whisper-large-v3 with QLoRA (4-bit)
- Target: WER ≤ 8%
- Saves merged model to `VietSpeech/weights/asr/`

**Memory note**: Requires ~14GB VRAM. On T4 (16GB), use batch_size=4 with grad_acc=4.

**Use locally after training**:
```bash
# Copy entire folder (several GB)
cp -r "VietSpeech/weights/asr/" storage/weights/asr/
```

### Step 4: PhoWhisper Fine-tuning (~4-6 hours on T4)

**Notebook**: `training/notebooks/04_asr_phowhisper.ipynb`

- Trains Vietnamese-specialized fallback ASR
- Smaller and faster than Whisper-large-v3

### Step 5: Translation mBART (~4-8 hours on T4)

**Notebook**: `training/notebooks/05_translation_mbart.ipynb`

> ⚠️ **Note**: The provided demo corpus is tiny (5 pairs). For production accuracy,
> obtain the full Ede-Vietnamese corpus (~10,000 pairs) from the paper authors
> (IJERT 2015: "Building a Vietnamese-Ede Machine Translation").

- Saves to `VietSpeech/weights/translation/`

### Step 6: TTS XTTS-v2 (~12-24 hours, recommended on A100)

**Notebook**: `training/notebooks/06_tts_xtts.ipynb`

> ⚠️ This is the most resource-intensive step. Consider **Colab Pro** with A100 (40GB).

- Fine-tunes XTTS-v2 on ViMD audio
- Extracts dialect speaker embeddings per province
- Saves to `VietSpeech/weights/xtts-v2/`

### Step 7: Voice Clone Evaluation

**Notebook**: `training/notebooks/07_voice_clone.ipynb`

- Validates speaker similarity ≥ 0.90
- Generates sample cloned voices

### Step 8: ONNX Export

**Notebook**: `training/notebooks/08_export_onnx.ipynb`

- Exports all models to ONNX + TorchScript
- Required for efficient local CPU inference

---

## Downloading Models from Colab

### Method A: Google Drive (Recommended)

Models are saved directly to your Google Drive. Access them at:
`MyDrive/VietSpeech/weights/`

Mount Drive locally on your machine:
```bash
# Install Google Drive sync app (google.com/drive/download/)
# Or use rclone:
rclone copy gdrive:"VietSpeech/weights/" storage/weights/ --progress
```

### Method B: Direct Download from Colab

In any notebook cell:
```python
# Zip the models for download (max 2GB per download)
!zip -r /content/dialect_model.zip /content/drive/MyDrive/VietSpeech/weights/dialect/
from google.colab import files
files.download('/content/dialect_model.zip')
```

### Method C: HuggingFace Hub (Optional)

Push to your HuggingFace Hub repo for easy sharing:
```python
# In notebook 08 or standalone:
from transformers import AutoModelForSequenceClassification
model.push_to_hub("your-username/vietspeech-dialect-classifier")
```

---

## Running Trained Models Locally

After downloading models:

```bash
# 1. Extract to storage/weights/
unzip models.zip -d storage/weights/

# Directory structure should be:
# storage/weights/
# ├── asr/           ← Whisper fine-tuned
# ├── dialect/       ← WavLM dialect classifier
# ├── xtts-v2/       ← XTTS TTS model
# └── translation/   ← mBART translation model

# 2. Set environment variables in .env:
ASR_MODEL_PATH=storage/weights/asr
DIALECT_MODEL_PATH=storage/weights/dialect/dialect_model.pt
TTS_MODEL_PATH=storage/weights/xtts-v2

# 3. Start the backend (will auto-load models):
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## Training on AWS (Alternative to Colab)

See `docs/TRAINING_GUIDE_AWS.md` for instructions on running training on AWS SageMaker or EC2 GPU instances.

---

## Resuming Training

All notebooks save checkpoints every 500 steps. To resume:

```python
# In any training notebook, find the trainer call and add:
trainer.train(resume_from_checkpoint=True)
# Or specify exact checkpoint:
trainer.train(resume_from_checkpoint=f"{CHECKPOINT_DIR}/checkpoint-2000")
```

---

## MLflow Experiment Tracking

Training metrics are logged to MLflow. Start the MLflow server to view:

```bash
# Run MLflow server locally:
docker-compose -f monitoring/mlflow/docker-compose.mlflow.yml up

# Open: http://localhost:5000
```

Or in Colab, the metrics are stored at `VietSpeech/mlflow/` on your Drive.
