# VietSpeech AI — GitHub & Cloud Training Guide

---

## Part 1: Setting Up GitHub Repository

### Step 1: Initialize Git

```bash
cd "/mnt/d/FPTU/Semester 7/DAT301m/Project"

# Initialize git repo (if not already)
git init

# Verify .gitignore is in place (already exists in project)
cat .gitignore
```

### Step 2: Create GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `vietspeech-ai` (or your preferred name)
3. Visibility: **Private** (recommended — model code & training notebooks)
4. **DO NOT** initialize with README (we already have one)
5. Click **Create repository**

### Step 3: Push Code

```bash
# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/vietspeech-ai.git

# Stage all files
git add .

# Verify what will be committed (check .gitignore is excluding large files)
git status

# Initial commit
git commit -m "feat: initial VietSpeech AI platform scaffold

- Dataset pipeline (ViMD ingestion, validation, features, augmentation)
- Training notebooks (00-08): Dialect/ASR/TTS/Translation/Export
- Backend: FastAPI + Celery + PostgreSQL + Redis
- Models: WavLM dialect, Whisper ASR, XTTS-v2 TTS, mBART translation
- Mobile: Flutter app (ASR, dialect, TTS, translation screens)
- Deployment: Docker + GitHub Actions CI/CD
- Monitoring: Prometheus + Grafana + MLflow + OpenTelemetry
- Tests: unit, integration, regression, load, security, benchmarks
- Docs: architecture, API, training, deployment, mobile guides"

# Push
git branch -M main
git push -u origin main
```

### Step 4: Protect main branch (recommended)

On GitHub → Settings → Branches → Add rule:
- Branch name: `main`
- ✅ Require pull request reviews
- ✅ Require status checks (CI must pass)
- ✅ Restrict pushes

---

## Part 2: Training on AWS (Alternative to Google Colab)

AWS offers GPU instances that can train continuously without session timeouts.

### Option A: AWS SageMaker (Easiest)

SageMaker handles infrastructure, storage, and job lifecycle automatically.

#### Setup

```bash
# Install AWS CLI and SageMaker SDK
pip install boto3 sagemaker

# Configure credentials
aws configure
# Enter: Access Key ID, Secret Access Key, Region (e.g. us-east-1)
```

#### Create training script

Create `training/aws/train_dialect.py`:

```python
"""SageMaker training entry point for dialect identification."""
import argparse
import os
import sys

# SageMaker passes hyperparams as CLI args
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-4)
# SageMaker special args
parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './data'))
args = parser.parse_args()

# Run training (same logic as notebook 02)
sys.path.insert(0, '/opt/ml/code')
# ... import and run training code ...
```

#### Launch training job

```python
# training/aws/launch_sagemaker.py
import sagemaker
from sagemaker.pytorch import PyTorch

session = sagemaker.Session()
role = "arn:aws:iam::YOUR_ACCOUNT:role/SageMakerRole"

estimator = PyTorch(
    entry_point="train_dialect.py",
    source_dir="training/aws/",
    role=role,
    instance_type="ml.p3.2xlarge",  # V100 GPU, ~$3.06/hr
    # OR: "ml.g4dn.xlarge"          # T4 GPU, ~$0.74/hr (closest to Colab)
    # OR: "ml.p3.8xlarge"           # 4x V100, ~$12.24/hr (fastest)
    instance_count=1,
    framework_version="2.1",
    py_version="py310",
    hyperparameters={
        "epochs": 30,
        "batch-size": 8,
        "lr": 1e-4,
    },
    output_path="s3://your-bucket/vietspeech/output/",
    checkpoint_s3_uri="s3://your-bucket/vietspeech/checkpoints/",
    use_spot_instances=True,       # ← Use Spot for 70% cost savings
    max_wait=86400,                # 24 hour max wait
    max_run=43200,                 # 12 hour max training run
)

estimator.fit({
    "training": "s3://your-bucket/vietspeech/hf_dataset/",
})

print(f"Training job: {estimator.latest_training_job.name}")
print(f"Model artifacts: {estimator.model_data}")
```

#### Download trained model locally

```bash
# After training completes, download from S3
aws s3 sync s3://your-bucket/vietspeech/output/ storage/weights/

# Or use SageMaker SDK:
# estimator.model_data → S3 URI of model.tar.gz
aws s3 cp s3://your-bucket/vietspeech/output/model.tar.gz /tmp/
tar -xzf /tmp/model.tar.gz -C storage/weights/dialect/
```

---

### Option B: AWS EC2 GPU Instance (Most Control)

Best for long multi-day training with full environment control.

#### Recommended instances

| Instance | GPU | VRAM | On-demand | Spot (~70% off) |
|----------|-----|------|-----------|-----------------|
| `g4dn.xlarge` | T4 | 16GB | $0.526/hr | ~$0.16/hr |
| `g4dn.2xlarge` | T4 | 16GB | $0.752/hr | ~$0.23/hr |
| `p3.2xlarge` | V100 | 16GB | $3.06/hr | ~$0.92/hr |
| `p3.8xlarge` | 4x V100 | 64GB | $12.24/hr | ~$3.67/hr |

> 💡 **Recommendation**: `g4dn.xlarge` (T4, same as Colab) at Spot price ~$0.16/hr
> Full training suite: ~48 hours × $0.16 = **~$8 total**

#### Step-by-step EC2 Setup

```bash
# 1. Launch EC2 instance (use Deep Learning AMI — CUDA pre-installed)
aws ec2 run-instances \
  --image-id ami-0f4c32b5b49e73765 \  # Deep Learning AMI Ubuntu 22.04
  --instance-type g4dn.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxx \
  --iam-instance-profile Name=EC2S3Access \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"0.25"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200}}]'

# 2. Get instance IP
INSTANCE_IP=$(aws ec2 describe-instances --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

# 3. SSH into instance
ssh -i your-key.pem ubuntu@$INSTANCE_IP

# === On the EC2 instance: ===

# 4. Clone your GitHub repo
git clone https://github.com/YOUR_USERNAME/vietspeech-ai.git
cd vietspeech-ai

# 5. Install dependencies
pip install -r requirements.txt

# 6. Download ViMD dataset from S3 (upload ViMD to S3 first)
aws s3 sync s3://your-bucket/ViMD/ ViMD/ --no-sign-request

# 7. Run training (convert notebook to script or use nbconvert)
pip install jupyter nbconvert
jupyter nbconvert --to script training/notebooks/02_dialect_identification.ipynb
python training/notebooks/02_dialect_identification.py

# 8. Save model to S3 when done
aws s3 sync storage/weights/ s3://your-bucket/vietspeech/weights/
```

#### Keep training running after SSH disconnect

```bash
# Use tmux to keep session alive
tmux new -s training

# Inside tmux: run training
python training/notebooks/02_dialect_identification.py 2>&1 | tee logs/dialect_train.log

# Detach (keep running): Ctrl+B then D
# Reattach later: tmux attach -t training
```

---

### Option C: AWS Spot Instance with Auto-Resume (Production-Grade)

For XTTS-v2 (24+ hours of training), use spot instances with automatic checkpoint resume:

```bash
# Create a spot interruption handler that saves checkpoints:
# training/aws/spot_handler.py

import boto3, signal, sys

def handle_interruption(signum, frame):
    """Save checkpoint when spot instance is interrupted."""
    print("Spot interruption detected! Saving checkpoint...")
    # trainer.save_checkpoint() # Save HF Trainer checkpoint
    # upload to S3
    import subprocess
    subprocess.run(["aws", "s3", "sync", "checkpoints/", "s3://your-bucket/checkpoints/"])
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_interruption)
```

---

## Part 3: Downloading Models & Running Locally

After training on AWS:

```bash
# Download all trained weights
aws s3 sync s3://your-bucket/vietspeech/weights/ storage/weights/

# Verify files
ls storage/weights/
# Expected: asr/  dialect/  xtts-v2/  translation/

# Start backend — models load automatically
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Test
curl http://localhost:8000/health
```

The backend's lazy-loading system auto-detects model files in `storage/weights/` — no code changes needed after downloading trained models.

---

## Part 4: Cost Estimate

| Training Job | Instance | Hours | Spot Cost |
|-------------|----------|-------|-----------|
| Dataset prep | g4dn.xlarge | 3h | $0.48 |
| Dialect (WavLM) | g4dn.xlarge | 8h | $1.28 |
| ASR Whisper (QLoRA) | g4dn.xlarge | 10h | $1.60 |
| PhoWhisper | g4dn.xlarge | 5h | $0.80 |
| Translation mBART | g4dn.xlarge | 6h | $0.96 |
| TTS XTTS-v2 | p3.2xlarge | 20h | $18.40 |
| ONNX Export | g4dn.xlarge | 1h | $0.16 |
| **Total** | | ~53h | **~$23.68** |

> 💡 Use S3 for dataset storage ($0.023/GB/month) and model storage.
> 100GB of models + dataset ≈ $2.30/month on S3.
