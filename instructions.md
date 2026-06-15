# MASTER BUILD SPECIFICATION
# VIETNAM MULTI-DIALECT SPEECH AI PLATFORM
Version: 2.0
Execution Target: Sonnet on Antigravity
Training Environment: Google Colab
Primary Objective: Maximum Accuracy + Production Readiness

---

# EXECUTION MODE

You are acting as:

- Principal ML Architect
- Senior Speech Engineer
- Senior MLOps Engineer
- Senior Backend Engineer
- Senior Mobile Engineer
- Senior Security Engineer

Generate implementation incrementally.

NEVER output placeholders.

Generate executable production-grade code.

Generate complete implementations.

Preserve backward compatibility.

Never overwrite existing authentication.

---

# PROJECT OBJECTIVE

Build a production-grade AI platform capable of:

1. Recognizing speech from all 63 provinces and cities of Vietnam.
2. Detecting province-level dialect and accent.
3. Recognizing ethnic-language influences including:
   - Ede
   - Bahnar
3. Converting speech into text.
4. Generating highly natural speech.
5. Preserving local pronunciation.
6. Preserving regional intonation.
7. Voice cloning while preserving accent.
8. Real-time inference inside mobile applications.
9. Integrating existing authentication.
10. Supporting multilingual future expansion.
11. Supporting continual retraining.

Primary KPI:

Province Accuracy:
≥ 70%

Regional Accuracy:
≥ 95%

ASR WER:
≤ 8%

CER:
≤ 4%

MOS:
≥ 4.3

Speaker Similarity:
≥ 0.90

Streaming Latency:
≤ 300 ms

---

# NON-NEGOTIABLE CONSTRAINTS

Training must run on:

Google Colab T4
Inference must support:

CPU
GPU
Mobile

Must support resume after disconnect.

Must automatically mount:

Google Drive

Must automatically save:

weights
logs
datasets
checkpoints

No manual intervention.

---

# EXISTING REPOSITORY

Repository already contains:

login
logout
authentication

DO NOT:

rewrite auth

replace auth

change routes

change session flow

Generate adapters only.

---

# SYSTEM ARCHITECTURE

Audio Input

↓

VAD

↓

Denoising

↓

Feature Extraction

↓

Dialect Detection

↓

ASR

↓

Semantic Layer

↓

Translation

↓

Prosody Modeling

↓

TTS

↓

Voice Conversion

↓

API

↓

Mobile

---

# MODEL STACK

## ASR

Primary:

Whisper Large-v3

Secondary:

PhoWhisper

Fallback:

Wav2Vec2

Implementation:

PEFT

LoRA

QLoRA

Flash Attention

Gradient Checkpointing

---

## DIALECT IDENTIFICATION

Encoder:

WavLM Large

Classifier:

Attention Pooling

Classification:

63 Provinces

Output:

{
province,
region,
confidence,
ethnic_probability
}

Loss:

CrossEntropy

Focal Loss

ArcFace

---

## ETHNIC LANGUAGE DETECTION

Targets:

Vietnamese

Ede

Bahnar

Architecture:

Shared Encoder

Adapter Heads

Output:

Language Probability

Code Switching Score

---

## PROSODY

Encoder:

GST

Variance Adaptor

Features:

Pitch

Energy

Duration

Output:

Style Embedding

---

## TTS

Primary:

XTTS-v2

Alternative:

CosyVoice2

Output:

24kHz

Controls:

speed

emotion

intonation

dialect

speaker

---

## VOICE CLONING

Input:

5–30 seconds

Output:

speaker_embedding

Constraints:

preserve dialect

preserve prosody

anti-overfit

---

# TRANSLATION

Implement:

Vietnamese ↔ Ede

Vietnamese ↔ Bahnar

Architecture:

mBART

Adapter Layers

Backtranslation

Synthetic Generation

Low Resource Learning

---

# DATA PIPELINE

Create:

dataset/

ingestion/

validation/

feature/

augmentation/

splits/

export/

---

# DATA SOURCES

Primary:

ViMD (/mnt/d/FPTU/Semester 7/DAT301m/Project/ViMD)

Additional:

Vietnamese speech datasets

Custom recordings

Synthetic data

Ethnic language corpora

---

# AUDIO STANDARDIZATION

Format:

wav

mono

16kHz

Normalization:

−23 LUFS

Remove:

duplicates

clipping

silence

invalid labels

---

# LABEL STRUCTURE

{
audio
transcript
speaker
gender
province
region
ethnicity
language
duration
pitch
energy
}

---

# DATA AUGMENTATION

Implement:

SpecAugment

Noise Injection

Room Simulation

Pitch Shift

Speed Shift

Codec Simulation

Reverberation

Frequency Masking

Time Masking

Mixup

---

# FEATURE EXTRACTION

Extract:

Mel

MFCC

F0

Pitch

Energy

Speaker Embeddings

Prosody

Store:

Parquet

Arrow

---

# TRAINING

Implement:

DeepSpeed

Distributed Training

Mixed Precision

Gradient Accumulation

Checkpointing

Early Stopping

Resume Training

EMA

Automatic LR Finder

---

# TRAINING CONFIG

batch:
auto

optimizer:
AdamW

scheduler:
Cosine

warmup:
10%

dropout:
0.1

weight_decay:
0.01

epochs:
adaptive

---

# ASR PIPELINE

train.py

evaluate.py

infer.py

export.py

Metrics:

WER

CER

Province WER

Gender WER

---

# DIALECT PIPELINE

Outputs:

province

region

confidence

Metrics:

F1

Recall

Top3 Accuracy

Confusion Matrix

---

# TTS PIPELINE

train.py

synthesize.py

clone.py

Metrics:

MOS

Speaker Similarity

Pitch Error

Latency

---

# EXPORT

Generate:

ONNX

TensorRT

TorchScript

GGUF

---

# BACKEND

FastAPI

PostgreSQL

Redis

Celery

Structure:

api/

services/

workers/

db/

middleware/

---

# API

Preserve existing:

/login
/logout
/register
/profile

Add:

POST /upload-audio

POST /transcribe

POST /detect-accent

POST /generate-speech

POST /voice-clone

POST /translate

GET /health

GET /metrics

---

# STORAGE

audio/

datasets/

checkpoints/

weights/

logs/

exports/

---

# SECURITY

JWT

Refresh Tokens

RBAC

Encryption

Rate Limiting

Secrets Manager

Audit Logs

CSRF

CORS

---

# MOBILE

Flutter

Android

iOS

Capabilities:

streaming

offline

cached inference

background uploads

quantized inference

---

# DEPLOYMENT

Docker

Docker Compose

Generate:

deployment

service

ingress

autoscaling

monitoring

---

# OBSERVABILITY

Prometheus

Grafana

MLflow

OpenTelemetry

Sentry

---

# TESTING

Generate:

Unit

Integration

Regression

Load

Security

Benchmark

Coverage:

≥90%

---

# CI/CD

GitHub Actions

Pipeline:

lint

test

train

build

deploy

rollback

---

# DOCUMENTATION

Generate:

README

Architecture

Training Guide

API Guide

Deployment Guide

Mobile Guide

Troubleshooting

---

# OUTPUT ORDER

Generate in this exact order:

Phase 0
Repository Inspection

Phase 1
Dataset

Phase 2
Training

Phase 3
Models

Phase 4
Backend

Phase 5
Mobile

Phase 6
Deployment

Phase 7
Testing

Phase 8
Documentation

Phase 9
Final Validation

Stop after each phase.

Provide:

tree

commands

files

verification

Do not continue until current phase passes.

END