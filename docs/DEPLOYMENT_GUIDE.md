# VietSpeech AI — Deployment Guide

## Quick Start (Development)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/vietspeech-ai.git
cd vietspeech-ai

# 2. Copy environment config
cp .env.example .env
# Edit .env with your settings

# 3. Start with Docker Compose (dev)
cd deployment/compose
docker-compose up --build

# 4. Verify the backend is running
curl http://localhost:8000/health
```

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required
DATABASE_URL=postgresql+asyncpg://vietspeech:secret@localhost:5432/vietspeech
REDIS_URL=redis://:secret@localhost:6379/0
SECRET_KEY=your-256-bit-secret-key-here

# Model paths (after training)
ASR_MODEL_PATH=storage/weights/asr
DIALECT_MODEL_PATH=storage/weights/dialect/dialect_model.pt
TTS_MODEL_PATH=storage/weights/xtts-v2
TRANSLATION_MODEL_PATH=storage/weights/translation

# Optional
SENTRY_DSN=https://...@sentry.io/...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
S3_BUCKET=vietspeech-models
```

---

## Production Deployment (Docker Compose)

```bash
# 1. Set up production secrets
echo "your-db-password" | docker secret create vietspeech_db_password -

# 2. Create production env file
cp .env.example deployment/compose/.env.prod
# Edit .env.prod with production values

# 3. Pre-populate model weights volume
docker volume create vietspeech_model_weights
# Copy trained models into the volume:
docker run --rm -v vietspeech_model_weights:/weights \
  -v $(pwd)/storage/weights:/source alpine \
  cp -r /source/. /weights/

# 4. Deploy production stack
cd deployment/compose
docker-compose -f docker-compose.prod.yml up -d

# 5. Verify deployment
docker-compose -f docker-compose.prod.yml ps
curl http://localhost:8000/health
```

---

## Infrastructure Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Backend CPU | 4 cores | 8 cores |
| Backend RAM | 16GB | 32GB |
| GPU (VRAM) | 8GB (RTX 3070) | 24GB (A10G) |
| Disk (models) | 50GB | 100GB |
| PostgreSQL | 2 cores, 4GB | 4 cores, 8GB |
| Redis | 1GB RAM | 2GB RAM |

---

## Monitoring

### Grafana Dashboard

Access at `http://localhost:3000` (admin/your-grafana-password).

Pre-built dashboard: `monitoring/grafana/dashboards/vietspeech_overview.json`

**Key panels**:
- API request rate & error rate
- ASR P95 latency (target ≤ 300ms)
- Dialect accuracy gauge
- GPU memory usage
- Active Celery tasks

### Prometheus Metrics

Access raw metrics at `http://localhost:8000/metrics`.

Key metrics:
- `asr_latency_ms_bucket` — ASR latency histogram
- `dialect_accuracy` — dialect classification accuracy
- `vietspeech_model_loaded` — model load status
- `celery_tasks_active` — active async tasks

---

## Database Migrations

```bash
# Initialize DB (first run)
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "add_new_table"

# Roll back
alembic downgrade -1
```

---

## SSL/TLS Setup

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Certificate auto-renews via cron — verify:
sudo certbot renew --dry-run
```

---

## Backup Strategy

```bash
# Database backup (run daily via cron)
docker exec vietspeech-postgres pg_dump -U vietspeech vietspeech \
  | gzip > backups/db_$(date +%Y%m%d).sql.gz

# Model weights backup (run after training)
aws s3 sync storage/weights/ s3://vietspeech-models/weights/ --delete

# Full system backup
make backup
```

---

## Scaling

### Horizontal Scaling (Multiple Workers)

```bash
# Scale Celery workers
docker-compose -f docker-compose.prod.yml up -d --scale worker=4
```

### GPU Scaling

Each worker requires one GPU. Add more GPU nodes and update `docker-compose.prod.yml`:
```yaml
deploy:
  replicas: 4  # 4 GPU nodes
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
```

---

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`):

1. **Lint**: `flake8`, `mypy` type checking
2. **Test**: `pytest tests/` with coverage ≥ 80%
3. **Security scan**: Bandit + Trivy container scan
4. **Build**: Docker images (backend + worker)
5. **Push**: to GitHub Container Registry (ghcr.io)
6. **Deploy**: SSH to production server, pull new images, restart services
7. **Rollback**: Automatic on health check failure

Trigger: Push to `main` or PR merge.

---

## Rollback

```bash
# Roll back to previous version
docker-compose -f docker-compose.prod.yml pull backend:previous
docker-compose -f docker-compose.prod.yml up -d --no-deps backend

# Or via Makefile
make rollback VERSION=1.2.3
```
