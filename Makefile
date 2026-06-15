.PHONY: help install install-dev setup env backend worker test lint format clean colab-push

PYTHON := python3
PIP    := pip
PROJECT_DIR := $(shell pwd)
VENV   := .venv
BIN    := $(VENV)/bin

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ─── Setup ────────────────────────────────────────────────────────────────────

install:  ## Install production dependencies
	$(PIP) install -r requirements-inference.txt

install-dev:  ## Install all dependencies including training
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	pre-commit install

env:  ## Copy .env.example to .env
	@test -f .env || (cp .env.example .env && echo "Created .env — edit with your secrets")

setup: env install-dev  ## Full local setup

# ─── Backend ──────────────────────────────────────────────────────────────────

backend:  ## Start FastAPI dev server
	uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

worker:  ## Start Celery worker
	celery -A backend.workers.celery_app worker --loglevel=info -Q default,audio,inference

beat:  ## Start Celery beat scheduler
	celery -A backend.workers.celery_app beat --loglevel=info

flower:  ## Start Celery Flower monitor
	celery -A backend.workers.celery_app flower --port=5555

# ─── Database ─────────────────────────────────────────────────────────────────

db-init:  ## Initialize the database
	alembic upgrade head

db-migrate:  ## Create a new migration
	alembic revision --autogenerate -m "$(msg)"

db-downgrade:  ## Rollback last migration
	alembic downgrade -1

# ─── Testing ──────────────────────────────────────────────────────────────────

test:  ## Run full test suite
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing

test-unit:  ## Run unit tests only
	pytest tests/unit/ -v

test-integration:  ## Run integration tests only
	pytest tests/integration/ -v

test-load:  ## Run load tests with Locust
	locust -f tests/load/locustfile.py --headless -u 50 -r 10 --run-time 60s --host http://localhost:8000

test-security:  ## Run security tests
	pytest tests/security/ -v

benchmark:  ## Run ASR/dialect benchmarks
	pytest tests/benchmarks/ -v -s

# ─── Code Quality ─────────────────────────────────────────────────────────────

lint:  ## Run ruff linter
	ruff check .

format:  ## Format with black + ruff
	black . --line-length 100
	ruff check . --fix

typecheck:  ## Run mypy type checker
	mypy backend/ models/ dataset/ --ignore-missing-imports

# ─── Docker ───────────────────────────────────────────────────────────────────

docker-build:  ## Build all Docker images
	docker compose -f deployment/compose/docker-compose.yml build

docker-up:  ## Start all services
	docker compose -f deployment/compose/docker-compose.yml up -d

docker-down:  ## Stop all services
	docker compose -f deployment/compose/docker-compose.yml down

docker-logs:  ## Follow logs
	docker compose -f deployment/compose/docker-compose.yml logs -f

docker-prod:  ## Start production stack
	docker compose -f deployment/compose/docker-compose.prod.yml up -d

# ─── MLflow ───────────────────────────────────────────────────────────────────

mlflow:  ## Start MLflow tracking server
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///storage/logs/mlflow.db --default-artifact-root storage/logs/mlflow-artifacts

# ─── Data Pipeline ────────────────────────────────────────────────────────────

data-ingest:  ## Run ViMD ingestion
	$(PYTHON) -m dataset.ingestion.vimd_loader

data-validate:  ## Validate dataset
	$(PYTHON) -m dataset.validation.audio_validator
	$(PYTHON) -m dataset.validation.label_validator

data-features:  ## Extract features
	$(PYTHON) -m dataset.feature.extractor

data-augment:  ## Run augmentation
	$(PYTHON) -m dataset.augmentation.audio_augment

data-pipeline:  ## Run full data pipeline
	$(MAKE) data-ingest data-validate data-features

# ─── Model Export ─────────────────────────────────────────────────────────────

export-onnx:  ## Export models to ONNX
	$(PYTHON) -m models.export.onnx_exporter

# ─── Cleanup ──────────────────────────────────────────────────────────────────

clean:  ## Remove build artifacts, caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage dist/ build/ *.egg-info/

clean-models:  ## Remove cached model weights (DANGER)
	@read -p "Delete all cached weights? [y/N] " confirm && [ "$$confirm" = "y" ] && rm -rf storage/weights/* || echo "Cancelled"
