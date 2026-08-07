.PHONY: install install-dev install-db install-full lint typecheck encoding-check test test-coverage \
        test-benchmark migrate migrate-create grpc docker-build docker-run clean pre-commit setup-hooks \
        ci security-check locust-run docs docs-serve help

PROJECT := noema
PYTHON := python
LOCUST := locust
ALEMBIC := alembic

help:
	@echo "$(PROJECT) Makefile"
	@echo ""
	@echo "Development:"
	@echo "  make install             Install base dependencies"
	@echo "  make install-dev         Install dev dependencies (pytest, ruff, mypy)"
	@echo "  make install-db          Install DB dependencies (SQLAlchemy, asyncpg, alembic)"
	@echo "  make install-full        Install all dependencies"
	@echo "  make lint                Run ruff linter + formatter"
	@echo "  make typecheck           Run mypy type checker"
	@echo "  make encoding-check      Check text files for encoding corruption"
	@echo "  make test                Run tests (excluding benchmarks)"
	@echo "  make test-coverage       Run tests with coverage report"
	@echo "  make test-benchmark      Run benchmarks"
	@echo "  make pre-commit          Run pre-commit hooks on all files"
	@echo "  make setup-hooks         Install pre-commit hooks"
	@echo ""
	@echo "Database:"
	@echo "  make migrate             Run alembic migrations up"
	@echo "  make migrate-create      Create new migration (msg=description)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build        Build Docker image"
	@echo "  make docker-run          Run with docker-compose"
	@echo ""
	@echo "Quality:"
	@echo "  make security-check      Run bandit + safety + pip-audit"
	@echo "  make locust-run          Run load tests"
	@echo ""
	@echo "Docs:"
	@echo "  make docs                Build documentation"
	@echo "  make docs-serve          Serve documentation locally"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean               Clean cache, build artifacts, __pycache__"
	@echo "  make ci                  Full CI pipeline (lint -> typecheck -> test -> security)"

# ─── Install ────────────────────────────────────────────────────────────

install:
	pip install --upgrade pip
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-db:
	pip install -e ".[db]"

install-full:
	pip install -e ".[dev,db,full,sentry]"

# ─── Lint & Type ────────────────────────────────────────────────────────

lint:
	ruff check noema/ tests/
	ruff format --check noema/ tests/

typecheck:
	mypy noema/ --ignore-missing-imports --no-error-summary

# ─── Encoding ─────────────────────────────────────────────────────────────

encoding-check:
	python scripts/check_encoding.py

# ─── Tests ──────────────────────────────────────────────────────────────

test:
	python -m pytest tests/ --ignore=tests/test_benchmarks.py -v --tb=short -W ignore::DeprecationWarning

test-coverage:
	python -m pytest tests/ --ignore=tests/test_benchmarks.py --tb=short \
		--cov=noema --cov-report=term --cov-report=html --cov-report=xml \
		-W ignore::DeprecationWarning

test-benchmark:
	python -m pytest tests/test_benchmarks.py --benchmark-enable --benchmark-warmup=on

# ─── Pre-commit ─────────────────────────────────────────────────────────

pre-commit:
	pre-commit run --all-files

setup-hooks:
	pre-commit install

# ─── Database ───────────────────────────────────────────────────────────

migrate:
	$(ALEMBIC) upgrade head

migrate-create:
ifndef msg
	$(error Usage: make migrate-create msg="description of migration")
endif
	$(ALEMBIC) revision --autogenerate -m "$(msg)"

# ─── gRPC ───────────────────────────────────────────────────────────────

grpc:
	python scripts/gen_grpc_stubs.py

# ─── Docker ─────────────────────────────────────────────────────────────

docker-build:
	docker build -t $(PROJECT):latest .

docker-run:
	docker-compose up --build -d

# ─── Security ───────────────────────────────────────────────────────────

security-check:
	bandit -r noema/ -f json -o bandit-report.json || true
	safety check --json --output safety-report.json || true
	pip-audit --format=json --output=pip-audit-report.json || true

# ─── Load Testing ───────────────────────────────────────────────────────

locust-run:
	$(LOCUST) --config tests/locust.conf

# ─── Documentation ──────────────────────────────────────────────────────

docs:
	mkdocs build

docs-serve:
	mkdocs serve

# ─── CI full pipeline ──────────────────────────────────────────────────

ci: lint typecheck encoding-check test security-check

# ─── Clean ──────────────────────────────────────────────────────────────

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf htmlcov/ coverage.xml .coverage
	rm -rf site/
	rm -rf bandit-report.json safety-report.json pip-audit-report.json
	rm -rf tests/loadtest_results* tests/loadtest_report.html
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Cleaned."
