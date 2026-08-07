"""Ядро DevOps — инфраструктура, CI/CD, мониторинг."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from noema.kernels.base import BaseKernel
from noema.logging import get_logger

if TYPE_CHECKING:
    from noema.core.types import Task

logger = get_logger(__name__)


class DevOpsKernel(BaseKernel):
    """Ядро DevOps и инфраструктуры."""

    @property
    def name(self) -> str:
        return "devops"

    @property
    def description(self) -> str:
        return "CI/CD, контейнеризация, оркестрация, мониторинг, IaC"

    async def execute(self, task: Task, **kwargs) -> dict[str, Any]:
        tags = {t.lower() for t in task.tags}

        return {
            "type": "devops",
            "containerization": self._containerization(task, tags),
            "ci_cd": self._ci_cd(task, tags),
            "orchestration": self._orchestration(task, tags),
            "monitoring": self._monitoring(task, tags),
            "iac": self._infrastructure_as_code(task, tags),
            "secrets": self._secrets_management(task),
            "scripts": self._generate_scripts(task, tags),
            "_confidence": 0.76,
        }

    def _containerization(self, task: Task, tags: set[str]) -> dict[str, Any]:
        multi_stage = """# Build stage
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Runtime stage
FROM python:3.12-slim AS runtime
RUN groupadd -r app && useradd -r -g app app
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
RUN chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        return {
            "approach": "multi-stage-build",
            "dockerfile": multi_stage,
            "best_practices": [
                "Non-root user",
                "Multi-stage build",
                "Health check",
                ".dockerignore",
                "Minimal base image (slim/alpine)",
                "Layer caching optimization",
            ],
        }

    def _ci_cd(self, task: Task, tags: set[str]) -> dict[str, Any]:
        github_actions = """name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: pytest --cov --cov-report=xml
      - uses: codecov/codecov-action@v3

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: kubectl apply -f k8s/
"""
        return {
            "provider": "GitHub Actions",
            "workflow": github_actions,
            "stages": ["lint", "test", "build", "push", "deploy"],
            "environments": ["staging", "production"],
        }

    def _orchestration(self, task: Task, tags: set[str]) -> dict[str, Any]:
        k8s_deployment = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
  labels:
    app: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: app
        image: ghcr.io/org/app:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: app-service
spec:
  selector:
    app: myapp
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
"""
        return {
            "platform": "Kubernetes",
            "manifests": k8s_deployment,
            "autoscaling": {"min": 2, "max": 10, "cpu_target": 70},
            "probes": {"liveness": "/health", "readiness": "/health"},
        }

    def _monitoring(self, task: Task, tags: set[str]) -> dict[str, Any]:
        return {
            "metrics": {
                "tool": "Prometheus",
                "dashboard": "Grafana",
                "key_metrics": [
                    "request_duration_seconds",
                    "requests_total",
                    "cpu_usage_percent",
                    "memory_usage_bytes",
                    "db_connection_pool_size",
                ],
            },
            "logging": {
                "tool": "Loki + Promtail",
                "format": "structured JSON",
                "levels": ["info", "warning", "error"],
            },
            "tracing": {
                "tool": "OpenTelemetry + Tempo",
                "sampling": "probabilistic 10%",
            },
            "alerting": {
                "tool": "Alertmanager",
                "rules": [
                    "error_rate > 5% for 5m",
                    "p99_latency > 2s for 5m",
                    "cpu > 80% for 10m",
                    "memory > 85% for 5m",
                ],
            },
        }

    def _infrastructure_as_code(self, task: Task, tags: set[str]) -> dict[str, Any]:
        return {
            "tool": "Terraform",
            "modules": [
                "vpc",
                "eks-cluster",
                "rds",
                "redis",
                "s3",
                "iam",
            ],
            "state_backend": "S3 + DynamoDB locking",
            "environments": ["dev", "staging", "production"],
        }

    def _secrets_management(self, task: Task) -> dict[str, Any]:
        return {
            "tool": "AWS Secrets Manager / HashiCorp Vault",
            "rotation": "90 days",
            "pattern": "Inject via K8s External Secrets Operator",
        }

    def _generate_scripts(self, task: Task, tags: set[str]) -> list[dict[str, str]]:
        return [
            {
                "name": "Makefile",
                "content": """.PHONY: dev test lint build deploy

dev:
\tpython -m uvicorn main:app --reload

test:
\tpytest tests/ -v --cov

lint:
\truff check .
\tmypy .

build:
\tdocker build -t app:latest .

deploy:
\tkubectl apply -f k8s/
""",
            },
            {
                "name": "scripts/health-check.sh",
                "content": """#!/bin/bash
set -e
curl -sf http://localhost:8000/health || exit 1
echo "Health check passed"
""",
            },
        ]
