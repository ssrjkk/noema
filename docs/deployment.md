# Deployment

## Docker

### Build

```bash
docker build -t noema .
```

Multi-stage build: builder stage installs dependencies, runtime stage runs as non-root `noema` user. Exposes ports `8000` (API) and `9090` (metrics).

### Run

```bash
docker run -d \
  --name noema \
  -p 8000:8000 \
  -p 9090:9090 \
  -e NOEMA_DB_URL=postgresql+asyncpg://noema:noema@host.docker.internal:5432/noema \
  -e NOEMA_REDIS_URL=redis://host.docker.internal:6379/0 \
  noema
```

## Docker Compose

Full stack with PostgreSQL 16 and Redis 7:

```bash
docker compose up -d
```

This starts:
- `noema` — the API server
- `postgres` — PostgreSQL 16 with health check
- `redis` — Redis 7 Alpine with health check

All services are on a dedicated `noema-network` bridge network.

## Monitoring Stack

```bash
docker compose -f deploy/monitoring/docker-compose-monitoring.yml up -d
```

Includes:
- **Prometheus** — metrics collection (config: `deploy/monitoring/prometheus-config.yml`)
- **Grafana** — dashboards (import `deploy/monitoring/grafana-dashboard.json`)
- **Loki + Promtail** — log aggregation (configs: `loki-config.yml`, `promtail-config.yml`)
- **Alerting** — rule-based alerts (`deploy/monitoring/prometheus-rules.yml`)

## Kubernetes

Kustomize manifests in `deploy/k8s/`:

```bash
kubectl apply -k deploy/k8s/
```

## Helm Chart

Helm chart in `deploy/helm/noema/`:

```bash
helm install noema deploy/helm/noema/
```

## Terraform

AWS infrastructure as code in `deploy/terraform/`:

| File | Purpose |
|------|---------|
| `main.tf` | Root module |
| `eks.tf` | EKS cluster definition |
| `rds.tf` | PostgreSQL RDS instance |
| `redis.tf` | ElastiCache Redis cluster |
| `helm.tf` | Helm chart deployment |
| `variables.tf` | Input variables |
| `outputs.tf` | Output values |

```bash
cd deploy/terraform
terraform init
terraform plan
terraform apply
```

## Postman Collection

Import from `deploy/postman/`:
- `Noema.postman_collection.json` — API endpoints
- `Noema-Environment.json` — environment variables
- `noema-openapi.yaml` — OpenAPI 3.0 spec

## Health Checks

All deployments should configure the built-in health endpoints:

- `/health` — liveness probe (always 200 when alive)
- `/ready` — readiness probe (checks LLM provider and worker pool)
- `/health/infra` — infrastructure health (Redis, PostgreSQL degradation status)

Default Docker health check interval: 30s, timeout: 5s, retries: 3, start period: 10s.
