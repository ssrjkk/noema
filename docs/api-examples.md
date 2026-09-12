# API Examples

Base URL `http://localhost:8000`, versioned under `/api/v1`. Auth header `X-API-Key`.
Every response carries `X-Request-ID` (send your own to correlate logs) and the
`X-RateLimit-*` headers.

## Health & readiness

```bash
curl -s localhost:8000/health       # status, version, uptime, db, llm, workers
curl -s localhost:8000/ready        # {ready: true, checks: {...}} — gate traffic on this
curl -s localhost:8000/health/infra # redis/postgres/queue/worker pool depth
curl -s localhost:8000/metrics      # Prometheus exposition
```

## Think (single solution)

```bash
curl -s -X POST localhost:8000/api/v1/think \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-key' \
  -d '{
    "title": "rate limiting middleware",
    "description": "Add a sliding-window rate limiter with burst to the FastAPI app",
    "complexity": "moderate",
    "tags": ["python", "fastapi"]
  }'
```

`POST /think/detail` returns the same result with `architecture`, `stack`,
`code_blocks`, `deployment`, and `performance_notes`/`security_notes`; pass
`{"detail": true}`-style requirements via `requirements` if your model expects them.

## Think (streaming SSE)

```bash
curl -N -X POST localhost:8000/api/v1/think/stream \
  -H 'Content-Type: application/json' \
  -d '{"title": "redis lock", "description": "Implement a distributed lock", "complexity": "simple"}'
```

Events: `step_start`, `step_end`, then `complete` with `solution_id`/`quality`/`summary`.
Cancel an in-flight task:

```bash
curl -s -X DELETE localhost:8000/api/v1/think/<task_id> \   # same id you passed above
  -H 'X-API-Key: your-key'
```

## Tasks & workers

```bash
curl -s localhost:8000/api/v1/tasks/active
curl -s localhost:8000/api/v1/workers/stats
curl -s -X POST localhost:8000/api/v1/tasks/enqueue \
  -H 'Content-Type: application/json' \
  -d '{"title": "background job", "description": "Run this without blocking the API"}'
```

## Knowledge

```bash
curl -s localhost:8000/api/v1/knowledge/stats
curl -s -X POST localhost:8000/api/v1/knowledge/search \
  -H 'Content-Type: application/json' -d '{"query": "merkle tree", "limit": 5}'
```

Run untrusted knowledge past `IngestionSanitizer.scan()` before ingestion (blocks
prompt injection and dangerous imports).

## Kernels & agents & features

```bash
curl -s localhost:8000/api/v1/kernels
curl -s localhost:8000/api/v1/agents
curl -s localhost:8000/api/v1/features
```

## Admin & diagnostics

```bash
curl -s localhost:8000/api/v1/admin/metrics
curl -s localhost:8000/api/v1/admin/tasks/history
curl -s localhost:8000/api/v1/admin/tenants/default/metrics
curl -s localhost:8000/api/v1/admin/diagnostics
curl -s localhost:8000/api/v1/admin/audit/proof/<task_id>   # Merkle proof for a task
curl -s -X POST localhost:8000/api/v1/admin/audit/verify    # verify chain integrity
curl -s localhost:8000/api/v1/admin/neurosymbolic/stats
```

## Webhooks (incident → autonomy)

```bash
# Register a webhook; the response contains the split secret
curl -s -X POST localhost:8000/api/v1/webhooks/register \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://yours/hook", "events": ["task.completed"]}'

curl -s localhost:8000/api/v1/webhooks
# Client sends events signed as: sha256=<hmac_sha256(secret, body)>
curl -s -X POST localhost:8000/api/v1/webhooks/incident \
  -H 'Content-Type: application/json' -H 'X-Signature: sha256=<sig>' -d '{...}'
```

## Experiments

```bash
curl -s -X POST localhost:8000/api/v1/experiments \
  -H 'Content-Type: application/json' -d '{"name": "reflexion-v2", "runs": 3}'
curl -s localhost:8000/api/v1/experiments
curl -s localhost:8000/api/v1/experiments/<id>/runs
curl -s localhost:8000/api/v1/experiments/<id>/runs/<run_id>
```

## Grid (distributed fleet)

```bash
curl -s localhost:8000/api/v1/grid   # node health, heartbeats, draining status
```

## Python (httpx)

```python
import httpx

with httpx.Client(base_url="http://localhost:8000") as c:
    r = c.post(
        "/api/v1/think",
        json={
            "title": "retry logic",
            "description": "Add backoff+circuit breaker",
            "complexity": "simple",
        },
        headers={"X-API-Key": "your-key", "X-Request-ID": "demo-1"},
    )
    print(r.status_code, r.json()["solution"]["summary"][:120])
```

## Errors

All failures are RFC 7807 problems, e.g.:

```json
{"type": "/errors/rate_limit", "title": "Rate limit exceeded", "status": 429,
 "detail": "...", "instance": "http://.../think"}
```

Statuses: `422` validation, `429` rate limit/quota, `504` handler timeout,
`413` payload too large (default 1 MB), `503` engine not ready.