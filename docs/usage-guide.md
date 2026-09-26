# Usage Guide

Complete reference for using Noema — from first demo to production deployment.

---

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
  - [think — generate a solution](#think--generate-a-solution)
  - [pipeline — run a kernel pipeline](#pipeline--run-a-kernel-pipeline)
  - [serve — launch the API server](#serve--launch-the-api-server)
  - [knowledge — search and manage knowledge](#knowledge--search-and-manage-knowledge)
  - [memory — query episodic memory](#memory--query-episodic-memory)
  - [graph — knowledge graph operations](#graph--knowledge-graph-operations)
  - [modules — domain modules](#modules--domain-modules)
  - [kernels & agents — discovery](#kernels--agents--discovery)
  - [evolve — self-evolution cycle](#evolve--self-evolution-cycle)
  - [grid — distributed federation](#grid--distributed-federation)
  - [audit — Merkle proof & verification](#audit--merkle-proof--verification)
  - [health — system checks](#health--system-checks)
- [API Server](#api-server)
  - [Endpoints](#endpoints)
  - [Authentication](#authentication)
  - [Streaming (SSE)](#streaming-sse)
  - [Async tasks](#async-tasks)
- [Python API (Library)](#python-api-library)
- [Common Workflows](#common-workflows)
  - [Generate and scaffold a project](#generate-and-scaffold-a-project)
  - [Run benchmarks](#run-benchmarks)
  - [Autonomous self-healing](#autonomous-self-healing)
- [Troubleshooting](#troubleshooting)

---

## Installation

```bash
git clone https://github.com/ssrjkk/noema && cd noema

# Minimal (demos + CLI)
pip install -e .

# Full (Z3 verification, vector search, all LLM providers)
pip install -e ".[full]"

# Everything (dev tools, DB, gRPC, vault)
pip install -e ".[dev,full,db,grpc,vault]"
```

**Python 3.11+** (3.12 recommended).

Verify:

```bash
python -m noema --help        # CLI entry point
python demo.py                # 12 live demos — no API keys required
```

If you see `=== ALL DEMOS COMPLETE ===` — installation is good.

---

## Configuration

Noema works out of the box with a **fallback provider** (template responses, no LLM). To use a real LLM, set one of:

### OpenAI

```bash
export NOEMA_LLM__PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

### Anthropic

```bash
export NOEMA_LLM__PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

### Ollama (local)

```bash
export NOEMA_LLM__PROVIDER=ollama
# Start Ollama first:
ollama serve
ollama pull llama3.1
```

### Fallback (no keys needed)

```bash
export NOEMA_LLM__PROVIDER=fallback
```

Returns template-based solutions. Useful for demos, CI, and testing the pipeline without burning tokens.

Full configuration reference: [configuration.md](configuration.md)

---

## CLI Reference

All commands support `--help` for full option lists.

```bash
noema <command> --help
python -m noema <command> --help    # alternative if `noema` is not in PATH
```

---

### `think` — generate a solution

The core command. Takes a task title, runs it through the neurosymbolic pipeline (LLM proposal → Z3 verification → AST check → sandbox execution), and returns a structured solution.

```bash
# Basic
noema think "Auth service on FastAPI"

# With tags and complexity
noema think "Real-time Chat App" \
  --tags "python,websocket,redis" \
  --complexity complex \
  --output full

# Scaffold generated files to disk
noema think "CRUD API" --scaffold --scaffold-dir ./my-project

# Use a specific provider/model for one command
noema think "Event bus" --llm openai --model gpt-4o

# JSON output (for scripting)
noema think "Database schema" --output json
```

**Options:**

| Flag | Description | Default |
|---|---|---|
| `--tags`, `-t` | Comma-separated tags | — |
| `--complexity`, `-c` | `trivial` / `simple` / `moderate` / `complex` / `extreme` | `moderate` |
| `--output`, `-o` | `summary` / `full` / `json` | `summary` |
| `--scaffold` | Write generated files to disk | off |
| `--scaffold-dir` | Target directory for scaffold | `.` |
| `--llm` | Provider override | from config |
| `--model` | Model override | from config |
| `--desc`, `-d` | Extended task description | — |
| `--stack`, `-s` | Comma-separated stack hint | — |

**Output includes:**
- Architecture pattern (monolith, microservices, event-driven, etc.)
- Tech stack (languages, frameworks, databases)
- Generated code files with syntax highlighting
- Optimization strategies (caching, pooling, etc.)
- Security analysis (hardcoded secrets, injection, dependency vulns)
- Thought process (5 steps: analysis → architecture → optimization → security → codegen)
- Confidence score and quality rating

---

### `pipeline` — run a kernel pipeline

Pre-configured multi-step workflows that combine kernels and agents.

```bash
# Full stack: architecture → codegen → security → deployment
noema pipeline fullstack --title "My SaaS" --tags "python,react,postgres"

# Quick: minimal viable solution
noema pipeline quick --title "URL shortener"

# Security-focused audit
noema pipeline security --title "Payment API"

# Architecture review
noema pipeline arch-review --title "Legacy migration"
```

**Available pipelines:** `fullstack`, `quick`, `security`, `arch-review`

---

### `serve` — launch the API server

```bash
noema serve                          # default: 0.0.0.0:8000
noema serve --host 127.0.0.1 --port 9000
noema serve --reload                 # auto-reload on code changes
```

Opens `http://localhost:8000` with all API endpoints. Interactive docs at `/docs` (Swagger UI).

---

### `knowledge` — search and manage knowledge

```bash
# Stats: how many entries, patterns, domain modules
noema knowledge stats

# Semantic search
noema knowledge search -q "database optimization"
noema knowledge search -q "distributed locking"
```

The knowledge base contains 22 domain modules (auth, database, gateway, graphql, ml_ops, mobile, terraform, websocket, and more) with pre-built patterns and best practices.

---

### `memory` — query episodic memory

Noema remembers past solutions, incidents, and decisions.

```bash
# Stats: episode counts by type
noema memory stats

# Search past episodes
noema memory search -q "rate limiting implementation"

# List recent episodes
noema memory episodes --limit 20
```

Memory types: **episodic** (past events), **semantic** (facts), **procedural** (how-to patterns).

---

### `graph` — knowledge graph operations

```bash
# Graph stats (nodes, edges, clusters)
noema graph stats

# Suggest compatible technologies
noema graph suggest --tags "python,fastapi,redis"

# Check compatibility of a specific tech
noema graph compatible --tech "GraphQL"
```

---

### `modules` — domain modules

```bash
# List all 22 modules
noema modules list

# Run a specific module
noema modules run --name auth --tags "jwt,rbac"

# Module stats
noema modules stats
```

Each module works independently and can be composed with others.

---

### `kernels` & `agents` — discovery

```bash
# List available kernels (Architect, Coder, Security, DevOps, DBA, AI Engineer)
noema kernels

# List available agents and their specializations
noema agents
```

---

### `evolve` — self-evolution cycle

Runs the self-improvement loop: mutates prompts/strategies, validates with tests, applies only if tests pass.

```bash
noema evolve
```

Evolution candidates are applied only when `evolution_test_before_apply` passes. No broken mutations reach production.

---

### `grid` — distributed federation

Split tasks across multiple Noema nodes.

```bash
# Federate a task across peers
noema grid federate \
  --title "Build platform" \
  --subtask "Auth service" \
  --subtask "API gateway" \
  --peer node1:50051 \
  --peer node2:50051

# Check fleet status (latency, tokens, errors per node)
noema grid status
noema grid status --redis redis://redis:6379/0
```

Without `--peer`, everything runs locally (fallback). Peers communicate via gRPC with circuit breaker and retry.

---

### `audit` — Merkle proof & verification

Cryptographic audit trail for every task.

```bash
# Generate a Merkle inclusion proof for a task
noema audit proof --tenant default --task <task_id> --output proof.json

# Verify a proof (client-side, no server trust needed)
noema audit verify proof.json

# Inspect the hash chain
noema audit chain --help
```

---

### `health` — system checks

```bash
# Run all health checks
noema health check

# Check LLM provider connectivity only
noema health check-llm
```

---

## API Server

All endpoints available at root (`/think`) and versioned prefix (`/api/v1/think`).

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| **Core** | | |
| POST | `/think` | Generate a full solution |
| POST | `/think/detail` | Solution + reasoning trace |
| POST | `/think/stream` | SSE stream of reasoning steps |
| DELETE | `/think/{task_id}` | Cancel a running task |
| **Tasks** | | |
| POST | `/tasks/enqueue` | Async task via Redis/arq |
| GET | `/tasks/active` | List active tasks |
| **Ops** | | |
| GET | `/health` | Health check (version, uptime, dependencies) |
| GET | `/ready` | Readiness gate (for load balancers) |
| GET | `/diagnostics` | Full system diagnostics |
| GET | `/features` | Feature flags |
| **Discovery** | | |
| GET | `/kernels` | Available kernels |
| GET | `/agents` | Available agents |
| GET | `/knowledge/stats` | Knowledge base stats |
| POST | `/knowledge/search` | Semantic search |
| **Admin** | | |
| GET | `/admin/metrics` | Prometheus-style metrics |
| GET | `/admin/tasks/history` | Task history |
| GET | `/admin/diagnostics` | Admin diagnostics |
| GET | `/admin/audit/proof/{task_id}` | Merkle proof |
| POST | `/admin/audit/verify` | Verify chain integrity |
| **Webhooks** | | |
| POST | `/webhooks/incident` | Incident → autonomous fix → PR |
| **Experiments** | | |
| POST | `/experiments` | Run benchmark matrix |
| GET | `/experiments` | List experiments |
| **Grid** | | |
| GET | `/grid` | Live fleet state |

### Authentication

Set `NOEMA_API__API_KEY` to enable. Pass via header:

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8000/think ...
```

Without the key set, auth is disabled (development mode).

### Streaming (SSE)

```bash
curl -N -X POST http://localhost:8000/think/stream \
  -H "Content-Type: application/json" \
  -d '{"title": "Redis lock", "complexity": "simple"}'
```

Events: `step_start` → `step_end` → ... → `complete` (with `solution_id`, `quality`, `summary`).

### Async tasks

```bash
# Enqueue (returns task_id immediately)
curl -X POST http://localhost:8000/tasks/enqueue \
  -H "Content-Type: application/json" \
  -d '{"title": "Background job", "description": "Run without blocking"}'

# Check status
curl http://localhost:8000/tasks/active
```

Requires Redis for multi-worker deployments.

---

## Python API (Library)

```python
import asyncio
from noema import NoemaEngine, Task


async def main():
    noema = NoemaEngine()
    await noema.initialize()

    solution, thought = await noema.think(
        Task(
            title="Build a REST API",
            description="FastAPI user management with registration, login, and RBAC.",
            tags=["api", "python", "fastapi", "auth"],
        )
    )

    print(f"Quality     : {solution.quality.value}")
    print(f"Confidence  : {solution.confidence:.0%}")
    print(f"Code blocks : {len(solution.code_blocks)}")
    print(f"Thought steps: {len(thought.steps)}")

    for block in solution.code_blocks:
        print(f"  {block.filename} ({block.language})")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Common Workflows

### Generate and scaffold a project

```bash
# Generate a full project structure
noema think "E-commerce API with auth, products, orders" \
  --tags "python,fastapi,postgres,redis" \
  --complexity complex \
  --scaffold \
  --scaffold-dir ./my-shop

# cd ./my-shop && ls
# auth/  app/  db/  requirements.txt  Dockerfile  ...
```

### Run benchmarks

```bash
# CLI
python -m noema.experiments.runner experiments/experiments.yaml --out results

# API
curl -X POST http://localhost:8000/experiments \
  -H "Content-Type: application/json" \
  -d '{"name": "model-comparison", "runs": 3}'
```

Output: `results/<experiment>/<run_id>/results.json` + CSV summaries with per-file token/cost breakdown.

### Autonomous self-healing

1. Incident arrives via Sentry or `POST /webhooks/incident`
2. Noema generates a fix through `NoemaEngine`
3. Fix is validated with tests (`validate_solution(run_tests=True)`)
4. If tests pass → GitHub branch + PR opened automatically
5. CI merge gate blocks if judge score is below threshold

```bash
# Configure
export NOEMA_AUTONOMY__GITHUB_TOKEN=ghp_...
export NOEMA_AUTONOMY__GITHUB_REPO=ssrjkk/my-app
export NOEMA_AUTONOMY__GITHUB_BASE_BRANCH=main
```

---

## Troubleshooting

### "Ollama is unreachable"

Ollama is the default provider. Either start it (`ollama serve`) or switch:

```bash
export NOEMA_LLM__PROVIDER=openai   # or anthropic, fallback
```

### "Service starting" (503 on /admin/metrics)

The engine hasn't finished initializing. Wait a few seconds and retry. Check `/health` for status.

### Z3 verification disabled

```bash
pip install -e ".[full]"
export NOEMA_NS__ENABLED=true
```

Without Z3, solutions skip formal verification (they still go through AST checks and sandbox).

### Rate limit errors (429)

Default: 60 requests/minute per API key. Adjust:

```bash
export NOEMA_API__RATE_LIMIT_RPM=120
export NOEMA_API__RATE_LIMIT_BURST=20
```

### Sandbox not available

Docker must be running for sandboxed execution. Without Docker, solutions run static analysis only (no runtime validation).

---

## Further Reading

- [Configuration Reference](configuration.md) — all env vars and YAML options
- [Deployment](deployment.md) — Docker / Compose / K8s / Helm / Terraform
- [API Examples](api-examples.md) — request/response examples for every endpoint
- [Production Checklist](production-checklist.md) — deploy readiness
- [Monitoring](monitoring-setup.md) — observability setup
- [Performance Tuning](performance-tuning.md) — optimization guide
- [Troubleshooting](troubleshooting.md) — common issues and fixes
