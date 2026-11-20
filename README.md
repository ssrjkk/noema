# Noema

[![CI](https://github.com/ssrjkk/noema/actions/workflows/ci.yml/badge.svg)](https://github.com/ssrjkk/noema/actions/workflows/ci.yml)
[![python-3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



**The engineering mind that verifies what it generates.**

Noema is not another code generator. It is a neurosymbolic system that proposes solutions with LLMs, verifies them against formal contracts with Z3, validates them in a sandboxed runtime, and maintains a fully auditable trail of every reasoning step — then fixes its own incidents end-to-end.

> [!NOTE]
> **Phase 1 — The Architect** is production-ready. Phase 2 (autonomous self-healing) and Phase 3 (multi-node grid federation) are merged and under test — `docs/ROADMAP.md` has no open items. Pointing the incident → PR loop at a real repository is a configuration step: it needs `NOEMA_AUTONOMY__GITHUB_*` credentials.

---

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#documentation">Docs</a> •
  <a href="README.ru.md">Русский</a>
</p>

---

## Quick Start

```bash
git clone https://github.com/ssrjkk/noema && cd noema
pip install -e .
python scripts/demo.py        # 12 live demos — no API keys required
```

If you see `=== ALL DEMOS COMPLETE ===` — you're good. Then:

```bash
noema think "Auth service on FastAPI" --tags "python,fastapi,auth"
```

## The Problem

LLMs are great at generation. Everything else they do poorly:

| Problem | What happens |
|---|---|
| **Confident but wrong** | Solutions look perfect, then fail on import errors, security holes, or logic bugs |
| **Nobody verifies** | Generated code goes straight to production without a single formal check |
| **No replay** | Bad output? No way to trace where the system went wrong |
| **Amnesia** | Every task starts from scratch — past solutions and incidents are forgotten |
| **No economics** | Tokens burn without accounting — cost per generated line is unknown |
| **Manual everything** | Bug fixes, reviews, gates — all human-operated |

Noema turns generation from a one-shot bet into an **engineering process with checkpoints, verification, and audit**.

## Features

### Core Loop

```
         ┌──────────────────────────────────────────────────┐
         │                   NoemaEngine                     │
         │   ChainOfThought (DAG)  ·  NeuroSymbolicEngine     │
         └───────┬───────────────────────┬───────────────────┘
                 │ propose               │ verify
     ┌───────────▼──────────┐   ┌────────▼────────────────────────┐
     │ NeuralInterface (LLM)│   │ SymbolicEngine (Z3) · static.py │
     └───────────┬──────────┘   └────────┬────────────────────────┘
                 │ hypothesis            │ verdict (fail-closed)
                 └───────► refine loop ◄─┘
                     ┌─────────────┼──────────────┐
                     │ trace       │ sandbox       │ memory / knowledge
                     │ replay      │ static+run    │ (episodic, domain)
```

The neural side **proposes**, the symbolic side **adjudicates**, and every exchange is recorded in an auditable artifact. No unverified result is ever accepted.

### What You Get

| Capability | Description |
|---|---|
| **Solution generation** | `noema think "Real-time Chat"` — full architecture, stack, code, optimizations, security |
| **Formal verification (Z3)** | Every hypothesis is checked against a symbolic contract extracted from requirements. Solver unavailable → solution rejected (fail-closed). A rejected candidate returns CEGIS feedback: the `unsat_core` naming the conflicting requirements, or a concrete counterexample to refine against |
| **Pre-run static analysis** | Pure AST pass: syntax, import hygiene, undefined names — verdict before untrusted code ever runs |
| **Sandboxed execution** | Docker isolation: no network, CPU/memory/time limits. Static verdict short-circuits execution |
| **Reasoning audit** | Every thought step, Z3 verdict, and AST check is recorded. Old verdicts are re-verified deterministically — no LLM needed |
| **Autonomous self-healing** | Incident (Sentry/webhook) → fix → branch → PR with passing validation. Merge gate blocks if judge score is below threshold |
| **Self-evolution** | Prompt/strategy mutations apply only when tests pass (`evolution_test_before_apply`) |
| **22 domain modules** | auth, database, gateway, graphql, ml_ops, mobile, terraform, websocket, and more — work independently and together |
| **Specialized kernels & agents** | Architect, Coder, Security, DevOps, DBA, AI Engineer. Pipelines: fullstack, quick, security, arch-review |
| **Memory & knowledge** | Episodic, semantic, and procedural memory + domain knowledge base |
| **Hybrid retrieval** | BM25 and dense scores fused by Reciprocal Rank Fusion, filtered by bi-temporal validity; Ebbinghaus retention ranks results instead of dropping them. Verified fixes are indexed by the incident's error signature (exception line + raising frame) and its Z3 `unsat_core`, and looked up before a new patch is proposed |
| **Token economics** | Every LLM call is traced, attributed, and converted to cost. Budgets and circuit breakers included |
| **Model routing (opt-in)** | `NOEMA_LLM__ROUTER=linucb` hands arm selection to a LinUCB contextual bandit updated from verified outcomes (Sherman-Morrison rank-1 inverse updates). Budget exhausted → `BudgetExceededError`, never an unmodelled call; a corrupt checkpoint raises instead of resetting |
| **Reproducible benchmarks** | One task matrix across providers/models → `results.json` + CSV summaries with per-file token/cost breakdown |
| **Verified training data** | `noema train export-dpo` / `export-grpo` build RLVR datasets from rollouts that already passed verification; `calibrate-judge` reports out-of-sample k-fold Brier/ECE for the judge |
| **Production API** | FastAPI: rate limiting, API keys, per-tenant quotas, request timeout guard (504 + exempt paths), RFC 7807, Prometheus metrics, SSE streaming |
| **Grid federation** | `noema grid federate`: subtasks delegated to peers via gRPC with circuit breaker and retry. Local fallback when peers are down. Every node's contribution recorded in an auditable ledger |
| **Grid dashboard** | `GET /grid` and `noema grid status`: live fleet state — latency, tokens, errors per node + cluster totals |

## Installation

```bash
# Minimal
pip install -e .

# Full (Z3 verification, vector search, LLM providers)
pip install -e ".[full]"

# Everything (dev tools, DB, gRPC, vault)
pip install -e ".[dev,full,db,grpc,vault]"
```

**Python 3.11+** declared in `pyproject.toml`; CI actually runs 3.12, 3.13 and 3.14 on Ubuntu + Windows, and the Docker image ships 3.14.

**LLM providers:** `openai`, `anthropic`, `ollama` + built-in **fallback provider** (works without API keys — for demos and CI).

Configure via env: `NOEMA_LLM__PROVIDER=openai`, `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`. Pin a model with
`noema think --model gpt-4o` per command, or `NOEMA_LLM__OLLAMA_MODEL` for Ollama. Env beats the YAML
file — see [Configuration](docs/configuration.md) for the full precedence.

### Verify Installation

```bash
noema --help                  # CLI entry point
python -m noema --help        # Module entry (more reliable on Windows)

python scripts/demo.py                # 12 live demos, zero API keys
# Expected: === ALL DEMOS COMPLETE ===
```

## Usage

### CLI

```bash
# Think — full solution generation
noema think "Real-time Chat App" --tags "python,websocket,redis" --complexity complex --output full

# Scaffold to disk
noema think "Auth service" --scaffold --scaffold-dir ./out

# Kernel pipelines
noema pipeline fullstack --title "My Project"
noema pipeline security --title "API audit"

# Knowledge, memory, graph
noema knowledge search -q "database optimization"
noema memory stats
noema graph suggest --tags "python,fastapi,redis"

# Autonomy & evolution
noema evolve
noema agents
noema modules list

# Grid federation
noema grid federate --peer localhost:50051
noema grid status
```

### API Server

```bash
noema serve
# http://localhost:8000
```

All endpoints available at root (`/think`) and versioned prefix (`/api/v1/think`).

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/think` | Full solution for a task |
| POST | `/think/detail` | Solution + reasoning trace |
| POST | `/think/stream` | SSE stream of reasoning steps |
| DELETE | `/think/{task_id}` | Cancel a running task |
| POST | `/tasks/enqueue` | Async task via Redis/arq |
| POST | `/experiments` | Benchmark matrix as a service |
| POST | `/webhooks/incident` | Incident → autonomous fix → PR |
| GET | `/health`, `/ready`, `/diagnostics`, `/features` | Ops endpoints |
| GET | `/kernels`, `/agents`, `/knowledge/stats`, `/knowledge/search` | Discovery |
| GET | `/grid` | Live grid fleet state |
| GET | `/admin/metrics` | Prometheus-style metrics |

```bash
curl -X POST http://localhost:8000/think \
  -H "Content-Type: application/json" \
  -d '{"title": "Event bus on RabbitMQ", "complexity": "complex", "tags": ["python", "rabbitmq"]}'
```

## Production Setup

**Default configuration is for development and demos, not production.** Before deploying:

### Required

1. **LLM provider** — the code default is `fallback` (template responses, not LLM); the `settings.yaml`
   shipped in this repo selects `ollama`, so name the provider explicitly instead of inheriting either
   ```bash
   export NOEMA_LLM__PROVIDER=openai
   export OPENAI_API_KEY=sk-...
   ```

2. **API key** — default is empty (auth disabled)
   ```bash
   export NOEMA_API__API_KEY=your-secret-key
   ```

3. **Neurosymbolic verification** — default is disabled
   ```bash
   export NOEMA_NS__ENABLED=true
   pip install -e ".[full]"   # requires Z3
   ```

4. **Webhook secret** — default HMAC verification is disabled
   ```bash
   export NOEMA_API__WEBHOOK_SECRET=your-webhook-secret
   ```

5. **Database** — for persistence, audit, and billing
   ```bash
   export NOEMA_DB__URL=postgresql+asyncpg://user:pass@localhost/noema
   pip install -e ".[db]"
   ```

### Recommended

- **Redis** — for rate limiting in multi-worker deployments
- **Sentry** — error monitoring (`NOEMA_OBS__SENTRY_DSN`)
- **Trusted proxies** — if behind reverse proxy (`NOEMA_API__TRUSTED_PROXIES`)

### Fail-Closed Behavior

Noema follows **fail-closed** principles: if a component is unavailable, the system refuses safely rather than fabricating an answer:

- LLM unavailable → `RuntimeError`, not a template response
- Webhook secret empty → request rejected
- Z3 solver unavailable → verification does not pass

## Autonomous Self-Healing

1. **Incident** — Sentry alert or `POST /webhooks/incident` normalized into `Incident`
2. **Fix** — task runs through `NoemaEngine`, result validated with tests (`validate_solution(run_tests=True)`). No PR without `all_valid`
3. **PR** — GitHub httpx client (no PyGithub) opens a branch and pull request
4. **Merge gate** — CI job blocks merge if `judge_score` is below threshold or sandbox fails
5. **Evolution** — prompt/strategy candidates apply only when tests pass

Configure: `NOEMA_AUTONOMY__GITHUB_TOKEN`, `NOEMA_AUTONOMY__GITHUB_REPO`, `NOEMA_AUTONOMY__GITHUB_BASE_BRANCH`.

## Experiments & Benchmarks

Reproducible runner: same task matrix across providers and models, collecting wall-time, tokens, judge scores, cost, and optional sandbox validation into `results/`.

```bash
# Demo without API keys (fallback provider)
python -m noema.experiments.runner experiments/experiments.yaml --out results

# Real models: set provider in experiments.yaml and export keys
# results/<experiment>/<run_id>/results.json — one record per (task, provider, model, repetition)
# results/<experiment>/<run_id>/runs.csv      — same in CSV
# results/<experiment>/<run_id>/summary.csv   — aggregates by (provider, model)
```

CI runs a smoke benchmark nightly on the fallback provider and uploads artifacts. The same runner is available as a service: `POST /experiments`.

## Project Structure

```
noema/
  autonomy/        # incidents → fixes → PRs
  neurosymbolic/   # Z3 verification + AST analysis pipeline
  sandbox/         # Docker sandbox + pre-run static checks
  tracing/         # reasoning trace: deterministic verdict replay
  experiments/     # reproducible benchmark runner + merge gate
  knowledge/       # knowledge base + 22 domain modules
  memory/          # episodic / semantic / procedural memory
  api/             # FastAPI: think, webhooks, experiments, admin, rate limits
  workers/         # arq workers, task hierarchy, pool
  modules/         # pluggable domain modules
  kernels/ agents/ # specialized kernels and agents
  llm/             # providers: openai, anthropic, ollama, fallback
  billing/ budget/ # token economics, quotas, budgets
  security/        # validation, schemas, tenant isolation
  grpc/            # gRPC server/client + protos
  observability/   # Prometheus metrics, Sentry
  vault/ audit/    # secrets, audit trail
```

## Quality & Testing

| Metric | Value |
|---|---|
| **Tests** | 2,769 collected — unit, hypothesis property and pytest-benchmark cases (2,752 passed + 17 skipped on Ubuntu / 3.12) |
| **Source files** | 225 Python files under `noema/` (160 modules + 65 package `__init__`) |
| **Lines of code** | 43,417 lines, 36,774 of them code (the rest blank and comments) |
| **CI** | Ruff lint + format, encoding/mojibake gate, mypy, bandit + pip-audit, golden eval, merge gate |
| **Platforms** | Ubuntu + Windows, Python 3.12 + 3.13 + 3.14 |
| **Encoding** | Unicode mojibake gate — broken strings fail CI |

## Documentation

- [Usage Guide](docs/usage-guide.md) — complete CLI, API, and library reference with examples
- [3D Connectome Showcase](docs/noema.html) — interactive neural-connectome page (static, DEMO SIM mode)
- [ML/AI Upgrade Plan](docs/noema_ml_ai_upgrade_plan.md) — v2: RLVR, hybrid memory, LinUCB routing, Z3 CEGIS (shipped to `main`)
- [Whitepaper](docs/WHITEPAPER.md) — vision and design principles
- [Roadmap](docs/ROADMAP.md) — three phases: Architect → Autopoietic Enterprise → Global Noema Grid
- [Configuration](docs/configuration.md) — all env vars and YAML options
- [Deployment](docs/deployment.md) — Docker / Compose / K8s / Helm / Terraform
- [Getting Started](docs/getting-started.md) — tutorial and walkthrough
- [API Examples](docs/api-examples.md) — request/response examples
- [Production Checklist](docs/production-checklist.md) — deploy readiness
- [Monitoring](docs/monitoring-setup.md) — observability setup
- [Performance Tuning](docs/performance-tuning.md) — optimization guide
- [Troubleshooting](docs/troubleshooting.md) — common issues and fixes

## Roadmap

| Phase | Name | Status |
|---|---|---|
| **1** | **The Architect** — generation + verification + sandbox + benchmarks + knowledge + reasoning trace | **Done** |
| **2** | **The Autopoietic Enterprise** — incident → PR, merge gate, evolution with auto-apply, benchmark service | **In progress** |
| **3** | **Global Noema Grid** — multi-node pool, gRPC federation, token/ledger economics, live dashboard | **In progress** |

## Author

**ssrjkk** — [GitHub](https://github.com/ssrjkk)

Project repository: [github.com/ssrjkk/noema](https://github.com/ssrjkk/noema)

## License

MIT. Open development — ideas, issues, and PRs welcome.

---

**SSRJKK** — [GitHub](https://github.com/ssrjkk) · [Telegram](https://t.me/ssrjkk) · [Projects hub](https://ssrjkk.github.io/) · [Bot catalogue](https://t.me/ssrjkk_bot)
