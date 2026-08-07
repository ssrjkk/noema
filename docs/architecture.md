# Architecture

Noema is structured around a core orchestration engine with pluggable modules, sub-agents, and infrastructure services.

## Core Engine

### NoemaEngine

The central orchestrator at `noema/core/noema.py`. Manages the full lifecycle: initialization, task reasoning, solution assembly, feedback recording, and self-evolution. Delegates reasoning to Chain-of-Thought or NeuroSymbolic engine based on configuration.

### ChainOfThought

DAG-based reasoning engine at `noema/core/chain_of_thought.py`. The `StepPlanner` dynamically selects steps based on task tags and complexity. Independent steps execute in parallel. Supports reflexion (up to 3 retry attempts with judge feedback) and checkpointing for resumable execution.

### Types

Core data models in `noema/core/types.py`. Includes `Task`, `Solution`, `TechStack`, `ArchitecturePattern`, `ThoughtProcess`, `CodeBlock`, and enums for `TaskComplexity`, `SolutionQuality`, `KernelType`, `AgentRole`.

### EventBus

Async pub/sub event bus at `noema/core/events.py`. Supports typed subscriptions, wildcard handlers (`*`), and a dead letter queue.

### CheckpointStore

File-based (or Redis) checkpoint persistence at `noema/core/checkpoint.py`. Stores DAG progress for resumable task execution.

## NeuroSymbolic Engine

Three-tier engine at `noema/neurosymbolic/`:

| Component | File | Role |
|-----------|------|------|
| `NeuroSymbolicEngine` | `engine.py` | Orchestrates parsing → hypothesis → verification → refinement loop |
| `SymbolicEngine` | `symbolic.py` | Z3-based formal verification with solver pool |
| `NeuralInterface` | `neural.py` | LLM-based hypothesis generation with circuit breaker + batching |
| `EvolutionEngine` | `evolution.py` | Records outcomes for prompt optimization |

## Sub-Agent System

`AgentOrchestrator` at `noema/agents/orchestrator.py` manages a pool of specialized agents (Architect, Developer, Security, DevOps, DBA, AI Engineer). Each agent implements `BaseAgent` and contributes domain expertise to solution generation.

## Module System

`ModuleRegistry` at `noema/modules/registry.py` auto-discovers and manages 22 built-in domain modules:

| Module | Module | Module |
|--------|--------|--------|
| monitoring | testing | documentation |
| database | queues | caching |
| auth | graphql | websocket |
| mobile | i18n | cli_generator |
| security_scanner | performance | config |
| events | quality | containers |
| terraform | data_pipeline | ml_ops |
| gateway | | |

## Kernels

9 reasoning kernels in `noema/kernels/`: analysis, architecture, codegen, optimization, security, frontend, devops, data, ai_ml. Each implements `BaseKernel` and provides structured JSON output consumed by later steps.

## Memory System

Three-tier memory at `noema/memory/store.py`:

- **Episodic** — task → outcome records with vector search
- **Semantic** — reusable knowledge fragments with embeddings
- **Procedural** — reusable step sequences (procedures)

Uses HNSW index for approximate nearest neighbor search.

## Resilience

At `noema/resilience/`:

| Component | File | Description |
|-----------|------|-------------|
| `CircuitBreaker` | `circuit_breaker.py` | CLOSED → OPEN → HALF_OPEN state machine |
| `RetryPolicy` | `circuit_breaker.py` | Exponential backoff with jitter |
| `ResilientExecutor` | `circuit_breaker.py` | Combined circuit breaker + retry |
| `GracefulDegradation` | `graceful_degradation.py` | Redis→memory, PostgreSQL→file fallback |
| `CancellationManager` | `cancellation.py` | Graceful task cancellation with cleanup |

## Observability

At `noema/observability/`:

- **Metrics** — Prometheus counters, histograms, gauges for HTTP, LLM, workers, memory, modules
- **Sentry** — Error tracking with structlog integration
- **Tracing** — OpenTelemetry-compatible tracer at `noema/tracing/tracer.py`

## Billing

At `noema/billing/`:

- **QuotaManager** — Per-tenant quotas (monthly budget, concurrent tasks, hourly rate, token limits)
- **CostTracker** — Cost attribution per tenant/task/step with Redis + in-memory backends

## Security

At `noema/security/`:

- **PII Redactor** — Pattern-based PII detection and masking
- **RAG Injection Sanitizer** — Input validation for knowledge injection
- **SandboxEngine** — Docker-based sandboxed code execution with AST/lint/type-check/run/test stages

## Infrastructure

- **PostgreSQL** — Persistent storage via SQLAlchemy 2.0 + asyncpg (with file fallback)
- **Redis** — Caching, rate limiting, cost tracking, active task management (with in-memory fallback)
- **LLM Providers** — Ollama, OpenAI, Anthropic with model routing and circuit breakers
