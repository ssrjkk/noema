# Noema

**Production-grade AI reasoning engine** with 22 domain modules, DAG-based CoT, neurosymbolic verification, tenant isolation, and full observability.

## Features

- **DAG-based Chain-of-Thought** — Dynamic step planner with parallel execution, reflexion, and checkpointing
- **NeuroSymbolic Engine** — Z3 formal verification + LLM hypothesis generation with circuit breaker protection
- **Multi-tenant** — contextvar isolation, per-tenant quotas, feature flags, audit logging
- **Resilience** — Circuit breakers, graceful degradation (Redis→memory, PostgreSQL→file), rate limiting
- **Observability** — Prometheus metrics, Sentry errors, structured logging (structlog), OpenTelemetry tracing
- **SSE Streaming** — Real-time step progress via Server-Sent Events
- **Security** — PII redaction, RAG injection sanitizer, sandboxed code execution
- **Self-Evolution** — Automated prompt optimization via trace analysis and judge feedback

## Quick Start

```bash
# Install
pip install -e ".[dev,db,full]"

# Run tests
make test

# Start API
uvicorn noema.api.server:app --reload
```

## Architecture

```
NoemaEngine
├── ChainOfThought (DAG-based)
├── NeuroSymbolicEngine
│   ├── SymbolicEngine (Z3)
│   └── NeuralInterface (LLM + CircuitBreaker)
├── ModelRouter
├── TokenBudget
├── CheckpointStore
├── MemoryStore
├── EvolutionEngine
└── SandboxEngine
```
