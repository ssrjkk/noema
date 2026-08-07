# Contributing

## Development Setup

```bash
git clone https://github.com/anomalyco/noema
cd noema
pip install -e ".[dev,db,full,sentry]"
```

## Pre-commit Hooks

The project uses [pre-commit](https://pre-commit.com) with the following hooks (`.pre-commit-config.yaml`):

```bash
pre-commit install
pre-commit run --all-files
```

| Hook | Purpose |
|------|---------|
| `trailing-whitespace` | Remove trailing whitespace |
| `end-of-file-fixer` | Ensure files end with newline |
| `check-yaml` | Validate YAML syntax |
| `check-json` | Validate JSON syntax |
| `check-toml` | Validate TOML syntax |
| `check-added-large-files` | Reject files >500 KB |
| `detect-private-key` | Prevent key leaks |
| `check-merge-conflict` | Detect unresolved merge markers |
| `ruff` | Lint + auto-fix |
| `ruff-format` | Code formatting |
| `mypy` | Static type checking (strict mode) |
| `bandit` | Security vulnerability scan |

## Code Style

- **Python**: 3.11+, type hints required
- **Line length**: 100 chars
- **Formatter**: `ruff format`
- **Linter**: `ruff check` with rulesets: E, F, W, I, N, UP, B, A, C4, SIM, TCH
- **Type checker**: `mypy --strict`
- **Imports**: isort-style (first-party: `noema`)
- **Docstrings**: Google style

## Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest --cov=noema tests/

# Specific test file
pytest tests/test_noema.py -v

# With asyncio mode (auto by default)
pytest tests/test_neurosymbolic.py -v

# Chaos tests (Redis/PostgreSQL failover)
pytest tests/chaos/ -v

# Adversarial / red-team tests
pytest tests/adversarial/ -v
```

## CI Pipeline

GitHub Actions automatically runs on push/PR:

1. **Lint** — ruff check + ruff format
2. **Type check** — mypy strict mode
3. **Test** — pytest with coverage
4. **Security** — bandit scan

## Project Structure

```
noema/
├── agents/          # Sub-agent system (Architect, Developer, Security, etc.)
├── api/             # FastAPI server, middleware, auth, rate limiting, admin
├── audit/           # Audit logging
├── billing/         # Quota management, cost tracking
├── budget/          # Token budget management
├── config/          # Pydantic settings, feature flags
├── core/            # NoemaEngine, ChainOfThought, types, events, checkpoint
├── db/              # SQLAlchemy models + engine
├── debug/           # Task replay
├── discovery/       # Key/resource discovery
├── evolution/       # Self-evolution engine
├── feedback/        # Solution feedback store
├── healer/          # Self-healing strategies
├── ingestion/       # Knowledge ingestion (files, directories, text)
├── kernels/         # Reasoning kernels (analysis, codegen, architecture, etc.)
├── knowledge/       # Knowledge store + graph
├── llm/             # LLM providers (Ollama, OpenAI, Anthropic)
├── memory/          # Three-tier memory (episodic, semantic, procedural)
├── modules/         # 22 pluggable domain modules
├── neurosymbolic/   # Z3 formal verification + LLM hypothesis engine
├── observability/   # Prometheus metrics, Sentry, tracing
├── persistence/     # Redis cache, PostgreSQL memory
├── pipelines/       # Multi-agent pipeline engine
├── plugins/         # Plugin system
├── resilience/      # Circuit breaker, retry, graceful degradation
├── routing/         # Model router
├── sandbox/         # Docker sandbox for code validation
├── scaffolder/      # Project scaffolding
├── security/        # PII redactor, RAG injection sanitizer
├── services/        # High-level services
├── tracing/         # OpenTelemetry tracer
├── utils/           # Helpers, atomic I/O
└── workers/         # Worker pool, hierarchy
```

## Making Changes

1. Create a feature branch from `main`
2. Make your changes with type hints and docstrings
3. Run pre-commit hooks
4. Add/update tests
5. Run `pytest tests/` to verify nothing is broken
6. Submit a PR with a clear description

## Adding a Module

1. Create `noema/modules/<name>/__init__.py` and `kernel.py`
2. Define a class with an `execute(task)` method and `DESCRIPTION` attribute
3. Register in `ModuleRegistry._load_builtin_modules()` in `noema/modules/registry.py`

## License

MIT
