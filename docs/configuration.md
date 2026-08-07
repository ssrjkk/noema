# Configuration

All settings are managed via `noema/config/settings.py` using Pydantic BaseSettings with environment variable overrides and optional YAML file loading.

## Environment Variables

Settings are organized into namespaced groups. Each group uses a unique environment prefix.

### Database (`NOEMA_DB_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `NOEMA_DB_URL` | `postgresql+asyncpg://noema:noema@localhost:5432/noema` | SQLAlchemy async connection URL |
| `NOEMA_DB_POOL_MIN` | `2` | Minimum connection pool size |
| `NOEMA_DB_POOL_MAX` | `10` | Maximum connection pool size |
| `NOEMA_DB_POOL_TIMEOUT` | `30.0` | Pool timeout in seconds |
| `NOEMA_DB_ECHO` | `false` | SQLAlchemy echo (debug SQL logging) |

### Redis (`NOEMA_REDIS_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `NOEMA_REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `NOEMA_REDIS_POOL_MAX` | `20` | Maximum connection pool size |
| `NOEMA_REDIS_SOCKET_TIMEOUT` | `5.0` | Socket timeout in seconds |

### LLM (`NOEMA_LLM_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `NOEMA_LLM_PROVIDER` | `ollama` | Provider: `ollama`, `openai`, `anthropic` |
| `NOEMA_LLM_OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `NOEMA_LLM_OLLAMA_MODEL` | `llama3.1` | Default Ollama model |
| `NOEMA_LLM_OPENAI_API_KEY` | `` | OpenAI API key |
| `NOEMA_LLM_ANTHROPIC_API_KEY` | `` | Anthropic API key |
| `NOEMA_LLM_TEMPERATURE` | `0.4` | LLM temperature (0.0–2.0) |
| `NOEMA_LLM_MAX_TOKENS` | `4096` | Max tokens per response |
| `NOEMA_LLM_REQUEST_TIMEOUT` | `120.0` | Request timeout in seconds |
| `NOEMA_LLM_CIRCUIT_BREAKER_THRESHOLD` | `5` | Failures before circuit opens |
| `NOEMA_LLM_CIRCUIT_BREAKER_RECOVERY` | `30.0` | Seconds before retry |
| `NOEMA_LLM_RETRY_MAX` | `3` | Max retry attempts |
| `NOEMA_LLM_RETRY_BASE_DELAY` | `1.0` | Base retry delay in seconds |
| `NOEMA_LLM_RETRY_MAX_DELAY` | `30.0` | Max retry delay in seconds |

### API (`NOEMA_API_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `NOEMA_API_HOST` | `0.0.0.0` | Bind address |
| `NOEMA_API_PORT` | `8000` | Listen port |
| `NOEMA_API_WORKERS` | `1` | Number of workers |
| `NOEMA_API_RELOAD` | `false` | Auto-reload on code changes |
| `NOEMA_API_API_KEY` | `` | Master API key (empty = auth disabled) |
| `NOEMA_API_API_KEY_HEADER` | `X-API-Key` | Header name for API key |
| `NOEMA_API_RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `NOEMA_API_RATE_LIMIT_RPM` | `60` | Requests per minute per key |
| `NOEMA_API_RATE_LIMIT_BURST` | `10` | Burst allowance |
| `NOEMA_API_CORS_ORIGINS` | `["*"]` | Allowed CORS origins |
| `NOEMA_API_CORS_METHODS` | `["GET","POST","PUT","DELETE"]` | Allowed CORS methods |
| `NOEMA_API_MAX_REQUEST_BODY` | `1048576` | Max request body in bytes (1 MB) |
| `NOEMA_API_MAX_TITLE_LENGTH` | `200` | Max task title length |
| `NOEMA_API_MAX_DESCRIPTION_LENGTH` | `10000` | Max task description length |
| `NOEMA_API_MAX_TAGS` | `20` | Max number of tags |

### Worker (`NOEMA_WORKER_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `NOEMA_WORKER_POOL_SIZE` | `10` | Worker pool size |
| `NOEMA_WORKER_MAX_QUEUE` | `100` | Max queue size |
| `NOEMA_WORKER_HIERARCHY_MAX_DEPTH` | `10` | Max worker hierarchy depth |
| `NOEMA_WORKER_HIERARCHY_MAX_CONCURRENT` | `50` | Max concurrent hierarchy tasks |
| `NOEMA_WORKER_TASK_TIMEOUT` | `300.0` | Task timeout in seconds |

### Memory (`NOEMA_MEMORY_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `NOEMA_MEMORY_DATA_DIR` | `./data` | Data directory for persistence |
| `NOEMA_MEMORY_BACKUP_ENABLED` | `true` | Enable rotating backups |
| `NOEMA_MEMORY_BACKUP_COUNT` | `5` | Number of rotating backups |
| `NOEMA_MEMORY_AUTO_SAVE_INTERVAL` | `30.0` | Seconds between auto-saves |

### Sandbox (`NOEMA_SANDBOX_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `NOEMA_SANDBOX_ENABLED` | `true` | Enable sandboxed code execution |
| `NOEMA_SANDBOX_DOCKER_IMAGE` | `python:3.12-slim` | Docker image for sandbox |
| `NOEMA_SANDBOX_TIMEOUT` | `60.0` | Execution timeout in seconds |
| `NOEMA_SANDBOX_MAX_MEMORY` | `256m` | Max memory per container |
| `NOEMA_SANDBOX_MAX_CPUS` | `0.5` | Max CPU cores |
| `NOEMA_SANDBOX_NETWORK_DISABLED` | `true` | Disable network in sandbox |
| `NOEMA_SANDBOX_READ_ONLY_ROOT` | `true` | Read-only root filesystem |

### Observability (`NOEMA_OBS_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `NOEMA_OBS_LOGGING_LEVEL` | `INFO` | Logging level |
| `NOEMA_OBS_LOGGING_FORMAT` | `json` | Log format: `json` or `console` |
| `NOEMA_OBS_METRICS_ENABLED` | `true` | Enable Prometheus metrics |
| `NOEMA_OBS_METRICS_PORT` | `9090` | Metrics HTTP port |
| `NOEMA_OBS_TRACING_ENABLED` | `false` | Enable OpenTelemetry tracing |
| `NOEMA_OBS_TRACING_ENDPOINT` | `http://localhost:4318` | OTLP HTTP endpoint |
| `NOEMA_OBS_SENTRY_DSN` | `` | Sentry DSN for error tracking |
| `NOEMA_OBS_SENTRY_ENVIRONMENT` | `production` | Sentry environment |
| `NOEMA_OBS_SENTRY_TRACES_SAMPLE_RATE` | `0.1` | Traces sample rate (0.0–1.0) |
| `NOEMA_OBS_SENTRY_PROFILES_SAMPLE_RATE` | `0.05` | Profiles sample rate (0.0–1.0) |

### NeuroSymbolic (`NOEMA_NS_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `NOEMA_NS_ENABLED` | `false` | Enable neurosymbolic engine |
| `NOEMA_NS_MAX_REFINEMENT_ATTEMPTS` | `3` | Max verification → refinement cycles |
| `NOEMA_NS_VERIFICATION_TIMEOUT` | `5.0` | Z3 verification timeout |
| `NOEMA_NS_EVOLUTION_ENABLED` | `true` | Enable outcome tracking |
| `NOEMA_NS_FALLBACK_TO_COT` | `true` | Fallback to CoT on NS failure |

### Global (`NOEMA_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `NOEMA_CONFIG` | `./settings.yaml` | Path to YAML config file |
| `NOEMA_COT_MAX_STEPS` | `12` | Max Chain-of-Thought steps |
| `NOEMA_COT_TEMPERATURE` | `0.4` | CoT LLM temperature |
| `NOEMA_EVOLUTION_ENABLED` | `true` | Enable self-evolution |
| `NOEMA_EVOLUTION_AUTO_APPLY` | `false` | Auto-apply evolution patches |
| `NOEMA_EVOLUTION_TEST_BEFORE_APPLY` | `true` | Run tests before applying patches |
| `NOEMA_KNOWLEDGE_PERSIST_PATH` | `./data/knowledge.json` | Knowledge base file path |
| `NOEMA_FEEDBACK_PERSIST_PATH` | `./data/feedback.json` | Feedback store file path |

## YAML Configuration

Settings can also be loaded from a YAML file specified via `NOEMA_CONFIG` environment variable.

```yaml
llm:
  provider: openai
  temperature: 0.3
  max_tokens: 8192

api:
  host: 0.0.0.0
  port: 8080
  rate_limit_rpm: 120

neurosymbolic:
  enabled: true
  max_refinement_attempts: 5
```

The configuration is loaded with env var override priority: environment variables > YAML file > defaults.
