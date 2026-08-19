"""Noema — centralised configuration (Pydantic BaseSettings + YAML)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# ─── Paths ───────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "settings.yaml"

_SUBMODEL_KEYS = frozenset(
    {
        "db",
        "redis",
        "llm",
        "api",
        "worker",
        "memory",
        "sandbox",
        "obs",
        "neurosymbolic",
        "audit",
        "autonomy",
    }
)


def _env_leaf_keys() -> set[str]:
    """Normalized env keys that actually map onto a settings field.

    Used to ignore stray ``NOEMA_*`` variables (including ``NOEMA_CONFIG``)
    that do not correspond to a real field — otherwise a leftover env var
    with a non-coercible value would break config loading entirely.
    """
    keys: set[str] = set()
    for name, field in NoemaSettings.model_fields.items():
        base = f"noema_{name.lower()}"
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, BaseSettings):
            for sub_name in ann.model_fields:
                keys.add(f"{base}_{sub_name.lower()}")
                keys.add(f"{base}__{sub_name.lower()}")
        else:
            keys.add(base)
    return keys


def _env_overrides() -> dict[str, Any]:
    """Map relevant ``NOEMA_*`` environment variables onto a nested config dict.

    Supports both ``NOEMA_LLM__PROVIDER`` (root prefix + nested delimiter) and
    ``NOEMA_LLM_PROVIDER`` (sub-model prefix) spellings; scalar root knobs like
    ``NOEMA_COT_MAX_STEPS`` map to their root field. Variables that do not
    correspond to a settings field are ignored, and empty values are treated
    as unset so an empty ``int``/``bool`` env var cannot crash pydantic.

    Container fields (``list``/``tuple``/``dict``, e.g.
    ``NOEMA_AUTONOMY__LEAN_VERIFIER_REQUIRED_PATHS``) expect a JSON value
    (``["crypto/", "auth/"]``); malformed JSON raises a clear ``ValueError``
    naming the variable rather than letting pydantic fail with a cryptic
    ``EnvSettingsSource`` error.
    """
    valid = _env_leaf_keys()
    out: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith("NOEMA_"):
            continue
        if key.lower() not in valid:
            continue
        if value == "":
            continue
        rest = key[len("NOEMA_") :]
        parts = rest.split("__")
        if len(parts) == 1 and "_" in rest:
            first, _, tail = rest.partition("_")
            if first.lower() in _SUBMODEL_KEYS:
                parts = [first, tail]
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part.lower(), {})
        leaf = parts[-1].lower()
        if _is_container_field([p.lower() for p in parts]):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                raise ValueError(
                    f"Env var {key} must be valid JSON for a list/dict/tuple field, got: {value!r}"
                ) from None
        node[leaf] = value
    return out


def _is_container_field(parts: list[str]) -> bool:
    """True when the env path resolves to a list/tuple/dict/set field."""
    node: Any = NoemaSettings
    for part in parts:
        field = node.model_fields.get(part)
        if field is None:
            return False
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, BaseSettings):
            node = ann
            continue
        origin = getattr(ann, "__origin__", None)
        return origin in (list, tuple, dict, set)
    return False


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` on top of ``base``; ``override`` wins."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ─── Sub-models ──────────────────────────────────────────────────────────
class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOEMA_DB_")

    url: str = Field(
        default="postgresql+asyncpg://noema:noema@localhost:5432/noema",
        description="SQLAlchemy async connection URL",
    )
    pool_min: int = Field(default=2, ge=1)
    pool_max: int = Field(default=10, ge=1)
    pool_timeout: float = Field(default=30.0, gt=0)
    echo: bool = Field(default=False, description="SQLAlchemy echo (debug)")


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOEMA_REDIS_")

    url: str = Field(default="redis://localhost:6379/0")
    pool_max: int = Field(default=20, ge=1)
    socket_timeout: float = Field(default=5.0, gt=0)


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOEMA_LLM_")

    provider: str = Field(default="ollama", description="ollama | openai | anthropic")
    ollama_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.1")
    openai_api_key: SecretStr = Field(default=SecretStr(""))
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))

    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    request_timeout: float = Field(default=120.0, gt=0)

    circuit_breaker_threshold: int = Field(
        default=5, ge=1, description="Failures before circuit opens"
    )
    circuit_breaker_recovery: float = Field(default=30.0, gt=0, description="Seconds before retry")
    retry_max: int = Field(default=3, ge=0)
    retry_base_delay: float = Field(default=1.0, gt=0)
    retry_max_delay: float = Field(default=30.0, gt=0)


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOEMA_API_")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)
    reload: bool = Field(default=False)

    # Auth
    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Master API key. Empty = auth disabled (dev only).",
    )
    api_key_header: str = Field(default="X-API-Key")

    # Inbound webhook verification
    webhook_secret: SecretStr = Field(
        default=SecretStr(""),
        description="HMAC secret for /webhooks/incident. Empty = verification disabled.",
    )

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_rpm: int = Field(default=60, ge=1, description="Requests per minute per key")
    rate_limit_burst: int = Field(default=10, ge=1, description="Burst allowance")
    trusted_proxies: list[str] = Field(
        default_factory=list,
        description="IPs/CIDRs of trusted reverse proxies that may set X-Forwarded-For. "
        "Empty = forwarded headers are ignored for client identity (anti-spoofing).",
    )

    # CORS
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    cors_methods: list[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])

    # Limits
    max_request_body: int = Field(default=1_048_576, description="Max request body in bytes (1 MB)")
    max_title_length: int = Field(default=200, ge=1)
    max_description_length: int = Field(default=10_000, ge=1)
    max_tags: int = Field(default=20, ge=1)


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOEMA_WORKER_")

    pool_size: int = Field(default=10, ge=1)
    max_queue: int = Field(default=100, ge=1)
    hierarchy_max_depth: int = Field(default=10, ge=1)
    hierarchy_max_concurrent: int = Field(default=50, ge=1)
    task_timeout: float = Field(default=300.0, gt=0)


class MemorySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOEMA_MEMORY_")

    data_dir: Path = Field(default=_PROJECT_ROOT / "data")
    backup_enabled: bool = Field(default=True)
    backup_count: int = Field(default=5, ge=0, description="Rotating backups")
    auto_save_interval: float = Field(default=30.0, gt=0, description="Seconds between auto-saves")


class SandboxSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOEMA_SANDBOX_")

    enabled: bool = Field(default=True)
    docker_image: str = Field(default="python:3.12-slim")
    timeout: float = Field(default=60.0, gt=0)
    max_memory: str = Field(default="256m")
    max_cpus: float = Field(default=0.5, gt=0)
    network_disabled: bool = Field(default=True)
    read_only_root: bool = Field(default=True)
    verify_think: bool = Field(
        default=True,
        description="Validate every generated Python block with the static pass "
        "before returning a solution from think()",
    )
    verify_think_enforce: bool = Field(
        default=False,
        description="Raise SandboxValidationError when the think() static gate "
        "rejects generated code (fail-closed)",
    )


class JudgeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOEMA_JUDGE_")

    enforce: bool = Field(
        default=False,
        description="Fail-closed: raise JudgeError when a solution fails the "
        "judge gate (score below threshold or judge crashed)",
    )


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOEMA_OBS_")

    logging_level: str = Field(default="INFO")
    logging_format: str = Field(default="json", description="json | console")
    metrics_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=9090, ge=1, le=65535)
    tracing_enabled: bool = Field(default=False)
    tracing_endpoint: str = Field(default="http://localhost:4318")
    sentry_dsn: str = Field(default="", description="Sentry DSN for error tracking")
    sentry_environment: str = Field(default="production")
    sentry_traces_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    sentry_profiles_sample_rate: float = Field(default=0.05, ge=0.0, le=1.0)


class NeurosymbolicSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOEMA_NS_")

    enabled: bool = Field(default=False)
    max_refinement_attempts: int = Field(default=3, ge=1, le=10)
    verification_timeout: float = Field(default=5.0, gt=0)
    evolution_enabled: bool = Field(default=True)
    fallback_to_cot: bool = Field(default=True)

    # Causal reasoning
    causal_enabled: bool = Field(default=True)
    causal_max_counterfactuals: int = Field(default=5, ge=1, le=20)


class AuditSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOEMA_AUDIT_")

    merkle_chain_enabled: bool = Field(default=True)
    merkle_chain_id: str = Field(default="")


class AutonomySettings(BaseSettings):
    """Configuration for the incident→PR autonomy loop (T2.1)."""

    model_config = SettingsConfigDict(env_prefix="NOEMA_AUTONOMY_")

    github_token: SecretStr = Field(
        default=SecretStr(""), description="GitHub personal access token"
    )
    github_repo: str = Field(default="", description="'owner/name' repository for fix PRs")
    github_base_branch: str = Field(default="main")

    auto_approve: bool = Field(
        default=False,
        description="Approve the fix PR (GitHub review) when the merge gate passes",
    )
    auto_merge: bool = Field(
        default=False,
        description="Merge the fix PR when the merge gate passes (requires auto_approve)",
    )
    lean_verifier: bool = Field(
        default=False,
        description="Require Lean 4 theorem-prover checks on PR specs (.lean files) "
        "before the merge gate can pass. Fail-closed: when the lean binary is "
        "missing, the gate blocks instead of skipping the formal stage",
    )
    lean_verifier_required_paths: list[str] = Field(
        default_factory=list,
        description="Path prefixes (e.g. 'crypto/') whose changed .py files must "
        "ship a matching .lean spec or the merge gate blocks with "
        "'missing_formal_spec'",
    )


# ─── Root config ─────────────────────────────────────────────────────────
class NoemaSettings(BaseSettings):
    """Single source of truth for every tunable knob in Noema."""

    model_config = SettingsConfigDict(
        env_prefix="NOEMA_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        """Env vars flow *only* through :meth:`from_yaml`'s ``_env_overrides``.

        Direct ``NoemaSettings()`` construction (init command, tests) must not
        be surprised by stray ``NOEMA_*`` variables; and the override layer
        rejects malformed JSON for container fields with a clear message
        instead of pydantic's cryptic ``EnvSettingsSource`` error.
        """
        return (init_settings,)

    # Sub-configs
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    api: APISettings = Field(default_factory=APISettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    judge: JudgeSettings = Field(default_factory=JudgeSettings)
    obs: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    neurosymbolic: NeurosymbolicSettings = Field(default_factory=NeurosymbolicSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    autonomy: AutonomySettings = Field(default_factory=AutonomySettings)

    # Chain-of-thought
    cot_max_steps: int = Field(default=12, ge=1)
    cot_temperature: float = Field(default=0.4, ge=0.0, le=2.0)

    # End-to-end think() ceiling: a runaway task (LLM retries × reflexion
    # attempts × steps) is cancelled fail-closed with checkpoints intact.
    think_timeout_seconds: float = Field(
        default=900.0,
        gt=0,
        description="Hard ceiling for one think() call; raises ThinkTimeoutError",
    )

    # Evolution
    evolution_enabled: bool = Field(default=True)
    evolution_auto_apply: bool = Field(default=False, description="Never auto-apply without tests")
    evolution_test_before_apply: bool = Field(default=True)

    # Knowledge
    knowledge_persist_path: Path = Field(default=_PROJECT_ROOT / "data" / "knowledge.json")
    feedback_persist_path: Path = Field(default=_PROJECT_ROOT / "data" / "feedback.json")
    ontology_persist_path: Path = Field(default=_PROJECT_ROOT / "data" / "ontology.json")

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> NoemaSettings:
        """Load settings from a YAML file, falling back to env vars.

        Precedence: environment variables > YAML file > field defaults.
        """
        cfg_path = Path(path) if path else _DEFAULT_CONFIG
        data: dict[str, Any] = {}
        if cfg_path.is_file():
            with open(cfg_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            data = {}
        merged = _deep_merge(data, _env_overrides())
        return cls(**merged)

    def dump_yaml(self, path: str | Path) -> None:
        """Write current settings to YAML (secrets masked, roundtrip-safe).

        ``model_dump(mode="json")`` converts Path/Enum/Secret fields to plain
        JSON primitives, so the emitted file is reloadable by ``from_yaml``.
        """
        from pydantic import SecretBytes, SecretStr

        def _mask(obj: Any) -> Any:
            if isinstance(obj, (SecretStr, SecretBytes)):
                return "***"
            if isinstance(obj, dict):
                return {k: _mask(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_mask(v) for v in obj]
            return obj

        dumped = self.model_dump(mode="json")
        masked = _mask(dumped)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(masked, f, default_flow_style=False, allow_unicode=True)


# ─── Singleton ───────────────────────────────────────────────────────────
_settings: NoemaSettings | None = None


def get_settings() -> NoemaSettings:
    """Get or create the global settings singleton."""
    global _settings
    if _settings is None:
        config_path = os.environ.get("NOEMA_CONFIG")
        _settings = NoemaSettings.from_yaml(config_path)
    return _settings


def reset_settings() -> None:
    """Reset the singleton (for testing)."""
    global _settings
    _settings = None
