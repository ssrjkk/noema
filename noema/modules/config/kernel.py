"""Configuration management module — 12-factor app, env-based config, feature flags, secrets management."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeatureFlag:
    name: str
    enabled: bool = False
    rollout_percentage: int = 0
    conditions: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigSchema:
    name: str
    type: str = "string"
    required: bool = False
    default: Any = None
    description: str = ""
    sensitive: bool = False


TWELVE_FACTOR_CHECKLIST = {
    "I. Codebase": "One codebase tracked in revision control, many deploys",
    "II. Dependencies": "Explicitly declare and isolate dependencies",
    "III. Config": "Store config in the environment",
    "IV. Backing services": "Treat backing services as attached resources",
    "V. Build, release, run": "Strictly separate the build and run stages",
    "VI. Processes": "Execute the app as one or more stateless processes",
    "VII. Port binding": "Export services via port binding",
    "VIII. Concurrency": "Scale out via the process model",
    "IX. Disposability": "Maximize robustness with fast startup and graceful shutdown",
    "X. Dev/prod parity": "Keep development, staging, and production as similar as possible",
    "XI. Logs": "Treat logs as event streams",
    "XII. Admin processes": "Run admin/management tasks as one-off processes",
}


class ConfigManager:
    def __init__(self) -> None:
        self._schemas: dict[str, list[ConfigSchema]] = {}
        self._configs: dict[str, dict[str, Any]] = {}

    def generate_config_schema(self, requirements: dict[str, Any]) -> list[ConfigSchema]:
        schemas = []
        for name, details in requirements.items():
            if isinstance(details, dict):
                schemas.append(
                    ConfigSchema(
                        name=name,
                        type=details.get("type", "string"),
                        required=details.get("required", False),
                        default=details.get("default", None),
                        description=details.get("description", ""),
                        sensitive=details.get("sensitive", False),
                    )
                )
            elif isinstance(details, str):
                schemas.append(
                    ConfigSchema(
                        name=name,
                        type="string",
                        required=True,
                        description=details,
                    )
                )
            elif isinstance(details, bool):
                schemas.append(
                    ConfigSchema(
                        name=name,
                        type="boolean",
                        required=False,
                        default=details,
                    )
                )
            elif isinstance(details, int):
                schemas.append(
                    ConfigSchema(
                        name=name,
                        type="integer",
                        required=False,
                        default=details,
                    )
                )
            elif isinstance(details, float):
                schemas.append(
                    ConfigSchema(
                        name=name,
                        type="number",
                        required=False,
                        default=details,
                    )
                )
            else:
                schemas.append(
                    ConfigSchema(
                        name=name,
                        type="string",
                        required=False,
                        default=str(details),
                    )
                )
        return schemas

    def validate_config(self, config: dict[str, Any], schema: list[ConfigSchema]) -> dict[str, Any]:
        errors = []
        warnings = []
        validated = {}

        {s.name: s for s in schema}

        for s in schema:
            value = config.get(s.name)

            if value is None and s.required:
                if s.default is not None:
                    validated[s.name] = s.default
                    warnings.append(f"'{s.name}' not provided, using default: {s.default}")
                else:
                    errors.append(f"'{s.name}' is required but not provided")
                continue

            if value is None:
                validated[s.name] = s.default
                continue

            if s.type == "integer":
                try:
                    validated[s.name] = int(value)
                except (ValueError, TypeError):
                    errors.append(f"'{s.name}' must be an integer, got '{value}'")
            elif s.type == "number":
                try:
                    validated[s.name] = float(value)
                except (ValueError, TypeError):
                    errors.append(f"'{s.name}' must be a number, got '{value}'")
            elif s.type == "boolean":
                if isinstance(value, bool):
                    validated[s.name] = value
                elif isinstance(value, str):
                    validated[s.name] = value.lower() in ("true", "1", "yes", "on")
                else:
                    validated[s.name] = bool(value)
            elif s.type == "list":
                if isinstance(value, list):
                    validated[s.name] = value
                elif isinstance(value, str):
                    validated[s.name] = [v.strip() for v in value.split(",")]
                else:
                    validated[s.name] = [value]
            elif s.type == "json":
                if isinstance(value, str):
                    try:
                        validated[s.name] = json.loads(value)
                    except json.JSONDecodeError:
                        errors.append(f"'{s.name}' must be valid JSON")
                        validated[s.name] = value
                else:
                    validated[s.name] = value
            else:
                validated[s.name] = str(value) if value is not None else value

        return {
            "valid": len(errors) == 0,
            "config": validated,
            "errors": errors,
            "warnings": warnings,
        }

    def generate_env_files(self, schema: list[ConfigSchema]) -> dict[str, str]:
        env_files = {}
        env_lines = []
        env_example_lines = []
        docker_env_lines = []
        k8s_secret_lines = []

        for s in schema:
            env_name = re.sub(r"[^A-Z0-9_]", "_", s.name.upper())

            if s.sensitive:
                placeholder = hashlib.md5(s.name.encode()).hexdigest()[:16].upper()
                env_lines.append(f"# {s.description}" if s.description else f"# {s.name}")
                env_lines.append(f"{env_name}=CHANGE_ME_{placeholder}")
                env_lines.append("")

                env_example_lines.append(f"# {s.description}" if s.description else f"# {s.name}")
                env_example_lines.append(f"{env_name}=")
                env_example_lines.append("")

                docker_env_lines.append(f'    {env_name}: "CHANGE_ME_{placeholder}"')

                safe_b64 = __import__("base64").b64encode(b"PLACEHOLDER").decode()
                k8s_secret_lines.append(f"  {env_name.lower()}: {safe_b64}")
            else:
                default_val = str(s.default) if s.default is not None else ""
                env_lines.append(f"# {s.description}" if s.description else f"# {s.name}")
                env_lines.append(f"{env_name}={default_val}")
                env_lines.append("")

                env_example_lines.append(f"# {s.description}" if s.description else f"# {s.name}")
                env_example_lines.append(f"{env_name}={default_val}")
                env_example_lines.append("")

                docker_env_lines.append(f'    {env_name}: "{default_val}"')

        env_files[".env"] = "\n".join(env_lines)
        env_files[".env.example"] = "\n".join(env_example_lines)

        docker_compose_env = "version: '3.8'\nservices:\n  app:\n    environment:\n"
        docker_compose_env += "\n".join(docker_env_lines)
        env_files["docker-compose.env.yml"] = docker_compose_env

        if k8s_secret_lines:
            k8s_yaml = "apiVersion: v1\nkind: Secret\nmetadata:\n  name: app-secrets\ntype: Opaque\ndata:\n"
            k8s_yaml += "\n".join(k8s_secret_lines)
            env_files["k8s-secret.yaml"] = k8s_yaml

        return env_files

    def generate_12factor_checklist(self, app_config: dict[str, Any]) -> dict[str, Any]:
        checks = {}
        for factor, description in TWELVE_FACTOR_CHECKLIST.items():
            status = "needs_review"
            notes = ""

            if factor == "III. Config":
                if app_config.get("env_based_config", False):
                    status = "pass"
                    notes = "Config stored in environment variables"
                else:
                    status = "fail"
                    notes = "Config should be moved to environment variables"

            elif factor == "VII. Port binding":
                if app_config.get("port_binding", False):
                    status = "pass"
                    notes = "Service exports via port binding"
                else:
                    status = "warning"
                    notes = "Ensure app binds to PORT env var"

            elif factor == "VIII. Concurrency":
                process_model = app_config.get("process_model", "")
                if process_model in ("process", "worker", "hybrid"):
                    status = "pass"
                    notes = f"Process model: {process_model}"
                else:
                    status = "warning"
                    notes = "Consider process model for horizontal scaling"

            elif factor == "XI. Logs":
                if app_config.get("logs_to_stdout", False):
                    status = "pass"
                    notes = "Logs written to stdout"
                else:
                    status = "fail"
                    notes = "Logs should go to stdout/stderr, not files"

            else:
                notes = f"Review: {description}"

            checks[factor] = {
                "status": status,
                "description": description,
                "notes": notes,
            }

        pass_count = sum(1 for c in checks.values() if c["status"] == "pass")
        total = len(checks)

        return {
            "checklist": checks,
            "score": f"{pass_count}/{total}",
            "score_percent": int(pass_count / total * 100) if total > 0 else 0,
            "summary": f"{pass_count} of {total} factors verified",
        }


class ConfigModule:
    NAME = "config"
    DESCRIPTION = "Configuration management: 12-factor app, env-based config, feature flags, secrets management"

    def __init__(self) -> None:
        self.manager: ConfigManager = ConfigManager()

    def execute(self, task: Any) -> dict[str, Any]:
        task_title = getattr(task, "title", "")
        task_tags = getattr(task, "tags", [])

        action = "schema"
        if "validate" in str(task_title).lower() or "validate" in task_tags:
            action = "validate"
        elif "env" in str(task_title).lower() or "env" in task_tags or "secrets" in task_tags:
            action = "env_files"
        elif (
            "12factor" in str(task_title).lower()
            or "twelve" in str(task_title).lower()
            or "12-factor" in task_tags
        ):
            action = "12factor"
        elif "feature" in str(task_title).lower() or "flag" in str(task_title).lower():
            action = "feature_flags"

        requirements = {}
        config = {}
        if hasattr(task, "metadata"):
            metadata = task.metadata
            requirements = metadata.get("requirements", {})
            config = metadata.get("config", {})

        if action == "schema":
            schemas = self.manager.generate_config_schema(
                requirements
                or {
                    "DATABASE_URL": {
                        "type": "string",
                        "required": True,
                        "sensitive": True,
                        "description": "Database connection URL",
                    },
                    "REDIS_URL": {
                        "type": "string",
                        "required": False,
                        "default": "redis://localhost:6379",
                        "description": "Redis connection URL",
                    },
                    "DEBUG": {
                        "type": "boolean",
                        "required": False,
                        "default": False,
                        "description": "Enable debug mode",
                    },
                    "LOG_LEVEL": {
                        "type": "string",
                        "required": False,
                        "default": "INFO",
                        "description": "Logging level",
                    },
                    "PORT": {
                        "type": "integer",
                        "required": False,
                        "default": 8000,
                        "description": "Server port",
                    },
                    "WORKERS": {
                        "type": "integer",
                        "required": False,
                        "default": 4,
                        "description": "Number of workers",
                    },
                    "SECRET_KEY": {
                        "type": "string",
                        "required": True,
                        "sensitive": True,
                        "description": "Application secret key",
                    },
                    "CORS_ORIGINS": {
                        "type": "list",
                        "required": False,
                        "default": ["http://localhost:3000"],
                        "description": "Allowed CORS origins",
                    },
                }
            )
            return {
                "action": "schema",
                "schema": [
                    {
                        "name": s.name,
                        "type": s.type,
                        "required": s.required,
                        "default": s.default,
                        "description": s.description,
                        "sensitive": s.sensitive,
                    }
                    for s in schemas
                ],
                "_confidence": 0.85,
            }
        elif action == "validate":
            schemas = self.manager.generate_config_schema(requirements or {})
            result = self.manager.validate_config(config or {}, schemas)
            result["_confidence"] = 0.90
            return result
        elif action == "env_files":
            schemas = self.manager.generate_config_schema(requirements or {})
            env_files = self.manager.generate_env_files(schemas)
            return {
                "action": "env_files",
                "files": env_files,
                "_confidence": 0.85,
            }
        elif action == "12factor":
            result = self.manager.generate_12factor_checklist(config or {})
            result["action"] = "12factor"
            result["_confidence"] = 0.80
            return result
        elif action == "feature_flags":
            flags = []
            if hasattr(task, "metadata"):
                flag_defs = task.metadata.get("flags", {})
                for name, details in flag_defs.items():
                    if isinstance(details, dict):
                        flags.append(
                            FeatureFlag(
                                name=name,
                                enabled=details.get("enabled", False),
                                rollout_percentage=details.get("rollout_percentage", 0),
                                conditions=details.get("conditions", {}),
                            )
                        )
                    else:
                        flags.append(FeatureFlag(name=name, enabled=bool(details)))
            return {
                "action": "feature_flags",
                "flags": [
                    {
                        "name": f.name,
                        "enabled": f.enabled,
                        "rollout_percentage": f.rollout_percentage,
                        "conditions": f.conditions,
                    }
                    for f in flags
                ],
                "_confidence": 0.75,
            }

        return {
            "action": "schema",
            "message": "No specific action matched. Defaulting to schema generation.",
            "_confidence": 0.50,
        }
