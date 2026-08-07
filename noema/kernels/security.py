"""Ядро безопасности — аудит, защита, best practices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from noema.kernels.base import BaseKernel
from noema.logging import get_logger

if TYPE_CHECKING:
    from noema.core.types import Task

logger = get_logger(__name__)


class SecurityKernel(BaseKernel):
    """Ядро анализа безопасности."""

    @property
    def name(self) -> str:
        return "security"

    @property
    def description(self) -> str:
        return "Аудит безопасности, защита, compliance"

    async def execute(self, task: Task, **kwargs) -> dict[str, Any]:
        tags = {t.lower() for t in task.tags}

        checks = self._base_security_checks()
        hardening = self._security_hardening(task)
        compliance = self._compliance_checks(task)
        secrets_mgmt = self._secrets_management(task)

        if "web" in tags or "api" in tags:
            checks.extend(self._web_security(task))
        if "auth" in tags or "user" in tags:
            checks.extend(self._auth_security(task))
        if "ml" in tags or "data" in tags:
            checks.extend(self._data_security(task))

        return {
            "type": "security",
            "checks": checks,
            "hardening": hardening,
            "compliance": compliance,
            "secrets_management": secrets_mgmt,
            "risk_score": self._calculate_risk(checks),
            "_confidence": 0.8,
        }

    def _base_security_checks(self) -> list[dict]:
        return [
            {
                "category": "dependency",
                "check": "Dependency Vulnerability Scan",
                "description": "Проверка зависимостей через safety/bandit/trivy",
                "tool": "safety check && bandit -r . && trivy fs .",
                "severity": "high",
            },
            {
                "category": "secrets",
                "check": "No Hardcoded Secrets",
                "description": "Проверка на захардкоженные секреты в коде",
                "tool": "trufflehog . && gitleaks detect",
                "severity": "critical",
            },
            {
                "category": "docker",
                "check": "Docker Image Scan",
                "description": "Сканирование Docker-образа на уязвимости",
                "tool": "trivy image <image:tag>",
                "severity": "high",
            },
        ]

    def _web_security(self, task: Task) -> list[dict]:
        return [
            {
                "category": "web",
                "check": "SQL Injection Prevention",
                "description": "РСЃРїРѕР»СЊР·РѕРІР°РЅРёРµ parameterized queries / ORM",
                "recommendation": "Всегда использовать prepared statements или ORM",
                "severity": "critical",
            },
            {
                "category": "web",
                "check": "XSS Prevention",
                "description": "Санитайзинг вывода и CSP headers",
                "recommendation": "Content-Security-Policy header + output encoding",
                "severity": "high",
            },
            {
                "category": "web",
                "check": "Rate Limiting",
                "description": "Ограничение частоты запросов",
                "implementation": """
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.get("/api/endpoint")
@limiter.limit("100/minute")
async def endpoint(request: Request):
    ...
""",
                "severity": "medium",
            },
            {
                "category": "web",
                "check": "CORS Configuration",
                "description": "Строгая настройка CORS",
                "recommendation": "Allow only specific origins, never use * in production",
                "severity": "medium",
            },
        ]

    def _auth_security(self, task: Task) -> list[dict]:
        return [
            {
                "category": "auth",
                "check": "JWT Best Practices",
                "description": "Короткие TTL, refresh tokens, ротация ключей",
                "recommendation": "Access token: 15min, Refresh token: 7d, rotation on use",
                "severity": "high",
            },
            {
                "category": "auth",
                "check": "Password Hashing",
                "description": "bcrypt/argon2 для хеширования паролей",
                "implementation": """
from argon2 import PasswordHasher
ph = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
)
hash = ph.hash(password)
verified = ph.verify(hash, password)
""",
                "severity": "critical",
            },
            {
                "category": "auth",
                "check": "MFA Support",
                "description": "Многофакторная аутентификация",
                "severity": "high",
            },
        ]

    def _data_security(self, task: Task) -> list[dict]:
        return [
            {
                "category": "data",
                "check": "Encryption at Rest",
                "description": "Шифрование данных в БД (AES-256)",
                "severity": "high",
            },
            {
                "category": "data",
                "check": "Encryption in Transit",
                "description": "TLS 1.3 для всех соединений",
                "recommendation": "Enforce HTTPS, HSTS header, certificate pinning",
                "severity": "critical",
            },
            {
                "category": "data",
                "check": "PII Data Handling",
                "description": "Маскирование и ротация персональных данных",
                "severity": "high",
            },
        ]

    def _security_hardening(self, task: Task) -> list[str]:
        return [
            "Non-root Docker container (USER appuser)",
            "Read-only filesystem in containers",
            "Network policies in Kubernetes",
            "Security context: drop ALL capabilities, add only needed",
            "Pod security standards: restricted",
            "Secrets via Vault / AWS Secrets Manager / K8s Secrets",
            "Audit logging enabled",
            "Fail2ban for brute-force protection",
        ]

    def _compliance_checks(self, task: Task) -> list[dict]:
        return [
            {"standard": "OWASP Top 10", "status": "check required"},
            {"standard": "GDPR", "status": "check if handling EU data"},
            {"standard": "SOC 2", "status": "required for enterprise"},
        ]

    def _secrets_management(self, task: Task) -> dict:
        return {
            "recommended_tool": "HashiCorp Vault or AWS Secrets Manager",
            "pattern": """
# .env (git-ignored)
DATABASE_URL=postgresql://...
JWT_SECRET=...
API_KEY=...

# Loading in code
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    jwt_secret: str
    api_key: str

    class Config:
        env_file = ".env"
""",
            "rotation": "Rotate secrets every 90 days",
            "audit": "Log all secret access with caller identity",
        }

    def _calculate_risk(self, checks: list[dict]) -> dict[str, Any]:
        critical = sum(1 for c in checks if c.get("severity") == "critical")
        high = sum(1 for c in checks if c.get("severity") == "high")
        medium = sum(1 for c in checks if c.get("severity") == "medium")

        score = min(100, (critical * 25) + (high * 10) + (medium * 3))
        level = (
            "critical"
            if score > 70
            else "high"
            if score > 40
            else "medium"
            if score > 20
            else "low"
        )

        return {
            "score": score,
            "level": level,
            "critical": critical,
            "high": high,
            "medium": medium,
            "total_checks": len(checks),
        }
