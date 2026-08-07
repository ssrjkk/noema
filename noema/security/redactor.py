"""PII/Secret Redactor — вычищает ключи, токены и PII из промптов."""

from __future__ import annotations

import re
from typing import Any

# Regex patterns for common secrets and PII
PATTERNS: list[tuple[str, str, str]] = [
    (
        "stripe_live",
        r"(sk_live|sk_test|pk_live|pk_test|rk_live|rk_test)_[A-Za-z0-9]{10,60}",
        "[REDACTED-STRIPE-KEY]",
    ),
    ("stripe_webhook", r"whsec_[A-Za-z0-9]{10,60}", "[REDACTED-STRIPE-WHSEC]"),
    ("openai_key", r"sk-(proj-)?[A-Za-z0-9]{20,60}", "[REDACTED-OPENAI-KEY]"),
    ("anthropic_key", r"sk-ant-[A-Za-z0-9]{20,60}", "[REDACTED-ANTHROPIC-KEY]"),
    ("aws_key", r"AKIA[0-9A-Z]{16}", "[REDACTED-AWS-KEY]"),
    (
        "aws_secret",
        r"(?i)aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{40}",
        "[REDACTED-AWS-SECRET]",
    ),
    ("github_token", r"ghp_[A-Za-z0-9]{36,40}", "[REDACTED-GITHUB-TOKEN]"),
    ("github_old", r"github_token\s*[=:]\s*['\"]?[A-Za-z0-9]{40}", "[REDACTED-GITHUB-TOKEN]"),
    ("jwt", r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}", "[REDACTED-JWT]"),
    ("bearer", r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]{20,100}", "Bearer [REDACTED-TOKEN]"),
    (
        "basic_auth",
        r"(?i)Authorization:\s*Basic\s+[A-Za-z0-9+/=]{10,200}",
        "Authorization: Basic [REDACTED]",
    ),
    (
        "password",
        r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?[^\s"\'<>{},;]{4,50}["\']?',
        r"\1=[REDACTED-PASSWORD]",
    ),
    (
        "api_key_generic",
        r"(?i)(api_key|apikey|api-secret|secret_key)\s*[=:]\s*['\"]?[A-Za-z0-9_\-./=+]{10,60}",
        r"\1=[REDACTED-API-KEY]",
    ),
    ("email", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED-EMAIL]"),
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED-SSN]"),
    ("cc_number", r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[REDACTED-CC]"),
    (
        "ip_private",
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b",
        "[REDACTED-PRIVATE-IP]",
    ),
    ("phone", r"\+\d{1,3}[\d\s-]{6,12}", "[REDACTED-PHONE]"),
]


class Redactor:
    """Redacts PII and secrets from text using regex patterns."""

    def __init__(self, patterns: list[tuple[str, str, str]] | None = None) -> None:
        self._rules = patterns or PATTERNS
        self._compiled = [
            (name, re.compile(pattern), replacement) for name, pattern, replacement in self._rules
        ]

    def redact(self, text: str) -> str:
        for _name, compiled, replacement in self._compiled:
            text = compiled.sub(replacement, text)
        return text

    def redact_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {k: self.redact(v) if isinstance(v, str) else v for k, v in m.items()} for m in messages
        ]


_default_redactor = Redactor()


def redact_text(text: str) -> str:
    return _default_redactor.redact(text)


def redact_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _default_redactor.redact_messages(messages)
