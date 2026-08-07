"""RAG Injection Sanitizer — проверка документов при ingest на prompt injection."""

from __future__ import annotations

import re
from typing import Any

from noema.logging import get_logger

log = get_logger(__name__)


# Injection patterns — text that suggests intent to override system behavior
INJECTION_PATTERNS: list[tuple[str, str, str]] = [
    (
        "ignore_previous",
        r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|directions)",
        "high",
    ),
    (
        "new_instructions",
        r"(?i)(you\s+are\s+(now|henceforth)\s+)|(from\s+now\s+on\s+you\s+are)",
        "high",
    ),
    ("system_override", r"(?i)system\s+(prompt|instruction|message)\s*[:=]", "high"),
    (
        "role_override",
        r"(?i)your\s+(role|function|purpose)\s+is\s+(to\s+)?(ignore|disregard|forget)",
        "high",
    ),
    (
        "backdoor_code",
        r"(?i)(backdoor|trap.?door|malicious|payload)\s*(in|inside|inserted)",
        "high",
    ),
    (
        "dangerous_import",
        r"(?i)(import\s+os|import\s+subprocess|from\s+os\s+import|eval\s*\(|exec\s*\()",
        "medium",
    ),
    ("data_exfil", r"(?i)(curl\s+http|wget\s+http|requests?\.(get|post)\s*\([\"']http)", "medium"),
    ("reverse_shell", r"(?i)(rev(erse)?.?shell|bind.?shell|nc\s+-[elve]|ncat)", "critical"),
    ("base64_decode", r"(?i)(base64\.(b64decode|decode)\s*\(|\.decode\([\"']base64)", "medium"),
]


SUSPICIOUS_FUNCTIONS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "execfile",
    "os.system",
    "subprocess.Popen",
}


class IngestionSanitizer:
    """Scans documents for prompt injection attempts before adding to KnowledgeStore."""

    def __init__(self) -> None:
        self._patterns = [
            (name, re.compile(pat, re.IGNORECASE), severity)
            for name, pat, severity in INJECTION_PATTERNS
        ]

    def scan(self, text: str) -> dict[str, Any]:
        findings = []
        for name, compiled, severity in self._patterns:
            match = compiled.search(text)
            if not match:
                continue
            matches = compiled.findall(text)
            findings.append(
                {
                    "pattern": name,
                    "severity": severity,
                    "matches": len(matches),
                    "sample": text[match.start() : match.end()][:100],
                }
            )

        code_issues = self._scan_code(text)

        return {
            "is_suspicious": len(findings) > 0,
            "is_blocked": any(f["severity"] == "critical" for f in findings),
            "findings": findings,
            "code_issues": code_issues,
            "summary": self._summarize(findings, code_issues),
        }

    def _scan_code(self, text: str) -> list[dict[str, Any]]:
        issues = []
        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            for func in SUSPICIOUS_FUNCTIONS:
                if func in stripped and not stripped.startswith("#"):
                    issues.append(
                        {
                            "line": i + 1,
                            "function": func,
                            "context": stripped[:100],
                        }
                    )
        return issues

    def _summarize(self, findings: list[dict], code_issues: list[dict]) -> str:
        parts = []
        if findings:
            sevs = {f["severity"] for f in findings}
            parts.append(
                f"Prompt injection patterns detected: {len(findings)} ({', '.join(sorted(sevs))})"
            )
        if code_issues:
            parts.append(f"Suspicious code constructs: {len(code_issues)}")
        return "; ".join(parts) if parts else "Clean"
