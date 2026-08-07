"""Security scanning module — SAST code analysis, vulnerability detection, dependency scanning, secret detection, OWASP Top 10 checks."""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Vulnerability:
    id: str
    severity: str  # critical, high, medium, low, info
    category: str
    file: str
    line: int
    description: str
    cwe: str
    recommendation: str


@dataclass
class ScanResult:
    vulnerabilities: list[Vulnerability] = field(default_factory=list)
    score: int = 100
    summary: dict[str, Any] = field(default_factory=dict)


KNOWN_VULNERABILITY_PATTERNS = {
    "sql_injection": {
        "patterns": [
            r'execute\s*\(\s*["\'].*%s.*["\']\s*%',
            r'execute\s*\(\s*f["\'].*\{.*\}.*["\']',
            r'execute\s*\(\s*["\'].*\+\s*\w+',
            r"cursor\.execute\s*\(.*\+",
            r'query\s*=\s*["\'].*\+\s*\w+',
            r'\.raw\s*\(\s*f["\']',
            r"WHERE.*\+\s*\w+",
            r"SELECT.*\+\s*\w+",
        ],
        "severity": "critical",
        "cwe": "CWE-89",
        "recommendation": "Use parameterized queries or ORM instead of string concatenation",
    },
    "xss": {
        "patterns": [
            r"innerHTML\s*=",
            r"document\.write\s*\(",
            r"\.html\s*\(\s*\w+\s*\)",
            r"v-html\s*=",
            r"dangerouslySetInnerHTML",
            r"\brender_template_string\s*\(",
            r"Markup\s*\(\s*\w+\s*\)",
            r"\\binnerHTML\\b",
            r"response\.write\s*\(.*\+",
        ],
        "severity": "high",
        "cwe": "CWE-79",
        "recommendation": "Sanitize and escape all user input; use templating engines with auto-escaping",
    },
    "path_traversal": {
        "patterns": [
            r"open\s*\(.*\+",
            r'open\s*\(\s*f["\']',
            r"send_file\s*\(.*\+",
            r"readFile\s*\(.*\+",
            r"os\.path\.join\s*\(.*request",
            r"Path\s*\(.*request",
            r"\.\.\/",
            r"\.\.\\\\",
        ],
        "severity": "high",
        "cwe": "CWE-22",
        "recommendation": "Validate and sanitize file paths; use allowlists for permitted directories",
    },
    "hardcoded_secrets": {
        "patterns": [
            r'(?i)(password|passwd|pwd)\s*=\s*["\'][^"\']+["\']',
            r'(?i)(api_key|apikey|api_secret)\s*=\s*["\'][^"\']+["\']',
            r'(?i)(secret_key|secret)\s*=\s*["\'][^"\']+["\']',
            r'(?i)(access_token|auth_token)\s*=\s*["\'][^"\']+["\']',
            r'(?i)(private_key)\s*=\s*["\'][^"\']+["\']',
            r'(?i)(database_url|db_url|dsn)\s*=\s*["\'][^"\']+["\']',
            r'AWS_ACCESS_KEY_ID\s*=\s*["\'][A-Z0-9]{20}["\']',
            r'AWS_SECRET_ACCESS_KEY\s*=\s*["\'][A-Za-z0-9/+=]{40}["\']',
        ],
        "severity": "critical",
        "cwe": "CWE-798",
        "recommendation": "Use environment variables or a secrets manager instead of hardcoding credentials",
    },
    "insecure_deserialization": {
        "patterns": [
            r"pickle\.loads?\s*\(",
            r"pickle\.load\s*\(",
            r"yaml\.load\s*\([^)]*\)(?!.*Loader)",
            r"marshal\.loads?\s*\(",
            r"shelve\.open\s*\(",
            r"jsonpickle\.decode\s*\(",
            r"eval\s*\(",
            r"exec\s*\(",
        ],
        "severity": "critical",
        "cwe": "CWE-502",
        "recommendation": "Avoid deserializing untrusted data; use safe alternatives like JSON or yaml.safe_load",
    },
    "ssrf": {
        "patterns": [
            r"requests\.(get|post|put|delete|patch|head)\s*\(.*\+",
            r"urllib\.request\.urlopen\s*\(.*\+",
            r"urlopen\s*\(.*\+",
            r"HTTPConnection\s*\(.*\+",
            r"httpx\.(get|post|put|delete)\s*\(.*\+",
            r"aiohttp\.ClientSession.*get\s*\(.*\+",
        ],
        "severity": "high",
        "cwe": "CWE-918",
        "recommendation": "Validate and allowlist target URLs; avoid passing user input directly to HTTP requests",
    },
    "command_injection": {
        "patterns": [
            r"os\.system\s*\(.*\+",
            r'os\.system\s*\(\s*f["\']',
            r"subprocess\.call\s*\(.*shell\s*=\s*True",
            r"subprocess\.Popen\s*\(.*shell\s*=\s*True",
            r"subprocess\.run\s*\(.*shell\s*=\s*True",
            r"os\.popen\s*\(",
            r"commands\.getoutput\s*\(",
            r"eval\s*\(",
            r"child_process\.exec\s*\(.*\+",
            r"child_process\.execSync\s*\(.*\+",
        ],
        "severity": "critical",
        "cwe": "CWE-78",
        "recommendation": "Use subprocess with list arguments; never pass user input to shell commands",
    },
    "weak_crypto": {
        "patterns": [
            r"hashlib\.md5\s*\(",
            r"hashlib\.sha1\s*\(",
            r"(?i)hashlib\.(md5|sha1)\b",
            r"MD5\s*\(",
            r"SHA1\s*\(",
            r"DES\s*\(",
            r"RC4\s*\(",
            r"Blowfish\s*\(",
            r"random\.random\s*\(",
            r"random\.randint\s*\(",
            r"random\.choice\s*\(",
            r"Math\.random\s*\(",
        ],
        "severity": "medium",
        "cwe": "CWE-327",
        "recommendation": "Use modern algorithms like SHA-256+, AES-256; use secrets module for cryptographic randomness",
    },
}

OWASP_TOP_10 = {
    "A01:2021 Broken Access Control": {
        "patterns": [
            r"@app\.route.*methods\s*=\s*\[.*PUT.*DELETE",
            r"@app\.route(?!.*login)(?!.*auth)",
            r"\.is_admin\s*=\s*True",
            r"bypass.*auth",
            r"without.*permission",
        ],
        "cwe": "CWE-862",
        "description": "Restrictions on what authenticated users are allowed to do are not properly enforced",
    },
    "A02:2021 Cryptographic Failures": {
        "patterns": [
            r"hashlib\.md5",
            r"hashlib\.sha1",
            r"DES\b",
            r"ECB\b",
            r'password.*=\s*["\'].*["\']',
            r"(?i)encrypt.*md5",
        ],
        "cwe": "CWE-310",
        "description": "Failures related to cryptography which often leads to sensitive data exposure",
    },
    "A03:2021 Injection": {
        "patterns": [
            r"execute\s*\(.*\+",
            r"execute\s*\(.*%",
            r'execute\s*\(.*f["\']',
            r"innerHTML",
            r"eval\s*\(",
            r"os\.system\s*\(",
            r"shell\s*=\s*True",
        ],
        "cwe": "CWE-89",
        "description": "User-supplied data is not validated, filtered, or sanitized",
    },
    "A04:2021 Insecure Design": {
        "patterns": [
            r"# TODO.*security",
            r"# FIXME.*auth",
            r"# HACK",
            r"except.*pass",
            r"except:.*pass",
        ],
        "cwe": "CWE-200",
        "description": "Risks related to design flaws, missing or ineffective security controls",
    },
    "A05:2021 Security Misconfiguration": {
        "patterns": [
            r"DEBUG\s*=\s*True",
            r"debug\s*=\s*true",
            r"ALLOWED_HOSTS\s*=\s*\[.*\*",
            r"CORS_ALLOW_ALL_ORIGINS\s*=\s*True",
            r"SECURE_SSL_REDIRECT\s*=\s*False",
        ],
        "cwe": "CWE-16",
        "description": "Missing appropriate security hardening across any part of the application",
    },
    "A06:2021 Vulnerable and Outdated Components": {
        "patterns": [
            r"version.*<\s*\d",
            r"jquery.*1\.\d",
            r"django.*[12]\.",
            r"flask.*0\.",
        ],
        "cwe": "CWE-1104",
        "description": "Using components with known vulnerabilities",
    },
    "A07:2021 Identification and Authentication Failures": {
        "patterns": [
            r"login.*password",
            r"auth.*token.*=",
            r"session.*=\s*True",
            r"max_age.*=\s*\d{6,}",
        ],
        "cwe": "CWE-287",
        "description": "Confirmation of the user's identity, authentication, and session management is not implemented correctly",
    },
    "A08:2021 Software and Data Integrity Failures": {
        "patterns": [
            r"pickle\.load",
            r"yaml\.load\s*\([^,)]+\)",
            r"eval\s*\(",
            r"exec\s*\(",
            r"allow_redirects\s*=\s*True",
        ],
        "cwe": "CWE-345",
        "description": "Code and infrastructure that does not protect against integrity violations",
    },
    "A09:2021 Security Logging and Monitoring Failures": {
        "patterns": [
            r"except.*:\s*$",
            r"except.*pass\s*$",
            r"print\s*\(.*error",
            r"logging\.(debug|info)\s*\(.*password",
        ],
        "cwe": "CWE-778",
        "description": "Insufficient logging, detection, monitoring, and active response",
    },
    "A10:2021 Server-Side Request Forgery (SSRF)": {
        "patterns": [
            r"requests\.(get|post)\s*\(.*\+",
            r"urlopen\s*\(.*\+",
            r"http\.get\s*\(.*\+",
            r"fetch\s*\(.*\+",
        ],
        "cwe": "CWE-918",
        "description": "SSRF flaws occur when a web application fetches a remote resource without validating the user-supplied URL",
    },
}


class SecurityScanner:
    def __init__(self) -> None:
        self.patterns = KNOWN_VULNERABILITY_PATTERNS
        self.owasp = OWASP_TOP_10
        self._vuln_counter = 0

    def _next_id(self) -> str:
        self._vuln_counter += 1
        return f"VULN-{self._vuln_counter:04d}"

    def _check_patterns(
        self, code: str, category: str, pattern_info: dict, file: str
    ) -> list[Vulnerability]:
        vulns = []
        lines = code.split("\n")
        for pattern_str in pattern_info["patterns"]:
            regex = re.compile(pattern_str)
            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    vulns.append(
                        Vulnerability(
                            id=self._next_id(),
                            severity=pattern_info["severity"],
                            category=category,
                            file=file,
                            line=i,
                            description=f"Potential {category.replace('_', ' ')} detected: {line.strip()[:80]}",
                            cwe=pattern_info["cwe"],
                            recommendation=pattern_info["recommendation"],
                        )
                    )
                    break
        return vulns

    def scan_code(self, code: str, language: str = "python") -> ScanResult:
        self._vuln_counter = 0
        vulns = []
        lang_lower = language.lower()

        lang_applies = {
            "sql_injection": lang_lower
            in ("python", "javascript", "typescript", "java", "php", "ruby"),
            "xss": lang_lower in ("python", "javascript", "typescript", "html", "jsx", "tsx"),
            "path_traversal": True,
            "hardcoded_secrets": True,
            "insecure_deserialization": lang_lower in ("python", "java"),
            "ssrf": lang_lower in ("python", "javascript", "typescript", "go"),
            "command_injection": True,
            "weak_crypto": lang_lower in ("python", "javascript", "typescript", "java", "go"),
        }

        filtered_patterns = {
            cat: info for cat, info in self.patterns.items() if lang_applies.get(cat, True)
        }

        for category, pattern_info in filtered_patterns.items():
            vulns.extend(self._check_patterns(code, category, pattern_info, "<input>"))

        score = max(0, 100 - len(vulns) * 5)
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for v in vulns:
            severity_counts[v.severity] = severity_counts.get(v.severity, 0) + 1

        return ScanResult(
            vulnerabilities=vulns,
            score=score,
            summary={
                "total": len(vulns),
                "severity_counts": severity_counts,
                "language": language,
                "score": score,
            },
        )

    def scan_dependencies(self, deps_file_content: str) -> ScanResult:
        self._vuln_counter = 0
        vulns = []
        known_vulnerable = {
            "django": {"below": "4.2", "cve": "CVE-2023-41164", "severity": "high"},
            "flask": {"below": "2.3", "cve": "CVE-2023-30861", "severity": "medium"},
            "requests": {"below": "2.31", "cve": "CVE-2023-32681", "severity": "medium"},
            "pillow": {"below": "10.0", "cve": "CVE-2023-44271", "severity": "high"},
            "jinja2": {"below": "3.1", "cve": "CVE-2024-22195", "severity": "medium"},
            "express": {"below": "4.19", "cve": "Multiple", "severity": "medium"},
            "lodash": {"below": "4.17.21", "cve": "CVE-2021-23337", "severity": "high"},
            "jquery": {"below": "3.6", "cve": "CVE-2020-23064", "severity": "medium"},
            "cryptography": {"below": "41.0", "cve": "CVE-2023-49083", "severity": "high"},
            "pyyaml": {"below": "6.0", "cve": "CVE-2020-1747", "severity": "critical"},
        }

        lines = deps_file_content.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for pkg, info in known_vulnerable.items():
                if pkg.lower() in line.lower():
                    vulns.append(
                        Vulnerability(
                            id=self._next_id(),
                            severity=info["severity"],
                            category="vulnerable_dependency",
                            file="deps",
                            line=0,
                            description=f"Potentially vulnerable dependency: {line.strip()} ({info['cve']})",
                            cwe="CWE-1395",
                            recommendation=f"Upgrade {pkg} to {info['below']} or later",
                        )
                    )

        score = max(0, 100 - len(vulns) * 10)
        return ScanResult(
            vulnerabilities=vulns,
            score=score,
            summary={"total": len(vulns), "packages_scanned": len(lines)},
        )

    def scan_secrets(self, code: str) -> ScanResult:
        self._vuln_counter = 0
        vulns = []
        lines = code.split("\n")
        secret_patterns = [
            (r'(?i)(password|passwd|pwd)\s*=\s*["\'][^"\']{4,}["\']', "Hardcoded password"),
            (r'(?i)(api_key|apikey|api_secret)\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded API key"),
            (r'(?i)(secret_key|secret)\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded secret key"),
            (r'(?i)(access_token|auth_token)\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded token"),
            (r'(?i)(private_key)\s*=\s*["\'][^"\']{16,}["\']', "Hardcoded private key"),
            (
                r'(?i)(database_url|db_url|dsn)\s*=\s*["\'][^"\']{12,}["\']',
                "Hardcoded database URL",
            ),
            (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
            (r"(?i)-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----", "Private key block"),
            (r"ghp_[A-Za-z0-9]{36}", "GitHub personal access token"),
            (r"(?i)xox[bpoa]-[0-9A-Za-z-]+", "Slack token"),
        ]

        for i, line in enumerate(lines, 1):
            for pattern_str, desc in secret_patterns:
                if re.search(pattern_str, line):
                    vulns.append(
                        Vulnerability(
                            id=self._next_id(),
                            severity="critical",
                            category="hardcoded_secret",
                            file="<input>",
                            line=i,
                            description=f"{desc}: {line.strip()[:60]}...",
                            cwe="CWE-798",
                            recommendation="Move secrets to environment variables or a secrets manager",
                        )
                    )
                    break

        score = max(0, 100 - len(vulns) * 10)
        return ScanResult(
            vulnerabilities=vulns,
            score=score,
            summary={"secrets_found": len(vulns)},
        )

    def owasp_check(self, code: str) -> ScanResult:
        self._vuln_counter = 0
        vulns = []
        lines = code.split("\n")

        for owasp_cat, info in self.owasp.items():
            for pattern_str in info["patterns"]:
                regex = re.compile(pattern_str)
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        vulns.append(
                            Vulnerability(
                                id=self._next_id(),
                                severity="high",
                                category=owasp_cat,
                                file="<input>",
                                line=i,
                                description=f"OWASP {owasp_cat}: {info['description']}",
                                cwe=str(info["cwe"]),
                                recommendation=f"Review code at line {i} for {owasp_cat} vulnerability",
                            )
                        )
                        break

        score = max(0, 100 - len(vulns) * 8)
        return ScanResult(
            vulnerabilities=vulns,
            score=score,
            summary={
                "owasp_findings": len(vulns),
                "categories_checked": len(self.owasp),
            },
        )


class SecurityScannerModule:
    NAME = "security_scanner"
    DESCRIPTION = "SAST code analysis, vulnerability detection, dependency scanning, secret detection, OWASP Top 10 checks"

    def __init__(self) -> None:
        self.scanner = SecurityScanner()

    def execute(self, task: Any) -> dict[str, Any]:
        task_title = getattr(task, "title", "")
        task_tags = getattr(task, "tags", [])

        code = ""
        language = "python"
        deps_content = ""
        scan_type = "code"

        if hasattr(task, "content"):
            code = task.content
        if hasattr(task, "metadata"):
            metadata = task.metadata
            language = metadata.get("language", language)
            deps_content = metadata.get("deps", deps_content)
            scan_type = metadata.get("scan_type", scan_type)

        if "secret" in task_tags or "secrets" in str(task_title).lower():
            scan_type = "secrets"
        elif "dependency" in str(task_title).lower() or "deps" in task_tags:
            scan_type = "dependencies"
        elif "owasp" in str(task_title).lower() or "owasp" in task_tags:
            scan_type = "owasp"

        if scan_type == "secrets":
            result = self.scanner.scan_secrets(code)
        elif scan_type == "dependencies":
            result = self.scanner.scan_dependencies(deps_content or code)
        elif scan_type == "owasp":
            result = self.scanner.owasp_check(code)
        else:
            result = self.scanner.scan_code(code, language)

        return {
            "scan_type": scan_type,
            "vulnerability_count": len(result.vulnerabilities),
            "vulnerabilities": [
                {
                    "id": v.id,
                    "severity": v.severity,
                    "category": v.category,
                    "line": v.line,
                    "description": v.description,
                    "cwe": v.cwe,
                    "recommendation": v.recommendation,
                }
                for v in result.vulnerabilities
            ],
            "score": result.score,
            "summary": result.summary,
            "_confidence": 0.85,
        }
