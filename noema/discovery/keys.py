from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DiscoveredKey:
    name: str
    value: str
    source: str  # env, file, keychain, cloud
    confidence: float = 0.9
    provider_hint: str = ""  # openai, anthropic, aws, etc.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceInfo:
    name: str
    kind: str  # cpu, gpu, ram, disk, network
    available: bool = True
    details: dict[str, Any] = field(default_factory=dict)


class KeyDiscovery:
    """Autonomous discovery of API keys, secrets, and resources."""

    # Map env var names to provider hints
    KNOWN_KEY_PATTERNS: dict[str, str] = {
        "OPENAI_API_KEY": "openai",
        "OPENAI_KEY": "openai",
        "ANTHROPIC_API_KEY": "anthropic",
        "ANTHROPIC_AUTH_TOKEN": "anthropic",
        "OLLAMA_BASE_URL": "ollama",
        "AWS_ACCESS_KEY_ID": "aws",
        "AWS_SECRET_ACCESS_KEY": "aws",
        "AWS_REGION": "aws",
        "GOOGLE_APPLICATION_CREDENTIALS": "google",
        "GITHUB_TOKEN": "github",
        "GITHUB_PERSONAL_ACCESS_TOKEN": "github",
        "HF_TOKEN": "huggingface",
        "HUGGING_FACE_HUB_TOKEN": "huggingface",
        "REPLICATE_API_TOKEN": "replicate",
        "COHERE_API_KEY": "cohere",
        "MISTRAL_API_KEY": "mistral",
        "DEEPSEEK_API_KEY": "deepseek",
        "GROQ_API_KEY": "groq",
        "TOGETHER_API_KEY": "together",
        "FIREWORKS_API_KEY": "fireworks",
        "AZURE_OPENAI_ENDPOINT": "azure",
        "AZURE_OPENAI_API_KEY": "azure",
    }

    CONFIG_FILE_NAMES = [
        ".env",
        ".env.local",
        ".env.development",
        ".env.production",
        ".noema/config.json",
    ]

    def __init__(self, project_root: str = ".") -> None:
        self.project_root = Path(project_root).resolve()

    def discover_keys(self) -> list[DiscoveredKey]:
        found: list[DiscoveredKey] = []
        found.extend(self._scan_env())
        found.extend(self._scan_files())
        found.extend(self._scan_keychain())
        found.extend(self._scan_cloud_metadata())
        # deduplicate by name, keep highest confidence
        best: dict[str, DiscoveredKey] = {}
        for key in found:
            existing = best.get(key.name)
            if not existing or key.confidence > existing.confidence:
                best[key.name] = key
        return list(best.values())

    def discover_resources(self) -> list[ResourceInfo]:
        resources: list[ResourceInfo] = []
        resources.append(self._check_cpu())
        resources.append(self._check_ram())
        resources.append(self._check_disk())
        resources.append(self._check_gpu())
        resources.append(self._check_network())
        return resources

    def discover_all(self) -> dict[str, Any]:
        keys = self.discover_keys()
        resources = self.discover_resources()
        providers: dict[str, list[str]] = {}
        for k in keys:
            if k.provider_hint:
                providers.setdefault(k.provider_hint, []).append(k.name)
        return {
            "keys": [
                {"name": k.name, "source": k.source, "provider": k.provider_hint} for k in keys
            ],
            "providers_available": list(providers.keys()),
            "resources": [
                {"name": r.name, "kind": r.kind, "available": r.available} for r in resources
            ],
        }

    # ── Internal scanners ─────────────────────────────────────

    def _scan_env(self) -> list[DiscoveredKey]:
        found = []
        for env_name, provider in self.KNOWN_KEY_PATTERNS.items():
            val = os.environ.get(env_name)
            if val:
                found.append(
                    DiscoveredKey(
                        name=env_name,
                        value=val[:8] + "..." if len(val) > 8 else val,
                        source="env",
                        confidence=1.0,
                        provider_hint=provider,
                    )
                )
        return found

    def _scan_files(self) -> list[DiscoveredKey]:
        found = []
        for name in self.CONFIG_FILE_NAMES:
            path = self.project_root / name
            if path.exists():
                content = path.read_text(encoding="utf-8", errors="ignore")
                found.extend(self._parse_env_file(content, source=f"file:{name}"))
        return found

    def _parse_env_file(self, content: str, source: str = "file") -> list[DiscoveredKey]:
        found = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                provider = self.KNOWN_KEY_PATTERNS.get(key, "")
                if provider and val:
                    found.append(
                        DiscoveredKey(
                            name=key,
                            value=val[:8] + "..." if len(val) > 8 else val,
                            source=source,
                            confidence=0.7,
                            provider_hint=provider,
                        )
                    )
        return found

    def _scan_keychain(self) -> list[DiscoveredKey]:
        found = []
        if shutil.which("security"):
            with contextlib.suppress(Exception):
                result = subprocess.run(
                    ["security", "find-generic-password", "-s", "noema", "-w"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    found.append(
                        DiscoveredKey(
                            name="keychain_noema",
                            value=result.stdout.strip()[:8] + "...",
                            source="keychain",
                            confidence=0.8,
                        )
                    )
        return found

    def _scan_cloud_metadata(self) -> list[DiscoveredKey]:
        found = []
        # AWS-style env vars already caught by _scan_env
        # Check for GCP metadata endpoint
        with contextlib.suppress(Exception):
            import urllib.request

            req = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/project/project-id",
                headers={"Metadata-Flavor": "Google"},
            )
            resp = urllib.request.urlopen(req, timeout=2)
            project_id = resp.read().decode().strip()
            if project_id:
                found.append(
                    DiscoveredKey(
                        name="GCP_PROJECT_ID",
                        value=project_id,
                        source="cloud:gcp",
                        confidence=0.9,
                        provider_hint="google",
                    )
                )
        return found

    # ── Resource checks ───────────────────────────────────────

    def _check_cpu(self) -> ResourceInfo:
        import multiprocessing

        return ResourceInfo(
            name="CPU",
            kind="cpu",
            available=True,
            details={
                "cores": multiprocessing.cpu_count(),
            },
        )

    def _check_ram(self) -> ResourceInfo:
        details: dict[str, Any] = {}
        try:
            if os.name == "nt":
                import ctypes

                kernel32 = getattr(ctypes, "windll").kernel32  # noqa: B009 (windll exists only on Windows)
                c_ulonglong = ctypes.c_ulonglong
                mem = c_ulonglong()
                kernel32.GetPhysicallyInstalledMemory(ctypes.byref(mem))
                total_gb = mem.value / (1024**3)
                details["total_gb"] = round(total_gb, 1)
            else:
                with contextlib.suppress(OSError), open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            kb = int(line.split()[1])
                            details["total_gb"] = round(kb / (1024**2), 1)
                            break
        except Exception:
            details["total_gb"] = 0.0
        return ResourceInfo(name="RAM", kind="ram", available=True, details=details)

    def _check_disk(self) -> ResourceInfo:
        details: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            usage = shutil.disk_usage("/")
            details["total_gb"] = round(usage.total / (1024**3), 1)
            details["free_gb"] = round(usage.free / (1024**3), 1)
        return ResourceInfo(name="Disk", kind="disk", available=True, details=details)

    def _check_gpu(self) -> ResourceInfo:
        gpu_available = False
        details: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpu_available = True
                details["gpus"] = result.stdout.strip().split("\n")
        return ResourceInfo(name="GPU", kind="gpu", available=gpu_available, details=details)

    def _check_network(self) -> ResourceInfo:
        details: dict[str, Any] = {}
        try:
            import urllib.request

            urllib.request.urlopen("https://api.openai.com", timeout=3)
            details["internet"] = True
        except Exception:
            details["internet"] = False
        return ResourceInfo(name="Network", kind="network", available=True, details=details)
