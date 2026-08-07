"""HashiCorp Vault integration for secrets management."""

from __future__ import annotations

from typing import Any, cast

import structlog

logger = structlog.get_logger(__name__)


class VaultClient:
    """Client for HashiCorp Vault KV v2 secrets engine.

    Usage:
        vault = VaultClient(url="http://vault:8200", token="hvs...")
        api_key = await vault.get_secret("noema/llm/openai-api-key")
        await vault.set_secret("noema/llm/anthropic-api-key", "sk-ant-...")
    """

    def __init__(self, url: str = "http://localhost:8200", token: str = "") -> None:
        self.url = url.rstrip("/")
        self.token = token
        self._client: Any = None

    async def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import hvac

            self._client = hvac.Client(url=self.url, token=self.token)
        except ImportError:
            logger.warning("hvac not installed, vault client unavailable")
            self._client = None

    async def get_secret(self, path: str, mount_point: str = "secret") -> str | None:
        """Get a secret value from Vault KV v2."""
        await self._ensure_client()
        if not self._client:
            return None
        try:
            secret = self._client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=mount_point,
            )
            return cast("str | None", secret.get("data", {}).get("data", {}).get("value"))
        except Exception as e:
            logger.error("vault_get_secret_failed", path=path, error=str(e))
            return None

    async def set_secret(self, path: str, value: str, mount_point: str = "secret") -> bool:
        """Set a secret value in Vault KV v2."""
        await self._ensure_client()
        if not self._client:
            return False
        try:
            self._client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret={"value": value},
                mount_point=mount_point,
            )
            logger.info("vault_secret_set", path=path)
            return True
        except Exception as e:
            logger.error("vault_set_secret_failed", path=path, error=str(e))
            return False

    async def delete_secret(self, path: str, mount_point: str = "secret") -> bool:
        """Delete a secret from Vault KV v2."""
        await self._ensure_client()
        if not self._client:
            return False
        try:
            self._client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=path,
                mount_point=mount_point,
            )
            logger.info("vault_secret_deleted", path=path)
            return True
        except Exception as e:
            logger.error("vault_delete_secret_failed", path=path, error=str(e))
            return False

    async def list_secrets(self, path: str = "", mount_point: str = "secret") -> list[str]:
        """List secrets at a given path."""
        await self._ensure_client()
        if not self._client:
            return []
        try:
            result = self._client.secrets.kv.v2.list_secrets(
                path=path,
                mount_point=mount_point,
            )
            return cast("list[str]", result.get("data", {}).get("keys", []))
        except Exception as e:
            logger.error("vault_list_secrets_failed", path=path, error=str(e))
            return []

    async def get_llm_api_key(self, provider: str) -> str | None:
        """Get LLM API key from Vault by provider name."""
        key_map = {"openai": "noema/llm/openai-api-key", "anthropic": "noema/llm/anthropic-api-key"}
        path = key_map.get(provider)
        if not path:
            logger.warning("vault_no_key_path_for_provider", provider=provider)
            return None
        return await self.get_secret(path)
