"""LLM providers — abstraction with resilience (circuit breaker + retry)."""

from __future__ import annotations

import abc
import json
import re
import time
from typing import TYPE_CHECKING, Any, cast

import aiohttp
from pydantic import BaseModel

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

from noema.cache import get_cache
from noema.config.settings import get_settings
from noema.context import get_tenant_id
from noema.logging import get_logger
from noema.resilience import CircuitBreaker, ResilientExecutor, RetryPolicy
from noema.tracing.tracer import get_tracer
from noema.utils.json_utils import strip_fences

log = get_logger(__name__)


class LLMProviderError(Exception):
    """A provider failure that must never be masked as model content.

    Raised when the provider SDK is missing or the client cannot be
    constructed, so callers fail closed instead of consuming a stub
    string as if it were a real model response.
    """


class LLMMessage(BaseModel):
    role: str  # system, user, assistant
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str = ""
    tokens_used: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    error: str = ""  # non-empty ⇒ the response is a failure, never content
    latency_ms: float = 0.0
    finish_reason: str = ""


class BaseLLMProvider(abc.ABC):
    """Abstract LLM provider with built-in resilience."""

    def __init__(self) -> None:
        settings = get_settings()
        self._resilient = ResilientExecutor(
            circuit_breaker=CircuitBreaker(
                failure_threshold=settings.llm.circuit_breaker_threshold,
                recovery_timeout=settings.llm.circuit_breaker_recovery,
                name=f"llm-{self.name}",
            ),
            retry_policy=RetryPolicy(
                max_retries=settings.llm.retry_max,
                base_delay=settings.llm.retry_base_delay,
                max_delay=settings.llm.retry_max_delay,
                non_retryable_exceptions=(LLMProviderError,),
                name=f"llm-{self.name}",
            ),
        )

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def model_name(self) -> str: ...

    @abc.abstractmethod
    async def _complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tenant_id: str = "",
    ) -> LLMResponse:
        """Complete with circuit breaker + retry + tracing + semantic cache."""
        tracer = get_tracer()
        cache = get_cache()
        effective_tenant = tenant_id or get_tenant_id()

        msg_dicts = [m.model_dump() for m in messages]

        cached = cache.get(msg_dicts, self.model_name, tenant_id=effective_tenant)
        if cached is not None:
            tracer.trace_llm_call(
                provider=self.name,
                model=self.model_name,
                messages=msg_dicts,
                response=cached,
                tokens_used=0,
                latency_ms=0.5,
            )
            return LLMResponse(content=cached, model=self.model_name)

        t0 = time.monotonic()
        try:
            response = await self._resilient.execute(
                self._complete, messages, temperature, max_tokens
            )
            latency = (time.monotonic() - t0) * 1000

            if response.error:
                raise LLMProviderError(response.error)

            if response.tokens_used > 0:
                cache.set(
                    msg_dicts,
                    response.content,
                    self.model_name,
                    response.tokens_used,
                    tenant_id=effective_tenant,
                )

            tracer.trace_llm_call(
                provider=self.name,
                model=self.model_name,
                messages=msg_dicts,
                response=response.content,
                tokens_used=response.tokens_used,
                latency_ms=latency,
                tokens_input=response.tokens_input,
                tokens_output=response.tokens_output,
            )
            return response
        except Exception as e:
            latency = (time.monotonic() - t0) * 1000
            tracer.trace_llm_call(
                provider=self.name,
                model=self.model_name,
                messages=msg_dicts,
                response="",
                tokens_used=0,
                latency_ms=latency,
                error=str(e),
            )
            raise

    async def generate_code(
        self,
        prompt: str,
        language: str = "python",
        context: str = "",
    ) -> str:
        system_msg = (
            f"You are an expert {language} developer. "
            "Generate production-ready, well-structured code. "
            "Do not include explanations, only code in markdown code blocks. "
            "Follow best practices and clean code principles."
        )
        if context:
            system_msg += f"\n\nContext:\n{context}"

        messages = [
            LLMMessage(role="system", content=system_msg),
            LLMMessage(role="user", content=prompt),
        ]
        response = await self.complete(messages, temperature=0.3, max_tokens=4096)
        return self._extract_code(response.content, language)

    async def generate_architecture(
        self,
        task_description: str,
        constraints: list[str] | None = None,
    ) -> dict[str, Any]:
        system_msg = (
            "You are a senior software architect. "
            "Analyze the task and produce a JSON architecture specification.\n"
            "Return ONLY valid JSON with these keys:\n"
            '- "pattern": {"name": "...", "description": "...", "pros": [...], "cons": [...]}\n'
            '- "components": [{"name": "...", "type": "...", "responsibility": "..."}]\n'
            '- "communication": {"sync": "...", "async": "..."}\n'
            '- "deployment": {"containerization": "...", "orchestration": "..."}\n'
            '- "tech_risks": [{"risk": "...", "mitigation": "..."}]'
        )
        user_msg = f"Task: {task_description}"
        if constraints:
            user_msg += f"\nConstraints: {', '.join(constraints)}"

        messages = [
            LLMMessage(role="system", content=system_msg),
            LLMMessage(role="user", content=user_msg),
        ]
        response = await self.complete(messages, temperature=0.4, max_tokens=2048)
        return self._extract_json(response.content)

    def _extract_code(self, text: str, language: str) -> str:
        def _between(open_fence: str, from_pos: int) -> str | None:
            start = text.find(open_fence, from_pos)
            if start == -1:
                return None
            start += len(open_fence)
            end = text.find("```", start)
            if end == -1:
                return None
            return text[start:end].strip()

        if f"```{language}" in text:
            extracted = _between(f"```{language}", 0)
            if extracted is not None:
                return extracted
        if "```" in text:
            extracted = _between("```", 0)
            if extracted is not None:
                return extracted
        return text.strip()

    def _extract_json(self, text: str) -> dict[str, Any]:
        text = strip_fences(text)
        try:
            return cast("dict[str, Any]", json.loads(text))
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON", "raw": text[:500]}

    def stats(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model_name,
            "resilience": self._resilient.stats(),
        }


# ── OpenAI ──────────────────────────────────────────────────────────────


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider (GPT-4o, GPT-4, etc)."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o") -> None:
        super().__init__()
        settings = get_settings()
        self.api_key = api_key or settings.llm.openai_api_key.get_secret_value()
        self._model = model

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    async def _complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise LLMProviderError("OpenAI not installed: pip install openai") from None

        settings = get_settings()
        try:
            client = AsyncOpenAI(api_key=self.api_key, timeout=settings.llm.request_timeout)
        except Exception as e:  # noqa: BLE001 - fail closed: never return a stub as content
            raise LLMProviderError(f"OpenAI unavailable: {e}") from e
        t0 = time.monotonic()

        chat_messages: list[dict[str, str]] = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        response = await client.chat.completions.create(
            model=self._model,
            messages=cast("list[ChatCompletionMessageParam]", chat_messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            tokens_input=response.usage.prompt_tokens if response.usage else 0,
            tokens_output=response.usage.completion_tokens if response.usage else 0,
            latency_ms=(time.monotonic() - t0) * 1000,
            finish_reason=choice.finish_reason or "",
        )


# ── Anthropic ────────────────────────────────────────────────────────────


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider (Claude)."""

    def __init__(
        self, api_key: str | None = None, model: str = "claude-sonnet-4-5-20250929"
    ) -> None:
        super().__init__()
        settings = get_settings()
        self.api_key = api_key or settings.llm.anthropic_api_key.get_secret_value()
        self._model = model

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    async def _complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        try:
            import anthropic
        except ImportError:
            raise LLMProviderError("Anthropic not installed: pip install anthropic") from None

        settings = get_settings()
        client = anthropic.AsyncAnthropic(
            api_key=self.api_key, timeout=settings.llm.request_timeout
        )
        t0 = time.monotonic()

        system_content = ""
        user_messages: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_content += m.content + "\n"
            else:
                user_messages.append({"role": m.role, "content": m.content})

        response = await client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system_content.strip(),
            messages=cast("Any", user_messages),
            temperature=temperature,
        )

        content = ""
        for block in response.content or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                content = text
                break

        return LLMResponse(
            content=content,
            model=response.model,
            tokens_used=(
                (response.usage.input_tokens + response.usage.output_tokens)
                if response.usage
                else 0
            ),
            tokens_input=response.usage.input_tokens if response.usage else 0,
            tokens_output=response.usage.output_tokens if response.usage else 0,
            latency_ms=(time.monotonic() - t0) * 1000,
            finish_reason=response.stop_reason or "",
        )


# ── Ollama (Local) ──────────────────────────────────────────────────────


class OllamaProvider(BaseLLMProvider):
    """Ollama — local models (llama3, codellama, deepseek-coder, etc)."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        super().__init__()
        settings = get_settings()
        self._model = model or settings.llm.ollama_model
        self._base_url = (base_url or settings.llm.ollama_url).rstrip("/")
        self._health_checked = False
        self._health_ok = False
        self._health_failed_at: float = 0.0
        self._health_recovery = settings.llm.circuit_breaker_recovery

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def _health_check(self) -> None:
        """Fast health-check against ``/api/tags``.

        Has a short (1s) timeout so an unreachable Ollama fails fast and the
        caller can degrade to the template-based fallback immediately instead
        of waiting for the full request timeout + retries. A failed check is
        cached only for ``circuit_breaker_recovery`` seconds, so a long-lived
        provider instance self-heals when Ollama comes back up instead of
        staying permanently poisoned by the first failure.
        """
        if self._health_ok:
            return
        if self._health_checked:
            since_failure = time.monotonic() - self._health_failed_at
            if since_failure < self._health_recovery:
                raise LLMProviderError(
                    f"Ollama is unreachable at {self._base_url}. "
                    "Start it locally (`ollama serve`) or switch providers."
                )
        self._health_checked = True
        self._health_ok = False
        timeout = aiohttp.ClientTimeout(total=1.0, connect=0.8)
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.get(f"{self._base_url}/api/tags") as resp,
            ):
                if resp.status == 200:
                    self._health_ok = True
                    self._health_failed_at = 0.0
                    return
                body_text = await resp.text()
                self._health_failed_at = time.monotonic()
                raise LLMProviderError(
                    f"Ollama health-check failed (HTTP {resp.status}): {body_text[:200]}"
                )
        except (aiohttp.ClientError, TimeoutError, ConnectionError) as exc:
            self._health_ok = False
            self._health_failed_at = time.monotonic()
            raise LLMProviderError(
                f"Ollama is unreachable at {self._base_url} ({type(exc).__name__}). "
                "Start it locally (`ollama serve`) or switch providers."
            ) from None

    async def _complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        await self._health_check()
        settings = get_settings()
        t0 = time.monotonic()
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        timeout = aiohttp.ClientTimeout(total=settings.llm.request_timeout)
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                ) as resp,
            ):
                if resp.status != 200:
                    raise ConnectionError(f"Ollama returned {resp.status}: {await resp.text()}")
                data = await resp.json()
        except (aiohttp.ClientError, TimeoutError, ConnectionError) as exc:
            self._health_ok = False
            self._health_failed_at = time.monotonic()
            raise LLMProviderError(
                f"Ollama request failed ({type(exc).__name__}): {str(exc)[:200]}"
            ) from None

        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=self._model,
            tokens_used=data.get("eval_count", 0) + data.get("prompt_eval_count", 0),
            tokens_input=data.get("prompt_eval_count", 0),
            tokens_output=data.get("eval_count", 0),
            latency_ms=(time.monotonic() - t0) * 1000,
            finish_reason="stop",
        )


# ── Fallback (No LLM) ──────────────────────────────────────────────────


class FallbackProvider(BaseLLMProvider):
    """Template-based demo engine — used when a real LLM provider is unavailable.

    Produces deterministic, structured JSON outputs so the rest of the pipeline
    (neurosymbolic converter, solution assembler, sandbox gate) sees a well-
    shaped answer instead of a stub string.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "fallback"

    @property
    def model_name(self) -> str:
        return "template-based"

    @staticmethod
    def _classify(messages: list[LLMMessage]) -> str:
        """Pick the best-matching template domain from the recent messages.

        Matching is token-aware (whole words/identifiers), never raw substring
        — ``"ui"`` must not light up inside ``"build"`` nor ``"api"`` inside
        ``"capital"``. Multi-word phrases (``"docker build"``) match as
        contiguous word runs; a trailing underscore keyword (``"test_"``)
        matches any identifier that starts with it.
        """
        text = "\n".join(m.content for m in messages[-3:]).lower()
        tokens = set(re.findall(r"\w+", text))

        def _hits(keyword: str) -> bool:
            if " " in keyword:
                return re.search(rf"\b{re.escape(keyword)}\b", text) is not None
            if keyword.endswith("_"):
                return any(w.startswith(keyword) for w in tokens)
            return keyword in tokens

        keywords = {
            "terraform": ["terraform", "hcl", "infrastructure", "aws", "gcp", "azure", "iac"],
            "docker": ["dockerfile", "container", "docker build", "image"],
            "database": ["schema", "postgres", "mysql", "database", "sql", "migration"],
            "auth": ["rbac", "auth", "login", "jwt", "permission", "role"],
            "pipeline": ["pipeline", "etl", "airflow", "dataflow", "dags"],
            "api": ["api", "endpoint", "rest", "fastapi", "flask", "backend"],
            "frontend": ["react", "ui", "frontend", "component", "html", "css"],
            "security": ["security", "vulnerability", "scan", "threat", "owasp"],
            "test": ["unit test", "pytest", "test_", "testing", "assert"],
        }
        best = "general"
        best_score = 0
        for label, kws in keywords.items():
            score = sum(1 for k in kws if _hits(k))
            if score > best_score:
                best_score = score
                best = label
        return best

    def _template(self, kind: str) -> dict:
        python_api = (
            "from fastapi import FastAPI, Depends, HTTPException, status\n"
            "from pydantic import BaseModel, Field\n"
            "from typing import Optional, List\n\n"
            "app = FastAPI(title='Noema Demo API', version='1.0.0')\n\n\n"
            "class Item(BaseModel):\n"
            "    id: Optional[int] = None\n"
            "    name: str = Field(..., min_length=2, max_length=100)\n"
            "    price: float = Field(..., gt=0)\n"
            "    tags: List[str] = Field(default_factory=list)\n\n\n"
            "ITEMS: dict[int, Item] = {}\n\n\n"
            "@app.post('/items', status_code=status.HTTP_201_CREATED)\n"
            "async def create_item(item: Item) -> Item:\n"
            "    new_id = max(ITEMS.keys(), default=0) + 1\n"
            "    item.id = new_id\n"
            "    ITEMS[new_id] = item\n"
            "    return item\n\n\n"
            "@app.get('/items')\n"
            "async def list_items() -> List[Item]:\n"
            "    return list(ITEMS.values())\n\n\n"
            "@app.get('/items/{item_id}')\n"
            "async def get_item(item_id: int) -> Item:\n"
            "    if item_id not in ITEMS:\n"
            "        raise HTTPException(status_code=404, detail='Item not found')\n"
            "    return ITEMS[item_id]\n"
        )
        docker_py = (
            "FROM python:3.12-slim\n\n"
            "ENV PYTHONDONTWRITEBYTECODE=1 \\\n"
            "    PYTHONUNBUFFERED=1 \\\n"
            "    PIP_NO_CACHE_DIR=1\n\n"
            "WORKDIR /app\n\n"
            "RUN apt-get update \\\n"
            " && apt-get install -y --no-install-recommends build-essential curl \\\n"
            " && rm -rf /var/lib/apt/lists/*\n\n"
            "COPY requirements.txt ./\n"
            "RUN pip install --upgrade pip && pip install -r requirements.txt\n\n"
            "COPY . .\n\n"
            "RUN useradd -m appuser && chown -R appuser /app\n"
            "USER appuser\n\n"
            "EXPOSE 8000\n\n"
            'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]\n'
        )
        tf_aws = (
            "terraform {\n"
            '  required_version = ">= 1.6.0"\n'
            "  required_providers {\n"
            '    aws = { source  = "hashicorp/aws"; version = "~> 5.0" }\n'
            "  }\n"
            "}\n\n"
            'provider "aws" { region = "us-east-1" }\n\n'
            'resource "aws_vpc" "main" {\n'
            '  cidr_block           = "10.0.0.0/16"\n'
            "  enable_dns_hostnames = true\n"
            '  tags = { Name = "noema-vpc" }\n'
            "}\n\n"
            'resource "aws_subnet" "public" {\n'
            "  count             = 2\n"
            "  vpc_id            = aws_vpc.main.id\n"
            "  cidr_block        = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)\n"
            "  availability_zone = element(data.aws_availability_zones.available.names, count.index)\n"
            '  tags = { Name = "noema-public-${count.index}" }\n'
            "}\n\n"
            'data "aws_availability_zones" "available" { state = "available" }\n'
        )
        rbac_py = (
            "from __future__ import annotations\n"
            "from dataclasses import dataclass, field\n"
            "from typing import Dict, Set, List\n\n\n"
            "ROLE_PERMISSIONS: Dict[str, Set[str]] = {\n"
            "    'admin': {'read', 'write', 'delete', 'manage_users', 'manage_roles'},\n"
            "    'editor': {'read', 'write', 'delete'},\n"
            "    'viewer': {'read'},\n"
            "}\n\n\n"
            "@dataclass\n"
            "class RBAC:\n"
            "    user_roles: Dict[str, Set[str]] = field(default_factory=dict)\n\n"
            "    def assign(self, user: str, role: str) -> None:\n"
            "        if role not in ROLE_PERMISSIONS:\n"
            "            raise ValueError(f'Unknown role: {role}')\n"
            "        self.user_roles.setdefault(user, set()).add(role)\n\n"
            "    def permissions(self, user: str) -> Set[str]:\n"
            "        perms: Set[str] = set()\n"
            "        for r in self.user_roles.get(user, set()):\n"
            "            perms |= ROLE_PERMISSIONS[r]\n"
            "        return perms\n\n"
            "    def can(self, user: str, permission: str) -> bool:\n"
            "        return permission in self.permissions(user)\n"
        )
        pytest_py = (
            "import pytest\n"
            "from fastapi.testclient import TestClient\n\n\n"
            "class TestHealth:\n"
            "    def test_ok(self, client: TestClient) -> None:\n"
            "        r = client.get('/health')\n"
            "        assert r.status_code == 200\n"
            "        assert r.json()['status'] == 'ok'\n\n\n"
            "class TestCreateItem:\n"
            "    def test_create(self, client: TestClient) -> None:\n"
            "        r = client.post('/items', json={'name': 'Widget', 'price': 9.99})\n"
            "        assert r.status_code == 201\n"
            "        body = r.json()\n"
            "        assert body['id'] == 1\n"
            "        assert body['name'] == 'Widget'\n\n"
            "    def test_invalid_price(self, client: TestClient) -> None:\n"
            "        r = client.post('/items', json={'name': 'Bad', 'price': -1})\n"
            "        assert r.status_code == 422\n"
        )
        sql_schema = (
            "CREATE EXTENSION IF NOT EXISTS pgcrypto;\n\n"
            "CREATE TABLE users (\n"
            "    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),\n"
            "    email         TEXT UNIQUE NOT NULL CHECK (email ~* '^.+@.+$'),\n"
            "    password_hash TEXT NOT NULL,\n"
            "    display_name  TEXT,\n"
            "    role          TEXT NOT NULL DEFAULT 'viewer' CHECK (role IN ('admin','editor','viewer')),\n"
            "    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),\n"
            "    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()\n"
            ");\n\n"
            "CREATE TABLE items (\n"
            "    id         BIGSERIAL PRIMARY KEY,\n"
            "    owner_id   UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,\n"
            "    name       TEXT NOT NULL CHECK (length(name) BETWEEN 2 AND 100),\n"
            "    price      NUMERIC(12,2) NOT NULL CHECK (price > 0),\n"
            "    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()\n"
            ");\n\n"
            "CREATE INDEX idx_items_owner ON items(owner_id);\n"
        )
        airflow_dag = (
            "from __future__ import annotations\n"
            "from datetime import datetime, timedelta\n"
            "from airflow import DAG\n"
            "from airflow.operators.python import PythonOperator\n\n\n"
            "DEFAULT_ARGS = {\n"
            "    'owner': 'noema',\n"
            "    'depends_on_past': False,\n"
            "    'email_on_failure': False,\n"
            "    'retries': 2,\n"
            "    'retry_delay': timedelta(minutes=5),\n"
            "}\n\n\n"
            "def _extract(**_):\n"
            "    print('extract: read from source')\n\n\n"
            "def _transform(**_):\n"
            "    print('transform: clean and aggregate')\n\n\n"
            "def _load(**_):\n"
            "    print('load: write to warehouse')\n\n\n"
            "with DAG(\n"
            "    dag_id='noema_etl',\n"
            "    default_args=DEFAULT_ARGS,\n"
            "    schedule_interval='@daily',\n"
            "    start_date=datetime(2025, 1, 1),\n"
            "    catchup=False,\n"
            "    tags=['noema', 'etl'],\n"
            ") as dag:\n"
            "    extract = PythonOperator(task_id='extract', python_callable=_extract)\n"
            "    transform = PythonOperator(task_id='transform', python_callable=_transform)\n"
            "    load = PythonOperator(task_id='load', python_callable=_load)\n"
            "    extract >> transform >> load\n"
        )
        react_component = (
            "import React, { useEffect, useState } from 'react';\n"
            "import type { Item } from './types';\n\n\n"
            "interface ItemListProps {\n"
            "  fetchItems: () => Promise<Item[]>;\n"
            "}\n\n"
            "export const ItemList: React.FC<ItemListProps> = ({ fetchItems }) => {\n"
            "  const [items, setItems] = useState<Item[]>([]);\n"
            "  const [loading, setLoading] = useState(true);\n"
            "  const [error, setError] = useState<string | null>(null);\n\n"
            "  useEffect(() => {\n"
            "    let cancelled = false;\n"
            "    setLoading(true);\n"
            "    fetchItems()\n"
            "      .then((data) => { if (!cancelled) setItems(data); })\n"
            "      .catch((err) => { if (!cancelled) setError(String(err)); })\n"
            "      .finally(() => { if (!cancelled) setLoading(false); });\n"
            "    return () => { cancelled = true; };\n"
            "  }, [fetchItems]);\n\n"
            "  if (loading) return <div className='spinner'>Loading…</div>;\n"
            "  if (error) return <div className='error'>{error}</div>;\n\n"
            "  return (\n"
            "    <ul className='item-list'>\n"
            "      {items.map((it) => (\n"
            "        <li key={it.id} className='item'>\n"
            "          <strong>{it.name}</strong>\n"
            "          <span>${Number(it.price).toFixed(2)}</span>\n"
            "        </li>\n"
            "      ))}\n"
            "    </ul>\n"
            "  );\n"
            "};\n\n"
            "export default ItemList;\n"
        )

        files: list[dict] = [
            {
                "path": "app/main.py",
                "content": python_api,
                "description": "FastAPI service scaffold (CRUD items)",
            },
            {
                "path": "tests/test_api.py",
                "content": pytest_py,
                "description": "Pytest cases for the API",
            },
            {
                "path": "Dockerfile",
                "content": docker_py,
                "description": "Production image for the service",
            },
            {"path": "infra/main.tf", "content": tf_aws, "description": "AWS VPC + subnet IaC"},
            {
                "path": "auth/rbac.py",
                "content": rbac_py,
                "description": "Role-based access control module",
            },
            {
                "path": "db/schema.sql",
                "content": sql_schema,
                "description": "PostgreSQL schema (users + items)",
            },
            {
                "path": "pipelines/etl_dag.py",
                "content": airflow_dag,
                "description": "Airflow DAG for ETL",
            },
            {
                "path": "web/ItemList.tsx",
                "content": react_component,
                "description": "React list component",
            },
        ]

        pick: dict[str, list[int]] = {
            "api": [0, 1, 2],
            "docker": [2, 0],
            "terraform": [3, 2],
            "database": [5, 0],
            "auth": [4, 0, 5],
            "pipeline": [6, 0, 1],
            "frontend": [7, 0, 1],
            "security": [4, 5, 1],
            "test": [1, 0],
            "general": [0, 2, 4, 5],
        }
        selected = [files[i] for i in pick.get(kind, pick["general"])]
        return {
            "summary": (
                f"Template-based fallback solution generated for '{kind}' scenario. "
                "Attach a real LLM provider (OpenAI/Anthropic/Ollama) for project-"
                "specific, context-aware output."
            ),
            "architecture": {
                "pattern": {
                    "name": "Layered Monolith (template)",
                    "description": (
                        "FastAPI REST service backed by PostgreSQL, containerized, "
                        "with RBAC, migrations and tests. IaC + Airflow DAGs "
                        "included for infrastructure and data flows."
                    ),
                    "pros": [
                        "Batteries included",
                        "Production-ready scaffolding",
                        "Tested defaults",
                    ],
                    "cons": [
                        "Generic template; adapt to your domain",
                        "Not tailored to edge cases",
                    ],
                },
                "high_level_design": (
                    "API → RBAC middleware → PostgreSQL; IaC provisions VPC; "
                    "Airflow runs daily ETL; Jest/pytest cover contracts."
                ),
            },
            "stack": {
                "languages": ["Python", "TypeScript", "SQL", "HCL"],
                "frameworks": ["FastAPI", "React", "Airflow", "Pytest"],
                "databases": ["PostgreSQL", "Redis"],
                "infrastructure": ["Docker", "Terraform", "AWS"],
            },
            "code": {"files": selected},
            "review": {
                "final_summary": (
                    "Degraded template solution. Connect an LLM API key or local "
                    "Ollama instance for bespoke reasoning and code generation."
                ),
                "weaknesses": ["LLM unavailable — generic template used"],
            },
        }

    async def _complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        t0 = time.monotonic()
        kind = self._classify(messages)
        payload = self._template(kind)
        fenced = "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"
        return LLMResponse(
            content=fenced,
            model="fallback/template-based",
            tokens_used=0,
            tokens_input=0,
            tokens_output=len(fenced),
            latency_ms=(time.monotonic() - t0) * 1000,
            finish_reason="stop",
        )


# ── Factory ──────────────────────────────────────────────────────────────


def create_llm_provider(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> BaseLLMProvider:
    """Factory for LLM providers."""
    settings = get_settings()
    provider = provider or settings.llm.provider

    if provider == "openai":
        return OpenAIProvider(api_key=api_key, model=model or "gpt-4o")
    elif provider == "anthropic":
        return AnthropicProvider(api_key=api_key, model=model or "claude-sonnet-4-5-20250929")
    elif provider == "ollama":
        return OllamaProvider(model=model, base_url=settings.llm.ollama_url)
    else:
        return FallbackProvider()
