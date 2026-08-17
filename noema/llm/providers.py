"""LLM providers — abstraction with resilience (circuit breaker + retry)."""

from __future__ import annotations

import abc
import json
import time
from typing import Any, cast

import aiohttp
from pydantic import BaseModel

from noema.cache import get_cache
from noema.config.settings import get_settings
from noema.context import get_tenant_id
from noema.logging import get_logger
from noema.resilience import CircuitBreaker, ResilientExecutor, RetryPolicy
from noema.tracing.tracer import get_tracer
from noema.utils.json_utils import strip_fences

log = get_logger(__name__)


class LLMMessage(BaseModel):
    role: str  # system, user, assistant
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str = ""
    tokens_used: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
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

        if temperature <= 0.1:
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

            if temperature <= 0.1 and response.tokens_used > 0:
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
            return LLMResponse(content="OpenAI not installed: pip install openai")

        settings = get_settings()
        try:
            client = AsyncOpenAI(api_key=self.api_key, timeout=settings.llm.request_timeout)
        except Exception as e:  # noqa: BLE001 - a stub beats a hard crash
            return LLMResponse(content=f"OpenAI unavailable: {e}")
        t0 = time.monotonic()

        response = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
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

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514") -> None:
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
            return LLMResponse(content="Anthropic not installed: pip install anthropic")

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

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def _complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
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
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.post(
                f"{self._base_url}/api/chat",
                json=payload,
            ) as resp,
        ):
            if resp.status != 200:
                body = await resp.text()
                raise ConnectionError(f"Ollama returned {resp.status}: {body[:200]}")
            data = await resp.json()

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
    """Stub — used when LLM is unavailable."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "fallback"

    @property
    def model_name(self) -> str:
        return "template-based"

    async def _complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        last_msg = messages[-1].content if messages else ""
        return LLMResponse(
            content=(
                f"[Fallback mode] Received prompt of length {len(last_msg)}. "
                "Install an LLM provider: pip install openai anthropic"
            ),
            model="fallback",
            tokens_used=0,
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
        return AnthropicProvider(api_key=api_key, model=model or "claude-sonnet-4-20250514")
    elif provider == "ollama":
        return OllamaProvider(model=model, base_url=settings.llm.ollama_url)
    else:
        return FallbackProvider()
