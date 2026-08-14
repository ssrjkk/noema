"""Deep-path coverage for EvolutionEngine: optimize_prompt, generate_patch, apply_patch."""

import json

import pytest

from noema.evolution.engine import EvolutionEngine, EvolutionPatch
from noema.llm.providers import BaseLLMProvider, FallbackProvider, LLMResponse


class _ScriptedLLM(BaseLLMProvider):
    def __init__(self, content: str) -> None:
        self._name = "scripted"
        self._model = "scripted-1"
        self._content = content
        super().__init__()

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._model

    async def _complete(self, messages, temperature=0.7, max_tokens=4096) -> LLMResponse:
        return LLMResponse(content=self._content, model=self._model, tokens_used=0)

    async def complete(
        self,
        messages,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tenant_id: str = "",
    ) -> LLMResponse:
        return await self._complete(messages, temperature, max_tokens)


class _RaisingLLM(_ScriptedLLM):
    def __init__(self) -> None:
        super().__init__("never used")

    async def _complete(self, messages, temperature=0.7, max_tokens=4096) -> LLMResponse:
        raise RuntimeError("llm exploded")


def _engine(tmp_path, llm: BaseLLMProvider) -> EvolutionEngine:
    return EvolutionEngine(llm_provider=llm, project_root=str(tmp_path))


def _traces() -> list[dict[str, object]]:
    return [
        {
            "output": "wrong answer",
            "judge_score": {"overall": 0.3},
            "weaknesses": ["misses constraints", "too verbose"],
        }
    ]


def _patch(
    target: str = "app.py",
    original: str = "x = 1\n",
    patched: str = "x = 2\n",
) -> EvolutionPatch:
    return EvolutionPatch(target=target, original_code=original, patched_code=patched)


@pytest.mark.asyncio
async def test_optimize_prompt_empty_traces_unchanged(tmp_path):
    engine = _engine(tmp_path, FallbackProvider())
    prompt = "System prompt v1"
    assert await engine.optimize_prompt(prompt, []) == prompt


@pytest.mark.asyncio
async def test_optimize_prompt_fallback_provider(tmp_path):
    engine = _engine(tmp_path, FallbackProvider())
    result = await engine.optimize_prompt("System prompt v1", _traces())
    assert isinstance(result, str)
    assert result


@pytest.mark.asyncio
async def test_optimize_prompt_strips_markdown_fence(tmp_path):
    engine = _engine(tmp_path, _ScriptedLLM("```\noptimized prompt text\n```"))
    result = await engine.optimize_prompt("System prompt v1", _traces())
    assert result == "optimized prompt text"


@pytest.mark.asyncio
async def test_optimize_prompt_plain_return(tmp_path):
    engine = _engine(tmp_path, _ScriptedLLM("  optimized prompt text  "))
    result = await engine.optimize_prompt("System prompt v1", _traces())
    assert result == "optimized prompt text"


@pytest.mark.asyncio
async def test_optimize_prompt_exception_returns_original(tmp_path):
    engine = _engine(tmp_path, _RaisingLLM())
    prompt = "System prompt v1"
    assert await engine.optimize_prompt(prompt, _traces()) == prompt


@pytest.mark.asyncio
async def test_optimize_prompt_truncates_long_traces(tmp_path):
    engine = _engine(tmp_path, _ScriptedLLM("optimized"))
    long_output = "o" * 1000
    traces = [{"output": long_output, "judge_score": {"overall": 0.1}, "weaknesses": ["w"]}]
    result = await engine.optimize_prompt("System prompt v1", traces)
    assert result == "optimized"


@pytest.mark.asyncio
async def test_generate_patch_parses_plain_json(tmp_path):
    target = tmp_path / "app.py"
    original = "def add(a, b):\n    return a + b\n"
    target.write_text(original, encoding="utf-8")
    payload = {
        "patched_code": "def add(a, b):\n    return a + b + 1\n",
        "rationale": "use float math",
        "confidence": 0.9,
        "description": "improve add",
    }
    engine = _engine(tmp_path, _ScriptedLLM(json.dumps(payload)))
    patch = await engine.generate_patch("app.py", {"type": "optimize", "severity": "medium"})
    assert patch is not None
    assert patch.target == "app.py"
    assert patch.original_code == original
    assert patch.patched_code == payload["patched_code"]
    assert patch.rationale == "use float math"
    assert patch.confidence == 0.9
    assert patch.description == "improve add"


@pytest.mark.asyncio
async def test_generate_patch_strips_code_fence(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("x = 1\n", encoding="utf-8")
    payload = {"patched_code": "x = 2\n", "description": "bump"}
    wrapped = f"```json\n{json.dumps(payload)}\n```"
    engine = _engine(tmp_path, _ScriptedLLM(wrapped))
    patch = await engine.generate_patch("app.py", {"type": "optimize"})
    assert patch is not None
    assert patch.patched_code == "x = 2\n"


@pytest.mark.asyncio
async def test_generate_patch_invalid_json_returns_none(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("x = 1\n", encoding="utf-8")
    engine = _engine(tmp_path, _ScriptedLLM("this is not json at all"))
    assert await engine.generate_patch("app.py", {"type": "optimize"}) is None


@pytest.mark.asyncio
async def test_generate_patch_missing_target_returns_none(tmp_path):
    engine = _engine(tmp_path, FallbackProvider())
    assert await engine.generate_patch("does_not_exist.py", {"type": "optimize"}) is None


@pytest.mark.asyncio
async def test_apply_patch_missing_target_returns_false(tmp_path):
    engine = _engine(tmp_path, FallbackProvider())
    assert await engine.apply_patch(_patch(target="nope.py")) is False


@pytest.mark.asyncio
async def test_apply_patch_dry_run_leaves_file_untouched(tmp_path):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    engine = _engine(tmp_path, FallbackProvider())
    patch = _patch(original="x = 1\n", patched="x = 2\n")
    assert await engine.apply_patch(patch, dry_run=True) is True
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "x = 1\n"
    assert patch.applied is False


@pytest.mark.asyncio
async def test_apply_patch_applies_valid_code(tmp_path):
    original = "def f():\n    return 1\n"
    patched = "def f():\n    return 2\n"
    (tmp_path / "app.py").write_text(original, encoding="utf-8")
    engine = _engine(tmp_path, FallbackProvider())
    patch = _patch(original=original, patched=patched)
    assert await engine.apply_patch(patch) is True
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == patched
    assert patch.applied is True
    assert not (tmp_path / "app.py.bak").exists()


@pytest.mark.asyncio
async def test_apply_patch_rolls_back_on_syntax_error(tmp_path):
    original = "def f():\n    return 1\n"
    (tmp_path / "app.py").write_text(original, encoding="utf-8")
    engine = _engine(tmp_path, FallbackProvider())
    patch = _patch(original=original, patched="def broken(:\n")
    assert await engine.apply_patch(patch) is False
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == original
    assert patch.applied is False
    assert patch.tests_passed is False
    assert not (tmp_path / "app.py.bak").exists()
