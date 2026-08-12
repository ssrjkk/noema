"""Self-evolution engine — automated prompt optimization via traces + judge.

Landing policy (T2.3): a mutation is only auto-applied after a **green test run
against the patched code**. ``evolution_enabled`` turns the cycle off entirely;
``evolution_test_before_apply`` (default on) forces the real test suite to pass on
the patched worktree before the patch is kept, reverting the file on failure;
``evolution_auto_apply`` (default off) makes the cycle propose patches for review
instead of mutating the tree.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from noema.config.settings import get_settings
from noema.logging import get_logger

from ..llm.providers import BaseLLMProvider, LLMMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

log = get_logger(__name__)


prompt_optimizer_system = """You are a prompt optimization engine (OPRO pattern).
You will receive an agent's system prompt, a trace of their failed outputs, and judge evaluations.

Your task: rewrite the system prompt to prevent these failures in the future.
Rules:
1. Keep the original purpose and structure
2. Add specific constraints that address the failures
3. Be explicit about what to AVOID
4. Add examples of correct vs incorrect output if helpful
5. Return ONLY the new system prompt, no explanations

Current system prompt:
```
{prompt}
```

Failed outputs (with judge scores):
```
{traces}
```

New system prompt (return ONLY the prompt text, no JSON, no markdown):"""


class EvolutionPatch(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = Field(default_factory=time.time)
    target: str = ""  # what module/component this patch targets
    description: str = ""
    original_code: str = ""
    patched_code: str = ""
    rationale: str = ""
    confidence: float = 0.5
    applied: bool = False
    tests_passed: bool = False
    tests_output: str = ""
    performance_delta: float = 0.0  # positive = improvement


class EvolutionResult(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = Field(default_factory=time.time)
    patches_generated: int = 0
    patches_applied: int = 0
    patches_passed: int = 0
    patches_proposed: int = 0
    patches_rejected: int = 0
    evolution_cycle: int = 0
    summary: str = ""
    improvements: list[str] = Field(default_factory=list)


class EvolutionEngine:
    """Self-improvement engine: analyzes own code, generates patches, optimizes prompts.

    Args:
        llm_provider: LLM used for patch generation and prompt optimization.
        project_root: Repository root the engine analyzes and mutates.
        max_patches_per_cycle: Upper bound on patches per cycle.
        enabled: Override for ``evolution_enabled`` (default: from settings).
        test_before_apply: Override for ``evolution_test_before_apply`` (default: settings).
        auto_apply: Override for ``evolution_auto_apply`` (default: settings).
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        project_root: str = ".",
        max_patches_per_cycle: int = 5,
        enabled: bool | None = None,
        test_before_apply: bool | None = None,
        auto_apply: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.enabled = settings.evolution_enabled if enabled is None else enabled
        self.test_before_apply = (
            settings.evolution_test_before_apply if test_before_apply is None else test_before_apply
        )
        self.auto_apply = settings.evolution_auto_apply if auto_apply is None else auto_apply
        self.llm = llm_provider
        self.project_root = project_root
        self.max_patches_per_cycle = max_patches_per_cycle
        self.history: list[EvolutionResult] = []
        self.patches: list[EvolutionPatch] = []
        self._cycle = 0

    async def analyze_self(self) -> dict[str, Any]:
        """Analyze own codebase for improvement opportunities."""
        from pathlib import Path

        root = Path(self.project_root)
        issues: list[dict[str, str]] = []
        improvements: list[dict[str, str]] = []

        for py_file in root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")
            except Exception:
                continue

            if len(content) < 50:
                continue

            if "TODO" in content or "FIXME" in content:
                issues.append(
                    {
                        "file": str(py_file.relative_to(root)),
                        "type": "todo_fixme",
                        "severity": "medium",
                    }
                )

            if "except:" in content or "except Exception:" in content:
                issues.append(
                    {
                        "file": str(py_file.relative_to(root)),
                        "type": "bare_except",
                        "severity": "low",
                    }
                )

            lines = content.split("\n")
            if len(lines) > 500:
                issues.append(
                    {
                        "file": str(py_file.relative_to(root)),
                        "type": "large_file",
                        "severity": "medium",
                    }
                )

            func_count = sum(
                1
                for line in lines
                if line.strip().startswith("def ") or line.strip().startswith("async def ")
            )
            if func_count > 20:
                improvements.append(
                    {
                        "file": str(py_file.relative_to(root)),
                        "type": "split_module",
                        "suggestion": f"Module has {func_count} functions, consider splitting",
                    }
                )

        return {
            "issues": issues,
            "improvements": improvements,
            "total_files": len(list(root.rglob("*.py"))),
        }

    async def generate_patch(
        self, target_file: str, issue: dict[str, str]
    ) -> EvolutionPatch | None:
        """Generate a patch for a specific issue."""
        from pathlib import Path

        path = Path(self.project_root) / target_file
        if not path.exists():
            return None

        content = path.read_text(encoding="utf-8")

        prompt = f"""You are a code improvement engine. Analyze this code and generate a patch.

File: {target_file}
Issue type: {issue["type"]}
Issue severity: {issue.get("severity", "unknown")}

Code:
```python
{content[:4000]}
```

Respond in JSON:
{{
    "patched_code": "the full improved code",
    "rationale": "why this change improves the code",
    "confidence": 0.0-1.0,
    "description": "what changed"
}}
"""

        try:
            messages = [LLMMessage(role="user", content=prompt)]
            response_obj = await self.llm.complete(messages)
            response = response_obj.content
            import json as _json

            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
            result = _json.loads(text)

            return EvolutionPatch(
                target=target_file,
                description=result.get("description", ""),
                original_code=content,
                patched_code=result.get("patched_code", content),
                rationale=result.get("rationale", ""),
                confidence=result.get("confidence", 0.5),
            )
        except Exception:
            return None

    async def apply_patch(self, patch: EvolutionPatch, dry_run: bool = False) -> bool:
        """Land a validated patch: write the file, compile-check it, mark it applied."""
        from pathlib import Path

        path = Path(self.project_root) / patch.target
        if not path.is_file():
            return False

        if dry_run:
            return True

        original = patch.original_code or path.read_text(encoding="utf-8")
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(original, encoding="utf-8")

        try:
            path.write_text(patch.patched_code, encoding="utf-8")
            import py_compile

            py_compile.compile(str(path), doraise=True)
            patch.applied = True
            backup.unlink(missing_ok=True)
            return True
        except Exception:
            path.write_text(original, encoding="utf-8")
            backup.unlink(missing_ok=True)
            patch.applied = False
            patch.tests_passed = False
            return False

    def _default_test_runner(self) -> Callable[[], Awaitable[tuple[bool, str]]]:
        """Real test runner: runs ``pytest`` in ``project_root`` via subprocess."""

        async def _run() -> tuple[bool, str]:
            proc = await asyncio.create_subprocess_exec(
                "python",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root),
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
            except TimeoutError:
                proc.kill()
                return False, "tests timed out after 300s"
            output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
            return proc.returncode == 0, output

        return _run

    async def _validate_and_land(
        self,
        patch: EvolutionPatch,
        test_runner: Callable[[], Awaitable[tuple[bool, str]]],
    ) -> bool:
        """Apply the patch provisionally, run the real tests, revert on failure.

        The patched code is written to the worktree *before* the tests run, so a
        green run genuinely validates the mutation. On a red run the file is
        restored and the patch is rejected — no mutation lands without a green run.
        """
        from pathlib import Path

        path = Path(self.project_root) / patch.target
        if not path.is_file():
            return False

        original = patch.original_code or path.read_text(encoding="utf-8")
        patch.original_code = original

        if self.test_before_apply:
            path.write_text(patch.patched_code, encoding="utf-8")
            try:
                ok, output = await test_runner()
            except Exception as e:
                ok, output = False, f"{type(e).__name__}: {e}"
            patch.tests_output = output
            if not ok:
                patch.tests_passed = False
                path.write_text(original, encoding="utf-8")
                log.warning(
                    "evolution_tests_failed",
                    proposal_id=patch.id,
                    file=patch.target,
                )
                return False
            patch.tests_passed = True
            log.info("evolution_tests_passed", proposal_id=patch.id, file=patch.target)

        if not self.auto_apply:
            # Propose for review: keep the worktree untouched.
            patch.applied = False
            path.write_text(original, encoding="utf-8")
            return True

        return await self.apply_patch(patch)

    async def run_evolution_cycle(
        self,
        test_runner: Callable[[], Awaitable[tuple[bool, str]]] | None = None,
        patch_generator: Callable[[str, dict[str, str]], Awaitable[EvolutionPatch | None]]
        | None = None,
    ) -> EvolutionResult:
        """Run one full evolution cycle.

        Honors the landing policy: disabled evolution produces no patches; with
        ``test_before_apply`` a mutation only lands on a green run of the real test
        suite executed against the patched code; without ``auto_apply`` patches are
        proposed for review and the worktree is left untouched.
        """
        self._cycle += 1
        result = EvolutionResult(evolution_cycle=self._cycle)

        if not self.enabled:
            result.summary = f"Cycle {self._cycle}: evolution disabled by settings"
            self.history.append(result)
            return result

        generate = patch_generator or self.generate_patch
        runner = test_runner or self._default_test_runner()
        analysis = await self.analyze_self()

        for issue in analysis.get("issues", [])[: self.max_patches_per_cycle]:
            patch = await generate(issue["file"], issue)
            if not patch:
                continue
            self.patches.append(patch)
            result.patches_generated += 1

            landed = await self._validate_and_land(patch, runner)
            if not landed:
                result.patches_rejected += 1
                continue
            if patch.applied:
                result.patches_applied += 1
                result.improvements.append(f"Applied patch to {patch.target}: {patch.description}")
            else:
                result.patches_proposed += 1

        result.summary = (
            f"Cycle {self._cycle}: {result.patches_generated} generated, "
            f"{result.patches_applied} applied, {result.patches_proposed} proposed, "
            f"{result.patches_rejected} rejected"
        )
        self.history.append(result)
        return result

    async def optimize_prompt(
        self,
        current_prompt: str,
        failed_traces: list[dict[str, Any]],
        agent_role: str = "",
    ) -> str:
        """Optimize a system prompt using traces of failed outputs (OPRO pattern).

        Args:
            current_prompt: The current system prompt to optimize.
            failed_traces: List of dicts with keys 'output', 'judge_score', 'weaknesses'.
            agent_role: Name of the agent (for logging).

        Returns:
            Optimized system prompt text.
        """
        if not failed_traces:
            return current_prompt

        traces_text = "\n---\n".join(
            f"Output: {t.get('output', '')[:500]}\n"
            f"Judge: {json.dumps(t.get('judge_score', {}))}\n"
            f"Weaknesses: {', '.join(t.get('weaknesses', []))}"
            for t in failed_traces[-5:]
        )

        prompt = prompt_optimizer_system.format(prompt=current_prompt, traces=traces_text)
        messages = [LLMMessage(role="user", content=prompt)]

        try:
            response = await self.llm.complete(messages, temperature=0.4, max_tokens=2048)
            optimized = response.content.strip()
            if optimized.startswith("```"):
                optimized = optimized.split("\n", 1)[1]
                if optimized.endswith("```"):
                    optimized = optimized[:-3]
            log.info(
                "prompt_optimized",
                agent=agent_role,
                original_len=len(current_prompt),
                optimized_len=len(optimized),
            )
            return optimized.strip()
        except Exception as e:
            log.error("prompt_optimize_failed", error=str(e))
            return current_prompt

    def get_stats(self) -> dict[str, Any]:
        total_patches = len(self.patches)
        applied = sum(1 for p in self.patches if p.applied)
        proposed = sum(1 for p in self.patches if not p.applied and p.tests_passed)
        rejected = sum(1 for p in self.patches if not p.applied and not p.tests_passed)
        return {
            "enabled": self.enabled,
            "test_before_apply": self.test_before_apply,
            "auto_apply": self.auto_apply,
            "total_cycles": self._cycle,
            "total_patches": total_patches,
            "applied": applied,
            "proposed": proposed,
            "rejected": rejected,
            "improvement_rate": applied / max(total_patches, 1),
        }
