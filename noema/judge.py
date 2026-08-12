"""LLM-as-a-Judge — pairwise evaluation, critic agent, reference-guided scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from noema.llm.providers import BaseLLMProvider, LLMMessage
from noema.logging import get_logger
from noema.utils.json_utils import strip_fences

if TYPE_CHECKING:
    from noema.core.types import Solution

log = get_logger(__name__)


@dataclass
class JudgeScore:
    architecture: float = 0.0
    code_quality: float = 0.0
    security: float = 0.0
    performance: float = 0.0
    maintainability: float = 0.0
    completeness: float = 0.0
    overall: float = 0.0


@dataclass
class JudgeVerdict:
    passed: bool = False
    scores: JudgeScore = field(default_factory=JudgeScore)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    production_readiness: str = "needs-work"
    summary: str = ""
    raw_feedback: str = ""


@dataclass
class PairwiseResult:
    winner: str = ""  # "A" or "B"
    winner_index: int = 0
    rationale: str = ""
    scores_a: JudgeScore = field(default_factory=JudgeScore)
    scores_b: JudgeScore = field(default_factory=JudgeScore)


JUDGE_SYSTEM_PROMPT = """You are a senior technical lead evaluating an AI-generated solution.
You must be CRITICAL and HONEST. Grade each category from 0.0 to 1.0.

Evaluation criteria:
- architecture: design quality, patterns, scalability, trade-offs
- code_quality: readability, structure, typing, error handling
- security: threat model, OWASP coverage, data protection
- performance: caching, optimization, async processing
- maintainability: modularity, documentation, testing strategy
- completeness: how well the solution addresses the original task

Return ONLY valid JSON:
{
  "scores": {"architecture": 0.0-1.0, "code_quality": 0.0-1.0, "security": 0.0-1.0, "performance": 0.0-1.0, "maintainability": 0.0-1.0, "completeness": 0.0-1.0, "overall": 0.0-1.0},
  "strengths": ["list of 2-3 key strengths"],
  "weaknesses": ["list of 2-3 key weaknesses"],
  "improvements": ["list of 2-3 actionable improvements"],
  "production_readiness": "ready|needs-work|major-issues",
  "summary": "one paragraph summary of the evaluation"
}

Be strict. A score above 0.8 should mean truly excellent. Below 0.4 means major issues."""

PAIRWISE_SYSTEM_PROMPT = """You are a senior technical lead comparing TWO AI-generated solutions (A and B).
Select the BETTER solution and explain your choice.

Consider: architecture quality, code quality, security, performance, maintainability, completeness.

Return ONLY valid JSON:
{
  "winner": "A" or "B",
  "rationale": "detailed comparison explaining why the winner is better",
  "scores_a": {"architecture": 0.0-1.0, "code_quality": 0.0-1.0, "security": 0.0-1.0, "performance": 0.0-1.0, "maintainability": 0.0-1.0, "completeness": 0.0-1.0, "overall": 0.0-1.0},
  "scores_b": {"architecture": 0.0-1.0, "code_quality": 0.0-1.0, "security": 0.0-1.0, "performance": 0.0-1.0, "maintainability": 0.0-1.0, "completeness": 0.0-1.0, "overall": 0.0-1.0}
}

Be specific about differences. A tie is not acceptable — choose one."""

CRITIC_SYSTEM_PROMPT = """You are a DEVIL'S ADVOCATE. Your job is to find hallucinations, logical holes, and vulnerabilities.
Be aggressive and critical. Assume the solution has hidden problems.

For each issue found, rate:
- severity: "low" (minor), "medium" (real concern), "high" (blocker), "critical" (must fix)
- type: "hallucination" (claim not supported by code/facts), "logic_hole" (missing case/edge), "vulnerability" (security), "overengineering" (unnecessary complexity)

Return ONLY valid JSON:
{
  "issues": [
    {
      "severity": "low|medium|high|critical",
      "type": "hallucination|logic_hole|vulnerability|overengineering",
      "description": "what the problem is",
      "evidence": "why you believe this is a problem",
      "suggestion": "how to fix it"
    }
  ],
  "summary": "overall assessment of trustworthiness",
  "trust_score": 0.0-1.0
}"""


async def evaluate_solution(
    llm: BaseLLMProvider,
    solution: Solution,
    task_description: str,
    task_tags: list[str],
    checklist: list[str] | None = None,
) -> JudgeVerdict:
    prompt_parts = [
        f"## Task\n{task_description}",
        f"Tags: {', '.join(task_tags)}",
        f"\n## Solution: {solution.title}",
        f"\n### Summary\n{solution.summary[:500]}",
    ]

    if checklist:
        prompt_parts.append(
            "\n### Required Checklist\n" + "\n".join(f"- [ ] {item}" for item in checklist)
        )

    if solution.architecture:
        prompt_parts.append(
            f"\n### Architecture\nPattern: {solution.architecture.name}\n"
            f"{solution.architecture.description[:500]}"
        )

    if solution.stack:
        prompt_parts.append(f"\n### Tech Stack\n{solution.stack.summary()}")

    if solution.code_blocks:
        code_summary = "\n".join(
            f"- {cb.filename} ({cb.language}, {len(cb.content)} chars)"
            for cb in solution.code_blocks[:5]
        )
        prompt_parts.append(f"\n### Generated Files\n{code_summary}")

    if solution.performance_notes:
        prompt_parts.append("\n### Performance\n" + "\n".join(solution.performance_notes[:5]))

    if solution.security_notes:
        prompt_parts.append("\n### Security\n" + "\n".join(solution.security_notes[:5]))

    user_prompt = "\n".join(prompt_parts)

    messages = [
        LLMMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_prompt),
    ]

    try:
        response = await llm.complete(messages, temperature=0.2, max_tokens=2048)
        return _parse_verdict(response.content, solution)
    except Exception as e:
        log.error("judge_failed", error=str(e))
        return JudgeVerdict(
            passed=False,
            summary=f"Judge evaluation skipped: {e}",
        )


async def evaluate_pairwise(
    llm: BaseLLMProvider,
    solution_a: Solution,
    solution_b: Solution,
    task_description: str,
    task_tags: list[str],
) -> PairwiseResult:
    def _summarize(s: Solution) -> str:
        parts = [f"Title: {s.title}", f"Summary: {s.summary[:300]}"]
        if s.architecture:
            parts.append(
                f"Architecture: {s.architecture.name} - {s.architecture.description[:200]}"
            )
        if s.stack:
            parts.append(f"Stack: {s.stack.summary()[:200]}")
        if s.code_blocks:
            parts.append(f"Files: {len(s.code_blocks)} files")
            for cb in s.code_blocks[:3]:
                parts.append(f"  {cb.filename} ({len(cb.content)} chars)")
        return "\n".join(parts)

    prompt_parts = [
        f"## Task\n{task_description}",
        f"Tags: {', '.join(task_tags)}",
        "\n## Solution A\n" + _summarize(solution_a),
        "\n## Solution B\n" + _summarize(solution_b),
        "\nCompare solutions A and B. Select the better one. Return ONLY valid JSON.",
    ]
    user_prompt = "\n".join(prompt_parts)

    messages = [
        LLMMessage(role="system", content=PAIRWISE_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_prompt),
    ]

    try:
        response = await llm.complete(messages, temperature=0.2, max_tokens=2048)
        return _parse_pairwise(response.content, solution_a, solution_b)
    except Exception as e:
        log.error("pairwise_judge_failed", error=str(e))
        return PairwiseResult(winner="", rationale=f"Pairwise judge error: {e}")


async def critique_solution(
    llm: BaseLLMProvider,
    solution: Solution,
    task_description: str,
) -> dict[str, Any]:
    prompt_parts = [
        f"## Task\n{task_description}",
        f"## Solution: {solution.title}",
        f"\n### Summary\n{solution.summary[:500]}",
    ]
    if solution.code_blocks:
        for cb in solution.code_blocks[:3]:
            prompt_parts.append(f"\n--- {cb.filename} ---\n{cb.content[:2000]}")
    user_prompt = "\n".join(prompt_parts)

    messages = [
        LLMMessage(role="system", content=CRITIC_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_prompt),
    ]

    try:
        response = await llm.complete(messages, temperature=0.3, max_tokens=2048)
        return cast("dict[str, Any]", json.loads(strip_fences(response.content)))
    except Exception as e:
        log.error("critic_failed", error=str(e))
        return {"issues": [], "summary": f"Critique failed: {e}", "trust_score": 0.5}


def _score(value: Any, default: float = 0.0) -> float:
    """Coerce an LLM-provided score to ``0.0..1.0``.

    Real models frequently return scores as strings (``"0.7"``), booleans or
    out-of-range numbers; those must not crash or distort the verdict.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:  # NaN
        return default
    return max(0.0, min(1.0, parsed))


def _parse_verdict(raw: str, solution: Solution) -> JudgeVerdict:
    text = strip_fences(raw)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return JudgeVerdict(
            passed=False,
            raw_feedback=raw[:500],
            summary="Raw (unparseable) judge feedback",
        )

    scores_data = data.get("scores", {})
    scores = JudgeScore(
        architecture=_score(scores_data.get("architecture")),
        code_quality=_score(scores_data.get("code_quality")),
        security=_score(scores_data.get("security")),
        performance=_score(scores_data.get("performance")),
        maintainability=_score(scores_data.get("maintainability")),
        completeness=_score(scores_data.get("completeness")),
        overall=_score(scores_data.get("overall")),
    )

    passed = scores.overall >= 0.5

    solution.metadata["judge_scores"] = {
        "architecture": scores.architecture,
        "code_quality": scores.code_quality,
        "security": scores.security,
        "performance": scores.performance,
        "maintainability": scores.maintainability,
        "completeness": scores.completeness,
        "overall": scores.overall,
    }
    solution.metadata["judge_weaknesses"] = data.get("weaknesses", [])

    log.info(
        "judge_verdict",
        passed=passed,
        overall=round(scores.overall, 2),
        readiness=data.get("production_readiness", "unknown"),
    )

    return JudgeVerdict(
        passed=passed,
        scores=scores,
        strengths=data.get("strengths", []),
        weaknesses=data.get("weaknesses", []),
        improvements=data.get("improvements", []),
        production_readiness=data.get("production_readiness", "needs-work"),
        summary=data.get("summary", "")[:500],
        raw_feedback=raw[:500],
    )


def _parse_pairwise(raw: str, solution_a: Solution, solution_b: Solution) -> PairwiseResult:
    text = strip_fences(raw)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return PairwiseResult(winner="A", rationale=raw[:300])

    winner = data.get("winner", "A")
    scores_a_data = data.get("scores_a", {})
    scores_b_data = data.get("scores_b", {})

    def _parse_scores(d: dict) -> JudgeScore:
        return JudgeScore(
            architecture=_score(d.get("architecture")),
            code_quality=_score(d.get("code_quality")),
            security=_score(d.get("security")),
            performance=_score(d.get("performance")),
            maintainability=_score(d.get("maintainability")),
            completeness=_score(d.get("completeness")),
            overall=_score(d.get("overall")),
        )

    return PairwiseResult(
        winner=winner,
        winner_index=0 if winner == "A" else 1,
        rationale=data.get("rationale", ""),
        scores_a=_parse_scores(scores_a_data),
        scores_b=_parse_scores(scores_b_data),
    )
