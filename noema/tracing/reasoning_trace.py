"""Verifiable reasoning traces — commit reasoning checkpoints + verification results
as run artifacts so any solution can be re-audited later without re-running the LLM.

The artifact is self-contained: it records the task input, every hypothesis that was
verified (one :class:`VerificationRound` per refine-verify attempt), the per-round AST
static verdict and symbolic (Z3) verdict, plus the terminal outcome. :func:`reverify_reasoning_trace`
replays the *deterministic* half of the pipeline — AST analysis and symbolic
verification — on the committed artifacts and reports whether the reproduced verdict
matches the recorded one. No LLM call is ever made during a re-audit.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from noema.logging import get_logger

if TYPE_CHECKING:
    from noema.neurosymbolic.symbolic import SymbolicEngine

log = get_logger(__name__)

_TRACE_SCHEMA_VERSION = 1


@dataclass
class VerificationRound:
    """One hypothesis → verification cycle of the refine-verify loop."""

    attempt: int
    hypothesis: dict[str, Any]
    static_verdict: dict[str, Any]
    symbolic_valid: bool
    violations: list[str]


@dataclass
class ReasoningTrace:
    """Self-contained, replayable record of one ``think`` run."""

    run_id: str
    correlation_id: str
    task: dict[str, Any]
    rounds: list[VerificationRound]
    outcome: str  # "completed" | "failed" | "error"
    attempts: int
    final_hypothesis: dict[str, Any] | None
    final_static_verdict: dict[str, Any] | None
    final_symbolic_valid: bool | None
    final_violations: list[str]
    error: str = ""
    schema_version: int = _TRACE_SCHEMA_VERSION
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "correlation_id": self.correlation_id,
            "created_at": self.created_at,
            "task": self.task,
            "outcome": self.outcome,
            "attempts": self.attempts,
            "error": self.error,
            "final_hypothesis": self.final_hypothesis,
            "final_static_verdict": self.final_static_verdict,
            "final_symbolic_valid": self.final_symbolic_valid,
            "final_violations": self.final_violations,
            "rounds": [asdict(r) for r in self.rounds],
        }


@dataclass
class ReplayVerdict:
    """Result of replaying a committed trace against the deterministic pipeline."""

    matches: bool
    static_matches: bool | None
    symbolic_matches: bool | None
    recorded_final_static_verdict: dict[str, Any] | None
    replayed_final_static_verdict: dict[str, Any] | None
    recorded_final_symbolic_valid: bool | None
    replayed_final_symbolic_valid: bool | None
    recorded_final_violations: list[str]
    replayed_final_violations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "matches": self.matches,
            "static_matches": self.static_matches,
            "symbolic_matches": self.symbolic_matches,
            "recorded_final_static_verdict": self.recorded_final_static_verdict,
            "replayed_final_static_verdict": self.replayed_final_static_verdict,
            "recorded_final_symbolic_valid": self.recorded_final_symbolic_valid,
            "replayed_final_symbolic_valid": self.replayed_final_symbolic_valid,
            "recorded_final_violations": self.recorded_final_violations,
            "replayed_final_violations": self.replayed_final_violations,
        }


def _safe_name(value: str) -> str:
    """Keep only filesystem-safe characters; separators could escape the dir."""
    value = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    value = value.strip("._")
    return value or "unknown"


def commit_reasoning_trace(trace: ReasoningTrace, directory: str | Path) -> Path:
    """Persist a reasoning trace as an atomic JSON artifact.

    Writes to a temp file in the same directory and renames over the target, so
    a crash mid-save can never leave a truncated trace behind.

    Returns:
        The path of the written artifact.
    """
    path = Path(directory) / f"{_safe_name(trace.run_id)}.json"
    Path(directory).mkdir(parents=True, exist_ok=True)
    payload = json.dumps(trace.to_dict(), ensure_ascii=False, default=str)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(tmp_path, path)
    log.debug("reasoning_trace_committed", run_id=trace.run_id, path=str(path))
    return path


def load_reasoning_trace(path: str | Path) -> ReasoningTrace | None:
    """Load a committed reasoning trace artifact.

    Returns:
        The trace, or ``None`` when the file is missing or malformed.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("reasoning_trace_load_failed", path=str(path), error=str(e))
        return None
    try:
        rounds = [
            VerificationRound(
                attempt=int(round_data["attempt"]),
                hypothesis=dict(round_data["hypothesis"]),
                static_verdict=dict(round_data["static_verdict"]),
                symbolic_valid=bool(round_data["symbolic_valid"]),
                violations=list(round_data["violations"]),
            )
            for round_data in data.get("rounds", [])
        ]
        final_hypothesis = data.get("final_hypothesis")
        return ReasoningTrace(
            run_id=str(data["run_id"]),
            correlation_id=str(data["correlation_id"]),
            task=dict(data["task"]),
            rounds=rounds,
            outcome=str(data.get("outcome", "error")),
            attempts=int(data.get("attempts", len(rounds))),
            final_hypothesis=dict(final_hypothesis) if final_hypothesis else None,
            final_static_verdict=(
                dict(data["final_static_verdict"]) if data.get("final_static_verdict") else None
            ),
            final_symbolic_valid=(
                bool(data["final_symbolic_valid"])
                if data.get("final_symbolic_valid") is not None
                else None
            ),
            final_violations=list(data.get("final_violations", [])),
            error=str(data.get("error", "")),
            schema_version=int(data.get("schema_version", _TRACE_SCHEMA_VERSION)),
            created_at=float(data.get("created_at", 0.0)),
        )
    except (KeyError, TypeError, ValueError) as e:
        log.warning("reasoning_trace_parse_failed", path=str(path), error=str(e))
        return None


async def reverify_reasoning_trace(
    trace: ReasoningTrace, symbolic_engine: SymbolicEngine
) -> ReplayVerdict:
    """Reproduce the trace's final verdict without any LLM call.

    Re-runs the deterministic checks — AST static analysis and symbolic
    (Z3) verification against the recorded task graph — on the recorded final
    hypothesis and compares the outcome with what was committed.

    Returns:
        A :class:`ReplayVerdict`; ``matches`` is ``True`` only when every
        recorded verdict field is reproduced exactly.
    """
    recorded_static = trace.final_static_verdict
    recorded_valid = trace.final_symbolic_valid

    replayed_static: dict[str, Any] | None = None
    static_matches: bool | None = None
    if trace.final_hypothesis is not None:
        # Imported lazily: the neurosymbolic package imports the engine, which
        # imports this module — a top-level import would be circular.
        from noema.neurosymbolic.static import analyze_solution_static

        replayed_static = analyze_solution_static(trace.final_hypothesis).as_dict()
        static_matches = replayed_static == recorded_static

    replayed_valid: bool | None = None
    replayed_violations: list[str] = list(trace.final_violations)
    symbolic_matches: bool | None = None
    if trace.final_hypothesis is not None and recorded_valid is not None:
        task_graph = await symbolic_engine.parse_task(trace.task)
        replayed_valid, replayed_violations = await symbolic_engine.verify_solution(
            trace.final_hypothesis, task_graph
        )
        symbolic_matches = replayed_valid == recorded_valid

    matches = True
    if static_matches is not None:
        matches = matches and static_matches
    if symbolic_matches is not None:
        matches = matches and symbolic_matches

    return ReplayVerdict(
        matches=matches,
        static_matches=static_matches,
        symbolic_matches=symbolic_matches,
        recorded_final_static_verdict=recorded_static,
        replayed_final_static_verdict=replayed_static,
        recorded_final_symbolic_valid=recorded_valid,
        replayed_final_symbolic_valid=replayed_valid,
        recorded_final_violations=list(trace.final_violations),
        replayed_final_violations=replayed_violations,
    )


async def reverify_trace_file(
    path: str | Path, symbolic_engine: SymbolicEngine
) -> ReplayVerdict | None:
    """Load a committed trace artifact and reverify it.

    Returns:
        The :class:`ReplayVerdict`, or ``None`` when the artifact cannot be loaded.
    """
    trace = load_reasoning_trace(path)
    if trace is None:
        return None
    return await reverify_reasoning_trace(trace, symbolic_engine)


def build_reasoning_trace(
    *,
    correlation_id: str,
    task: dict[str, Any],
    rounds: list[VerificationRound],
    outcome: str,
    attempts: int,
    final_hypothesis: dict[str, Any] | None,
    final_static_verdict: dict[str, Any] | None,
    final_symbolic_valid: bool | None,
    final_violations: list[str],
    error: str = "",
) -> ReasoningTrace:
    """Build a :class:`ReasoningTrace` with a fresh ``run_id``."""
    return ReasoningTrace(
        run_id=uuid.uuid4().hex,
        correlation_id=correlation_id,
        task=task,
        rounds=list(rounds),
        outcome=outcome,
        attempts=attempts,
        final_hypothesis=final_hypothesis,
        final_static_verdict=final_static_verdict,
        final_symbolic_valid=final_symbolic_valid,
        final_violations=list(final_violations),
        error=error,
    )
