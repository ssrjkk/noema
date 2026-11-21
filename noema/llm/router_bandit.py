"""Contextual LinUCB Bandit Router for cost/latency/accuracy Pareto-optimal routing.

Designed as an opt-in routing policy for ``noema/routing/model_router.py``.
Uses exact Sherman-Morrison rank-1 inverse updates O(d^2) with thread safety
(``RLock``), strict fail-closed budget enforcement (``BudgetExceededError``),
and validated checkpoint deserialization.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class BudgetExceededError(RuntimeError):
    """Raised under fail-closed budget policy when no arm fits remaining budget."""


@dataclass(frozen=True)
class TaskContext:
    """Observable engineering features of a task/subtask before LLM invocation."""

    prompt_tokens: int
    tag_count: int
    domain_modules_count: int
    complexity_level: str
    historical_unsat_rate: float = 0.15
    requires_formal_z3: bool = True

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0:
            raise ValueError("TaskContext.prompt_tokens must be >= 0")

    def to_feature_vector(self) -> list[float]:
        """Convert task context into a normalized 7-D feature vector with bias term."""
        comp_map = {"simple": 0.2, "medium": 0.55, "complex": 0.95}
        comp_val = comp_map.get(self.complexity_level.lower(), 0.55)
        log_tokens = min(1.0, math.log1p(max(0, self.prompt_tokens)) / 10.0)
        norm_tags = min(1.0, max(0, self.tag_count) / 8.0)
        norm_mods = min(1.0, max(0, self.domain_modules_count) / 6.0)
        unsat = max(0.0, min(1.0, self.historical_unsat_rate))
        z3_flag = 1.0 if self.requires_formal_z3 else 0.0
        return [1.0, log_tokens, norm_tags, norm_mods, comp_val, unsat, z3_flag]


@dataclass(frozen=True)
class ModelArm:
    """Candidate inference target (local LoRA, Ollama, Frontier API, or gRPC peer)."""

    arm_id: str
    provider: str
    model: str
    cost_per_1k_tokens_usd: float
    avg_latency_sec: float
    is_local: bool = False


@dataclass(frozen=True)
class RoutingDecision:
    """Auditable routing decision logged into Noema's reasoning trace & billing."""

    arm: ModelArm
    predicted_pass_prob: float
    ucb_bonus: float
    utility_score: float
    estimated_cost_usd: float


def _dot(u: Sequence[float], v: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(u, v, strict=True))


def _mat_vec(mat: Sequence[Sequence[float]], vec: Sequence[float]) -> list[float]:
    return [sum(row[j] * vec[j] for j in range(len(vec))) for row in mat]


class _LinUCBArmState:
    """Internal per-arm ridge state (A^{-1}, b) updated via Sherman-Morrison."""

    def __init__(self, dim: int, ridge_lambda: float = 1.0, prior_bias: float = 0.65) -> None:
        self.dim = dim
        inv_l = 1.0 / max(1e-6, ridge_lambda)
        self.a_inv: list[list[float]] = [
            [inv_l if i == j else 0.0 for j in range(dim)] for i in range(dim)
        ]
        self.b: list[float] = [prior_bias * ridge_lambda] + [0.0] * (dim - 1)
        self.pulls: int = 0
        self.passes: int = 0
        self.total_cost_usd: float = 0.0

    def theta(self) -> list[float]:
        return _mat_vec(self.a_inv, self.b)

    def predict(self, x: Sequence[float], alpha: float) -> tuple[float, float]:
        theta_vec = self.theta()
        mean_pred = max(0.0, min(1.0, _dot(theta_vec, x)))
        a_inv_x = _mat_vec(self.a_inv, x)
        variance = max(0.0, _dot(x, a_inv_x))
        bonus = alpha * math.sqrt(variance)
        return mean_pred, bonus

    def update(self, x: Sequence[float], reward_pass: float, cost_usd: float) -> None:
        """Exact Sherman-Morrison rank-1 inverse update."""
        a_inv_x = _mat_vec(self.a_inv, x)
        denom = 1.0 + _dot(x, a_inv_x)
        if denom > 1e-12:
            for i in range(self.dim):
                row_i = self.a_inv[i]
                factor = a_inv_x[i] / denom
                for j in range(self.dim):
                    row_i[j] -= factor * a_inv_x[j]

        for i in range(self.dim):
            self.b[i] += reward_pass * x[i]

        self.pulls += 1
        if reward_pass >= 0.5:
            self.passes += 1
        self.total_cost_usd += max(0.0, cost_usd)


class LinUCBModelRouter:
    """Thread-safe Contextual Bandit router optimizing Pass@1 vs. Cost & Latency."""

    FEATURE_DIM = 7

    def __init__(
        self,
        arms: Sequence[ModelArm],
        alpha: float = 0.35,
        cost_weight: float = 18.0,
        latency_weight: float = 0.02,
        ridge_lambda: float = 1.0,
    ) -> None:
        if not arms:
            raise ValueError("LinUCBModelRouter requires at least one ModelArm")
        seen_ids: set[str] = set()
        duplicates: set[str] = set()
        for arm in arms:
            if arm.arm_id in seen_ids:
                duplicates.add(arm.arm_id)
            seen_ids.add(arm.arm_id)
        if duplicates:
            raise ValueError(
                f"Duplicate arm_id(s) {sorted(duplicates)} would silently drop an arm "
                "from the routing pool"
            )
        self._lock = threading.RLock()
        self.arms: dict[str, ModelArm] = {a.arm_id: a for a in arms}
        self.alpha = alpha
        self.cost_weight = cost_weight
        self.latency_weight = latency_weight
        self._states: dict[str, _LinUCBArmState] = {
            a.arm_id: _LinUCBArmState(
                self.FEATURE_DIM,
                ridge_lambda=ridge_lambda,
                prior_bias=0.72 if not a.is_local else 0.62,
            )
            for a in arms
        }

    def select_arm(
        self,
        context: TaskContext,
        budget_remaining_usd: float | None = None,
        exclude_arms: set[str] | None = None,
        *,
        allow_free_local_when_exhausted: bool = False,
    ) -> RoutingDecision:
        """Choose the Pareto-optimal model arm; fail-closed if budget is exceeded."""
        x = context.to_feature_vector()
        est_tokens = max(256, context.prompt_tokens * 2)
        excluded = exclude_arms or set()
        if budget_remaining_usd is not None and budget_remaining_usd < 0.0:
            raise ValueError("budget_remaining_usd must be >= 0.0")
        exhausted = budget_remaining_usd is not None and budget_remaining_usd <= 0.0

        with self._lock:
            best_decision: RoutingDecision | None = None
            for arm_id, arm in self.arms.items():
                if arm_id in excluded:
                    continue
                est_cost = (est_tokens / 1000.0) * arm.cost_per_1k_tokens_usd
                if budget_remaining_usd is not None and est_cost > budget_remaining_usd:
                    continue
                if exhausted and not (arm.is_local and allow_free_local_when_exhausted):
                    continue

                mean_pass, bonus = self._states[arm_id].predict(x, self.alpha)
                utility = (
                    (mean_pass + bonus)
                    - self.cost_weight * est_cost
                    - self.latency_weight * arm.avg_latency_sec
                )
                decision = RoutingDecision(
                    arm=arm,
                    predicted_pass_prob=mean_pass,
                    ucb_bonus=bonus,
                    utility_score=utility,
                    estimated_cost_usd=est_cost,
                )
                if best_decision is None or decision.utility_score > best_decision.utility_score:
                    best_decision = decision

            if best_decision is None:
                raise BudgetExceededError(
                    f"No eligible model arm fits budget_remaining_usd={budget_remaining_usd!r} "
                    f"(excluded={sorted(excluded)})"
                )
            return best_decision

    def record_outcome(
        self,
        arm_id: str,
        context: TaskContext,
        verified_pass: bool,
        actual_cost_usd: float,
    ) -> None:
        """Online update after Z3 + Docker Sandbox returns ground-truth verification."""
        x = context.to_feature_vector()
        with self._lock:
            if arm_id not in self._states:
                raise KeyError(f"Unknown arm_id: {arm_id}")
            self._states[arm_id].update(x, 1.0 if verified_pass else 0.0, actual_cost_usd)

    def pareto_summary(self) -> list[dict[str, object]]:
        """Return empirical Pass@1 vs. Cost metrics per arm for dashboards and benchmarks."""
        with self._lock:
            ranked: list[tuple[float, float, dict[str, object]]] = []
            for arm_id, arm in self.arms.items():
                st = self._states[arm_id]
                pass_rate = st.passes / st.pulls if st.pulls > 0 else 0.0
                avg_cost = (
                    st.total_cost_usd / st.pulls if st.pulls > 0 else arm.cost_per_1k_tokens_usd
                )
                row: dict[str, object] = {
                    "arm_id": arm_id,
                    "provider": arm.provider,
                    "model": arm.model,
                    "pulls": st.pulls,
                    "pass_rate": round(pass_rate, 4),
                    "avg_cost_usd": round(avg_cost, 6),
                    "avg_latency_sec": arm.avg_latency_sec,
                    "is_local": arm.is_local,
                }
                ranked.append((-pass_rate, avg_cost, row))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in ranked]

    def save(self, path: str | Path) -> None:
        """Persist router weights and empirical counts to JSON."""
        with self._lock:
            payload = {
                "alpha": self.alpha,
                "cost_weight": self.cost_weight,
                "latency_weight": self.latency_weight,
                "arms": [asdict(a) for a in self.arms.values()],
                "states": {
                    arm_id: {
                        "a_inv": st.a_inv,
                        "b": st.b,
                        "pulls": st.pulls,
                        "passes": st.passes,
                        "total_cost_usd": st.total_cost_usd,
                    }
                    for arm_id, st in self._states.items()
                },
            }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> LinUCBModelRouter:
        """Restore persisted LinUCB router weights with strict shape & finiteness validation."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("arms"), list):
            raise ValueError("Invalid LinUCB checkpoint: missing 'arms' list")

        arms = [ModelArm(**item) for item in data["arms"]]
        alpha = float(data["alpha"])
        cost_weight = float(data["cost_weight"])
        latency_weight = float(data["latency_weight"])
        if not all(math.isfinite(v) and v >= 0.0 for v in (alpha, cost_weight, latency_weight)):
            raise ValueError("Invalid LinUCB hyperparameters: must be finite and non-negative")

        router = cls(
            arms=arms,
            alpha=alpha,
            cost_weight=cost_weight,
            latency_weight=latency_weight,
        )
        states = data.get("states")
        if not isinstance(states, dict) or set(states.keys()) != set(router.arms.keys()):
            raise ValueError("Invalid LinUCB checkpoint: states keys must match configured arms")

        dim = cls.FEATURE_DIM
        for arm_id, st_data in states.items():
            a_inv = st_data.get("a_inv")
            b_vec = st_data.get("b")
            pulls = int(st_data["pulls"])
            passes = int(st_data["passes"])
            total_cost = float(st_data["total_cost_usd"])

            if (
                not isinstance(a_inv, list)
                or len(a_inv) != dim
                or any(
                    not isinstance(row, list)
                    or len(row) != dim
                    or any(not math.isfinite(float(val)) for val in row)
                    for row in a_inv
                )
            ):
                raise ValueError(f"Invalid a_inv matrix shape or values for arm {arm_id!r}")
            if (
                not isinstance(b_vec, list)
                or len(b_vec) != dim
                or any(not math.isfinite(float(val)) for val in b_vec)
            ):
                raise ValueError(f"Invalid b vector shape or values for arm {arm_id!r}")
            if (
                pulls < 0
                or passes < 0
                or passes > pulls
                or not math.isfinite(total_cost)
                or total_cost < 0.0
            ):
                raise ValueError(f"Invalid empirical counters for arm {arm_id!r}")

            st = router._states[arm_id]
            st.a_inv = [[float(val) for val in row] for row in a_inv]
            st.b = [float(val) for val in b_vec]
            st.pulls = pulls
            st.passes = passes
            st.total_cost_usd = total_cost
        return router
