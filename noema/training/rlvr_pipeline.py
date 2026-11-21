"""RLVR (Reinforcement Learning with Verifiable Rewards) & Judge Calibration Pipeline.

Converts Noema's audited reasoning traces (AST + Z3 + MKS + Docker Sandbox) into:
1. DPO preference datasets: (prompt, chosen_verified, rejected_unverified, margin).
2. GRPO rollout groups with within-group advantage normalization.
3. Platt-calibrated Judge probabilities with honest out-of-sample k-fold CV
   Brier Score and Expected Calibration Error (ECE).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class TraceRollout:
    """Single audited reasoning/code rollout recorded by noema/tracing/."""

    task_id: str
    prompt: str
    response: str
    ast_ok: bool
    z3_sat: bool
    mutation_kill_score: float
    docker_pass_rate: float
    normalized_cost: float = 0.0
    raw_judge_score: float = 0.5


@dataclass(frozen=True)
class RewardWeights:
    """Weights for Noema's composite verifiable reward function."""

    w_ast: float = 0.15
    w_z3_mks: float = 0.35
    w_docker: float = 0.50
    w_cost_penalty: float = 0.10


def compute_verifiable_reward(
    rollout: TraceRollout,
    weights: RewardWeights | None = None,
) -> float:
    """Compute objective reward R(y) grounded purely in formal + runtime verification."""
    w = weights or RewardWeights()
    ast_term = 1.0 if rollout.ast_ok else 0.0
    z3_term = max(0.0, min(1.0, rollout.mutation_kill_score)) if rollout.z3_sat else 0.0
    docker_term = max(0.0, min(1.0, rollout.docker_pass_rate))
    cost_term = max(0.0, min(1.0, rollout.normalized_cost))
    reward = (
        w.w_ast * ast_term
        + w.w_z3_mks * z3_term
        + w.w_docker * docker_term
        - w.w_cost_penalty * cost_term
    )
    return max(0.0, min(1.0, reward))


class RLVRDatasetBuilder:
    """Exports DPO pairs and GRPO advantage-normalized groups from verified Noema traces."""

    def __init__(
        self,
        weights: RewardWeights | None = None,
        min_reward_margin: float = 0.25,
    ) -> None:
        self.weights = weights or RewardWeights()
        self.min_reward_margin = min_reward_margin

    def build_dpo_pairs(
        self,
        rollouts: Sequence[TraceRollout],
        max_pairs_per_task: int | None = None,
    ) -> list[dict[str, object]]:
        """Pair high-reward verified traces against rejected/UNSAT traces for DPO training.

        ``max_pairs_per_task`` bounds the otherwise O(n^2) per-task pairing blow-up.
        """
        by_task: dict[str, list[tuple[TraceRollout, float]]] = defaultdict(list)
        for r in rollouts:
            by_task[r.task_id].append((r, compute_verifiable_reward(r, self.weights)))

        pairs: list[dict[str, object]] = []
        for task_id, items in by_task.items():
            items.sort(key=lambda pair: pair[1], reverse=True)
            emitted = 0
            for idx_hi, (hi, r_hi) in enumerate(items):
                if not (hi.ast_ok and hi.z3_sat and hi.docker_pass_rate >= 1.0):
                    continue
                for lo, r_lo in items[idx_hi + 1 :]:
                    margin = r_hi - r_lo
                    if margin >= self.min_reward_margin:
                        pairs.append(
                            {
                                "task_id": task_id,
                                "prompt": hi.prompt,
                                "chosen": hi.response,
                                "rejected": lo.response,
                                "reward_chosen": round(r_hi, 4),
                                "reward_rejected": round(r_lo, 4),
                                "margin": round(margin, 4),
                            }
                        )
                        emitted += 1
                        if max_pairs_per_task is not None and emitted >= max_pairs_per_task:
                            break
                if max_pairs_per_task is not None and emitted >= max_pairs_per_task:
                    break
        return pairs

    def build_grpo_groups(
        self,
        rollouts: Sequence[TraceRollout],
        eps: float = 1e-6,
    ) -> list[dict[str, object]]:
        """Compute group-relative advantages A_i = (r_i - mean(r)) / (std(r) + eps)."""
        by_task: dict[str, list[tuple[TraceRollout, float]]] = defaultdict(list)
        for r in rollouts:
            by_task[r.task_id].append((r, compute_verifiable_reward(r, self.weights)))

        groups: list[dict[str, object]] = []
        for task_id, items in by_task.items():
            if len(items) < 2:
                continue
            rewards = [rw for _, rw in items]
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((rw - mean_r) ** 2 for rw in rewards) / len(rewards)
            std_r = math.sqrt(var_r)
            groups.append(
                {
                    "task_id": task_id,
                    "prompt": items[0][0].prompt,
                    "mean_reward": round(mean_r, 4),
                    "std_reward": round(std_r, 4),
                    "completions": [
                        {
                            "response": r.response,
                            "reward": round(rw, 4),
                            "advantage": round((rw - mean_r) / (std_r + eps), 4),
                        }
                        for r, rw in items
                    ],
                }
            )
        return groups

    def export_jsonl(
        self,
        records: Sequence[dict[str, object]],
        output_path: str | Path,
        *,
        allowed_root: str | Path | None = None,
        create_parents: bool = False,
    ) -> Path:
        """Write dataset records as UTF-8 JSONL.

        ``allowed_root`` confines the destination: a path that resolves outside it raises
        ValueError, so a caller cannot smuggle ``..`` out of the export directory.
        Parent directories are not created unless ``create_parents=True`` — callers that
        own the path (the CLI) pass it explicitly.
        """
        out = Path(output_path)
        if allowed_root is not None:
            root_resolved = Path(allowed_root).resolve()
            out_resolved = out.resolve()
            if not out_resolved.is_relative_to(root_resolved):
                raise ValueError(f"Output path {out_resolved} escapes allowed_root {root_resolved}")
        if create_parents:
            out.parent.mkdir(parents=True, exist_ok=True)
        elif not out.parent.exists():
            raise FileNotFoundError(
                f"Parent directory does not exist: {out.parent} "
                "(pass create_parents=True to create it)"
            )
        with out.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return out


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass(frozen=True)
class CalibrationMetrics:
    """Calibration quality summary (Brier Score and Expected Calibration Error)."""

    brier_score: float
    ece: float
    samples: int


class JudgeCalibrator:
    """Platt Scaling calibrator mapping raw LLM judge_score to true Docker pass probability."""

    def __init__(self, slope: float = 1.0, intercept: float = 0.0) -> None:
        self.slope = slope
        self.intercept = intercept

    def fit(
        self,
        raw_scores: Sequence[float],
        docker_outcomes: Sequence[int],
        lr: float = 0.25,
        epochs: int = 400,
        l2_reg: float = 1e-3,
    ) -> CalibrationMetrics:
        """Fit logistic Platt parameters (a, b) minimizing regularized binary cross-entropy.

        Note: The returned metrics are training-set (in-sample) diagnostics. For unbiased
        evaluation use ``cross_validate()`` or evaluate on a held-out validation split.
        """
        if len(raw_scores) != len(docker_outcomes) or not raw_scores:
            raise ValueError("raw_scores and docker_outcomes must be non-empty and equal length")

        n = len(raw_scores)
        a = self.slope
        b = self.intercept

        for _ in range(epochs):
            grad_a = 0.0
            grad_b = 0.0
            for s, y in zip(raw_scores, docker_outcomes, strict=True):
                p = _sigmoid(a * s + b)
                err = p - float(y)
                grad_a += err * s
                grad_b += err
            a -= lr * (grad_a / n + l2_reg * a)
            b -= lr * (grad_b / n)

        self.slope = a
        self.intercept = b
        probs = [self.calibrate(s) for s in raw_scores]
        return self.evaluate(probs, docker_outcomes)

    @classmethod
    def cross_validate(
        cls,
        raw_scores: Sequence[float],
        docker_outcomes: Sequence[int],
        k_folds: int = 5,
        lr: float = 0.25,
        epochs: int = 400,
        l2_reg: float = 1e-3,
    ) -> CalibrationMetrics:
        """Compute honest out-of-sample k-fold CV Brier Score and ECE."""
        n = len(raw_scores)
        if n != len(docker_outcomes) or n < 2:
            raise ValueError("Need at least 2 paired samples for cross-validation")
        folds = min(max(2, k_folds), n)

        oof_probs: list[float] = []
        oof_outcomes: list[int] = []

        for fold_idx in range(folds):
            train_s: list[float] = []
            train_y: list[int] = []
            val_s: list[float] = []
            val_y: list[int] = []
            for idx, (s, y) in enumerate(zip(raw_scores, docker_outcomes, strict=True)):
                if idx % folds == fold_idx:
                    val_s.append(s)
                    val_y.append(y)
                else:
                    train_s.append(s)
                    train_y.append(y)
            if not val_s or not train_s:
                continue
            fold_cal = cls()
            fold_cal.fit(train_s, train_y, lr=lr, epochs=epochs, l2_reg=l2_reg)
            oof_probs.extend(fold_cal.calibrate(s) for s in val_s)
            oof_outcomes.extend(val_y)

        return cls.evaluate(oof_probs, oof_outcomes)

    def calibrate(self, raw_judge_score: float) -> float:
        """Return calibrated pass probability in (0, 1)."""
        return _sigmoid(self.slope * raw_judge_score + self.intercept)

    @staticmethod
    def evaluate(
        predicted_probs: Sequence[float],
        outcomes: Sequence[int],
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """Compute exact Brier Score and Expected Calibration Error (ECE)."""
        n = len(predicted_probs)
        if n == 0 or n != len(outcomes):
            raise ValueError("predicted_probs and outcomes must be non-empty and equal length")

        brier = sum((p - float(y)) ** 2 for p, y in zip(predicted_probs, outcomes, strict=True)) / n

        bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
        for p, y in zip(predicted_probs, outcomes, strict=True):
            idx = min(n_bins - 1, max(0, int(p * n_bins)))
            bins[idx].append((p, y))

        ece = 0.0
        for bucket in bins:
            if not bucket:
                continue
            avg_conf = sum(p for p, _ in bucket) / len(bucket)
            avg_acc = sum(y for _, y in bucket) / len(bucket)
            ece += (len(bucket) / n) * abs(avg_conf - avg_acc)

        return CalibrationMetrics(brier_score=round(brier, 6), ece=round(ece, 6), samples=n)
