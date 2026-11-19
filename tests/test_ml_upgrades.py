"""Comprehensive pytest suite for Noema's CI-clean ML & Neuro-Symbolic upgrades."""

from __future__ import annotations

import ast
import importlib
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from noema.llm.router_bandit import BudgetExceededError, LinUCBModelRouter, ModelArm, TaskContext
from noema.memory.hybrid_retriever import FailurePattern, HybridCognitiveMemory, MemoryRecord
from noema.neurosymbolic.cegis_mutator import (
    Z3_AVAILABLE,
    ASTContractMutator,
    VerifierGuidedMCTS,
    Z3CEGISVerifier,
    run_static_ast_gate,
)
from noema.training.rlvr_pipeline import (
    JudgeCalibrator,
    RLVRDatasetBuilder,
    TraceRollout,
    compute_verifiable_reward,
)

if TYPE_CHECKING:
    from pathlib import Path


def _ast_has_positive_divisor_guard(code: str) -> tuple[bool, str]:
    """Pure AST test helper (zero exec) validating safe_ratio guards."""
    st = run_static_ast_gate(code)
    if not st.ok:
        return False, "AST_FAIL"
    tree = ast.parse(code)
    has_guard = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "b"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.LtE)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == 0
        for node in ast.walk(tree)
    )
    has_div = any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) for node in ast.walk(tree)
    )
    if not has_guard:
        return False, "UNSAT: divisor=0 not guarded"
    if not has_div:
        return False, "UNSAT: division operator missing"
    return True, "SAT"


def test_static_ast_gate_allows_valid_python_and_blocks_indirect_exec() -> None:
    valid = (
        "def parse_items(items):\n"
        "    fn = lambda v: v + 1\n"
        "    out = [fn(x) for x in items]\n"
        "    try:\n"
        "        match out:\n"
        "            case [first, *rest]:\n"
        "                return first + len(rest)\n"
        "    except ValueError as err:\n"
        "        return len(str(err))\n"
        "    return 0\n"
    )
    assert run_static_ast_gate(valid).ok

    for bad in (
        "def f(x):\n    return eval(unknown_var)\n",
        "import builtins\ndef f(s):\n    return builtins.exec(s)\n",
        "import os\ndef f(c):\n    return os.system(c)\n",
        "import subprocess\ndef f(c):\n    return subprocess.run(c, shell=True)\n",
        "import builtins\ndef f():\n    return getattr(builtins, 'exec')\n",
    ):
        assert not run_static_ast_gate(bad).ok


def test_ast_mutator_rejects_broken_and_weak_verifiers() -> None:
    src = (
        "def safe_ratio(a: int, b: int) -> float:\n"
        "    if b <= 0:\n"
        "        raise ValueError('b must be > 0')\n"
        "    return a / b\n"
    )
    mutator = ASTContractMutator()

    weak = mutator.evaluate_contract_strength(src, lambda _code: True, min_kill_score=0.80)
    assert weak.baseline_passed
    assert weak.mutation_kill_score == 0.0
    assert not weak.passed

    always_false = mutator.evaluate_contract_strength(src, lambda _code: False, min_kill_score=0.80)
    assert not always_false.baseline_passed
    assert always_false.mutation_kill_score == 0.0
    assert not always_false.passed

    def broken_verifier(_code: str) -> bool:
        raise RuntimeError("broken solver")

    broken = mutator.evaluate_contract_strength(src, broken_verifier, min_kill_score=0.80)
    assert not broken.baseline_passed
    assert broken.verifier_errors >= 1
    assert broken.mutation_kill_score == 0.0
    assert not broken.passed

    strong = mutator.evaluate_contract_strength(
        src,
        lambda c: _ast_has_positive_divisor_guard(c)[0],
        min_kill_score=0.80,
    )
    assert strong.baseline_passed
    assert strong.verifier_errors == 0
    assert strong.passed
    assert strong.mutation_kill_score == 1.0


def test_mcts_cegis_repairs_buggy_seed_and_verifies_mks() -> None:
    buggy = "def ratio(a, b):\n    return a / b\n"
    fixed = "def ratio(a, b):\n    if b <= 0:\n        raise ValueError('b<=0')\n    return a / b\n"

    engine = VerifierGuidedMCTS(min_mks=0.80)
    result = engine.search(
        task="safe ratio",
        initial_code=buggy,
        propose_fn=lambda _t, _c, _fb: [fixed],
        verify_fn=_ast_has_positive_divisor_guard,
        max_rollouts=5,
    )
    assert result.verified
    assert result.cegis_repairs >= 1
    assert result.mutation_kill_score >= 0.80


def test_z3_cegis_fail_closed_or_unsat_core() -> None:
    verifier = Z3CEGISVerifier(fail_closed=True)
    if not Z3_AVAILABLE:
        fb = verifier.verify_named_constraints({"c1": object()})
        assert fb.verdict == "FAIL_CLOSED"
        assert not fb.is_sat
    else:
        z3 = importlib.import_module("z3")
        x = z3.Int("x")
        fb = verifier.verify_named_constraints({"c_pos": x > 10, "c_neg": x < 0})
        assert fb.verdict == "UNSAT"
        assert set(fb.unsat_core) == {"c_pos", "c_neg"}


def test_linucb_cost_sensitivity_budget_fail_closed_and_load_validation(tmp_path: Path) -> None:
    arms = (
        ModelArm("local-lora", "ollama", "qwen-7b-lora", 0.0002, 0.7, is_local=True),
        ModelArm("frontier-expensive", "openai", "gpt-4o", 0.0500, 0.7, is_local=False),
    )
    ctx = TaskContext(
        prompt_tokens=1000,
        tag_count=2,
        domain_modules_count=1,
        complexity_level="simple",
    )

    router = LinUCBModelRouter(arms, alpha=0.0, cost_weight=15.0, latency_weight=0.0)
    for _ in range(10):
        router.record_outcome("local-lora", ctx, verified_pass=True, actual_cost_usd=0.0004)
        router.record_outcome("frontier-expensive", ctx, verified_pass=True, actual_cost_usd=0.1)
    assert router.select_arm(ctx).arm.arm_id == "local-lora"

    zero_cost_router = LinUCBModelRouter(arms, alpha=0.0, cost_weight=0.0, latency_weight=0.0)
    assert zero_cost_router.select_arm(ctx).arm.arm_id == "frontier-expensive"

    with pytest.raises(BudgetExceededError):
        router.select_arm(ctx, budget_remaining_usd=0.00001)

    save_path = tmp_path / "router_state.json"
    router.save(save_path)
    restored = LinUCBModelRouter.load(save_path)
    assert restored.select_arm(ctx).arm.arm_id == "local-lora"

    corrupted = json.loads(save_path.read_text(encoding="utf-8"))
    corrupted["states"]["local-lora"]["b"] = [1.0, 2.0]
    bad_path = tmp_path / "bad_state.json"
    bad_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid b vector"):
        LinUCBModelRouter.load(bad_path)


def test_hybrid_bitemporal_memory_strict_supersede_and_non_destructive_retention() -> None:
    mem = HybridCognitiveMemory()
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = t0 + timedelta(days=5)
    t_far = t0 + timedelta(days=365)

    mem.add(
        MemoryRecord(
            record_id="cfg_old",
            content="PaymentGateway timeout is 5 seconds",
            valid_from=t0,
        )
    )
    mem.supersede(
        "cfg_old",
        MemoryRecord(
            record_id="cfg_new",
            content="PaymentGateway timeout is 30 seconds after retry storm fix",
        ),
        at_time=t1,
    )

    with pytest.raises(KeyError, match="nonexistent_id"):
        mem.supersede(
            "nonexistent_id",
            MemoryRecord(record_id="cfg_x", content="orphan"),
            at_time=t1,
        )

    # Even 1 year later, valid semantic facts are NOT silently erased (default min_retention=0.0)
    hits = mem.retrieve("PaymentGateway timeout seconds", top_k=5, at_time=t_far)
    hit_ids = [h.record.record_id for h in hits]
    assert "cfg_new" in hit_ids
    assert "cfg_old" not in hit_ids

    mem.index_failure_fix(
        FailurePattern(
            pattern_id="fix_div0",
            error_signature="ZeroDivisionError in quota_router.py",
            unsat_core=("c_nonzero_divisor",),
            verified_patch_diff="if divisor <= 0: raise ValueError()",
            z3_contract="(assert (> divisor 0))",
        )
    )
    found = mem.lookup_similar_failures("ZeroDivisionError quota_router", ("c_nonzero_divisor",))
    assert len(found) == 1
    assert found[0][0].pattern_id == "fix_div0"


def test_rlvr_dataset_builder_and_honest_out_of_sample_calibration(tmp_path: Path) -> None:
    rollouts = [
        TraceRollout("t1", "quota guard", "good_code", True, True, 0.95, 1.0, 0.05, 0.95),
        TraceRollout("t1", "quota guard", "bad_code", True, False, 0.0, 0.0, 0.10, 0.80),
        TraceRollout("t1", "quota guard", "syntax_err", False, False, 0.0, 0.0, 0.02, 0.20),
    ]
    assert compute_verifiable_reward(rollouts[0]) > 0.9
    assert compute_verifiable_reward(rollouts[1]) < 0.2

    builder = RLVRDatasetBuilder(min_reward_margin=0.30)
    dpo = builder.build_dpo_pairs(rollouts)
    grpo = builder.build_grpo_groups(rollouts)
    assert len(dpo) == 2
    assert len(grpo) == 1
    completions = grpo[0]["completions"]
    assert isinstance(completions, list)
    first_comp = completions[0]
    assert isinstance(first_comp, dict)
    assert float(first_comp["advantage"]) > 0

    out_file = builder.export_jsonl(dpo, tmp_path / "dpo.jsonl")
    assert out_file.exists()

    train_scores = [0.98, 0.95, 0.91, 0.89, 0.86, 0.84, 0.80, 0.76, 0.72, 0.65, 0.55, 0.45]
    train_y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    val_scores = [0.97, 0.93, 0.88, 0.85, 0.81, 0.74, 0.62, 0.50]
    val_y = [1, 1, 1, 0, 0, 0, 0, 0]

    raw_val_metrics = JudgeCalibrator.evaluate(val_scores, val_y)
    cal = JudgeCalibrator()
    cal.fit(train_scores, train_y, lr=0.4, epochs=600)
    cal_val_probs = [cal.calibrate(s) for s in val_scores]
    cal_val_metrics = JudgeCalibrator.evaluate(cal_val_probs, val_y)

    assert cal_val_metrics.brier_score < raw_val_metrics.brier_score

    all_scores = train_scores + val_scores
    all_y = train_y + val_y
    cv_metrics = JudgeCalibrator.cross_validate(all_scores, all_y, k_folds=5)
    assert cv_metrics.samples == 20
    assert cv_metrics.brier_score < JudgeCalibrator.evaluate(all_scores, all_y).brier_score


# ---------------------------------------------------------------------------
# Mutation-battery gap closers: each test makes one previously-surviving
# mutation of the 15-mutation battery fail (kill), proving the gate bites.
# ---------------------------------------------------------------------------


def test_static_ast_gate_blocks_direct_dangerous_builtins_by_name() -> None:
    """Kill: removing the ast.Name (node.func.id) forbidden ban would let these pass.

    Every sample calls a forbidden builtin directly while all names are defined,
    so the ONLY rejection mechanism is the direct-name ban, not undefined_names
    and not the attribute ban.
    """
    for bad in (
        "def f(s):\n    return eval(s)\n",
        "def f(s):\n    return exec(s)\n",
        "def f(s):\n    return compile(s, '<s>', 'exec')\n",
        "def f(m):\n    return __import__(m)\n",
    ):
        st = run_static_ast_gate(bad)
        assert not st.ok
        assert st.forbidden_calls
        assert not st.undefined_names


def test_mutation_strength_never_counts_verifier_exceptions_as_kills() -> None:
    """Kill: counting verifier crashes as kills would inflate killed_mutants."""
    src = (
        "def safe_ratio(a: int, b: int) -> float:\n"
        "    if b <= 0:\n"
        "        raise ValueError('b must be > 0')\n"
        "    return a / b\n"
    )
    mutator = ASTContractMutator()

    def raises_on_mutants(code: str) -> bool:
        if code == src:
            return True
        raise RuntimeError("solver crashed on mutant")

    rep = mutator.evaluate_contract_strength(src, raises_on_mutants, min_kill_score=0.80)
    assert rep.baseline_passed
    assert rep.total_mutants > 0
    assert rep.verifier_errors == rep.total_mutants
    assert rep.killed_mutants == 0
    assert rep.mutation_kill_score == 0.0
    assert not rep.passed


def test_router_load_rejects_non_list_a_inv_matrix(tmp_path: Path) -> None:
    """Kill: disabling isinstance(a_inv, list) would raise TypeError instead of ValueError."""
    arms = (
        ModelArm("local-lora", "ollama", "qwen-7b-lora", 0.0002, 0.7, is_local=True),
        ModelArm("frontier-expensive", "openai", "gpt-4o", 0.0500, 0.7, is_local=False),
    )
    router = LinUCBModelRouter(arms, alpha=0.0, cost_weight=15.0, latency_weight=0.0)
    save_path = tmp_path / "router_state.json"
    router.save(save_path)

    corrupted = json.loads(save_path.read_text(encoding="utf-8"))
    corrupted["states"]["local-lora"]["a_inv"] = 12345
    bad_path = tmp_path / "bad_ainv.json"
    bad_path.write_text(json.dumps(corrupted), encoding="utf-8")

    with pytest.raises(ValueError, match="a_inv"):
        LinUCBModelRouter.load(bad_path)


def test_router_load_rejects_non_finite_or_negative_hyperparameters(tmp_path: Path) -> None:
    """Kill: disabling the finiteness check would silently load a corrupt router."""
    arms = (ModelArm("local-lora", "ollama", "qwen-7b-lora", 0.0002, 0.7, is_local=True),)
    router = LinUCBModelRouter(arms, alpha=0.0, cost_weight=15.0, latency_weight=0.0)
    save_path = tmp_path / "hp_state.json"
    router.save(save_path)

    for field, bad_value in (
        ("alpha", float("nan")),
        ("cost_weight", float("inf")),
        ("latency_weight", -3.0),
    ):
        corrupted = json.loads(save_path.read_text(encoding="utf-8"))
        corrupted[field] = bad_value
        bad_path = tmp_path / f"bad_{field}.json"
        bad_path.write_text(json.dumps(corrupted), encoding="utf-8")
        with pytest.raises(ValueError, match="hyperparameters"):
            LinUCBModelRouter.load(bad_path)


def test_cross_validate_never_fits_a_fold_on_the_full_dataset() -> None:
    """Kill: the train+val leakage mutation would make every fold train on all n points."""

    class _FoldSpy(JudgeCalibrator):
        seen_train_sizes: list[int] = []

        def fit(self, raw_scores, docker_outcomes, *args, **kwargs):
            _FoldSpy.seen_train_sizes.append(len(raw_scores))
            return super().fit(raw_scores, docker_outcomes, *args, **kwargs)

    scores = [0.95, 0.40, 0.88, 0.30, 0.78, 0.22, 0.65, 0.12]
    outcomes = [1, 0, 1, 0, 1, 0, 1, 0]
    n = len(scores)

    _FoldSpy.seen_train_sizes = []
    metrics = _FoldSpy.cross_validate(scores, outcomes, k_folds=4)

    assert metrics.samples == n
    assert len(_FoldSpy.seen_train_sizes) == 4
    assert all(size < n for size in _FoldSpy.seen_train_sizes)


def test_task_context_rejects_negative_prompt_tokens() -> None:
    with pytest.raises(ValueError, match="prompt_tokens"):
        TaskContext(
            prompt_tokens=-1,
            tag_count=1,
            domain_modules_count=0,
            complexity_level="simple",
        )


def test_router_budget_validation_and_zero_fail_closed() -> None:
    arms = (ModelArm("local", "ollama", "qwen", 0.0, 0.7, is_local=True),)
    router = LinUCBModelRouter(arms, alpha=0.0, cost_weight=1.0, latency_weight=0.0)
    ctx = TaskContext(
        prompt_tokens=1000,
        tag_count=1,
        domain_modules_count=1,
        complexity_level="simple",
    )

    with pytest.raises(ValueError, match="budget_remaining_usd must be >= 0"):
        router.select_arm(ctx, budget_remaining_usd=-0.5)

    with pytest.raises(BudgetExceededError):
        router.select_arm(ctx, budget_remaining_usd=0.0)

    decision = router.select_arm(
        ctx,
        budget_remaining_usd=0.0,
        allow_free_local_when_exhausted=True,
    )
    assert decision.arm.arm_id == "local"


def test_memory_add_refuses_silent_overwrite() -> None:
    mem = HybridCognitiveMemory()
    t = datetime(2026, 1, 1, tzinfo=UTC)

    mem.add(MemoryRecord(record_id="dup", content="alpha", valid_from=t))
    assert mem.record_count == 1

    with pytest.raises(ValueError, match="already exists"):
        mem.add(MemoryRecord(record_id="dup", content="beta", valid_from=t))
    assert mem.record_count == 1

    mem.add(MemoryRecord(record_id="dup", content="gamma", valid_from=t), overwrite=True)
    assert mem.record_count == 1


def test_dpo_builder_caps_pairs_per_task() -> None:
    rollouts = [
        TraceRollout("t1", "p", "good", True, True, 1.0, 1.0, 0.0, 0.95),
        TraceRollout("t1", "p", "b1", True, False, 0.0, 0.0, 0.5, 0.10),
        TraceRollout("t1", "p", "b2", True, False, 0.0, 0.0, 0.5, 0.10),
        TraceRollout("t1", "p", "b3", False, False, 0.0, 0.0, 0.5, 0.10),
    ]
    builder = RLVRDatasetBuilder(min_reward_margin=0.25)

    uncapped = builder.build_dpo_pairs(rollouts)
    assert len(uncapped) == 3

    capped = builder.build_dpo_pairs(rollouts, max_pairs_per_task=2)
    assert len(capped) == 2
