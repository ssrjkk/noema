"""CEGIS verification, AST contract mutation strength testing, and Verifier-Guided MCTS.

Upgrades Noema's linear propose->verify refine loop into:
1. Fail-closed Z3 CEGIS with unsat_core() and counterexample model extraction.
2. Mutation-Based Contract Strength Testing (Mutation Kill Score, MKS) that validates
   baseline source first and never counts verifier exceptions as mutant kills.
3. Verifier-Guided MCTS (PUCT) over ChainOfThought DAG nodes with a pluggable
   Process Reward Model (PRM).
"""

from __future__ import annotations

import ast
import builtins
import copy
import importlib
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

z3: Any = None
try:
    z3 = importlib.import_module("z3")
    Z3_AVAILABLE = True
except ImportError:  # pragma: no cover - fail-closed path tested explicitly
    Z3_AVAILABLE = False


class Z3UnavailableError(RuntimeError):
    """Raised under Noema's fail-closed invariant when Z3 is requested but missing."""


@dataclass(frozen=True)
class StaticCheckResult:
    """Result of pre-run pure AST hygiene analysis."""

    ok: bool
    syntax_error: str | None = None
    undefined_names: tuple[str, ...] = ()
    forbidden_calls: tuple[str, ...] = ()


@dataclass(frozen=True)
class CEGISFeedback:
    """Structured counterexample feedback extracted from Z3 UNSAT/SAT runs."""

    verdict: str
    unsat_core: tuple[str, ...] = ()
    counterexample: dict[str, str] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    reason: str = ""

    @property
    def is_sat(self) -> bool:
        return self.verdict == "SAT"


@dataclass(frozen=True)
class CodeMutant:
    """Single semantic AST mutant used to stress-test contract strength."""

    mutant_id: str
    operator: str
    lineno: int
    mutated_source: str


@dataclass(frozen=True)
class ContractStrengthReport:
    """Measures whether a formal contract actually rejects faulty mutations."""

    total_mutants: int
    killed_mutants: int
    survived_mutant_ids: tuple[str, ...]
    min_required_mks: float
    baseline_passed: bool = True
    verifier_errors: int = 0

    @property
    def mutation_kill_score(self) -> float:
        if self.total_mutants == 0 or not self.baseline_passed:
            return 0.0
        return self.killed_mutants / self.total_mutants

    @property
    def passed(self) -> bool:
        return (
            self.baseline_passed
            and self.verifier_errors == 0
            and self.total_mutants > 0
            and self.mutation_kill_score >= self.min_required_mks
        )


def run_static_ast_gate(source_code: str) -> StaticCheckResult:
    """Fast pre-sandbox AST hygiene filter (syntax, dangerous names/attrs, scope).

    NOT a security boundary: it only rejects obvious unsafe constructs cheaply
    before an expensive run. The isolation boundary is the Docker sandbox that
    actually executes the candidate code; never rely on this gate alone.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return StaticCheckResult(ok=False, syntax_error=f"Line {exc.lineno}: {exc.msg}")

    forbidden_direct = {"eval", "exec", "__import__", "compile"}
    forbidden_attrs = {"system", "popen", "exec", "eval", "__import__"}
    shell_runners = {"run", "Popen", "call", "check_call", "check_output"}
    forbidden_call_names = forbidden_direct | forbidden_attrs | shell_runners
    found_forbidden: list[str] = []
    defined: set[str] = set(dir(builtins))
    loaded: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                defined.add(bound)
                # A plain Call check only sees `os.system(...)`. Once the shell entry
                # point is rebound by an import (`from os import system`), the call site
                # is a bare Name and used to slip through this gate entirely.
                if isinstance(node, ast.ImportFrom) and bound in forbidden_call_names:
                    found_forbidden.append(f"from {node.module or '.'} import {bound}")
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            defined.add(node.rest)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            defined.update(node.names)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_direct:
                found_forbidden.append(node.func.id)
            elif (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in forbidden_direct
            ):
                found_forbidden.append(f"getattr:{node.args[1].value}")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_attrs:
                found_forbidden.append(node.func.attr)
            elif isinstance(node.func, ast.Attribute) and node.func.attr in shell_runners:
                for kw in node.keywords:
                    if (
                        kw.arg == "shell"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                    ):
                        found_forbidden.append(f"{node.func.attr}(shell=True)")

    undefined = tuple(sorted(name for name in loaded if name not in defined))
    ok = not found_forbidden and not undefined
    return StaticCheckResult(
        ok=ok,
        undefined_names=undefined,
        forbidden_calls=tuple(sorted(set(found_forbidden))),
    )


class ASTContractMutator:
    """Generates targeted semantic AST mutants to test Z3 contract strength (MKS)."""

    _COMPARE_SWAPS: dict[type[ast.cmpop], type[ast.cmpop]] = {
        ast.Lt: ast.GtE,
        ast.LtE: ast.Gt,
        ast.Gt: ast.LtE,
        ast.GtE: ast.Lt,
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
    }
    _BINOP_SWAPS: dict[type[ast.operator], type[ast.operator]] = {
        ast.Add: ast.Sub,
        ast.Sub: ast.Add,
        ast.Mult: ast.Div,
        ast.Div: ast.Mult,
    }
    _BOOLOP_SWAPS: dict[type[ast.boolop], type[ast.boolop]] = {
        ast.And: ast.Or,
        ast.Or: ast.And,
    }

    def generate_mutants(self, source_code: str, max_mutants: int = 16) -> list[CodeMutant]:
        """Produce deterministic, syntactically valid semantic mutants of source_code."""
        base_tree = ast.parse(source_code)
        sites: list[tuple[int, str, int]] = []

        for idx, node in enumerate(ast.walk(base_tree)):
            lineno = getattr(node, "lineno", 0)
            if (
                isinstance(node, ast.Compare)
                and node.ops
                and type(node.ops[0]) in self._COMPARE_SWAPS
            ):
                sites.append((idx, "compare_flip", lineno))
            elif isinstance(node, ast.BinOp) and type(node.op) in self._BINOP_SWAPS:
                sites.append((idx, "binop_flip", lineno))
            elif isinstance(node, ast.BoolOp) and type(node.op) in self._BOOLOP_SWAPS:
                sites.append((idx, "boolop_flip", lineno))
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)
            ):
                sites.append((idx, "const_off_by_one", lineno))

        mutants: list[CodeMutant] = []
        for target_idx, op_name, lineno in sites[:max_mutants]:
            cloned = copy.deepcopy(base_tree)
            for idx, node in enumerate(ast.walk(cloned)):
                if idx != target_idx:
                    continue
                if op_name == "compare_flip" and isinstance(node, ast.Compare):
                    node.ops[0] = self._COMPARE_SWAPS[type(node.ops[0])]()
                elif op_name == "binop_flip" and isinstance(node, ast.BinOp):
                    node.op = self._BINOP_SWAPS[type(node.op)]()
                elif op_name == "boolop_flip" and isinstance(node, ast.BoolOp):
                    node.op = self._BOOLOP_SWAPS[type(node.op)]()
                elif (
                    op_name == "const_off_by_one"
                    and isinstance(node, ast.Constant)
                    and isinstance(node.value, (int, float))
                ):
                    node.value = node.value + 1 if node.value == 0 else node.value - 1
                break
            ast.fix_missing_locations(cloned)
            mutated_src = ast.unparse(cloned)
            if mutated_src != source_code:
                mutants.append(
                    CodeMutant(
                        mutant_id=f"mut_{len(mutants) + 1:02d}_{op_name}",
                        operator=op_name,
                        lineno=lineno,
                        mutated_source=mutated_src,
                    )
                )
        return mutants

    def evaluate_contract_strength(
        self,
        source_code: str,
        verifier_fn: Callable[[str], bool],
        min_kill_score: float = 0.80,
        max_mutants: int = 16,
    ) -> ContractStrengthReport:
        """Verify baseline first, then run verifier_fn on every semantic mutant.

        Exceptions in verifier_fn are counted in ``verifier_errors`` and NEVER as kills.
        """
        mutants = self.generate_mutants(source_code, max_mutants=max_mutants)
        try:
            baseline_ok = bool(verifier_fn(source_code))
        except Exception:
            return ContractStrengthReport(
                total_mutants=len(mutants),
                killed_mutants=0,
                survived_mutant_ids=tuple(m.mutant_id for m in mutants),
                min_required_mks=min_kill_score,
                baseline_passed=False,
                verifier_errors=1,
            )

        if not baseline_ok:
            return ContractStrengthReport(
                total_mutants=len(mutants),
                killed_mutants=0,
                survived_mutant_ids=tuple(m.mutant_id for m in mutants),
                min_required_mks=min_kill_score,
                baseline_passed=False,
                verifier_errors=0,
            )

        killed = 0
        errors = 0
        survived: list[str] = []

        for mutant in mutants:
            try:
                mutant_accepted = bool(verifier_fn(mutant.mutated_source))
            except Exception:
                errors += 1
                survived.append(mutant.mutant_id)
                continue

            if not mutant_accepted:
                killed += 1
            else:
                survived.append(mutant.mutant_id)

        return ContractStrengthReport(
            total_mutants=len(mutants),
            killed_mutants=killed,
            survived_mutant_ids=tuple(survived),
            min_required_mks=min_kill_score,
            baseline_passed=True,
            verifier_errors=errors,
        )


class Z3CEGISVerifier:
    """Fail-closed Z3 SMT verifier with unsat_core() and counterexample extraction."""

    def __init__(self, timeout_ms: int = 2500, fail_closed: bool = True) -> None:
        self.timeout_ms = timeout_ms
        self.fail_closed = fail_closed

    def verify_named_constraints(
        self,
        constraints: dict[str, object],
        check_negated_safety: bool = False,
    ) -> CEGISFeedback:
        """Verify named Z3 boolean constraints and extract the unsat_core on UNSAT."""
        started = time.perf_counter()
        if not Z3_AVAILABLE or z3 is None:
            elapsed = (time.perf_counter() - started) * 1000.0
            if self.fail_closed:
                return CEGISFeedback(
                    verdict="FAIL_CLOSED",
                    elapsed_ms=elapsed,
                    reason="z3-solver is unavailable; fail-closed gate rejects unverified output.",
                )
            raise Z3UnavailableError("z3-solver is not installed")

        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)
        solver.set("unsat_core", True)

        for label, expr in constraints.items():
            solver.assert_and_track(expr, label)

        status = solver.check()
        elapsed = (time.perf_counter() - started) * 1000.0

        if status == z3.sat:
            model = solver.model()
            # Extract over the constraints' real free variables, not model.decls():
            # assert_and_track tracking literals share the requirement labels and
            # would shadow a same-named variable (yielding "True" instead of its value).
            variables: dict[str, Any] = {}
            for expr in constraints.values():
                for var in z3.z3util.get_vars(expr):
                    variables[var.decl().name()] = var
            cex = {
                name: str(model.eval(var, model_completion=True)) for name, var in variables.items()
            }
            if check_negated_safety:
                return CEGISFeedback(
                    verdict="UNSAT",
                    counterexample=cex,
                    elapsed_ms=elapsed,
                    reason="Counterexample found violating negated safety property.",
                )
            return CEGISFeedback(
                verdict="SAT",
                counterexample=cex,
                elapsed_ms=elapsed,
                reason="All formal constraints are satisfiable.",
            )

        if status == z3.unsat:
            core = tuple(str(c) for c in solver.unsat_core())
            if check_negated_safety:
                return CEGISFeedback(
                    verdict="SAT",
                    unsat_core=core,
                    elapsed_ms=elapsed,
                    reason="No counterexample exists (safety property proved Q.E.D.).",
                )
            return CEGISFeedback(
                verdict="UNSAT",
                unsat_core=core,
                elapsed_ms=elapsed,
                reason=f"Conflicting constraints in unsat_core: {core}",
            )

        return CEGISFeedback(
            verdict="UNKNOWN",
            elapsed_ms=elapsed,
            reason="Z3 solver returned unknown/timeout; treated as unverified (fail-closed).",
        )


class ProcessRewardModel(Protocol):
    """Scores an intermediate ChainOfThought / code candidate before sandbox execution."""

    def score_step(
        self,
        task: str,
        candidate_code: str,
        feedback_history: Sequence[str],
    ) -> float:
        """Return prior quality probability P_PRM in [0.0, 1.0]."""


class HeuristicLinearPRM:
    """Fast local baseline PRM (drop-in interface for fine-tuned ModernBERT/DeBERTa PRM)."""

    def score_step(
        self,
        task: str,
        candidate_code: str,
        feedback_history: Sequence[str],
    ) -> float:
        static_res = run_static_ast_gate(candidate_code)
        if not static_res.ok:
            return 0.05

        score = 0.45
        if "def " in candidate_code:
            score += 0.15
        if "raise " in candidate_code or "if " in candidate_code:
            score += 0.15
        if '"""' in candidate_code or "->" in candidate_code:
            score += 0.10
        if len(feedback_history) > 0:
            score += 0.08
        if any(tok in candidate_code for tok in ("TODO", "pass\n", "NotImplementedError")):
            score -= 0.25
        return max(0.01, min(0.99, score))


@dataclass
class MCTSNode:
    """Node in the Verifier-Guided ChainOfThought search tree."""

    node_id: str
    code: str
    prior_prm: float
    parent: MCTSNode | None = None
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    feedback: str = ""
    verified: bool = False
    mks: float = 0.0

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def puct_score(self, c_puct: float = 1.41) -> float:
        parent_visits = self.parent.visits if self.parent is not None else 1
        exploration = (
            c_puct * self.prior_prm * (math.sqrt(max(1, parent_visits)) / (1 + self.visits))
        )
        return self.q_value + exploration


@dataclass(frozen=True)
class MCTSSearchResult:
    """Final output of Verifier-Guided MCTS + CEGIS."""

    best_code: str
    verified: bool
    mutation_kill_score: float
    rollouts_used: int
    cegis_repairs: int
    trace: list[dict[str, object]]


class VerifierGuidedMCTS:
    """PUCT Tree Search guided by PRM priors, static AST hygiene, Z3 CEGIS, and MKS."""

    def __init__(
        self,
        prm: ProcessRewardModel | None = None,
        mutator: ASTContractMutator | None = None,
        c_puct: float = 1.41,
        min_mks: float = 0.80,
    ) -> None:
        self.prm = prm or HeuristicLinearPRM()
        self.mutator = mutator or ASTContractMutator()
        self.c_puct = c_puct
        self.min_mks = min_mks

    def search(
        self,
        task: str,
        initial_code: str,
        propose_fn: Callable[[str, str, str], list[str]],
        verify_fn: Callable[[str], tuple[bool, str]],
        max_rollouts: int = 8,
    ) -> MCTSSearchResult:
        """Explore candidate refinements via PUCT selection and CEGIS counterexample feedback."""
        root = MCTSNode(
            node_id="root",
            code=initial_code,
            prior_prm=self.prm.score_step(task, initial_code, ()),
        )
        best_node = root
        trace: list[dict[str, object]] = []
        cegis_repairs = 0

        for rollout in range(1, max_rollouts + 1):
            node = root
            while node.children:
                node = max(node.children, key=lambda child: child.puct_score(self.c_puct))

            static_res = run_static_ast_gate(node.code)
            if not static_res.ok:
                reward = 0.0
                node.feedback = (
                    f"AST_FAIL syntax={static_res.syntax_error} "
                    f"undef={static_res.undefined_names} forbidden={static_res.forbidden_calls}"
                )
            else:
                ok, fb = verify_fn(node.code)
                node.feedback = fb
                if ok:
                    mks_report = self.mutator.evaluate_contract_strength(
                        node.code,
                        lambda mutant_src: verify_fn(mutant_src)[0],
                        min_kill_score=self.min_mks,
                    )
                    node.mks = mks_report.mutation_kill_score
                    if mks_report.passed:
                        node.verified = True
                        reward = 0.70 + 0.30 * node.mks
                    else:
                        reward = 0.45 * node.mks
                        node.feedback = (
                            f"WEAK_CONTRACT MKS={node.mks:.2f} < {self.min_mks:.2f}; "
                            f"survived={mks_report.survived_mutant_ids}"
                        )
                else:
                    reward = 0.20 * node.prior_prm
                    cegis_repairs += 1

            trace.append(
                {
                    "rollout": rollout,
                    "node_id": node.node_id,
                    "prior_prm": round(node.prior_prm, 4),
                    "reward": round(reward, 4),
                    "verified": node.verified,
                    "mks": round(node.mks, 4),
                    "feedback": node.feedback,
                }
            )

            cursor: MCTSNode | None = node
            while cursor is not None:
                cursor.visits += 1
                cursor.value_sum += reward
                cursor = cursor.parent

            if (node.verified and not best_node.verified) or (
                node.verified == best_node.verified and node.q_value > best_node.q_value
            ):
                best_node = node

            if node.verified:
                break

            children_src = propose_fn(task, node.code, node.feedback)
            for idx, child_code in enumerate(children_src):
                prior = self.prm.score_step(task, child_code, (node.feedback,))
                node.children.append(
                    MCTSNode(
                        node_id=f"{node.node_id}.{rollout}_{idx}",
                        code=child_code,
                        prior_prm=prior,
                        parent=node,
                    )
                )

        return MCTSSearchResult(
            best_code=best_node.code,
            verified=best_node.verified,
            mutation_kill_score=best_node.mks,
            rollouts_used=len(trace),
            cegis_repairs=cegis_repairs,
            trace=trace,
        )
