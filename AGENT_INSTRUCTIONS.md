# AGENT INSTRUCTIONS (v2 — Post-Audit CI-Clean Bundle) — ssrjkk/noema

This archive contains the audited, CI-clean (`ruff`, `mypy`, `bandit`, `pytest`, UTF-8 LF)
ML/AI upgrade modules and standalone 3D Connectome showcase page for `ssrjkk/noema`.

Insecure/demo wrappers from v1 (such as unauthenticated mutating `atlas_dashboard.py`
and pattern-matching fake verifiers) have been intentionally REMOVED.

## Files Included (9)
- `AGENT_INSTRUCTIONS.md` — this execution plan
- `docs/noema_ml_ai_upgrade_plan.md` — full ML/AI engineering roadmap
- `docs/noema.html` — standalone static 3D neural connectome showcase (for GitHub Pages, NOT mounted as an unauthenticated mutating API route)
- `noema/training/__init__.py`
- `noema/training/rlvr_pipeline.py` — DPO/GRPO dataset builder + Platt `JudgeCalibrator` with honest out-of-sample `cross_validate()`
- `noema/memory/hybrid_retriever.py` — BM25 + Dense + RRF + bi-temporal `supersede(strict=True)` + non-destructive `min_retention=0.0` + `FailurePattern` index
- `noema/llm/router_bandit.py` — thread-safe (`RLock`) LinUCB Contextual Bandit with strict fail-closed `BudgetExceededError` and validated `load()`
- `noema/neurosymbolic/cegis_mutator.py` — fail-closed `Z3CEGISVerifier` (`unsat_core`), hardened `run_static_ast_gate`, `ASTContractMutator` (checks baseline & never counts verifier exceptions as kills), `VerifierGuidedMCTS`
- `tests/test_ml_upgrades.py` — comprehensive `pytest` suite covering all invariants and regression edge cases

## Recommended 2-PR Integration Plan (Zero Parallel Stacks)

### PR 1: Training Calibration + Hybrid Bi-Temporal Memory (Immediate)
1. **Wire `noema/training/rlvr_pipeline.py` into `noema/experiments/gate.py`:**
   - Keep full backward compatibility when no calibrator is configured.
   - Add an optional `JudgeCalibrator` hook before comparing against `judge_threshold`.
   - Always report out-of-sample `cross_validate()` or holdout Brier/ECE metrics (never in-sample only).
2. **Wire `noema/memory/hybrid_retriever.py` over existing `noema/memory/store.py`:**
   - Avoid a disconnected parallel memory stack: reuse `store.py` persistence/records and layer BM25 + RRF + bi-temporal `supersede(strict=True)` + `FailurePattern` lookup on top.
   - Keep default `min_retention=0.0` so Ebbinghaus decay adjusts ranking without silently erasing older valid semantic facts.
3. **Verify CI gates:**
   ```bash
   python -m pytest tests/test_ml_upgrades.py -v
   ruff check noema/training/ noema/memory/hybrid_retriever.py tests/test_ml_upgrades.py
   ruff format --check noema/training/ noema/memory/hybrid_retriever.py tests/test_ml_upgrades.py
   python -m mypy noema/training/ noema/memory/hybrid_retriever.py
   python -m bandit -r noema/training/ noema/memory/hybrid_retriever.py
   ```

### PR 2: LinUCB Routing Policy + Z3 CEGIS/MKS (Follow-up)
1. **Integrate `LinUCBModelRouter` inside existing `noema/routing/model_router.py`** (single unified router used by `core/engine.py` and `core/chain_of_thought.py`), preserving fail-closed `BudgetExceededError`.
2. **Integrate `Z3CEGISVerifier` and `ASTContractMutator` into existing `noema/neurosymbolic/symbolic.py`** and add a CI job installing `pip install -e ".[dev,full]"` so Z3 paths execute in CI.

### Separate Static Frontend (Optional)
- `docs/noema.html` is a standalone interactive 3D showcase page (`DEMO SIM` mode by default, bounded generation loops, `prefers-reduced-motion`, `aria-*`). Deploy only as static documentation/Pages; do NOT expose unauthenticated mutating POST routes in FastAPI.
