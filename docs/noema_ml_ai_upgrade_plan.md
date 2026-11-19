# NOEMA -> SOTA Neuro-Symbolic ML & AI System
### Инженерный аудит `ssrjkk/noema` и план апгрейда в сильный ML/AI-проект (v2)

## 1. Архитектурный фокус
У `ssrjkk/noema` уже есть автоматический генератор верифицируемых наград (Verifiable Rewards) и готовые рабочие ядра:
- `noema/memory/store.py` (трёхуровневая память с HNSW),
- `noema/experiments/gate.py` (`judge_threshold` merge gate),
- `noema/routing/model_router.py` (маршрутизация для `core/engine.py` и `chain_of_thought.py`),
- `noema/neurosymbolic/symbolic.py` (Z3 fail-closed),
- `Docker Sandbox` + `tracing/`.

Все новые ML/AI-компоненты должны встраиваться **в эти существующие ядра**, а не создавать параллельные стеки.

## 2. Внедряемые ML/AI-апгрейды
1. **RLVR Pipeline & Honest Judge Calibration (`noema/training/rlvr_pipeline.py` -> `experiments/gate.py`)**:
   - Экспорт DPO пар и GRPO групп с нормализацией advantage на верифицируемых наградах (AST + Z3*MKS + Docker - Cost).
   - Platt-калибровка `judge_score` с честной out-of-sample k-fold кросс-валидацией (`JudgeCalibrator.cross_validate()`).
2. **Hybrid Bi-Temporal Memory & Failure Index (`noema/memory/hybrid_retriever.py` -> `memory/store.py`)**:
   - BM25 + Dense + Reciprocal Rank Fusion (RRF) + опциональный Cross-Encoder Reranker.
   - Строгий би-темпоральный `supersede(..., strict=True)` и неразрушающее ранжирование Эббингауза (`min_retention=0.0`).
   - Failure-Indexed Memory: поиск проверенных патчей по стектрейсу и Z3 `unsat_core`.
3. **Contextual LinUCB Bandit Router (`noema/llm/router_bandit.py` -> `routing/model_router.py`)**:
   - Онлайн Ridge-регрессия с обновлением Шермана-Моррисона, потокобезопасностью (`RLock`), строгим `fail-closed` бюджетом (`BudgetExceededError`) и валидацией форм/конечности матриц в `load()`.
4. **CEGIS + AST Contract Mutation Testing + Verifier-Guided MCTS (`noema/neurosymbolic/cegis_mutator.py` -> `neurosymbolic/symbolic.py`)**:
   - Извлечение минимального `unsat_core()` и контрпримера в Z3 (fail-closed).
   - Честный `ASTContractMutator`: сначала проверяет baseline, раздельно учитывает `verifier_errors` и никогда не засчитывает падение верификатора как убийство мутанта.
   - PUCT MCTS с подключаемой Process Reward Model (PRM).
5. **Standalone 3D Neural Connectome (`docs/noema.html`)**:
   - Интерактивная визуализация 95 000 нейронов и 135 000 синапсов в честном режиме `DEMO SIM` (без неавторизованного мутирующего API и без фейкового `VERIFIED`).
