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

## 2. ML/AI-апгрейды v2: статус
Пункты 1–4 слиты в `main` бандлом `bcfe489` (PR #37), CI зелёный на Python 3.12–3.14, ubuntu и windows.
Провязка в существующих ядрах уже сделана (формулировка «no wiring yet» в описании squash-коммита
описывает более раннее состояние ветки):
`experiments/gate.py:32,379` — `JudgeCalibrator.cross_validate`; `neurosymbolic/symbolic.py:29,355` —
`CEGISFeedback` с `unsat_core` и контрпримером; `routing/model_router.py:21,129` — `LinUCBModelRouter`;
`memory/store.py:28,299,355` — `HybridCognitiveMemory` поверх трёх уровней, `lookup_similar_failures` вызывает
`autonomy/fixer.py:258`.
Из этого по умолчанию включено не всё: маршрутизация остаётся `static`, bandit — только с
`NOEMA_LLM__ROUTER=linucb`; `search_hybrid` доступен как API хранилища, основную цепочку поиска он не замещает.
Пункт 5 (`docs/noema.html`) в main отдельно от этого бандла и остаётся в режиме `DEMO SIM`.
Фазовые формулировки README (Фаза 2 и Фаза 3 — в активной разработке) этот бандл не меняет.

### 2.1 Состав ML/AI-апгрейдов
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
   - Извлечение `unsat_core()` и контрпримера в Z3 (fail-closed). Ядро возвращается тем, что даёт
     `solver.unsat_core()`; отдельной минимизации поверх кода Z3 нет, поэтому гарантию «минимальности»
     в README и docs не формулируем.
   - Честный `ASTContractMutator`: сначала проверяет baseline, раздельно учитывает `verifier_errors` и никогда не засчитывает падение верификатора как убийство мутанта.
   - PUCT MCTS с подключаемой Process Reward Model (PRM).
5. **Standalone 3D Neural Connectome (`docs/noema.html`)**:
   - Интерактивная визуализация 95 000 нейронов и 135 000 синапсов в честном режиме `DEMO SIM` (без неавторизованного мутирующего API и без фейкового `VERIFIED`).
