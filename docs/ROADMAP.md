# Noema Roadmap

> From a reasoning framework to an autonomous engineering mind.
>
> Status legend: `[x]` done · `[~]` partial · `[ ]` not started.
> File references point at the real modules so every task has a concrete entry point.

---

## Phase 1 — The Architect

Noema works as a CLI/API for engineers: `noema think "Real-time Chat App"` produces a full
solution, benchmarked through the reproducible runner.

### Already in place
- [x] CLI: `think`, `pipeline`, `graph`, `knowledge`, `feedback` — `noema/cli/main.py`
- [x] API server (FastAPI): `/v1/think`, streaming, `/enqueue`, health/readiness — `noema/api/server.py`
- [x] Async queue worker over Redis (arq): `noema worker` / `enqueue` — `noema/cli/arq.py`, `noema/workers/arq_worker.py`
- [x] gRPC server + client — `noema/grpc/server.py`, `noema/grpc/client.py`, `protos/`
- [x] Isolated Docker sandbox (network disabled, resource caps) — `noema/sandbox/engine.py`
- [x] NeuroSymbolic verification fails closed (no silent pass without Z3) — `noema/neurosymbolic/symbolic.py`
- [x] Reproducible benchmark runner: per-cell timeout, incremental `runs.jsonl`, isolated per-run
      workspace, full `results.json` + `summary.json`/`summary.csv` — `noema/experiments/runner.py`
- [x] Economy of computation: token tracing (`noema/tracing/tracer.py`) + cost estimation
      (`noema/billing/cost_tracker.py`)
- [x] Vault integration, observability (Prometheus metrics), audit trail — `noema/vault`,
      `noema/observability`, `noema/audit`
- [x] MkDocs site with API reference — `mkdocs.yml`

### Backlog
- [x] **T1.1 Sandbox quotas from settings.** `max_memory`, `max_cpus`, `max_cpu_seconds` now flow
      from `SandboxConfig` into the Docker flags (`--memory`/`--cpus`) and the direct-run
      rlimit `preexec` (RLIMIT_AS/CPU); rlimit errors are tolerated so a failed limit never
      kills the run. Done when: exceeding `max_memory` is killed and recorded in the run record.
      Verified by `tests/test_refactor_sandbox.py` (flags + preexec binding).
- [x] **T1.2 Static validation pre-sandbox.** Pure-AST pass (`noema/sandbox/static_check.py`):
      syntax, import hygiene (wildcard/relative/not-in-sandbox imports) and scope-aware
      undefined-name call-graph analysis. Wired into `validate_code_block`/`validate_files`
      (`static_check_enabled` config); a failing file short-circuits the run stage so the
      verdict comes from the static pass, not the sandbox. Done when: a sample of broken
      outputs is rejected by the static pass, not the sandbox.
      Verified by `tests/test_refactor_sandbox.py` (rejection + no false positives).
- [x] **T1.3 Requirement → symbolic contract extraction.** `_extract_bounds`
      (`noema/neurosymbolic/symbolic.py`) now covers phrases ("at least"/"at most"/"must stay
      below/above"/"must not exceed"/"exactly"/"no more|less than"/min/max/below/above/up to),
      percentages and time units (ms/us scale to seconds), and scans description/requirement/
      text/condition/statement fields — not just `constraints`. Requirements also render a
      human-readable description (`"x in [0, 100]"`) instead of an empty string.
      Done when: a corpus of requirement sentences yields a non-vacuous contract for ≥ 90% of
      them. Verified by `tests/test_refactor_bounds.py` (30-sentence corpus at ≥90%, exact
      bound semantics, narrowing, unit scaling).
- [x] **T1.4 Domain-module knowledge seeding.** New `noema/knowledge/domains.py` seeds one
      English knowledge entry per built-in domain module (keyed by the registry name, domain
      vocabulary + concrete patterns + the module name as first tag), appended to
      `BUILTIN_KNOWLEDGE` in `noema/knowledge/store.py`. Also fixed two corrupt/garbled
      best-practice entries (12-Factor, DB optimization) to clean English.
      Done when: `noema knowledge query` returns relevant hits for each domain without tuning.
      Verified by `tests/test_knowledge_domains.py` (22/22 modules seeded; every domain query
      surfaces its own entry on top — 100% ≥ 90%; bare module-name queries; score sanity) and
      end-to-end via `noema knowledge search`.

---

## Phase 2 — The Autopoietic Enterprise

Noema integrates with CI/CD and starts fixing itself: incidents become PRs, PRs are validated
in the sandbox, and only provably-good changes merge.

- [x] **T2.1 Incident → PR loop.** New `noema/autonomy/` package: `incidents.py` normalizes a
      Sentry alert or webhook into an `Incident`; `github.py` is a minimal httpx REST client
      (branch refs, contents API, pulls — transport-injectable for tests, no PyGithub dep);
      `fixer.py` `IncidentFixer` runs the fix task through `NoemaEngine`, gates the PR on a
      **passing run** (`validate_solution(run_tests=True)` — no PR without `all_valid`), and
      submits the fix files as a branch + PR. `POST /webhooks/incident` (`noema/api/webhooks.py`)
      consumes the alert and enqueues a `fix_incident_task` on the arq worker, falling back to
      inline execution when Redis is unavailable. Config via `autonomy` settings
      (`NOEMA_AUTONOMY__GITHUB_TOKEN` / `github_repo` / `github_base_branch`).
      Done when: a synthetic crash produces a branch + PR with the fix and a passing run.
      Verified by `tests/test_autonomy.py` (Sentry/generic/webhook parsing, REST payloads via
      mock transport, fixer flow, validation gate blocks the PR, no-changes path).
- [x] **T2.2 Merge gate by judge score.** Wire the existing judge + sandbox into a CI job that
      blocks merges when `judge_score` < threshold or the sandbox fails.
      Done when: the gate is part of `.github/workflows/ci.yml` and documented.
- [x] **T2.3 Evolution auto-apply.** Honor `evolution_enabled` + `evolution_test_before_apply`
      (`noema/config/settings.py`) so a mutation only lands when its tests pass.
      Done when: no mutation is auto-applied without a green test run, enforced by a test.
- [x] **T2.4 Benchmark as a service.** Expose the runner matrix (`noema/experiments/runner.py`)
      behind an API endpoint producing the existing `results.json`/`summary.json` schema.
      `noema/api/experiments.py`: `POST /experiments` accepts the experiment config as
      YAML/JSON, runs the matrix off-loop (`asyncio.to_thread`, runner serializes env
      mutation with a thread lock) and lands artifacts in `results/`; `GET /experiments`,
      `GET /experiments/{id}/runs`, `GET /experiments/{id}/runs/{run_id}` serve them.
      Done when: `POST /experiments` returns a run id and the artifacts land in `results/`.
      Verified by `tests/test_experiments_api.py` (real fallback-provider run, artifacts
      on disk, list/get endpoints, config rejection, path-traversal guard).
- [x] **T2.5 Exact cost per artifact line.** `attribute_file_costs`
      (`noema/experiments/runner.py`) distributes each run's measured
      `tokens_input`/`tokens_output` across the solution's code blocks
      weighted by generated line count (the same line-weighted model as PR
      cost attribution); the last file absorbs rounding drift so per-file
      rows sum back exactly to the run totals. `RunRecord.file_costs` flows
      into `results.json` (`runs.jsonl` keeps the incremental copy);
      `runs.csv` stays a flat projection without the nested rows.
      Done when: `results.json` includes per-file token/cost breakdown.
      Verified by `tests/test_file_costs.py` (line-weighted split, exact
      sums, empty inputs, on-disk artifacts) and the schema test.

---

## Phase 3 — Global Noema Grid

Noema becomes a decentralized network: heavy reasoning is delegated across nodes, coordinated
by a shared Redis queue and a gRPC mesh.

- [x] **T3.1 Multi-node worker pool.** `noema arq worker` publishes liveness
      to Redis (`NodeHeartbeat` in `noema/workers/arq_worker.py`) under an
      auto-expiring key with `draining` and `metrics_port` fields; graceful
      drain marks the node draining and removes the key on shutdown, and a
      dead node disappears when its TTL lapses. `noema arq workers` lists
      the live fleet from the registry.
      Done when: a stress run distributes 100 tasks across 3 nodes with zero
      double-execution. Verified by `tests/test_grid_workers.py` (100 tasks
      over 3 heterogeneous nodes: no double-claims, no loss, even split,
      heartbeats drained) and `tests/test_arq_worker.py`.
- [x] **T3.2 Federation protocol.** New `noema/federation/` package:
      `FederationRouter.execute` splits sub-tasks, delegates each to the next
      healthy peer over the existing gRPC Think RPC, and re-joins results
      positionally. Per-peer `ResilientExecutor` (circuit breaker +
      exponential-backoff retries; an open circuit fails fast) with a hard
      request timeout; when no healthy peer remains — or a delegation
      ultimately fails — the sub-task falls back to the local executor
      instead of failing the hierarchy. Every delegation lands in the
      contribution ledger. Config: `NOEMA_FEDERATION__*` (peers, timeouts,
      thresholds); CLI: `noema grid federate`.
      Done when: a hierarchy of sub-tasks is split across two nodes and
      re-joined correctly. Verified by `tests/test_federation.py`
      (round-robin split, retry-then-success, local fallback, circuit opens
      at threshold and routes around the dead peer, positional re-join,
      ledger entries).
- [x] **T3.3 Token/ledger economy.** `noema/billing/ledger.py`
      `ContributionLedger`: append-only JSONL entries (node, task, kind,
      model, in/out tokens, cost, peer, artifact) with bounded memory;
      `per_node()`/`entries_for()`/`audit()` answer "who generated what
      value". The arq worker records every completed `think` job into its
      ledger (`NOEMA_WORKER_LEDGER` path), and federated delegations are
      recorded by the router; `noema arq ledger` audits a node's file.
      Done when: a run produces an auditable ledger of who (which
      node/task) generated what value. Verified by `tests/test_ledger.py`
      (aggregation, task trails, JSONL round-trip, corrupt-line tolerance,
      bounded memory, write failures never break the run).
- [x] **T3.4 Grid dashboard.** `noema/observability/grid.py`
      `GridDashboard` joins the Redis worker heartbeats with each node's
      Prometheus `/metrics` endpoint (advertised `metrics_port`) into a
      per-node latency/token/error view plus cluster totals; exposition
      parsing degrades gracefully without `prometheus_client`, and an
      unreachable node is reported, never raised. Exposed as
      `GET /grid` (`noema/api/grid.py`) and `noema grid status` (CLI);
      `noema arq workers` renders the live fleet.
      Done when: a dashboard renders live grid health from the existing
      metric endpoints. Verified by `tests/test_grid_dashboard.py`
      (aggregation, unreachable nodes, dead-node expiry, snapshot stream)
      and `tests/test_grid_api.py` (endpoint shape, never-500 contract).

---

## Cross-cutting — the NeuroSymbolic track

- [x] **T4.1 AST verification.** New `noema/neurosymbolic/static.py` bridges the pure-AST checks
      (syntax, import hygiene, undefined-name call-graph analysis from `noema/sandbox/static_check.py`)
      into the symbolic pipeline: `NeuroSymbolicEngine.think` analyzes every Python snippet inside
      the hypothesis (recursive code extraction; bare values are skipped) *before* Z3 verification,
      and reports `static_analyzed`/`static_passed`/`static_issues` on the verification events plus a
      `static_verdict` on the terminal event — the static verdict always accompanies the Z3 verdict.
      Done when: the NeuroSymbolic pipeline reports a static-analysis verdict alongside the Z3 verdict.
      Verified by `tests/test_static_verdict.py` (verdict extraction, alongside-Z3 reporting,
      refinement flips the verdict).
- [x] **T4.2 Verifiable reasoning traces.** New `noema/tracing/reasoning_trace.py` commits every
      `think` run's reasoning checkpoints + verification results as a self-contained JSON artifact
      (task input, one `VerificationRound` per refine-verify attempt with the verified hypothesis,
      the AST static verdict and the symbolic Z3 verdict, plus the terminal outcome).
      `NeuroSymbolicEngine` accepts `trace_dir` and commits on `completed`/`failed`/`error` (atomic
      write, I/O off the event loop; disabled by default). `reverify_reasoning_trace`/`reverify_trace_file`
      replay the deterministic half of the pipeline (AST + Z3, never the LLM) on a previous run's
      artifact and report `matches`. Done when: a previous run's artifacts fully reproduce its
      verdict without re-running the LLM. Verified by `tests/test_reasoning_trace.py` (round-trip,
      verdict reproduction, drift detection, engine commit on all terminal outcomes).
- [x] **T4.3 Whitepaper.** `docs/WHITEPAPER.md` states the project vision and design
      principles (verify-don't-trust, structure-before-execution, re-auditable verdicts,
      zero-trust inputs, economy of computation, self-improvement under proof) and maps the
      roadmap phases (Architect → Autopoietic Enterprise → Global Noema Grid) onto them.
      Done when: the whitepaper exists and is linked from the docs nav and README. Linked as
      `Whitepaper` in `mkdocs.yml` nav and from `README.md`.

---

## Tracking

Each item is a checklist entry; completed items are ticked in the doc. To keep the roadmap in
sync with code, prefer merging a task only when its "Done when" criteria hold.

New ideas from the community/owners go at the bottom of the matching phase with `[ ]` and a
one-line rationale.
