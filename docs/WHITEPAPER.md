# Noema Whitepaper

> The design principles behind Noema — from a reasoning framework to an
> autonomous engineering mind.
>
> Status: living document. Sections mirror the [Roadmap](ROADMAP.md) phases so
> the whitepaper always states where Noema is and where it is going.

---

## 1. Vision

Large-language-model reasoning systems are trapped between two failure modes:
purely neural pipelines that are fluent but unverifiable, and purely symbolic
systems that are precise but brittle. Noema's thesis is that the two must not
be blended ad hoc — they must be **composed under a strict contract**: the
neural side proposes, the symbolic side disposes, and the record of that
exchange must be auditable.

We call the result an *engineering mind*: a system that not only generates
technical solutions but can justify them, defend them against formal
verification, and — over time — repair its own codebase without human
supervision.

## 2. Principles

### 2.1 Verify, don't trust

Every claim a neural model makes about a solution is a hypothesis until a
deterministic check confirms it. Noema fails **closed**: if formal
verification (Z3) is unavailable, the engine refuses to mark a solution valid
rather than silently passing everything (`noema/neurosymbolic/symbolic.py`).
There is no path from "unverified" to "accepted".

### 2.2 Structure before execution

Code is analyzed as a tree before it is run. A pure-AST pass
(`noema/sandbox/static_check.py`) rejects broken syntax, policy-violating
imports, and undefined names *before* any sandbox launch, so a verdict can be
produced without executing untrusted code at all. The same structural analysis
is embedded in the reasoning pipeline itself: every Python snippet inside a
hypothesis is statically judged alongside the symbolic verdict
(`noema/neurosymbolic/static.py`).

### 2.3 Every verdict must be re-auditable

Reasoning is only trustworthy if it can be replayed. Noema commits each run's
checkpoints and verification results as self-contained artifacts
(`noema/tracing/reasoning_trace.py`); a later re-audit reproduces the verdict
from the artifact using only deterministic analysis — no LLM call is ever
needed to re-check an old decision.

### 2.4 Zero-trust inputs

Hostile or malformed input is rejected at the boundary with strict schema
validation, quota enforcement (task size, tag count), and per-tenant
isolation. Sandboxed execution enforces resource caps (memory, CPU, time) and
denies network access.

### 2.5 Economy of computation

Token spend is a first-class cost. Every LLM call is traced, attributed to a
tenant, task and step, converted into a monetary estimate, and bounded by
token budgets and circuit breakers.

### 2.6 The system improves itself — under proof

Self-evolution is allowed only when it can be validated: prompt candidates run
in shadow mode and are promoted on statistical evidence; code mutations land
only when their tests pass (`evolution_enabled`,
`evolution_test_before_apply`). The engineering mind never merges unproven
change.

## 3. Architecture in one view

```
            ┌──────────────────────────────────────────────────┐
            │                   NoemaEngine                     │
            │   ChainOfThought (DAG)  ·  NeuroSymbolicEngine     │
            └───────┬───────────────────────┬───────────────────┘
                    │ propose               │ verify
        ┌───────────▼──────────┐   ┌────────▼────────────────────────┐
        │ NeuralInterface (LLM)│   │ SymbolicEngine (Z3) · static.py │
        └───────────┬──────────┘   └────────┬────────────────────────┘
                    │ hypothesis            │ verdict (fail-closed)
                    └───────► refine loop ◄─┘
                        ┌─────────────┼──────────────┐
                        │ trace (T4.2) │ sandbox       │ memory / knowledge
                        │ replay       │ static+run    │ (episodic, domain)
```

The reasoning loop is bounded by `max_refinement_attempts`: parse the task
into a symbolic graph, generate a hypothesis, verify it against the graph,
refine on violation, stop on success or exhaustion.

## 4. The verification contract

1. **Parsing** — the task is coerced through a strict input model; unknown keys
   and wrong types are rejected, priorities and bounds are normalized.
2. **Structure** — code snippets in the hypothesis get an AST verdict (syntax,
   imports, undefined names) reported *alongside* the formal verdict.
3. **Formal** — the candidate is checked against the accumulated symbolic
   requirements with a bounded-time Z3 solve. `unsat` → violation; no matched
   variables → violation; degraded solver → violation.
4. **Evidence** — the whole round (hypothesis, static verdict, symbolic
   verdict, violations) is committed to the run artifact.

## 5. From architect to autopoietic enterprise

- **Phase 1 — The Architect (current).** Noema as a CLI/API tool: generates
  solutions, validates them in a sandbox, and benchmarks them reproducibly.
  Sandbox quotas, pre-sandbox static validation, symbolic contract extraction,
  domain knowledge seeding, AST-in-pipeline verification and verifiable
  reasoning traces are complete.
- **Phase 2 — The Autopoietic Enterprise.** Noema consumes incidents and opens
  pull requests; merges are gated by judge score and sandbox results; mutations
  apply only when tests pass; the benchmark becomes a service; cost is traced
  per artifact line.
- **Phase 3 — The Global Noema Grid.** Reasoning is delegated across a
  multi-node pool coordinated by a shared queue and a gRPC mesh, with a
  token/ledger economy and a live grid dashboard.

## 6. Why auditable neurosymbolic composition matters

Formal verification alone cannot design a system; language models alone cannot
prove one. By making the neural proposer and the symbolic verifier peer
components of one loop — and by making every round's verdict replayable —
Noema converts generation from a one-shot gamble into an engineering process
with checkpoints, gates, and an audit trail. That is the difference between a
chatbot that writes code and an engineering mind that takes responsibility for
it.

---

*See also: [Roadmap](ROADMAP.md) · [Architecture](architecture.md) · [Changelog](changelog.md)*
