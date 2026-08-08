# Noema

Фреймворк генерации мощных технических решений на любом стеке.

## Quick Start

```bash
pip install -e ".[dev]"

# CLI
noema think "Real-time Chat App" --tags "python,websocket,redis" --complexity complex

# API
noema serve
# POST http://localhost:8000/think

# Tests
pytest tests/
```

## Experiments and Benchmarks

Reproducible benchmark runner: runs the same tasks across providers/models and
collects wall time, tokens, judge score, cost estimate and optional sandbox
validation into `results/` (JSON per run + CSV summaries).

```bash
# Run the demo experiment (uses the built-in fallback provider, no API keys)
python -m noema.experiments.runner experiments/experiments.yaml --out results

# Results
#   results/<experiment>/<run_id>/results.json   — one record per (task, provider, model, repetition)
#   results/<experiment>/<run_id>/runs.csv       — same, as CSV
#   results/<experiment>/<run_id>/summary.csv    — per (provider, model) aggregates
```

Benchmarking real models: list the provider in `experiments/experiments.yaml`
and export the matching key (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`). Each
model may define `cost_per_token` for the cost estimate column.

Running in CI: `.github/workflows/experiments.yml` executes the smoke benchmark
nightly (or on `workflow_dispatch`) with the mock provider and uploads the
results as artifacts. Experiments are forward-compatible: the `settings`
section accepts `neurosymbolic` (wired), plus `retrieval` and `sandbox` knobs
that later phases connect to the retrieval and sandbox pipelines.

