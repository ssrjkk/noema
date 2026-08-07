"""Performance benchmarks for noema modules."""

import asyncio
import contextlib
import tempfile

import pytest

from noema.core.checkpoint import CheckpointStore, DAGCheckpoint
from noema.core.types import ArchitecturePattern, Solution, TechStack
from noema.judge import evaluate_solution
from noema.llm.providers import FallbackProvider
from noema.neurosymbolic import (
    CircuitBreaker,
    CircuitOpenError,
    Constraint,
    NeuralInterface,
    NeuroSymbolicEngine,
    SymbolicEngine,
    TaskGraph,
)

try:
    import z3  # noqa: F401

    has_z3 = True
except ImportError:
    has_z3 = False


async def _dummy_async():
    return "ok"


# ── 1. BenchmarkNeuralInterface ────────────────────────────────────────


class BenchmarkNeuralInterface:
    @pytest.mark.benchmark(min_rounds=100, warmup=True)
    def bm_neural_backoff(self, benchmark):
        ni = NeuralInterface(base_delay=1.0, max_delay=10.0)

        def _run():
            for i in range(10000):
                ni._calculate_backoff(i % 10)

        benchmark(_run)

    @pytest.mark.benchmark(min_rounds=100, warmup=True)
    def bm_circuit_breaker_transitions(self, benchmark):
        async def _run():
            cb = CircuitBreaker(failure_threshold=5)
            for _ in range(1000):
                with contextlib.suppress(CircuitOpenError):
                    await cb.call(_dummy_async)

        benchmark(lambda: asyncio.run(_run()))


# ── 2. BenchmarkCheckpointStore ────────────────────────────────────────


class BenchmarkCheckpointStore:
    @pytest.mark.benchmark(min_rounds=100, warmup=True)
    def bm_checkpoint_save(self, benchmark):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CheckpointStore(persist_dir=tmpdir)
            cp = DAGCheckpoint(
                task_id="bench_task",
                tenant_id="bench_tenant",
                completed_steps=[f"step_{i}" for i in range(100)],
                step_results={f"step_{i}": f"result_{i}" for i in range(100)},
            )

            async def _save():
                await store.save(cp)

            benchmark(lambda: asyncio.run(_save()))

    @pytest.mark.benchmark(min_rounds=100, warmup=True)
    def bm_checkpoint_load(self, benchmark):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CheckpointStore(persist_dir=tmpdir)
            cp = DAGCheckpoint(
                task_id="bench_task",
                tenant_id="bench_tenant",
                completed_steps=[f"step_{i}" for i in range(100)],
                step_results={f"step_{i}": f"result_{i}" for i in range(100)},
            )

            async def _save():
                await store.save(cp)

            asyncio.run(_save())

            async def _load():
                await store.load("bench_task", "bench_tenant")

            benchmark(lambda: asyncio.run(_load()))

    @pytest.mark.benchmark(min_rounds=100, warmup=True)
    def bm_checkpoint_roundtrip(self, benchmark):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CheckpointStore(persist_dir=tmpdir)
            cp = DAGCheckpoint(
                task_id="bench_task",
                tenant_id="bench_tenant",
                completed_steps=[f"step_{i}" for i in range(100)],
                step_results={f"step_{i}": f"result_{i}" for i in range(100)},
            )

            async def _roundtrip():
                await store.save(cp)
                await store.load("bench_task", "bench_tenant")

            benchmark(lambda: asyncio.run(_roundtrip()))


# ── 3. BenchmarkJudge ──────────────────────────────────────────────────


class BenchmarkJudge:
    @pytest.mark.benchmark(min_rounds=100, warmup=True)
    def bm_judge_fallback(self, benchmark):
        llm = FallbackProvider()
        solution = Solution(
            task_id="bench_task",
            title="Test Solution",
            summary="A simple test solution for benchmarking",
            architecture=ArchitecturePattern(
                name="microservices",
                description="Microservices architecture pattern",
            ),
            stack=TechStack(
                languages=["python"],
                frameworks=["fastapi"],
            ),
        )

        async def _run():
            for _ in range(50):
                await evaluate_solution(
                    llm=llm,
                    solution=solution,
                    task_description="Build a simple API",
                    task_tags=["api", "python"],
                )

        benchmark(lambda: asyncio.run(_run()))


# ── 4. BenchmarkSymbolic ───────────────────────────────────────────────


@pytest.mark.skipif(not has_z3, reason="z3 not installed")
class BenchmarkSymbolic:
    @pytest.mark.benchmark(min_rounds=100, warmup=True)
    def bm_parse_task(self, benchmark):
        engine = SymbolicEngine()

        async def _run():
            task = {
                "requirements": [
                    {
                        "name": f"req_{i}",
                        "type": "numeric",
                        "min": 0,
                        "max": 100,
                        "priority": 1,
                    }
                    for i in range(100)
                ],
                "constraints": [],
                "goals": ["goal_1"],
            }
            await engine.parse_task(task)

        benchmark(lambda: asyncio.run(_run()))

    @pytest.mark.benchmark(min_rounds=100, warmup=True)
    def bm_verify_solution(self, benchmark):
        from z3 import And, Int

        engine = SymbolicEngine()

        async def _run():
            variables = {}
            constraints = []
            for i in range(50):
                var = Int(f"x_{i}")
                variables[f"x_{i}"] = var
                constraints.append(
                    Constraint(
                        name=f"x_{i}",
                        expression=And(var >= 0, var <= 100),
                        description=f"x_{i} in [0, 100]",
                    )
                )
            tg = TaskGraph(requirements=constraints, variables=variables)
            await engine.verify_solution({f"x_{i}": 50 for i in range(50)}, tg)

        benchmark(lambda: asyncio.run(_run()))


# ── 5. BenchmarkNeuroSymbolicEngine ────────────────────────────────────


class BenchmarkNeuroSymbolicEngine:
    @pytest.mark.benchmark(min_rounds=100, warmup=True)
    def bm_engine_metrics(self, benchmark):
        engine = NeuroSymbolicEngine()
        engine._metrics["tasks_processed"] = 1000
        engine._metrics["tasks_successful"] = 950
        engine._metrics["tasks_failed"] = 50
        engine._metrics["total_refinements"] = 200
        engine._metrics["total_llm_calls"] = 1500

        def _run():
            for _ in range(10000):
                engine.get_metrics()

        benchmark(_run)
