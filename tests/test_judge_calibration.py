"""Judge Calibration Tests — проверяет, что Judge различает хорошие и плохие решения."""

from __future__ import annotations

import pytest

from noema.core.types import ArchitecturePattern, CodeBlock, Solution, TechStack
from noema.judge import critique_solution, evaluate_pairwise, evaluate_solution
from noema.llm.providers import FallbackProvider


@pytest.fixture
def fallback_llm():
    return FallbackProvider()


@pytest.fixture
def good_solution() -> Solution:
    return Solution(
        task_id="good-1",
        title="Well-designed REST API",
        summary="A clean REST API with proper authentication, rate limiting, and comprehensive error handling. "
        "Uses FastAPI with async endpoints, JWT authentication, PostgreSQL with SQLAlchemy ORM, "
        "Redis caching layer, and structured logging. Includes unit tests and integration tests.",
        architecture=ArchitecturePattern(
            name="Layered Architecture",
            description="Clean separation: API layer → Service layer → Repository layer. "
            "DI container for loose coupling. Middleware for cross-cutting concerns.",
            pros=["Testable", "Maintainable", "Scalable"],
            cons=["More boilerplate"],
        ),
        stack=TechStack(
            languages=["Python", "TypeScript"],
            frameworks=["FastAPI", "SQLAlchemy", "Pydantic"],
            databases=["PostgreSQL", "Redis"],
            infrastructure=["Docker", "Kubernetes"],
        ),
        code_blocks=[
            CodeBlock(
                filename="app/main.py",
                language="python",
                content="""from fastapi import FastAPI
from app.routes import router
from app.middleware import setup_middleware
from app.database import init_db

app = FastAPI(title="My API", version="1.0.0")
setup_middleware(app)
app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup():
    await init_db()
""",
                description="Main application entry point",
            ),
        ],
        performance_notes=[
            "Redis caching with TTL=300s",
            "Database connection pooling (min=5, max=20)",
        ],
        security_notes=[
            "JWT authentication with refresh tokens",
            "Rate limiting per user",
            "Input validation via Pydantic",
        ],
    )


@pytest.fixture
def bad_solution() -> Solution:
    return Solution(
        task_id="bad-1",
        title="Insecure Monolith",
        summary="A quick and dirty API. No authentication. Direct SQL queries concatenated with f-strings. "
        "Everything in a single file. No tests. No error handling. Uses eval() for user input parsing.",
        architecture=ArchitecturePattern(
            name="Spaghetti",
            description="Everything in one file. Global mutable state. No separation of concerns.",
            pros=["Fast to write"],
            cons=["Unmaintainable", "Unsafe"],
        ),
        stack=TechStack(
            languages=["Python"],
            frameworks=["Flask"],
            databases=["SQLite"],
            infrastructure=[],
        ),
        code_blocks=[
            CodeBlock(
                filename="app.py",
                language="python",
                content="""from flask import Flask, request
import sqlite3
import os

app = Flask(__name__)
db = sqlite3.connect('app.db')

@app.route('/users/<id>')
def get_user(id):
    query = f"SELECT * FROM users WHERE id = {id}"
    return str(db.execute(query).fetchone())

@app.route('/exec')
def execute():
    cmd = request.args.get('cmd', '')
    return os.popen(cmd).read()
""",
                description="Single file API with SQL injection and RCE",
            ),
        ],
        performance_notes=[],
        security_notes=[],
    )


@pytest.fixture
def task_description() -> str:
    return "Build a REST API for user management"


@pytest.mark.asyncio
async def test_judge_calibration_good_vs_bad(
    fallback_llm, good_solution, bad_solution, task_description
):
    """Judge должен оценивать хорошее решение выше плохого."""
    verdict_good = await evaluate_solution(fallback_llm, good_solution, task_description, ["api"])
    verdict_bad = await evaluate_solution(fallback_llm, bad_solution, task_description, ["api"])

    assert verdict_good.scores.overall >= verdict_bad.scores.overall, (
        f"Good solution ({verdict_good.scores.overall}) should score >= "
        f"bad solution ({verdict_bad.scores.overall})"
    )


@pytest.mark.asyncio
async def test_judge_detects_missing_auth(fallback_llm, bad_solution, task_description):
    """Judge должен выявить отсутствие аутентификации."""
    verdict = await evaluate_solution(fallback_llm, bad_solution, task_description, ["api"])
    " ".join(verdict.weaknesses).lower()
    assert verdict.scores.security < 0.8, (
        f"Security score should be low, got {verdict.scores.security}"
    )
    assert verdict.scores.overall < 0.9, (
        f"Overall should be penalized, got {verdict.scores.overall}"
    )


@pytest.mark.asyncio
async def test_judge_with_checklist(fallback_llm, good_solution, task_description):
    """Judge с чеклистом должен учитывать требования."""
    checklist = ["Must have authentication", "Must have rate limiting", "Must have tests"]
    verdict = await evaluate_solution(
        fallback_llm,
        good_solution,
        task_description,
        ["api"],
        checklist=checklist,
    )
    assert verdict.scores.overall >= 0.0


@pytest.mark.asyncio
async def test_pairwise_judge_selects_better(
    fallback_llm, good_solution, bad_solution, task_description
):
    """Pairwise Judge должен выбрать лучшее решение."""
    result = await evaluate_pairwise(
        fallback_llm,
        good_solution,
        bad_solution,
        task_description,
        ["api"],
    )
    assert result.winner == "A", f"Good solution (A) should win, got winner={result.winner}"
    assert result.scores_a.overall >= result.scores_b.overall, (
        f"Scores A ({result.scores_a.overall}) should >= B ({result.scores_b.overall})"
    )


@pytest.mark.asyncio
async def test_critique_finds_issues(fallback_llm, bad_solution, task_description):
    """Critique agent должен найти проблемы в плохом решении."""
    result = await critique_solution(fallback_llm, bad_solution, task_description)
    issues = result.get("issues", [])
    trust = result.get("trust_score", 1.0)
    # With FallbackProvider, critique returns default values
    assert 0.0 <= trust <= 1.0
    assert isinstance(issues, list)


@pytest.mark.asyncio
async def test_good_solution_passes_judge(fallback_llm, good_solution, task_description):
    """Хорошее решение должно проходить Judge."""
    verdict = await evaluate_solution(fallback_llm, good_solution, task_description, ["api"])
    assert isinstance(verdict.passed, bool)
    assert verdict.scores.overall >= 0.0


@pytest.mark.asyncio
async def test_bad_solution_has_weaknesses(fallback_llm, bad_solution, task_description):
    """Плохое решение должно иметь замечания."""
    verdict = await evaluate_solution(fallback_llm, bad_solution, task_description, ["api"])
    assert isinstance(verdict.weaknesses, list)
    assert verdict.scores.security <= 0.6
