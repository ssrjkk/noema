"""Golden Eval Pipeline — автоматическая оценка качества рассуждений Noema.

Запуск:
    python -m tests.eval.run_eval [--tasks golden_tasks.json] [--output results.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from noema.core.engine import NoemaEngine
from noema.core.types import Task, TaskComplexity
from noema.judge import evaluate_solution


async def run_single_eval(
    noema: NoemaEngine,
    task_data: dict[str, Any],
) -> dict[str, Any]:
    task = Task(
        title=task_data["task"][:50],
        description=task_data["task"],
        tags=task_data.get("tags", []),
        complexity=TaskComplexity(task_data.get("complexity", "moderate")),
    )
    task.requirements = [
        {"description": r, "category": "eval", "priority": 5}
        for r in task_data.get("key_requirements", [])
    ]

    t0 = time.time()
    solution, thought = await noema.think(task)
    duration = time.time() - t0

    verdict = await evaluate_solution(
        noema.llm,
        solution,
        task_data["task"],
        task_data.get("tags", []),
    )

    red_flags_found = [
        rf
        for rf in task_data.get("red_flags", [])
        if rf.lower() in solution.summary.lower()
        or any(rf.lower() in w.lower() for w in verdict.weaknesses)
    ]

    return {
        "task_id": task_data["id"],
        "passed": verdict.passed,
        "duration_sec": round(duration, 1),
        "judge_overall": verdict.scores.overall,
        "judge_weaknesses": verdict.weaknesses,
        "red_flags_found": red_flags_found,
        "red_flags_missed": [
            rf for rf in task_data.get("red_flags", []) if rf not in red_flags_found
        ],
        "steps": len(thought.steps),
        "summary": solution.summary[:200],
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Noema Eval Pipeline")
    parser.add_argument("--tasks", default=str(Path(__file__).parent / "golden_tasks.json"))
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        print(f"ERROR: Tasks file not found: {tasks_path}")
        sys.exit(1)

    with open(tasks_path) as f:
        golden_tasks = json.load(f)

    print(f"Golden Eval Pipeline — {len(golden_tasks)} tasks")
    print(f"LLM: {args.provider or 'default'} / {args.model or 'default'}")
    print("─" * 60)

    noema = NoemaEngine(
        llm_provider=args.provider,
        llm_model=args.model,
        project_root=str(Path(__file__).resolve().parent.parent.parent),
    )
    await noema.initialize()

    results = []
    passed = 0
    for task_data in golden_tasks:
        print(f"  [{task_data['id']}] {task_data['task'][:60]}...")
        result = await run_single_eval(noema, task_data)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"    → {status} | judge: {result['judge_overall']:.2f} | "
            f"red_flags: {len(result['red_flags_found'])}/{len(task_data.get('red_flags', []))} | "
            f"{result['duration_sec']}s"
        )
        if result["passed"]:
            passed += 1

    await noema.shutdown()

    summary = {
        "total": len(golden_tasks),
        "passed": passed,
        "failed": len(golden_tasks) - passed,
        "pass_rate": round(passed / max(len(golden_tasks), 1), 3),
        "avg_judge_score": round(
            sum(r["judge_overall"] for r in results) / max(len(results), 1), 3
        ),
        "avg_duration_sec": round(
            sum(r["duration_sec"] for r in results) / max(len(results), 1), 1
        ),
        "results": results,
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("─" * 60)
    print(f"Results: {passed}/{len(golden_tasks)} passed (rate={summary['pass_rate']:.1%})")
    print(f"Avg judge score: {summary['avg_judge_score']:.3f}")
    print(f"Avg duration: {summary['avg_duration_sec']}s")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
