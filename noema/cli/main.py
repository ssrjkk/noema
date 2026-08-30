"""CLI entry point for Noema (Typer + Rich design system).

Architecture:
- Every command is a thin synchronous wrapper that delegates to an ``async``
  helper via :func:`asyncio.run`; the async helpers own all I/O.
- All rendering flows through :mod:`noema.cli.ui` so panels, tables, colors,
  and spinners stay visually consistent across commands.

Concurrency contract:
- Exactly one event loop per command (``asyncio.run``); no loop is leaked or
  reused across commands. Long-running LLM work runs inside the noema engine.

Complexity:
- Per-command helpers are ``O(output)`` in what they print; the heavy lifting
  is delegated to NoemaEngine / pipelines / stores.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import typer

from noema.cli.arq import arq_app
from noema.cli.audit import audit_app
from noema.cli.grid import grid_app
from noema.cli.grpc import grpc_app
from noema.cli.health import health_app
from noema.cli.init_cmd import init_app
from noema.cli.ui import (
    ARROW,
    ELLIPSIS,
    MIDDOT,
    TAGLINE,
    console,
    data_table,
    fmt_duration,
    info,
    kv_panel,
    ok,
    panel,
    print_banner,
    section,
    spinner,
    task_progress,
    warn,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from noema.agents.base import BaseAgent
    from noema.core.types import Solution, ThoughtProcess
    from noema.kernels.base import BaseKernel
    from noema.workers.hierarchy import HierarchyTask

app = typer.Typer(
    name="noema",
    help=TAGLINE,
    no_args_is_help=False,
    invoke_without_command=True,
    rich_markup_mode="rich",
)


@app.callback()
def _main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None and not ctx.resilient_parsing:
        print_banner()
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command(rich_help_panel="Core")
def think(
    title: str = typer.Argument(..., help="Название задачи"),
    description: str = typer.Option("", "--desc", "-d", help="Описание задачи"),
    tags: str = typer.Option("", "--tags", "-t", help="Теги через запятую"),
    complexity: str = typer.Option(
        "moderate", "--complexity", "-c", help="trivial|simple|moderate|complex|extreme"
    ),
    stack: str = typer.Option("", "--stack", "-s", help="Стек через запятую"),
    output: str = typer.Option("summary", "--output", "-o", help="summary|full|json"),
    scaffold: bool = typer.Option(False, "--scaffold", help="Экспорт в структуру проекта"),
    scaffold_dir: str = typer.Option(".", "--scaffold-dir", help="Директория для экспорта"),
    llm: str = typer.Option("", "--llm", help="LLM провайдер: openai|anthropic|ollama"),
    model: str = typer.Option("", "--model", help="Модель LLM"),
) -> None:
    """Generate a technical solution for a task."""
    asyncio.run(
        _think(
            title, description, tags, complexity, stack, output, scaffold, scaffold_dir, llm, model
        )
    )


async def _think(
    title: str,
    description: str,
    tags: str,
    complexity: str,
    stack: str,
    output: str,
    scaffold: bool,
    scaffold_dir: str,
    llm: str,
    model: str,
) -> None:
    from noema import NoemaEngine
    from noema.core.types import Task, TaskComplexity, TechStack

    noema = NoemaEngine(llm_provider=llm or None, llm_model=model or None)
    await noema.initialize()

    valid_complexities = {c.value for c in TaskComplexity}
    if complexity not in valid_complexities:
        console.print(
            f"[err]Invalid complexity: {complexity}. "
            f"Use one of: {', '.join(sorted(valid_complexities))}[/err]"
        )
        await noema.shutdown()
        return

    task = Task(
        title=title,
        description=description or title,
        complexity=TaskComplexity(complexity),
        tags=[t.strip() for t in tags.split(",") if t.strip()] if tags else [],
    )

    if stack:
        langs = [s.strip() for s in stack.split(",") if s.strip()]
        task.preferred_stack = TechStack(languages=langs)

    section("Thinking")
    progress = task_progress(f"[accent]Queued{ELLIPSIS}[/accent]")
    task_progress_id = progress.add_task(f"[accent]Queued{ELLIPSIS}[/accent]", total=None)

    async def on_step_start(name: str, label: str, done: int, total: int) -> None:
        progress.update(
            task_progress_id, description=f"[accent]{label}[/accent]", total=total, completed=done
        )

    async def on_step_end(name: str, result: str, done: int, total: int) -> None:
        status = "OK" if not result.startswith("FAILED") else "ERR"
        color = "ok" if status == "OK" else "err"
        preview = result[:80].replace("\n", " ")
        console.print(f"  [{color}]{status:3}[/{color}] [bold]{name}[/bold] {preview}")
        progress.update(task_progress_id, completed=done, total=total)

    noema.on_step_start(on_step_start)
    noema.on_step_end(on_step_end)

    with progress:
        solution, thought = await noema.think(task)

    if output == "json":
        console.print_json(solution.model_dump_json(indent=2))
    elif output == "full":
        _print_full(solution, thought)
    else:
        _print_summary(solution, thought)

    if scaffold:
        result = await noema.scaffold_project(solution, task, scaffold_dir)
        ok(f"Project scaffolded: {result['project_dir']}")
        info(f"Files created: {result['files_created']}")

    await noema.shutdown()


def _print_summary(solution: Solution, thought: ThoughtProcess) -> None:
    kv_panel(
        "Noema Solution",
        [
            ("Title", solution.title),
            ("Quality", solution.quality.value),
            ("Confidence", f"{solution.confidence:.0%}"),
            ("Stack", solution.stack.summary()),
            ("Code blocks", str(len(solution.code_blocks))),
            ("Time", fmt_duration(thought.duration_ms)),
            ("Thought steps", str(len(thought.steps))),
        ],
    )

    if solution.architecture:
        section("Architecture")
        console.print(f"[accent]{solution.architecture.name}[/accent]")
        console.print(f"  {solution.architecture.description}")

    if solution.code_blocks:
        section("Generated files")
        data_table(
            "Generated files",
            ["File", "Language"],
            [[b.filename, b.language] for b in solution.code_blocks],
        )

    if solution.performance_notes:
        section("Optimizations")
        for note in solution.performance_notes[:5]:
            console.print(f"  [ok]{ARROW}[/ok] {note[:100]}")

    if solution.security_notes:
        section("Security")
        for note in solution.security_notes[:5]:
            console.print(f"  [err]{ARROW}[/err] {note[:100]}")


def _print_full(solution: Solution, thought: ThoughtProcess) -> None:
    _print_summary(solution, thought)

    section("Thought Process")
    for step in thought.steps:
        console.print(
            f"  [accent]Step {step.step_number}[/accent]: [{step.kernel}] "
            f"-> {step.output_summary[:120]} (confidence: {step.confidence:.0%})"
        )

    section("Generated Code")
    lang_map = {
        "python": "python",
        "typescript": "typescript",
        "go": "go",
        "rust": "rust",
        "yaml": "yaml",
    }
    from rich.syntax import Syntax

    for block in solution.code_blocks:
        console.print(f"\n[path]{block.filename}[/path] ({block.language})")
        lang = lang_map.get(block.language, "text")
        console.print(Syntax(block.content, lang, theme="monokai", line_numbers=True))


@app.command(rich_help_panel="Core")
def pipeline(
    name: str = typer.Argument("fullstack", help="Pipeline: fullstack|quick|security|arch-review"),
    title: str = typer.Option("My Project", "--title", help="Название задачи"),
    tags: str = typer.Option("", "--tags", "-t", help="Теги через запятую"),
) -> None:
    """Запустить пайплайн ядер."""
    asyncio.run(_pipeline(name, title, tags))


async def _pipeline(name: str, title: str, tags: str) -> None:
    from noema import NoemaEngine
    from noema.core.types import Task, TaskComplexity
    from noema.pipelines.engine import (
        create_architecture_review_pipeline,
        create_fullstack_pipeline,
        create_quick_prototype_pipeline,
        create_security_audit_pipeline,
    )

    pipelines = {
        "fullstack": create_fullstack_pipeline,
        "quick": create_quick_prototype_pipeline,
        "security": create_security_audit_pipeline,
        "arch-review": create_architecture_review_pipeline,
    }

    if name not in pipelines:
        console.print(f"[err]Unknown pipeline: {name}. Available: {list(pipelines.keys())}[/err]")
        return

    noema = NoemaEngine()
    await noema.initialize()

    task = Task(
        title=title,
        complexity=TaskComplexity.MODERATE,
        tags=[t.strip() for t in tags.split(",") if t.strip()] if tags else [],
    )

    pipeline = pipelines[name]()
    async with spinner(f"Running pipeline [bold]{name}[/bold]{ELLIPSIS}"):
        result = await pipeline.execute(task, noema=noema)

    kv_panel(
        "Pipeline Result",
        [
            ("Pipeline", result.pipeline_name),
            ("Steps", f"{result.completed_steps}/{len(result.steps)}"),
            ("Failed", str(result.failed_steps)),
            ("Duration", fmt_duration(result.total_duration_ms)),
            ("Success", str(result.success)),
        ],
        border="ok",
    )

    rows = []
    for step in result.steps:
        marker = (
            "[ok]completed[/ok]"
            if step.status.value == "completed"
            else "[err]failed[/err]"
            if step.status.value == "failed"
            else "[warn]skipped[/warn]"
        )
        rows.append([marker, step.name, fmt_duration(step.duration_ms)])
    data_table("Pipeline Steps", ["Status", "Step", "Duration"], rows, border="ok")

    await noema.shutdown()


@app.command(rich_help_panel="Knowledge")
def graph(
    action: str = typer.Argument("stats", help="stats|suggest|compatible"),
    tech: str = typer.Option("", "--tech", help="Технология для анализа"),
    tags: str = typer.Option("", "--tags", "-t", help="Теги для suggestions"),
) -> None:
    """Работа с графом знаний."""
    asyncio.run(_graph(action, tech, tags))


async def _graph(action: str, tech: str, tags: str) -> None:
    from noema.knowledge.graph import KnowledgeGraph

    kg = KnowledgeGraph()

    if action == "stats":
        data_table(
            "Knowledge Graph Stats",
            ["Metric", "Value"],
            [[k, v] for k, v in kg.get_stats().items()],
        )

    elif action == "suggest" and tags:
        tag_list = [t.strip() for t in tags.split(",")]
        section(f"Suggestions for {', '.join(tag_list)}")
        result = kg.suggest_architecture(tag_list)
        rows = [[c["from"], c["to"], c["relationship"]] for c in result["components"][:10]]
        data_table("Architecture Suggestions", ["From", "To", "Relationship"], rows)

    elif action == "compatible" and tech:
        section(f"Compatible with {tech}")
        compat = kg.get_compatible_technologies(tech)
        for category, techs in compat.items():
            console.print(f"  [accent]{category}:[/accent] {', '.join(techs)}")

    else:
        console.print("[warn]Usage: noema graph suggest --tags 'python,fastapi,redis'[/warn]")


@app.command(rich_help_panel="Knowledge")
def knowledge(
    action: str = typer.Argument("stats", help="stats|search|add"),
    query: str = typer.Option("", "--query", "-q", help="Поисковый запрос"),
) -> None:
    """Работа с базой знаний."""
    asyncio.run(_knowledge(action, query))


async def _knowledge(action: str, query: str) -> None:
    from noema.knowledge.store import KnowledgeStore

    store = KnowledgeStore()
    await store.load()

    if action == "stats":
        data_table(
            "Knowledge Store Stats",
            ["Metric", "Value"],
            [[k, v] for k, v in store.get_stats().items()],
        )

    elif action == "search":
        if not query:
            console.print("[err]Specify --query[/err]")
            return
        results = await store.search(query)
        for r in results:
            kind = r.get("type", "unknown")
            title = r.get("title", r.get("name", "N/A"))
            score = r.get("score", 0)
            console.print(
                f"  [path]{title}[/path] [dim]{MIDDOT} {kind}[/dim] [accent]({score:.2f})[/accent]"
            )
            if r.get("content"):
                console.print(f"    {r['content'][:150]}...")
            console.print()


@app.command(rich_help_panel="Knowledge")
def feedback(
    action: str = typer.Argument("stats", help="stats|analyze"),
) -> None:
    """Анализ обратной связи."""
    asyncio.run(_feedback(action))


async def _feedback(action: str) -> None:
    from noema.feedback.store import FeedbackStore

    store = FeedbackStore()
    await store.load()

    if action == "stats":
        data_table(
            "Feedback Stats", ["Metric", "Value"], [[k, v] for k, v in store.get_stats().items()]
        )

    elif action == "analyze":
        analysis = store.analyze_patterns()
        if analysis.get("status") == "no_data":
            warn("No feedback data yet")
            return
        data_table(
            "Feedback Analysis",
            ["Metric", "Value"],
            [[k, v] for k, v in analysis.items() if not isinstance(v, (dict, list))],
        )


@app.command(rich_help_panel="System")
def serve(
    host: str | None = typer.Option(None, "--host", "-h"),
    port: int | None = typer.Option(None, "--port", "-p"),
    reload: bool | None = typer.Option(None, "--reload"),
) -> None:
    """Launch API server."""
    import uvicorn

    from noema.config.settings import get_settings

    settings = get_settings()
    h = host or settings.api.host
    p = port or settings.api.port
    r = reload if reload is not None else settings.api.reload
    panel(
        f"[val]http://{h}:{p}[/val]  {MIDDOT}  [dim]reload={r}[/dim]",
        title="Noema API",
        border="ok",
    )
    uvicorn.run("noema.api.server:app", host=h, port=p, reload=r)


@app.command(rich_help_panel="Core")
def kernels() -> None:
    """Показать доступные ядра."""
    from noema.kernels import (
        AIMLKernel,
        AnalysisKernel,
        ArchitectureKernel,
        CodegenKernel,
        DataKernel,
        DevOpsKernel,
        FrontendKernel,
        OptimizationKernel,
        SecurityKernel,
    )

    kernel_factories: list[Callable[..., BaseKernel]] = [
        ArchitectureKernel,
        CodegenKernel,
        OptimizationKernel,
        SecurityKernel,
        AnalysisKernel,
        FrontendKernel,
        DevOpsKernel,
        DataKernel,
        AIMLKernel,
    ]
    rows = [[k.name, k.description] for factory in kernel_factories for k in [factory()]]
    data_table("Available Kernels", ["Name", "Description"], rows)


@app.command(rich_help_panel="Core")
def agents() -> None:
    """Показать доступных агентов."""
    from noema.agents.base import (
        AIEngineerAgent,
        ArchitectAgent,
        DBAAgent,
        DeveloperAgent,
        DevOpsAgent,
        SecurityAgent,
    )

    agent_factories: list[Callable[..., BaseAgent]] = [
        ArchitectAgent,
        DeveloperAgent,
        SecurityAgent,
        DevOpsAgent,
        DBAAgent,
        AIEngineerAgent,
    ]
    rows = []
    for factory in agent_factories:
        a = factory()
        rows.append([a.name, a.role.value, ", ".join(a.expertise)])
    data_table("Available Agents", ["Name", "Role", "Expertise"], rows)


@app.command(rich_help_panel="Knowledge")
def memory(
    action: str = typer.Argument("stats", help="stats|search|episodes"),
    query: str = typer.Option("", "--query", "-q", help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Result limit"),
) -> None:
    """Memory system - episodic/semantic/procedural memory."""
    asyncio.run(_memory(action, query, limit))


async def _memory(action: str, query: str, limit: int) -> None:
    from noema.memory.store import MemoryStore

    store = MemoryStore()
    stats = store.stats()

    if action == "stats":
        rows = [[k.replace("_", " ").title(), str(v)] for k, v in stats.items()]
        data_table("Memory Stats", ["Metric", "Value"], rows)

    elif action == "search":
        if not query:
            console.print("[err]Specify --query[/err]")
            return
        episodes = store.search_episodes(query, limit=limit)
        knowledge = store.search_knowledge(query, limit=limit)
        procedures = store.search_procedures(query)

        if episodes:
            section("Episodic Memory")
            for ep in episodes:
                console.print(
                    f"  [path]{ep.task_description[:80]}[/path] [dim]({ep.tech_stack})[/dim]"
                )

        if knowledge:
            section("Semantic Memory")
            for k in knowledge:
                console.print(
                    f"  [path]{k.fact[:80]}[/path] [dim](confidence: {k.confidence:.0%})[/dim]"
                )

        if procedures:
            section("Procedural Memory")
            for p in procedures:
                console.print(
                    f"  {p.procedure_name}: [ok]success_rate={p.success_rate:.0%}[/ok] "
                    f"[dim]applied={p.times_applied}[/dim]"
                )

    elif action == "episodes":
        episodes = store.get_recent_episodes(limit=limit)
        section(f"Recent {len(episodes)} Episodes")
        for ep in episodes:
            console.print(f"  [path]{ep.task_description[:60]}[/path] [dim]| {ep.tech_stack}[/dim]")


@app.command(rich_help_panel="Core")
def evolve() -> None:
    """Run self-evolution cycle."""
    asyncio.run(_evolve())


async def _evolve() -> None:
    from noema import NoemaEngine

    noema = NoemaEngine()
    await noema.initialize()
    async with spinner(f"Running self-evolution cycle{ELLIPSIS}"):
        result = await noema.evolve()

    kv_panel(
        f"Evolution Cycle {result.get('evolution_cycle', 0)}",
        [
            ("Patches generated", str(result.get("patches_generated", 0))),
            ("Patches applied", str(result.get("patches_applied", 0))),
            ("Patches rejected", str(result.get("patches_rejected", 0))),
            ("Summary", result.get("summary", "")),
        ],
        border="warn",
    )

    await noema.shutdown()


@app.command(rich_help_panel="System")
def discover() -> None:
    """Discover available keys and resources."""
    asyncio.run(_discover())


async def _discover() -> None:
    from noema.discovery.keys import KeyDiscovery

    disc = KeyDiscovery()
    async with spinner(f"Discovering keys and resources{ELLIPSIS}"):
        result = disc.discover_all()

    providers = result.get("providers_available", [])
    kv_panel(
        "Resource Discovery",
        [("Providers available", ", ".join(providers) or "none")],
        border="warn",
    )

    if result.get("keys"):
        data_table(
            "Discovered Keys",
            ["Name", "Source", "Provider"],
            [[k["name"], k["source"], k["provider"]] for k in result["keys"]],
        )

    if result.get("resources"):
        rows = [
            [r["name"], r["kind"], "yes" if r["available"] else "no"] for r in result["resources"]
        ]
        data_table("System Resources", ["Resource", "Kind", "Available"], rows)


@app.command(rich_help_panel="Knowledge")
def ingest(
    source: str = typer.Argument(..., help="File, directory, or URL to ingest"),
) -> None:
    """Ingest knowledge from files, directories, or URLs."""
    asyncio.run(_ingest(source))


async def _ingest(source: str) -> None:
    from noema.ingestion.loader import KnowledgeLoader
    from noema.memory.store import MemoryStore

    store = MemoryStore()
    loader = KnowledgeLoader(knowledge_store=store)

    if os.path.isfile(source):
        async with spinner(f"Ingesting file [bold]{source}[/bold]{ELLIPSIS}"):
            result = await loader.ingest_file(source, tags=["cli-ingested"])
    elif os.path.isdir(source):
        async with spinner(f"Ingesting directory [bold]{source}[/bold]{ELLIPSIS}"):
            result = await loader.ingest_directory(source, tags=["cli-ingested"])
    elif source.startswith("http"):
        async with spinner(f"Ingesting URL [bold]{source}[/bold]{ELLIPSIS}"):
            result = await loader.ingest_url(source, tags=["cli-ingested"])
    else:
        console.print(f"[err]Unknown source: {source}[/err]")
        return

    kv_panel(
        "Knowledge Ingestion",
        [
            ("Source", result.source),
            ("Type", result.source_type),
            ("Ingested", f"{result.entries_ingested} entries"),
            ("Skipped", str(result.entries_skipped)),
            ("Topics", ", ".join(result.topics_extracted[:10]) or "N/A"),
            ("Errors", str(len(result.errors))),
        ],
        border="ok",
    )


@app.command(rich_help_panel="Core")
def hierarchy(
    description: str = typer.Argument(
        "Build a web application with authentication", help="Task description"
    ),
) -> None:
    """Execute task through infinite worker hierarchy."""
    asyncio.run(_hierarchy(description))


async def _hierarchy(description: str) -> None:
    from noema.workers.hierarchy import WorkerHierarchy

    async def decompose(task: HierarchyTask) -> list[str]:
        if task.depth >= 2:
            return []
        return [
            f"Analyze requirements for: {task.description}",
            f"Design architecture for: {task.description}",
            f"Implement components for: {task.description}",
        ]

    async def execute(task: HierarchyTask) -> dict[str, str | int]:
        return {"status": "done", "task": task.description, "depth": task.depth}

    hierarchy = WorkerHierarchy(max_depth=3)
    async with spinner(f"Decomposing and executing hierarchy{ELLIPSIS}"):
        result = await hierarchy.execute(description, decomposer=decompose, executor=execute)

    def count_subtasks(task: HierarchyTask) -> int:
        count = len(task.subtasks)
        for st in task.subtasks:
            count += count_subtasks(st)
        return count

    total = count_subtasks(result)

    kv_panel(
        "Worker Hierarchy",
        [
            ("Task", result.description),
            ("State", result.state.value),
            ("Subtasks spawned", str(total)),
            ("Stats", str(hierarchy.get_stats())),
        ],
    )


@app.command(rich_help_panel="Core")
def modules(
    action: str = typer.Argument("list", help="list|run|stats"),
    name: str = typer.Option("", "--name", "-n", help="Module name to run"),
    tags: str = typer.Option("", "--tags", "-t", help="Filter tags for run"),
) -> None:
    """Noema modules - pluggable domain modules that work standalone and together."""
    asyncio.run(_modules(action, name, tags))


async def _modules(action: str, name: str, tags: str) -> None:
    from noema.modules.registry import get_registry

    registry = get_registry()

    if action == "list":
        mods = registry.list_modules()
        data_table(
            "Noema Modules",
            ["Name", "Description"],
            [[m["name"], m["description"]] for m in mods],
        )

    elif action == "stats":
        stats = registry.stats()
        kv_panel(
            "Module Stats",
            [
                ("Total modules", str(stats["total_modules"])),
                ("Modules", ", ".join(stats["modules"])),
            ],
            border="warn",
        )

    elif action == "run":
        if not name:
            console.print("[err]Specify --name[/err]")
            return
        from noema.core.types import Task

        task = Task(
            title=f"Module test: {name}",
            tags=[t.strip() for t in tags.split(",") if t.strip()] if tags else [name],
        )
        result = registry.execute_module(name, task)
        kv_panel(
            f"Module: {name}",
            [(k, str(v)) for k, v in result.items() if k != "_confidence"],
            border="ok",
        )

    else:
        console.print(f"[err]Unknown action: {action}. Use list, run, or stats[/err]")


app.add_typer(health_app, name="health", rich_help_panel="System")
app.add_typer(init_app, name="init", rich_help_panel="System")
app.add_typer(arq_app, name="arq", rich_help_panel="Background")
app.add_typer(grid_app, name="grid", rich_help_panel="Background")
app.add_typer(grpc_app, name="grpc", rich_help_panel="Background")
app.add_typer(audit_app, name="audit", rich_help_panel="Audit")

if __name__ == "__main__":
    app()
