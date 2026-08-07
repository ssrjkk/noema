"""CLI Generator Module — scaffold CLI tools with argparse, click, typer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CLIFramework(StrEnum):
    TYPER = "typer"
    CLICK = "click"
    ARGPARSE = "argparse"
    OCLIF = "oclif"
    CLOMAD = "clomad"


@dataclass
class CLICommand:
    name: str = ""
    description: str = ""
    arguments: list[dict[str, str]] = field(default_factory=list)
    options: list[dict[str, str]] = field(default_factory=list)
    subcommands: list[CLICommand] = field(default_factory=list)
    handler_code: str = ""


@dataclass
class CLIApp:
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    framework: CLIFramework = CLIFramework.TYPER
    commands: list[CLICommand] = field(default_factory=list)
    global_options: list[dict[str, str]] = field(default_factory=list)


class CLIGenerator:
    """Generate CLI tool code."""

    def generate_typer(self, app: CLIApp) -> str:
        lines = [
            '"""' + app.description + '"""',
            "",
            "import typer",
            "from rich.console import Console",
            "",
            "app = typer.Typer(name='" + app.name + "', help='" + app.description + "')",
            "console = Console()",
            "",
        ]

        for cmd in app.commands:
            params = []
            for arg in cmd.arguments:
                params.append(arg["name"] + ": " + arg.get("type", "str"))
            for opt in cmd.options:
                flag = "--" + opt["name"].replace("_", "-")
                params.append(
                    opt["name"]
                    + ": "
                    + opt.get("type", "str")
                    + " = typer.Option("
                    + opt.get("default", '""')
                    + ", '"
                    + flag
                    + "')"
                )
            params_str = ", ".join(params)
            if params_str:
                params_str = ", " + params_str

            lines.append("")
            lines.append("@app.command()")
            lines.append("def " + cmd.name + "(" + params_str + "):")
            lines.append('    """' + cmd.description + '"""')
            if cmd.handler_code:
                lines.append("    " + cmd.handler_code)
            else:
                lines.append("    console.print('[bold]" + cmd.name + "[/bold]')")

        lines.append("")
        lines.append('if __name__ == "__main__":')
        lines.append("    app()")
        return "\n".join(lines)

    def generate_click(self, app: CLIApp) -> str:
        lines = [
            '"""' + app.description + '"""',
            "",
            "import click",
            "from rich.console import Console",
            "",
            "console = Console()",
            "",
            "@click.group()",
            "@click.version_option('" + app.version + "')",
            "def cli():",
            '    """' + app.description + '"""',
            "    pass",
            "",
        ]

        for cmd in app.commands:
            lines.append("@cli.command()")
            for arg in cmd.arguments:
                lines.append("@click.argument('" + arg["name"].upper() + "')")
            for opt in cmd.options:
                lines.append(
                    "@click.option('--"
                    + opt["name"].replace("_", "-")
                    + "', default='"
                    + opt.get("default", "")
                    + "')"
                )

            args_str = ", ".join(a["name"] for a in cmd.arguments)
            opts_str = ", ".join(o["name"] for o in cmd.options)
            all_params = ", ".join(filter(None, [args_str, opts_str]))

            lines.append("def " + cmd.name + "(" + all_params + "):")
            lines.append('    """' + cmd.description + '"""')
            lines.append("    console.print('[bold]" + cmd.name + "[/bold]')")
            lines.append("")

        lines.append("if __name__ == '__main__':")
        lines.append("    cli()")
        return "\n".join(lines)

    def generate_argparse(self, app: CLIApp) -> str:
        lines = [
            '"""' + app.description + '"""',
            "",
            "import argparse",
            "import sys",
            "",
            "def main():",
            "    parser = argparse.ArgumentParser(description='" + app.description + "')",
            "    parser.add_argument('--version', action='version', version='" + app.version + "')",
            "    subparsers = parser.add_subparsers(dest='command')",
            "",
        ]

        for cmd in app.commands:
            lines.append("    # Command: " + cmd.name)
            lines.append(
                "    sub_"
                + cmd.name
                + " = subparsers.add_parser('"
                + cmd.name
                + "', help='"
                + cmd.description
                + "')"
            )
            for arg in cmd.arguments:
                lines.append(
                    "    sub_"
                    + cmd.name
                    + ".add_argument('"
                    + arg["name"]
                    + "', help='"
                    + arg.get("description", "")
                    + "')"
                )
            for opt in cmd.options:
                lines.append(
                    "    sub_"
                    + cmd.name
                    + ".add_argument('--"
                    + opt["name"].replace("_", "-")
                    + "', default='"
                    + opt.get("default", "")
                    + "')"
                )
            lines.append("")

        lines.append("    args = parser.parse_args()")
        lines.append("    if args.command:")
        lines.append("        print(f'Running: {args.command}')")
        lines.append("    else:")
        lines.append("        parser.print_help()")
        lines.append("")
        lines.append("if __name__ == '__main__':")
        lines.append("    main()")
        return "\n".join(lines)

    def execute(self, task: Any, framework: str = "typer") -> str:
        tags = getattr(task, "tags", [])
        title = getattr(task, "title", "cli-tool") if hasattr(task, "title") else "cli-tool"

        app = CLIApp(
            name=title.lower().replace(" ", "-"),
            description=title,
        )

        if "deploy" in tags:
            app.commands.append(CLICommand(name="deploy", description="Deploy application"))
        if "migrate" in tags or "database" in tags:
            app.commands.append(CLICommand(name="migrate", description="Run migrations"))
            app.commands.append(CLICommand(name="rollback", description="Rollback migrations"))
        if "test" in tags:
            app.commands.append(CLICommand(name="test", description="Run tests"))
        if "serve" in tags or "api" in tags:
            app.commands.append(CLICommand(name="serve", description="Start dev server"))

        if not app.commands:
            app.commands = [
                CLICommand(name="init", description="Initialize project"),
                CLICommand(name="build", description="Build project"),
                CLICommand(name="run", description="Run project"),
            ]

        if framework == "typer":
            return self.generate_typer(app)
        elif framework == "click":
            return self.generate_click(app)
        return self.generate_argparse(app)


class CLIGeneratorModule:
    """Standalone CLI generator module."""

    NAME = "cli_generator"
    DESCRIPTION = "Generate CLI tools with argparse, click, or typer"

    def __init__(self) -> None:
        self.generator = CLIGenerator()

    def execute(self, task: Any) -> dict[str, Any]:
        tags = getattr(task, "tags", [])
        framework = "typer"
        if "click" in tags:
            framework = "click"
        elif "argparse" in tags:
            framework = "argparse"

        code = self.generator.execute(task, framework)
        return {
            "type": "cli_generator",
            "framework": framework,
            "code_preview": code[:500],
            "features": ["auto_complete", "help_text", "version", "subcommands"],
            "_confidence": 0.85,
        }
