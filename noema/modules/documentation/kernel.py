"""Documentation Module — auto-docs, API specs, changelogs, README generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DocFormat(StrEnum):
    MARKDOWN = "markdown"
    RST = "rst"
    HTML = "html"
    OPENAPI_JSON = "openapi_json"
    OPENAPI_YAML = "openapi_yaml"


@dataclass
class DocSection:
    title: str = ""
    content: str = ""
    level: int = 1
    subsections: list[DocSection] = field(default_factory=list)


@dataclass
class APIDoc:
    endpoint: str = ""
    method: str = "GET"
    summary: str = ""
    description: str = ""
    parameters: list[dict[str, str]] = field(default_factory=list)
    request_body: dict[str, Any] | None = None
    response_example: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class ChangelogEntry:
    version: str = ""
    date: str = ""
    changes: dict[str, list[str]] = field(default_factory=dict)  # category -> items


class DocGenerator:
    """Generate documentation from code and specs."""

    def generate_readme(
        self,
        project_name: str,
        description: str,
        features: list[str] | None = None,
        install_cmd: str = "",
        usage_example: str = "",
        api_endpoints: list[dict] | None = None,
    ) -> str:
        sections = [f"# {project_name}\n", f"{description}\n"]

        if features:
            sections.append("## Features\n")
            for f in features:
                sections.append(f"- {f}")
            sections.append("")

        if install_cmd:
            sections.append("## Installation\n")
            sections.append(f"```bash\n{install_cmd}\n```\n")

        if usage_example:
            sections.append("## Usage\n")
            sections.append(f"```python\n{usage_example}\n```\n")

        if api_endpoints:
            sections.append("## API Endpoints\n")
            sections.append("| Method | Path | Description |")
            sections.append("|--------|------|-------------|")
            for ep in api_endpoints:
                sections.append(
                    f"| {ep.get('method', 'GET')} | {ep.get('path', '/')} | {ep.get('description', '')} |"
                )
            sections.append("")

        sections.append("## License\n")
        sections.append("MIT\n")
        return "\n".join(sections)

    def generate_api_docs(self, endpoints: list[dict[str, Any]]) -> list[APIDoc]:
        docs = []
        for ep in endpoints:
            doc = APIDoc(
                endpoint=ep.get("path", "/"),
                method=ep.get("method", "GET").upper(),
                summary=ep.get("summary", ep.get("description", "")),
                description=ep.get("docstring", ""),
                parameters=ep.get("parameters", []),
                request_body=ep.get("request_body"),
                response_example=ep.get("response"),
                tags=ep.get("tags", []),
            )
            docs.append(doc)
        return docs

    def generate_openapi_spec(
        self, title: str, version: str, endpoints: list[dict[str, Any]]
    ) -> dict[str, Any]:
        paths: dict[str, Any] = {}
        for ep in endpoints:
            path = ep.get("path", "/")
            method = ep.get("method", "get").lower()
            if path not in paths:
                paths[path] = {}
            operation: dict[str, Any] = {
                "summary": ep.get("summary", ep.get("description", "")),
                "tags": ep.get("tags", []),
                "responses": {
                    "200": {"description": "Success"},
                },
            }
            if ep.get("parameters"):
                operation["parameters"] = [
                    {
                        "name": p.get("name", ""),
                        "in": "query",
                        "schema": {"type": p.get("type", "string")},
                    }
                    for p in ep["parameters"]
                ]
            if ep.get("request_body"):
                operation["requestBody"] = {
                    "content": {"application/json": {"schema": ep["request_body"]}}
                }
            paths[path][method] = operation

        return {
            "openapi": "3.0.3",
            "info": {"title": title, "version": version},
            "paths": paths,
        }

    def generate_changelog(self, entries: list[ChangelogEntry]) -> str:
        lines = [
            "# Changelog\n",
            "All notable changes to this project will be documented in this file.\n",
        ]
        for entry in entries:
            lines.append(f"## [{entry.version}] - {entry.date}\n")
            for category, items in entry.changes.items():
                lines.append(f"### {category}\n")
                for item in items:
                    lines.append(f"- {item}")
                lines.append("")
        return "\n".join(lines)

    def generate_docstring(
        self,
        function_name: str,
        params: list[dict[str, str]],
        return_type: str = "Any",
        description: str = "",
        examples: list[str] | None = None,
    ) -> str:
        lines = [f'"""{description or function_name}.\n']
        if params:
            lines.append("Args:\n")
            for p in params:
                lines.append(
                    f"    {p.get('name', 'param')} ({p.get('type', 'Any')}): {p.get('description', '')}"
                )
        if return_type and return_type != "None":
            lines.append(f"\nReturns:\n    {return_type}: Description of return value.")
        if examples:
            lines.append("\nExamples:\n")
            for ex in examples:
                lines.append(f"    >>> {ex}")
        lines.append('"""')
        return "\n".join(lines)

    def execute(self, task: Any) -> dict[str, Any]:
        tags = getattr(task, "tags", [])
        return {
            "type": "documentation",
            "readme_generated": True,
            "formats_available": [f.value for f in DocFormat],
            "recommended_docs": self._suggest_docs(tags),
            "_confidence": 0.85,
        }

    def _suggest_docs(self, tags: list[str]) -> list[str]:
        docs = ["README.md", "CHANGELOG.md", "CONTRIBUTING.md"]
        if "api" in tags:
            docs.extend(["openapi.yaml", "API.md"])
        if "python" in tags:
            docs.append("docs/conf.py")
        if "docker" in tags:
            docs.extend(["Dockerfile.md", "docker-compose.md"])
        return docs


class DocumentationModule:
    """Standalone documentation module."""

    NAME = "documentation"
    DESCRIPTION = "Auto-generate README, API docs, OpenAPI specs, changelogs, docstrings"

    def __init__(self) -> None:
        self.generator = DocGenerator()

    def generate_project_docs(
        self, project_name: str, description: str, endpoints: list[dict] | None = None
    ) -> dict[str, str]:
        docs = {}
        docs["README.md"] = self.generator.generate_readme(project_name, description)
        if endpoints:
            openapi = self.generator.generate_openapi_spec(project_name, "1.0.0", endpoints)
            import json

            docs["openapi.json"] = json.dumps(openapi, indent=2)
        return docs

    def execute(self, task: Any) -> dict[str, Any]:
        return self.generator.execute(task)
