"""Скаффолдер — экспорт решений в реальные файлы проекта."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from noema.logging import get_logger

if TYPE_CHECKING:
    from noema.core.types import CodeBlock, Solution, Task

logger = get_logger(__name__)


class ProjectScaffolder:
    """
    Генератор структуры проекта из решения.

    Создаёт полную структуру файлов: код, конфиги, Dockerfile, tests, CI/CD.
    """

    def __init__(self, output_dir: str = ".") -> None:
        self.output_dir = Path(output_dir)
        self._files_created: list[str] = []

    async def scaffold(self, solution: Solution, task: Task) -> dict[str, Any]:
        """Создать полную структуру проекта из решения."""
        project_name = self._slugify(task.title)
        project_dir = self.output_dir / project_name
        self._files_created = []

        # Основная структура
        dirs = self._create_directories(project_dir, solution)

        # Кодовые блоки
        project_root = project_dir.resolve()
        for block in solution.code_blocks:
            file_path = project_dir / self._map_filename(block, dirs)
            target = file_path.resolve()
            if not target.is_relative_to(project_root):
                raise ValueError(f"Unsafe filename escapes project directory: {block.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(block.content, encoding="utf-8")
            self._files_created.append(str(target.relative_to(project_root)))

        # Конфиги
        self._create_configs(project_dir, solution, task)

        # README
        self._create_readme(project_dir, solution, task)

        # pyproject.toml / package.json
        self._create_project_manifest(project_dir, solution)

        return {
            "project_dir": str(project_dir),
            "files_created": len(self._files_created),
            "files": self._files_created,
            "structure": self._tree(project_dir),
        }

    def _create_directories(self, project_dir: Path, solution: Solution) -> dict[str, str]:
        """Создание структуры директорий."""
        lang = solution.stack.languages[0].lower() if solution.stack.languages else "python"

        dirs = {
            "src": "src",
            "tests": "tests",
            "docs": "docs",
            "config": "config",
        }

        if lang in ("python",):
            slug = self._slugify(solution.title)
            dirs["src"] = f"src/{slug}"
            dirs["tests"] = "tests"
        elif lang in ("typescript",):
            dirs["src"] = "src"
            dirs["tests"] = "tests"

        for d in dirs.values():
            (project_dir / d).mkdir(parents=True, exist_ok=True)
            (project_dir / d / "__init__.py").touch(exist_ok=True) if lang == "python" else None

        (project_dir / "tests").mkdir(parents=True, exist_ok=True)

        return dirs

    @staticmethod
    def _sanitize_filename(raw: str) -> str:
        """Strip traversal/absolute components from an untrusted filename.

        The filename comes from LLM/NS output (``block.filename``) and must
        never escape ``project_dir``: backslashes are normalized, and the
        path is re-parsed as a pure POSIX path so root markers (``/``,
        ``\\``) and drive prefixes (``C:``) are dropped on every platform.
        """
        from pathlib import PurePosixPath

        fn = raw.replace("\\", "/").strip()
        parts = [
            p for p in PurePosixPath(fn).parts if p not in ("", ".", "..", "/") and ":" not in p
        ]
        if not parts:
            parts = ["generated.py"]
        return "/".join(parts)

    def _map_filename(self, block: CodeBlock, dirs: dict[str, str]) -> str:
        """Маппинг имён файлов в структуру проекта."""
        fn = self._sanitize_filename(block.filename)

        test_indicators = ("test_", "_test", "spec_")
        if any(fn.startswith(ind) or ind in fn for ind in test_indicators):
            return f"tests/{fn}"

        if fn in ("Dockerfile", "docker-compose.yml", ".dockerignore"):
            return fn
        if fn.endswith((".yml", ".yaml", ".toml", ".json")):
            return fn
        if fn.startswith("."):
            return fn

        lang = block.language.lower()
        if lang == "python":
            return f"{dirs['src']}/{fn}" if "/" not in fn else fn
        return fn

    def _create_configs(self, project_dir: Path, solution: Solution, task: Task) -> None:
        """Создание конфигурационных файлов."""
        # .gitignore
        gitignore_content = self._generate_gitignore(solution)
        (project_dir / ".gitignore").write_text(gitignore_content, encoding="utf-8")
        self._files_created.append(".gitignore")

        # .env.example
        env_content = self._generate_env_example(solution)
        (project_dir / ".env.example").write_text(env_content, encoding="utf-8")
        self._files_created.append(".env.example")

        # ruff.toml для Python
        if solution.stack.languages and solution.stack.languages[0].lower() == "python":
            ruff_content = '[tool.ruff]\nline-length = 100\ntarget-version = "py312"\n\n[tool.ruff.lint]\nselect = ["E", "F", "I", "N", "UP"]\n'
            (project_dir / "ruff.toml").write_text(ruff_content, encoding="utf-8")
            self._files_created.append("ruff.toml")

    def _create_project_manifest(self, project_dir: Path, solution: Solution) -> None:
        """Создание манифеста проекта (pyproject.toml / package.json)."""
        lang = solution.stack.languages[0].lower() if solution.stack.languages else "python"
        name = self._slugify(solution.title)

        if lang == "python":
            deps = self._python_deps(solution)
            content = f'''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = {deps}

[tool.hatch.build.targets.wheel]
packages = ["src/{name}"]
'''
            (project_dir / "pyproject.toml").write_text(content, encoding="utf-8")
            self._files_created.append("pyproject.toml")

        elif lang == "typescript":
            content = f'''{{
  "name": "{name}",
  "version": "0.1.0",
  "private": true,
  "scripts": {{
    "dev": "tsx watch src/index.ts",
    "build": "tsc",
    "start": "node dist/index.js",
    "test": "vitest",
    "lint": "eslint src/",
    "typecheck": "tsc --noEmit"
  }},
  "dependencies": {{}},
  "devDependencies": {{
    "typescript": "^5.4",
    "tsx": "^4.7",
    "vitest": "^1.3"
  }}
}}
'''
            (project_dir / "package.json").write_text(content, encoding="utf-8")
            self._files_created.append("package.json")

    def _create_readme(self, project_dir: Path, solution: Solution, task: Task) -> None:
        """Создание README.md."""
        stack_line = solution.stack.summary()
        arch_line = f"Architecture: {solution.architecture.name}" if solution.architecture else ""
        quality_line = f"Quality: {solution.quality.value} | Confidence: {solution.confidence:.0%}"

        readme = f"""# {solution.title}

{task.description or task.title}

## Stack

{stack_line}

{arch_line}

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run
python -m {self._slugify(solution.title)}.main

# Test
pytest tests/
```

## Project Structure

```
{self._tree(project_dir)}
```

---
Generated by Noema | {quality_line}
"""
        (project_dir / "README.md").write_text(readme, encoding="utf-8")
        self._files_created.append("README.md")

    def _generate_gitignore(self, solution: Solution) -> str:
        lang = solution.stack.languages[0].lower() if solution.stack.languages else "python"
        base = """__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.env
.venv/
venv/
*.log
node_modules/
.DS_Store
"""
        if lang == "python":
            return base
        if lang == "typescript":
            return base + "\nnode_modules/\ndist/\n.env*\n"
        return base

    def _generate_env_example(self, solution: Solution) -> str:
        lines = ["# Environment Configuration"]
        for db in solution.stack.databases:
            lines.append(f"{db.upper().replace(' ', '_')}_URL=connection_string")
        lines.append("LOG_LEVEL=INFO")
        lines.append("SECRET_KEY=change-me-in-production")
        return "\n".join(lines) + "\n"

    def _python_deps(self, solution: Solution) -> str:
        deps = ['"pydantic>=2.0"']
        for fw in solution.stack.frameworks:
            fw_lower = fw.lower()
            if "fastapi" in fw_lower:
                deps.append('"fastapi>=0.100"')
                deps.append('"uvicorn>=0.23"')
            elif "flask" in fw_lower:
                deps.append('"flask>=3.0"')
        for db in solution.stack.databases:
            if "postgres" in db.lower():
                deps.append('"asyncpg>=0.29"')
                deps.append('"sqlalchemy>=2.0"')
            elif "redis" in db.lower():
                deps.append('"redis>=5.0"')
        return "[" + ", ".join(deps) + "]"

    def _slugify(self, text: str) -> str:
        import re
        import unicodedata

        # Normalize unicode, strip non-ascii, lowercase
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = text.strip("_")
        return text or "project"

    def _tree(self, path: Path, prefix: str = "", max_depth: int = 4) -> str:
        """Генерация ASCII-дерева."""
        if max_depth <= 0:
            return ""
        lines = []
        entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name))
        entries = [
            e
            for e in entries
            if e.name not in ("__pycache__", ".pytest_cache", "node_modules", ".git", "egg-info")
        ]
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                lines.append(self._tree(entry, prefix + extension, max_depth - 1))
        return "\n".join(lines)
