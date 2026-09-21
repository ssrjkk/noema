# Getting Started

## Installation

```bash
# Clone
git clone https://github.com/ssrjkk/noema
cd noema

# Basic install (достаточно для демо и CLI)
pip install -e .

# Full install — все опции: верификация Z3, векторный поиск, провайдеры LLM, БД
pip install -e ".[dev,db,full,sentry]"

# Или через make (Unix / Git Bash / WSL)
make install-full
```

### Verify installation

```bash
# Вариант 1: через установленную команду CLI
noema --help

# Вариант 2: через python -m (всегда работает, даже если скрипт не в PATH)
python -m noema --help

# Запустить живую демонстрацию всех 12 модулей (без ключей LLM — sandbox mode)
python demo.py
```

## Configuration

Скопируйте пример `.env`-файла и поправьте значения:

```bash
# Unix / Linux / macOS / Git Bash
cp compose.env.example compose.env
# Windows (PowerShell / CMD)
copy compose.env.example compose.env

# Правите compose.env: задайте POSTGRES_PASSWORD, REDIS_PASSWORD, API-ключи LLM
```

Для локальной разработки без Docker / БД env-файл **не обязателен** — работает fallback-провайдер и in-memory хранилища.

## Running

```bash
# Вариант 1: CLI — самый простой способ
noema serve
# или
python -m noema serve
# http://localhost:8000

# Вариант 2: uvicorn напрямую
uvicorn noema.api.server:app --reload

# Вариант 3: Docker Compose (требуется compose.env)
make docker-run
```

## Your First Task

Сохраните как `first_run.py` и запустите `python first_run.py`:

```python
import asyncio
from noema import NoemaEngine, Task


async def main():
    noema = NoemaEngine()
    await noema.initialize()

    solution, thought = await noema.think(
        Task(
            title="Build a REST API",
            description="Design a FastAPI-based user management API "
                        "with registration, login and RBAC roles.",
            tags=["api", "python", "fastapi", "auth"],
        )
    )

    print(f"Solution ID : {solution.id}")
    print(f"Quality     : {solution.quality.value}")
    print(f"Confidence  : {solution.confidence:.0%}")
    print(f"Code blocks : {len(solution.code_blocks)}")
    print(f"Thought steps: {len(thought.steps)}")
    if solution.summary:
        print(f"Summary     : {solution.summary[:200]}...")


if __name__ == "__main__":
    asyncio.run(main())
```
