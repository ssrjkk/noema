# Getting Started

## Installation

```bash
# Clone
git clone https://github.com/anomalyco/noema
cd noema

# Basic install
pip install -e .

# Full install with all extras
pip install -e ".[dev,db,full,sentry]"

# Or use make
make install-full
```

## Configuration

Copy `compose.env` and adjust:

```bash
cp compose.env .env
# Edit .env with your settings
```

## Running

```bash
# Start API server
uvicorn noema.api.server:app --reload

# Or with Docker
make docker-run
```

## Your First Task

```python
from noema import NoemaEngine, Task

noema = NoemaEngine()
await noema.initialize()

solution, thought = await noema.think(Task(
    title="Build a REST API",
    description="Design a FastAPI-based user management API",
    tags=["api", "python"],
))

print(f"Quality: {solution.quality.value}")
print(f"Confidence: {solution.confidence:.0%}")
```
