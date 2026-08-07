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
