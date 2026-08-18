"""Утилиты фреймворка."""

from __future__ import annotations

import hashlib
import time
from functools import wraps
from typing import TYPE_CHECKING, Any

from noema.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

log = get_logger(__name__)


def timer(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
    """Декоратор для измерения времени выполнения (async)."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.debug("timer", func=func.__qualname__, duration_ms=round(elapsed_ms, 2))

    return wrapper


def generate_id(data: str) -> str:
    """Генерация короткого ID из данных."""
    return hashlib.sha256(data.encode()).hexdigest()[:12]


def deep_merge(base: dict, override: dict) -> dict:
    """Глубокое слияние словарей."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def truncate(text: str, max_len: int = 100, suffix: str = "...") -> str:
    """Обрезка текста."""
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


def chunk_list(lst: list, size: int) -> list[list]:
    """Разбиение списка на чанки."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]
