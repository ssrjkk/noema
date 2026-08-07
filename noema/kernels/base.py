"""Базовый класс ядра — интерфейс для всех kernels."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, cast

from noema.logging import get_logger

if TYPE_CHECKING:
    from noema.core.types import Task, TechStack

logger = get_logger(__name__)


class BaseKernel(abc.ABC):
    """
    Абстрактное ядро (kernel).

    Каждое ядро отвечает за свою область генерации решений:
    - Архитектура
    - Генерация кода
    - Оптимизация
    - Безопасность
    - Анализ
    и т.д.
    """

    def __init__(self, knowledge: Any = None, **kwargs: Any) -> None:
        self.knowledge = knowledge
        self.config = kwargs
        self._hooks: dict[str, list] = {}

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """РРјСЏ СЏРґСЂР°."""
        ...

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Описание ядра."""
        ...

    @abc.abstractmethod
    async def execute(self, task: Task, **kwargs: Any) -> dict[str, Any]:
        """Основной метод выполнения ядра."""
        ...

    async def execute_subtask(
        self, subtask: dict, stack: TechStack | None = None
    ) -> dict[str, Any]:
        """Выполнение подзадачи (для параллельной обработки)."""
        return {"subtask": subtask, "status": "processed"}

    def on(self, event: str, callback: Any) -> None:
        """Регистрация хука на событие."""
        self._hooks.setdefault(event, []).append(callback)

    async def _emit(self, event: str, data: Any) -> None:
        """Эмиссия события."""
        for cb in self._hooks.get(event, []):
            if callable(cb):
                result = cb(data)
                if hasattr(result, "__await__"):
                    await result

    async def _query_knowledge(self, query: str, top_k: int = 5) -> list[dict]:
        """Запрос к базе знаний."""
        if self.knowledge:
            return cast("list[dict]", await self.knowledge.search(query, top_k=top_k))
        return []

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
