"""Система обратной связи — обучение на результатах."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from noema.logging import get_logger

if TYPE_CHECKING:
    from noema.core.types import Solution, Task

logger = get_logger(__name__)


@dataclass
class FeedbackEntry:
    """Запись обратной связи."""

    solution_id: str
    task_title: str
    rating: int  # 1-5
    quality_assessment: str = ""  # actual quality from user
    comments: str = ""
    used_stack: str = ""
    would_use_again: bool = True
    improvements_suggested: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class SolutionMetrics:
    """Метрики решения для анализа."""

    task_tags: list[str] = field(default_factory=list)
    quality_given: str = ""
    quality_actual: str = ""
    rating: int = 0
    confidence: float = 0.0
    generation_time_ms: float = 0.0
    code_blocks_count: int = 0
    would_reuse: bool = True


class FeedbackStore:
    """
    Хранилище обратной связи.

    Анализирует паттерны успеха/неудачи для улучшения генерации.
    """

    def __init__(self, persist_path: str | None = None) -> None:
        self.persist_path = Path(persist_path) if persist_path else Path("noema_feedback.json")
        self.entries: list[FeedbackEntry] = []
        self._metrics: list[SolutionMetrics] = []

    async def load(self) -> None:
        """Загрузка данных."""
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text(encoding="utf-8"))
                for entry_data in data.get("entries", []):
                    self.entries.append(FeedbackEntry(**entry_data))
                for m in data.get("metrics", []):
                    self._metrics.append(SolutionMetrics(**m))
                logger.info(f"Загружено {len(self.entries)} feedback entries")
            except Exception as e:
                logger.warning(f"Ошибка загрузки feedback: {e}")

    async def persist(self) -> None:
        """Сохранение данных (атомарно: tmp + rename, перезапись не рвёт файл)."""
        data = {
            "entries": [asdict(e) for e in self.entries],
            "metrics": [asdict(m) for m in self._metrics],
        }
        from noema.utils.atomic_io import atomic_write_json

        atomic_write_json(self.persist_path, data)

    async def record_feedback(
        self,
        solution: Solution,
        task: Task,
        rating: int,
        comments: str = "",
        would_use_again: bool = True,
        improvements: list[str] | None = None,
    ) -> FeedbackEntry:
        """Записать обратную связь."""
        entry = FeedbackEntry(
            solution_id=solution.id,
            task_title=task.title,
            rating=max(1, min(5, rating)),
            comments=comments,
            used_stack=solution.stack.summary(),
            would_use_again=would_use_again,
            improvements_suggested=improvements or [],
            tags=task.tags,
        )
        self.entries.append(entry)

        metrics = SolutionMetrics(
            task_tags=task.tags,
            quality_given=solution.quality.value,
            rating=entry.rating,
            confidence=solution.confidence,
            code_blocks_count=len(solution.code_blocks),
            would_reuse=would_use_again,
        )
        self._metrics.append(metrics)

        await self.persist()
        return entry

    def analyze_patterns(self) -> dict[str, Any]:
        """Анализ паттернов в обратной связи."""
        if not self.entries:
            return {"status": "no_data"}

        avg_rating = sum(e.rating for e in self.entries) / len(self.entries)
        reuse_rate = sum(1 for e in self.entries if e.would_use_again) / len(self.entries)

        # Анализ по тегам
        tag_ratings: dict[str, list[int]] = {}
        for entry in self.entries:
            for tag in entry.tags:
                tag_ratings.setdefault(tag, []).append(entry.rating)

        best_tags = {}
        worst_tags = {}
        for tag, ratings in tag_ratings.items():
            avg = sum(ratings) / len(ratings)
            if avg >= 4.0:
                best_tags[tag] = avg
            elif avg <= 2.5:
                worst_tags[tag] = avg

        # Топ улучшений
        all_improvements = []
        for entry in self.entries:
            all_improvements.extend(entry.improvements_suggested)

        improvement_freq: dict[str, int] = {}
        for imp in all_improvements:
            improvement_freq[imp] = improvement_freq.get(imp, 0) + 1

        return {
            "total_feedback": len(self.entries),
            "avg_rating": round(avg_rating, 2),
            "reuse_rate": round(reuse_rate, 2),
            "best_performing_tags": best_tags,
            "worst_performing_tags": worst_tags,
            "top_improvements": sorted(improvement_freq.items(), key=lambda x: -x[1])[:5],
            "rating_distribution": {
                str(i): sum(1 for e in self.entries if e.rating == i) for i in range(1, 6)
            },
        }

    def get_stack_recommendations(self, tags: list[str]) -> dict[str, Any]:
        """Рекомендации стека на основе обратной связи."""
        relevant = [m for m in self._metrics if any(t in m.task_tags for t in tags)]
        if not relevant:
            return {"recommendation": "no_data"}

        successful = [m for m in relevant if m.rating >= 4 and m.would_reuse]
        failed = [m for m in relevant if m.rating <= 2]

        return {
            "total_solutions": len(relevant),
            "success_rate": len(successful) / len(relevant) if relevant else 0,
            "avg_confidence_successful": (
                sum(m.confidence for m in successful) / len(successful) if successful else 0
            ),
            "recommended_approach": "proven" if len(successful) > len(failed) else "experiment",
        }

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_entries": len(self.entries),
            "total_metrics": len(self._metrics),
            "avg_rating": sum(e.rating for e in self.entries) / len(self.entries)
            if self.entries
            else 0,
        }
