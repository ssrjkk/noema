"""Graph-based knowledge — NetworkX граф связей между технологиями и паттернами."""

from __future__ import annotations

import contextlib
from typing import Any

import networkx as nx

from noema.logging import get_logger

logger = get_logger(__name__)


class KnowledgeGraph:
    """
    Граф знаний на основе NetworkX.

    Связывает технологии, паттерны, проблемы и решения
    в граф для поиска оптимальных путей.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._build_default_graph()

    def _build_default_graph(self) -> None:
        """Построение графа по умолчанию."""

        # ── Технологии ──────────────────────────────────────────────────────
        techs = {
            "python": {"type": "language", "category": "backend"},
            "typescript": {"type": "language", "category": "fullstack"},
            "go": {"type": "language", "category": "backend"},
            "rust": {"type": "language", "category": "systems"},
            "java": {"type": "language", "category": "enterprise"},
            "kotlin": {"type": "language", "category": "backend"},
            "dart": {"type": "language", "category": "mobile"},
            "c++": {"type": "language", "category": "systems"},
            "c#": {"type": "language", "category": "game"},
            "scala": {"type": "language", "category": "data"},
            "swift": {"type": "language", "category": "mobile"},
        }

        frameworks = {
            "fastapi": {"type": "framework", "category": "api"},
            "django": {"type": "framework", "category": "web"},
            "flask": {"type": "framework", "category": "web"},
            "express": {"type": "framework", "category": "api"},
            "nextjs": {"type": "framework", "category": "frontend"},
            "react": {"type": "framework", "category": "frontend"},
            "vue": {"type": "framework", "category": "frontend"},
            "svelte": {"type": "framework", "category": "frontend"},
            "gin": {"type": "framework", "category": "api"},
            "chi": {"type": "framework", "category": "api"},
            "axum": {"type": "framework", "category": "api"},
            "actix": {"type": "framework", "category": "api"},
            "spring-boot": {"type": "framework", "category": "enterprise"},
            "flutter": {"type": "framework", "category": "mobile"},
            "django-rest": {"type": "framework", "category": "api"},
            "pytorch": {"type": "framework", "category": "ml"},
            "tensorflow": {"type": "framework", "category": "ml"},
        }

        databases = {
            "postgresql": {"type": "database", "category": "relational"},
            "mysql": {"type": "database", "category": "relational"},
            "sqlite": {"type": "database", "category": "embedded"},
            "mongodb": {"type": "database", "category": "document"},
            "redis": {"type": "database", "category": "cache"},
            "elasticsearch": {"type": "database", "category": "search"},
            "cassandra": {"type": "database", "category": "wide-column"},
            "neo4j": {"type": "database", "category": "graph"},
            "dynamodb": {"type": "database", "category": "key-value"},
            "clickhouse": {"type": "database", "category": "analytics"},
            "influxdb": {"type": "database", "category": "time-series"},
        }

        infra = {
            "docker": {"type": "infrastructure", "category": "containerization"},
            "kubernetes": {"type": "infrastructure", "category": "orchestration"},
            "terraform": {"type": "infrastructure", "category": "iac"},
            "github-actions": {"type": "infrastructure", "category": "ci-cd"},
            "gitlab-ci": {"type": "infrastructure", "category": "ci-cd"},
            "prometheus": {"type": "infrastructure", "category": "monitoring"},
            "grafana": {"type": "infrastructure", "category": "monitoring"},
            "kafka": {"type": "infrastructure", "category": "messaging"},
            "rabbitmq": {"type": "infrastructure", "category": "messaging"},
            "nginx": {"type": "infrastructure", "category": "proxy"},
            "envoy": {"type": "infrastructure", "category": "proxy"},
        }

        for nodes in [techs, frameworks, databases, infra]:
            for name, attrs in nodes.items():
                self.graph.add_node(name, **attrs)

        # ── Связи: язык → фреймворк ────────────────────────────────────────
        lang_fw = [
            ("python", "fastapi"),
            ("python", "django"),
            ("python", "flask"),
            ("python", "django-rest"),
            ("python", "pytorch"),
            ("python", "tensorflow"),
            ("typescript", "express"),
            ("typescript", "nextjs"),
            ("typescript", "react"),
            ("go", "gin"),
            ("go", "chi"),
            ("rust", "axum"),
            ("rust", "actix"),
            ("java", "spring-boot"),
            ("dart", "flutter"),
            ("kotlin", "spring-boot"),
        ]
        for src, dst in lang_fw:
            self.graph.add_edge(src, dst, relationship="supports")

        # ── Фреймворк → БД ─────────────────────────────────────────────────
        fw_db = [
            ("fastapi", "postgresql"),
            ("fastapi", "redis"),
            ("fastapi", "mongodb"),
            ("django", "postgresql"),
            ("django", "sqlite"),
            ("django", "redis"),
            ("express", "mongodb"),
            ("express", "postgresql"),
            ("express", "redis"),
            ("spring-boot", "postgresql"),
            ("spring-boot", "mysql"),
            ("spring-boot", "redis"),
            ("axum", "postgresql"),
            ("axum", "redis"),
        ]
        for src, dst in fw_db:
            self.graph.add_edge(src, dst, relationship="integrates_with")

        # ── БД → инфра ─────────────────────────────────────────────────────
        db_infra = [
            ("redis", "kubernetes"),
            ("postgresql", "kubernetes"),
            ("kafka", "kubernetes"),
            ("elasticsearch", "kubernetes"),
        ]
        for src, dst in db_infra:
            self.graph.add_edge(src, dst, relationship="deploys_on")

        # ── Фреймворк → инфра ──────────────────────────────────────────────
        fw_infra = [
            ("fastapi", "docker"),
            ("fastapi", "kubernetes"),
            ("nextjs", "docker"),
            ("express", "docker"),
            ("gin", "docker"),
            ("axum", "docker"),
        ]
        for src, dst in fw_infra:
            self.graph.add_edge(src, dst, relationship="containerized_by")

        # ── Проблемы и решения ──────────────────────────────────────────────
        problems = {
            "high-load": {"type": "problem", "category": "scalability"},
            "real-time": {"type": "problem", "category": "latency"},
            "data-intensive": {"type": "problem", "category": "throughput"},
            "security-critical": {"type": "problem", "category": "security"},
            "ml-inference": {"type": "problem", "category": "ml-serving"},
        }
        for name, attrs in problems.items():
            self.graph.add_node(name, **attrs)

        problem_solutions = [
            ("high-load", "kafka", {"solution": "event-driven decoupling"}),
            ("high-load", "redis", {"solution": "caching layer"}),
            ("high-load", "kubernetes", {"solution": "horizontal scaling"}),
            ("real-time", "redis", {"solution": "pub/sub"}),
            ("real-time", "kafka", {"solution": "streaming"}),
            ("data-intensive", "clickhouse", {"solution": "analytics engine"}),
            ("data-intensive", "cassandra", {"solution": "distributed storage"}),
            ("security-critical", "postgresql", {"solution": "ACID compliance"}),
            ("ml-inference", "pytorch", {"solution": "model serving"}),
            ("ml-inference", "redis", {"solution": "feature store cache"}),
        ]
        for problem, solution, attrs in problem_solutions:
            self.graph.add_edge(problem, solution, **attrs)

    def find_optimal_stack(
        self,
        requirements: list[str],
        max_depth: int = 3,
    ) -> list[list[str]]:
        """Найти оптимальный путь стека по требованиям."""
        candidates = []
        for req in requirements:
            req_lower = req.lower()
            if req_lower in self.graph:
                # BFS от проблемы к решениям
                paths = []
                for target in self.graph.nodes():
                    if self.graph.nodes[target].get("type") in ("framework", "database"):
                        path: list[str] | None = None
                        with contextlib.suppress(nx.NetworkXNoPath, nx.NodeNotFound):
                            path = nx.shortest_path(self.graph, req_lower, target)
                        if path and len(path) <= max_depth + 1:
                            paths.append(path)
                candidates.extend(paths)

        # Сортировка по длине (короткий путь = лучше)
        candidates.sort(key=len)
        return candidates[:5]

    def get_compatible_technologies(self, tech: str) -> dict[str, list[str]]:
        """Получить совместимые технологии."""
        if tech.lower() not in self.graph:
            return {}

        node = tech.lower()
        result: dict[str, list[str]] = {
            "supports": [],
            "integrates_with": [],
            "containerized_by": [],
        }

        for _, target, data in self.graph.out_edges(node, data=True):
            rel = data.get("relationship", "related")
            if rel in result:
                result[rel].append(target)
            else:
                result.setdefault("related", []).append(target)

        return result

    def suggest_architecture(self, tags: list[str]) -> dict[str, Any]:
        """Предложить архитектуру на основе тегов."""
        tag_nodes = [t.lower() for t in tags if t.lower() in self.graph]

        components = []
        for node in tag_nodes:
            compatible = self.get_compatible_technologies(node)
            for category, techs in compatible.items():
                for tech in techs[:3]:
                    components.append(
                        {
                            "from": node,
                            "to": tech,
                            "relationship": category,
                            "confidence": 0.7,
                        }
                    )

        # Кластеризация по типам
        by_type: dict[str, list[str]] = {}
        for node in tag_nodes:
            ntype = self.graph.nodes[node].get("type", "unknown")
            by_type.setdefault(ntype, []).append(node)

        return {
            "input_techs": tag_nodes,
            "components": components,
            "clusters": by_type,
            "total_suggestions": len(components),
        }

    def get_stats(self) -> dict[str, Any]:
        types: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("type", "unknown")
            types[t] = types.get(t, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": types,
            "density": nx.density(self.graph),
        }
