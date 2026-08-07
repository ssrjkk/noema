"""Ядро Frontend — генерация клиентских решений."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from noema.kernels.base import BaseKernel
from noema.logging import get_logger

if TYPE_CHECKING:
    from noema.core.types import Task

logger = get_logger(__name__)


class FrontendKernel(BaseKernel):
    """Ядро frontend-генерации."""

    @property
    def name(self) -> str:
        return "frontend"

    @property
    def description(self) -> str:
        return "Генерация frontend решений: React, Vue, Svelte, Next.js, Tailwind"

    async def execute(self, task: Task, **kwargs) -> dict[str, Any]:
        tags = {t.lower() for t in task.tags}

        framework = self._select_framework(tags)
        styling = self._select_styling(tags)
        state = self._select_state_management(tags)
        components = self._design_components(task, tags)
        project_structure = self._design_structure(framework)
        performance = self._performance_strategies(tags)

        return {
            "type": "frontend",
            "framework": framework,
            "styling": styling,
            "state_management": state,
            "components": components,
            "project_structure": project_structure,
            "performance": performance,
            "_confidence": 0.74,
        }

    def _select_framework(self, tags: set[str]) -> dict[str, str]:
        if "nextjs" in tags or "next" in tags or "ssr" in tags:
            return {"name": "Next.js 14", "approach": "App Router + RSC", "lang": "TypeScript"}
        if "vue" in tags:
            return {
                "name": "Vue 3",
                "approach": "Composition API + script setup",
                "lang": "TypeScript",
            }
        if "svelte" in tags or "sveltekit" in tags:
            return {"name": "SvelteKit", "approach": "File-based routing", "lang": "TypeScript"}
        if "solid" in tags:
            return {"name": "SolidJS", "approach": "Signals", "lang": "TypeScript"}
        return {"name": "React 18", "approach": "App Router + Hooks", "lang": "TypeScript"}

    def _select_styling(self, tags: set[str]) -> dict[str, str]:
        if "tailwind" in tags or "utility" in tags:
            return {"name": "Tailwind CSS", "approach": "utility-first"}
        if "css-modules" in tags:
            return {"name": "CSS Modules", "approach": "scoped styles"}
        if "styled-components" in tags or "css-in-js" in tags:
            return {"name": "styled-components", "approach": "CSS-in-JS"}
        if "chakra" in tags:
            return {"name": "Chakra UI", "approach": "component library"}
        if "shadcn" in tags:
            return {"name": "shadcn/ui + Tailwind", "approach": "copy-paste components"}
        return {"name": "Tailwind CSS + shadcn/ui", "approach": "utility-first + components"}

    def _select_state_management(self, tags: set[str]) -> dict[str, str]:
        if "zustand" in tags:
            return {"name": "Zustand", "approach": "minimal store"}
        if "redux" in tags:
            return {"name": "Redux Toolkit", "approach": "normalized state"}
        if "jotai" in tags or "atomic" in tags:
            return {"name": "Jotai", "approach": "atomic state"}
        if "server-state" in tags or "tanstack" in tags:
            return {"name": "TanStack Query + Zustand", "approach": "server + client state"}
        return {"name": "Zustand + TanStack Query", "approach": "client + server state"}

    def _design_components(self, task: Task, tags: set[str]) -> list[dict[str, Any]]:
        base = [
            {
                "name": "AppShell",
                "type": "layout",
                "description": "Основной layout с навигацией",
            },
            {
                "name": "ThemeProvider",
                "type": "provider",
                "description": "Провайдер темы (dark/light)",
            },
        ]

        if "web" in tags or "dashboard" in tags:
            base.extend(
                [
                    {
                        "name": "Sidebar",
                        "type": "navigation",
                        "description": "Боковая навигация",
                    },
                    {
                        "name": "Header",
                        "type": "navigation",
                        "description": "Верхняя панель",
                    },
                    {
                        "name": "DataTable",
                        "type": "data-display",
                        "description": "Таблица с сортировкой/фильтрацией",
                    },
                    {
                        "name": "Charts",
                        "type": "data-visualization",
                        "description": "Графики и диаграммы",
                    },
                ]
            )

        if "auth" in tags or "user" in tags:
            base.extend(
                [
                    {"name": "LoginForm", "type": "form", "description": "Форма входа"},
                    {
                        "name": "RegisterForm",
                        "type": "form",
                        "description": "Форма регистрации",
                    },
                    {
                        "name": "ProfilePage",
                        "type": "page",
                        "description": "Профиль пользователя",
                    },
                ]
            )

        if "ecommerce" in tags or "shop" in tags:
            base.extend(
                [
                    {
                        "name": "ProductCard",
                        "type": "card",
                        "description": "Карточка товара",
                    },
                    {"name": "Cart", "type": "widget", "description": "Корзина"},
                    {
                        "name": "Checkout",
                        "type": "page",
                        "description": "Оформление заказа",
                    },
                ]
            )

        base.append(
            {
                "name": "ErrorBoundary",
                "type": "error-handling",
                "description": "Обработка ошибок",
            }
        )
        return base

    def _design_structure(self, framework: dict) -> dict[str, list[str]]:
        name = framework.get("name", "React")
        if "Next.js" in name:
            return {
                "app/": ["layout.tsx", "page.tsx", "loading.tsx", "error.tsx"],
                "app/(auth)/": ["login/page.tsx", "register/page.tsx"],
                "app/(dashboard)/": ["page.tsx", "settings/page.tsx"],
                "components/": ["ui/", "layout/", "forms/"],
                "lib/": ["utils.ts", "api.ts", "auth.ts"],
                "hooks/": ["useAuth.ts", "useApi.ts"],
                "stores/": ["authStore.ts"],
            }
        return {
            "src/": ["App.tsx", "main.tsx"],
            "src/components/": ["ui/", "layout/"],
            "src/pages/": ["Home.tsx", "Dashboard.tsx"],
            "src/hooks/": [],
            "src/stores/": [],
            "src/lib/": ["api.ts", "utils.ts"],
        }

    def _performance_strategies(self, tags: set[str]) -> list[dict]:
        strategies = [
            {
                "strategy": "Code Splitting",
                "description": "Lazy loading routes and heavy components",
            },
            {
                "strategy": "Image Optimization",
                "description": "next/image or lazy loading with blur placeholder",
            },
            {"strategy": "Font Optimization", "description": "next/font for zero layout shift"},
            {
                "strategy": "Bundle Analysis",
                "description": "next/bundle-analyzer or webpack-bundle-analyzer",
            },
        ]
        if "seo" in tags:
            strategies.append(
                {"strategy": "SSR/SSG", "description": "Server-side rendering for SEO"}
            )
        if "pwa" in tags:
            strategies.append(
                {"strategy": "PWA", "description": "Service worker + offline support"}
            )
        return strategies
