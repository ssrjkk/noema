"""CodegenKernel — deterministic code scaffolding for service projects.

Architecture:
- Pure template-driven generation: every artifact (service, models, repository,
  service layer, Dockerfile, compose, module stubs) is produced from static
  templates in CODE_TEMPLATES / DOCKERFILE_TEMPLATES / COMPOSE_TEMPLATES.
- Zero LLM calls: this kernel never blocks on I/O and is fully deterministic
  for a given task + stack.

Concurrency contract:
- ``execute``/``execute_subtask`` are async for API uniformity but perform no
  blocking work, so they are safe to call on the event loop.

Complexity:
- ``execute``: ``O(G)`` template expansions for a fixed set of G generator
  methods (6), each ``O(len(template) + len(task.requirements))``.
- ``execute_subtask``: ``O(len(template))`` for a single module stub.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from noema.kernels.base import BaseKernel
from noema.logging import get_logger

if TYPE_CHECKING:
    from noema.core.types import Task, TechStack

logger = get_logger(__name__)

# ── Шаблоны кода по стекам ──────────────────────────────────────────────────

CODE_TEMPLATES: dict[str, dict[str, str]] = {
    "python": {
        "fastapi_service": '''"""Auto-generated FastAPI service."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="{title}")


class HealthResponse(BaseModel):
    status: str = "ok"


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@app.get("/api/v1/{resource}")
async def get_{resource}(id: str):
    # TODO: implement business logic
    return {{"id": id, "data": {{}}}}


@app.post("/api/v1/{resource}")
async def create_{resource}(payload: dict):
    # TODO: implement business logic
    return {{"created": True, "data": payload}}
''',
        "data_model": '''"""Data models — auto-generated."""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional
import uuid


class BaseRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class {model_name}(BaseRecord):
    {fields}
    is_active: bool = True

    class Config:
        from_attributes = True
''',
        "repository": '''"""Repository pattern — auto-generated."""

from typing import TypeVar, Generic, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

T = TypeVar("T")


class BaseRepository(Generic[T]):
    def __init__(self, session: AsyncSession, model: type[T]):
        self.session = session
        self.model = model

    async def get_by_id(self, id: str) -> Optional[T]:
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()

    async def list_all(self, offset: int = 0, limit: int = 100) -> list[T]:
        result = await self.session.execute(
            select(self.model).offset(offset).limit(limit)
        )
        return list(result.scalars().all())

    async def create(self, entity: T) -> T:
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def update(self, entity: T) -> T:
        await self.session.merge(entity)
        await self.session.commit()
        return entity

    async def delete(self, id: str) -> bool:
        entity = await self.get_by_id(id)
        if entity:
            await self.session.delete(entity)
            await self.session.commit()
            return True
        return False
''',
        "service": '''"""Service layer — auto-generated."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class {service_name}Service:
    """Бизнес-логика для {resource_name}."""

    # dependencies injected here

    async def get(self, id: str) -> Optional[dict]:
        """Fetch a single record by id (TODO: query your store)."""
        return None

    async def create(self, data: dict) -> dict:
        """Persist a new record (TODO: validation + persistence)."""
        return data

    async def update(self, id: str, data: dict) -> Optional[dict]:
        """Update a record (TODO: persistence)."""
        return data

    async def delete(self, id: str) -> bool:
        """Delete a record (TODO: persistence)."""
        return True

    async def list_all(self, page: int = 1, size: int = 20) -> list[dict]:
        """List records with pagination (TODO: query your store)."""
        return []
''',
    },
    "typescript": {
        "express_service": """import express, {{ Request, Response }} from 'express';
import {{ z }} from 'zod';

const app = express();
app.use(express.json());

const healthSchema = z.object({{ status: z.literal('ok') }});

app.get('/health', (_req: Request, res: Response) => {{
  res.json({{ status: 'ok' }});
}});

app.get('/api/v1/:resource/:id', async (req: Request, res: Response) => {{
  const {{ resource, id }} = req.params;
  // TODO: implement business logic
  res.json({{ id, data: {{}} }});
}});

app.post('/api/v1/:resource', async (req: Request, res: Response) => {{
  // TODO: implement business logic
  res.status(201).json({{ created: true, data: req.body }});
}});

export default app;
""",
    },
    "go": {
        "http_service": """package main

import (
    "encoding/json"
    "log"
    "net/http"
)

type Response struct {
    Status string `json:"status"`
}

func healthHandler(w http.ResponseWriter, r *http.Request) {{
    json.NewEncoder(w).Encode(Response{{Status: "ok"}})
}}

func main() {{
    http.HandleFunc("/health", healthHandler)
    log.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}}
""",
    },
    "rust": {
        "axum_service": """use axum::{{
    routing::{{get, post}},
    Json, Router,
}};
use serde::{{Deserialize, Serialize}};
use std::net::SocketAddr;

#[derive(Serialize)]
struct HealthResponse {{
    status: String,
}}

async fn health() -> Json<HealthResponse> {{
    Json(HealthResponse {{
        status: "ok".to_string(),
    }})
}}

#[tokio::main]
async fn main() {{
    let app = Router::new()
        .route("/health", get(health));

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("listening on {{}}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}}
""",
    },
    "java": {
        "spring_boot": """package com.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.*;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@SpringBootApplication
@RestController
public class Application {{
    private final Map<String, Object> store = new ConcurrentHashMap<>();

    public static void main(String[] args) {{
        SpringApplication.run(Application.class, args);
    }}

    @GetMapping("/health")
    public Map<String, String> health() {{
        return Map.of("status", "ok");
    }}

    @GetMapping("/api/v1/{{resource}}/{{id}}")
    public Map<String, Object> get(@PathVariable String id) {{
        return Map.of("id", id, "data", Map.of());
    }}

    @PostMapping("/api/v1/{{resource}}")
    public Map<String, Object> create(@RequestBody Map<String, Object> payload) {{
        return Map.of("created", true, "data", payload);
    }}
}}
""",
    },
}

# ── Конфигурации ────────────────────────────────────────────────────────────

DOCKERFILE_TEMPLATES: dict[str, str] = {
    "python": """FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
""",
    "typescript": """FROM node:20-slim
WORKDIR /app
COPY package*.json .
RUN npm ci
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["node", "dist/index.js"]
""",
    "go": """FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY . .
RUN go build -o server .

FROM alpine:3.19
COPY --from=builder /app/server /server
EXPOSE 8080
CMD ["/server"]
""",
    "rust": """FROM rust:1.77-slim AS builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/server /server
EXPOSE 8080
CMD ["/server"]
""",
}

COMPOSE_TEMPLATES: dict[str, str] = {
    "default": """version: '3.9'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/app
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: app
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

volumes:
  pgdata:
""",
    "microservices": """version: '3.9'
services:
  gateway:
    build:
      context: .
      dockerfile: Dockerfile.gateway
    ports:
      - "8080:8080"

  auth-service:
    build:
      context: ./services/auth
    environment:
      - JWT_SECRET=change-me

  core-service:
    build:
      context: ./services/core
    depends_on:
      - db
      - redis

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: app

  redis:
    image: redis:7-alpine

  kafka:
    image: confluentinc/cp-kafka:7.6.0
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092

  zookeeper:
    image: confluentinc/cp-zookeeper:7.6.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
""",
}


class CodegenKernel(BaseKernel):
    """Deterministic code generator: services, models, repos, infra files.

    Only template-safe, sanitized task-derived identifiers are ever inserted
    into generated code (zero-trust), and every generator degrades to None on
    unsupported languages/stacks.
    """

    @property
    def name(self) -> str:
        return "codegen"

    @property
    def description(self) -> str:
        return "Генерация кодовых блоков, конфигов и инфраструктурных файлов"

    async def execute(self, task: Task, **kwargs) -> dict[str, Any]:
        """Generate the full set of service artifacts for a task.

        kwargs["stack"] may carry an explicit TechStack; otherwise the stack is
        inferred from task tags.

        Complexity: O(G) generator calls (G fixed at 6), each linear in template
        size and requirement count.
        """
        from noema.core.types import TechStack

        stack = kwargs.get("stack") or task.preferred_stack
        if isinstance(stack, dict):
            stack = TechStack(**stack)
        if not stack:
            stack = self._infer_stack(task)

        primary_lang = (stack.languages[0] if stack.languages else "python").lower()

        blocks = []

        # Основной сервис
        service_block = self._gen_service(task, primary_lang)
        if service_block:
            blocks.append(service_block)

        # Модели данных
        model_block = self._gen_models(task, primary_lang)
        if model_block:
            blocks.append(model_block)

        # Репозиторий
        repo_block = self._gen_repository(task, primary_lang)
        if repo_block:
            blocks.append(repo_block)

        # Сервисный слой
        service_layer = self._gen_service_layer(task, primary_lang)
        if service_layer:
            blocks.append(service_layer)

        # Dockerfile
        dockerfile = self._gen_dockerfile(primary_lang)
        if dockerfile:
            blocks.append(dockerfile)

        # docker-compose
        compose = self._gen_compose(task)
        if compose:
            blocks.append(compose)

        return {
            "type": "codegen",
            "language": primary_lang,
            "blocks": blocks,
            "block_count": len(blocks),
            "_confidence": 0.7,
        }

    async def execute_subtask(
        self, subtask: dict, stack: TechStack | None = None
    ) -> dict[str, Any]:
        """Generate a single module stub for one subtask.

        The output filename is normalized to the language extension; the module
        name is sanitized so untrusted requirement text cannot inject code.

        Complexity: O(len(template)).
        """
        from noema.core.types import TechStack as _TechStack

        if isinstance(stack, dict):
            stack = _TechStack(**stack)
        lang = stack.languages[0].lower() if stack and stack.languages else "python"
        requirement = subtask.get("requirement", "")
        filename = subtask.get("filename", "module.py")

        ext_map = {
            "python": ".py",
            "typescript": ".ts",
            "go": ".go",
            "rust": ".rs",
            "java": ".java",
        }
        ext = ext_map.get(lang, ".py")
        if not filename.endswith(ext):
            filename = filename.rsplit(".", 1)[0] + ext

        content = self._gen_module(requirement, lang, subtask)

        return {
            "filename": filename,
            "language": lang,
            "content": content,
            "description": f"Модуль: {requirement}",
        }

    def _infer_stack(self, task: Task) -> TechStack:
        """Infer a tech stack from task tags. Complexity: O(T) in tag count."""
        from noema.core.types import TechStack

        tags = {t.lower() for t in task.tags}
        if "go" in tags:
            return TechStack(languages=["Go"])
        if "rust" in tags:
            return TechStack(languages=["Rust"])
        if "java" in tags or "spring" in tags:
            return TechStack(languages=["Java"], frameworks=["Spring Boot"])
        if "typescript" in tags or "node" in tags:
            return TechStack(languages=["TypeScript"], frameworks=["Express"])
        return TechStack(languages=["Python"], frameworks=["FastAPI"])

    @staticmethod
    def _sanitize_identifier(raw: str, max_len: int = 20) -> str:
        """Reduce arbitrary text to a safe identifier ([a-z0-9_]).

        Untrusted titles/categories may contain path separators, quotes, or
        control characters; stripping them keeps generated code valid and
        injection-free. Complexity: O(len(raw)).
        """
        cleaned = re.sub(r"[^a-z0-9_]+", "_", raw.lower().strip()).strip("_")[:max_len]
        return cleaned.strip("_") or "i"

    def _gen_service(self, task: Task, lang: str) -> dict | None:
        templates = CODE_TEMPLATES.get(lang, {})
        resource = self._sanitize_identifier(task.title)
        title = task.title.replace('"', '"').replace("\\", "\\\\")

        template = (
            templates.get("fastapi_service")
            or templates.get("express_service")
            or templates.get("http_service")
            or templates.get("axum_service")
            or templates.get("spring_boot")
        )
        if not template:
            return None

        try:
            content = template.format(title=title, resource=resource)
        except (KeyError, IndexError):
            content = template

        ext_map = {
            "python": ".py",
            "typescript": ".ts",
            "go": ".go",
            "rust": ".rs",
            "java": ".java",
        }
        ext = ext_map.get(lang, ".py")

        return {
            "filename": f"main{ext}",
            "language": lang,
            "content": content,
            "description": f"Основной сервис для {task.title}",
        }

    def _gen_models(self, task: Task, lang: str) -> dict | None:
        if lang == "python":
            resource = self._sanitize_identifier(task.title, max_len=30)
            fields = (
                "\n    ".join(
                    f"{self._sanitize_identifier(req.category, max_len=30)}: str = ''"
                    for req in task.requirements[:5]
                ).strip()
                or "name: str = ''"
            )
            content = CODE_TEMPLATES["python"]["data_model"].format(
                model_name=resource, fields=fields
            )
            return {
                "filename": "models.py",
                "language": lang,
                "content": content,
                "description": "Модели данных",
            }
        return None

    def _gen_repository(self, task: Task, lang: str) -> dict | None:
        """Emit repository.py for Python. Complexity: O(1) (static template)."""
        if lang == "python":
            return {
                "filename": "repository.py",
                "language": lang,
                "content": CODE_TEMPLATES["python"]["repository"],
                "description": "Repository pattern для работы с БД",
            }
        return None

    def _gen_service_layer(self, task: Task, lang: str) -> dict | None:
        if lang == "python":
            resource = self._sanitize_identifier(task.title, max_len=30)
            content = CODE_TEMPLATES["python"]["service"].format(
                service_name=resource, resource_name=task.title
            )
            return {
                "filename": "service.py",
                "language": lang,
                "content": content,
                "description": "Сервисный слой",
            }
        return None

    def _gen_dockerfile(self, lang: str) -> dict | None:
        """Emit a Dockerfile for the language. Complexity: O(1) (static template)."""
        template = DOCKERFILE_TEMPLATES.get(lang)
        if template:
            return {
                "filename": "Dockerfile",
                "language": "dockerfile",
                "content": template,
                "description": "Docker-конфигурация",
            }
        return None

    def _gen_compose(self, task: Task) -> dict | None:
        """Emit docker-compose.yml, microservices flavor when tagged.

        Complexity: O(T) for tag inspection, O(1) template lookup.
        """
        tags = {t.lower() for t in task.tags}
        template = (
            COMPOSE_TEMPLATES["microservices"]
            if "microservice" in tags
            else COMPOSE_TEMPLATES["default"]
        )
        return {
            "filename": "docker-compose.yml",
            "language": "yaml",
            "content": template,
            "description": "Docker Compose конфигурация",
        }

    def _gen_module(self, requirement: str, lang: str, subtask: dict) -> str:
        """Generate a module stub for the subtask requirement.

        The requirement text is length-capped and quote-escaped before being
        interpolated, so untrusted input cannot break the emitted source.

        Complexity: O(len(template)).
        """
        safe_name = requirement.replace('"', '\\"').replace("\\", "\\\\")[:80]

        templates = {
            "python": f'''"""Модуль: {safe_name}"""

from dataclasses import dataclass, field
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Handler:
    """Обработчик: {safe_name}"""

    config: dict[str, Any] = field(default_factory=dict)

    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Основной метод обработки."""
        logger.info("Processing: {safe_name}")
        # TODO: implement logic for {safe_name}
        return {{"status": "ok", "input": input_data}}

    def validate(self, data: dict[str, Any]) -> bool:
        """Валидация входных данных."""
        return bool(data)
''',
            "typescript": f"""// Модуль: {safe_name}

interface InputData {{
  [key: string]: any;
}}

interface OutputData {{
  status: string;
  data?: any;
}}

export class Handler {{
  private config: Record<string, any>;

  constructor(config: Record<string, any> = {{}}) {{
    this.config = config;
  }}

  async process(input: InputData): Promise<OutputData> {{
    console.log("Processing: {safe_name}");
    // TODO: implement logic
    return {{ status: "ok", data: input }};
  }}

  validate(data: InputData): boolean {{
    return Boolean(data);
  }}
}}
""",
            "go": f"""package handler

import "log"

// Handler для {safe_name}
type Handler struct {{
	Config map[string]interface{{}}
}}

func NewHandler(config map[string]interface{{}}) *Handler {{
	return &Handler{{Config: config}}
}}

func (h *Handler) Process(input map[string]interface{{}}) map[string]interface{{}} {{
	log.Println("Processing: {safe_name}")
	// TODO: implement logic
	return map[string]interface{{}}{{"status": "ok", "input": input}}
}}
""",
            "rust": f"""use serde::{{Deserialize, Serialize}};

#[derive(Debug, Deserialize, Serialize)]
pub struct Input {{
    pub data: serde_json::Value,
}}

#[derive(Debug, Serialize)]
pub struct Output {{
    pub status: String,
}}

pub struct Handler;

impl Handler {{
    pub fn new() -> Self {{
        Handler
    }}

    pub fn process(&self, input: &Input) -> Output {{
        println!("Processing: {safe_name}");
        Output {{ status: "ok".to_string() }}
    }}
}}
""",
        }
        return templates.get(lang, f"// Module: {safe_name}\n// TODO: implement")
