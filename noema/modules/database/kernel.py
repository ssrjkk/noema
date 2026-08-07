"""Database Module — migrations, ORM, query optimization, schema design."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DBEngine(StrEnum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    REDIS = "redis"
    CLICKHOUSE = "clickhouse"
    DYNAMODB = "dynamodb"


class ORMType(StrEnum):
    SQLALCHEMY = "sqlalchemy"
    TORTOISE = "tortoise"
    PRISMA = "prisma"
    DRIZZLE = "drizzle"
    TYPEORM = "typeorm"
    MONGOMODEL = "mongomodel"
    DATAMAPPER = "datamapper"


@dataclass
class Column:
    name: str = ""
    col_type: str = "VARCHAR(255)"
    nullable: bool = True
    primary_key: bool = False
    unique: bool = False
    default: str = ""
    foreign_key: str = ""
    index: bool = False


@dataclass
class Table:
    name: str = ""
    columns: list[Column] = field(default_factory=list)
    indexes: list[list[str]] = field(default_factory=list)
    description: str = ""


@dataclass
class Migration:
    version: str = ""
    name: str = ""
    up_sql: str = ""
    down_sql: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class QueryOptimization:
    query: str = ""
    issue: str = ""
    suggestion: str = ""
    impact: str = ""  # high, medium, low
    fixed_query: str = ""


class SchemaDesigner:
    """Design database schemas from requirements."""

    COMMON_PATTERNS: dict[str, list[Table]] = {
        "user": [
            Table(
                name="users",
                columns=[
                    Column(name="id", col_type="UUID", primary_key=True, nullable=False),
                    Column(
                        name="email",
                        col_type="VARCHAR(255)",
                        unique=True,
                        nullable=False,
                        index=True,
                    ),
                    Column(name="username", col_type="VARCHAR(100)", unique=True),
                    Column(name="password_hash", col_type="VARCHAR(255)", nullable=False),
                    Column(name="is_active", col_type="BOOLEAN", default="true"),
                    Column(name="created_at", col_type="TIMESTAMP", default="NOW()"),
                    Column(name="updated_at", col_type="TIMESTAMP", default="NOW()"),
                ],
            ),
        ],
        "auth": [
            Table(
                name="users",
                columns=[
                    Column(name="id", col_type="UUID", primary_key=True),
                    Column(name="email", col_type="VARCHAR(255)", unique=True, nullable=False),
                    Column(name="password_hash", col_type="VARCHAR(255)", nullable=False),
                    Column(name="role", col_type="VARCHAR(50)", default="'user'"),
                ],
            ),
            Table(
                name="sessions",
                columns=[
                    Column(name="id", col_type="UUID", primary_key=True),
                    Column(name="user_id", col_type="UUID", foreign_key="users.id", index=True),
                    Column(name="token", col_type="VARCHAR(500)", unique=True, index=True),
                    Column(name="expires_at", col_type="TIMESTAMP", nullable=False),
                ],
            ),
            Table(
                name="roles",
                columns=[
                    Column(name="id", col_type="UUID", primary_key=True),
                    Column(name="name", col_type="VARCHAR(50)", unique=True),
                    Column(name="permissions", col_type="JSONB"),
                ],
            ),
        ],
        "ecommerce": [
            Table(
                name="products",
                columns=[
                    Column(name="id", col_type="UUID", primary_key=True),
                    Column(name="name", col_type="VARCHAR(255)", nullable=False),
                    Column(name="description", col_type="TEXT"),
                    Column(name="price", col_type="DECIMAL(10,2)", nullable=False),
                    Column(name="stock", col_type="INTEGER", default="0"),
                    Column(
                        name="category_id", col_type="UUID", foreign_key="categories.id", index=True
                    ),
                ],
            ),
            Table(
                name="orders",
                columns=[
                    Column(name="id", col_type="UUID", primary_key=True),
                    Column(name="user_id", col_type="UUID", foreign_key="users.id", index=True),
                    Column(name="total", col_type="DECIMAL(10,2)"),
                    Column(name="status", col_type="VARCHAR(50)", default="'pending'"),
                    Column(name="created_at", col_type="TIMESTAMP", default="NOW()"),
                ],
            ),
            Table(
                name="order_items",
                columns=[
                    Column(name="id", col_type="UUID", primary_key=True),
                    Column(name="order_id", col_type="UUID", foreign_key="orders.id", index=True),
                    Column(name="product_id", col_type="UUID", foreign_key="products.id"),
                    Column(name="quantity", col_type="INTEGER", nullable=False),
                    Column(name="price", col_type="DECIMAL(10,2)"),
                ],
            ),
            Table(
                name="categories",
                columns=[
                    Column(name="id", col_type="UUID", primary_key=True),
                    Column(name="name", col_type="VARCHAR(100)", nullable=False),
                    Column(name="parent_id", col_type="UUID", foreign_key="categories.id"),
                ],
            ),
        ],
        "blog": [
            Table(
                name="posts",
                columns=[
                    Column(name="id", col_type="UUID", primary_key=True),
                    Column(name="title", col_type="VARCHAR(255)", nullable=False),
                    Column(name="slug", col_type="VARCHAR(255)", unique=True, index=True),
                    Column(name="content", col_type="TEXT"),
                    Column(name="author_id", col_type="UUID", foreign_key="users.id", index=True),
                    Column(name="published", col_type="BOOLEAN", default="false"),
                    Column(name="created_at", col_type="TIMESTAMP", default="NOW()"),
                ],
            ),
            Table(
                name="comments",
                columns=[
                    Column(name="id", col_type="UUID", primary_key=True),
                    Column(name="post_id", col_type="UUID", foreign_key="posts.id", index=True),
                    Column(name="author_id", col_type="UUID", foreign_key="users.id"),
                    Column(name="content", col_type="TEXT", nullable=False),
                    Column(name="created_at", col_type="TIMESTAMP", default="NOW()"),
                ],
            ),
            Table(
                name="tags",
                columns=[
                    Column(name="id", col_type="UUID", primary_key=True),
                    Column(name="name", col_type="VARCHAR(50)", unique=True),
                ],
            ),
            Table(
                name="post_tags",
                columns=[
                    Column(name="post_id", col_type="UUID", foreign_key="posts.id"),
                    Column(name="tag_id", col_type="UUID", foreign_key="tags.id"),
                ],
            ),
        ],
        "messaging": [
            Table(
                name="conversations",
                columns=[
                    Column(name="id", col_type="UUID", primary_key=True),
                    Column(name="created_at", col_type="TIMESTAMP", default="NOW()"),
                ],
            ),
            Table(
                name="messages",
                columns=[
                    Column(name="id", col_type="UUID", primary_key=True),
                    Column(
                        name="conversation_id",
                        col_type="UUID",
                        foreign_key="conversations.id",
                        index=True,
                    ),
                    Column(name="sender_id", col_type="UUID", foreign_key="users.id"),
                    Column(name="content", col_type="TEXT", nullable=False),
                    Column(name="created_at", col_type="TIMESTAMP", default="NOW()"),
                ],
            ),
            Table(
                name="conversation_participants",
                columns=[
                    Column(name="conversation_id", col_type="UUID", foreign_key="conversations.id"),
                    Column(name="user_id", col_type="UUID", foreign_key="users.id"),
                ],
            ),
        ],
    }

    def design_schema(self, requirements: list[str], tags: list[str] | None = None) -> list[Table]:
        tables: list[Table] = []
        tags = tags or []

        for req in requirements:
            req_lower = req.lower()
            for pattern_name, pattern_tables in self.COMMON_PATTERNS.items():
                if pattern_name in req_lower or pattern_name in tags:
                    for table in pattern_tables:
                        if not any(t.name == table.name for t in tables):
                            tables.append(table)

        if not tables:
            tables = self._generate_generic_schema(requirements)

        return tables

    def _generate_generic_schema(self, requirements: list[str]) -> list[Table]:
        tables = []
        for _i, req in enumerate(requirements[:5]):
            name = req.lower().replace(" ", "_").replace("-", "_")[:50]
            tables.append(
                Table(
                    name=f"{name}s",
                    columns=[
                        Column(name="id", col_type="UUID", primary_key=True),
                        Column(name="name", col_type="VARCHAR(255)", nullable=False),
                        Column(name="description", col_type="TEXT"),
                        Column(name="created_at", col_type="TIMESTAMP", default="NOW()"),
                        Column(name="updated_at", col_type="TIMESTAMP", default="NOW()"),
                    ],
                )
            )
        return tables

    def to_sql(self, tables: list[Table], engine: DBEngine = DBEngine.POSTGRESQL) -> str:
        lines = []
        for table in tables:
            lines.append(f"CREATE TABLE {table.name} (")
            col_defs = []
            for col in table.columns:
                parts = [f"    {col.name} {col.col_type}"]
                if not col.nullable:
                    parts.append("NOT NULL")
                if col.primary_key:
                    parts.append("PRIMARY KEY")
                if col.unique:
                    parts.append("UNIQUE")
                if col.default:
                    parts.append(f"DEFAULT {col.default}")
                if col.foreign_key:
                    parts.append(f"REFERENCES {col.foreign_key}")
                col_defs.append(" ".join(parts))
            lines.append(",\n".join(col_defs))
            lines.append(");\n")

            for idx_cols in table.indexes:
                idx_name = f"idx_{table.name}_{'_'.join(idx_cols)}"
                lines.append(f"CREATE INDEX {idx_name} ON {table.name} ({', '.join(idx_cols)});")

        return "\n".join(lines)


class MigrationGenerator:
    """Generate database migrations."""

    def create_migration(self, name: str, tables: list[Table], version: str = "") -> Migration:
        up_lines = []
        down_lines = []
        for table in tables:
            up_lines.append(f"CREATE TABLE IF NOT EXISTS {table.name} (")
            col_defs = []
            for col in table.columns:
                parts = [f"    {col.name} {col.col_type}"]
                if not col.nullable:
                    parts.append("NOT NULL")
                if col.primary_key:
                    parts.append("PRIMARY KEY")
                col_defs.append(" ".join(parts))
            up_lines.append(",\n".join(col_defs) + ");")
            down_lines.append(f"DROP TABLE IF EXISTS {table.name};")

        return Migration(
            version=version or f"{int(time.time())}",
            name=name,
            up_sql="\n\n".join(up_lines),
            down_sql="\n\n".join(down_lines),
        )


class QueryOptimizer:
    """Analyze and optimize database queries."""

    KNOWN_ISSUES: list[tuple[str, str, str, str]] = [
        (r"SELECT \*", "Selecting all columns", "Select only needed columns", "high"),
        (
            r"WHERE.*LIKE '%",
            "Leading wildcard in LIKE",
            "Use full-text search or trigram index",
            "high",
        ),
        (
            r"ORDER BY.*LIMIT.*OFFSET \d+",
            "OFFSET-based pagination",
            "Use cursor-based pagination",
            "medium",
        ),
        (r"NOT IN\s*\(", "NOT IN subquery", "Use NOT EXISTS or LEFT JOIN ... IS NULL", "medium"),
        (r"SELECT.*FROM.*WHERE.*OR ", "OR in WHERE clause", "Consider UNION or index", "low"),
        (
            r"(?i)SELECT.*JOIN.*JOIN.*JOIN",
            "Many joins",
            "Consider materialized view or denormalization",
            "high",
        ),
    ]

    def analyze_query(self, query: str) -> list[QueryOptimization]:
        optimizations = []
        for pattern, issue, suggestion, impact in self.KNOWN_ISSUES:
            if re.search(pattern, query):
                optimizations.append(
                    QueryOptimization(
                        query=query[:200],
                        issue=issue,
                        suggestion=suggestion,
                        impact=impact,
                    )
                )
        return optimizations

    def suggest_indexes(self, queries: list[str], tables: list[Table]) -> list[dict[str, str]]:
        suggestions = []
        for query in queries:
            where_match = re.findall(r"WHERE\s+(\w+\.\w+|\w+)\s*=", query, re.IGNORECASE)
            for col_ref in where_match:
                col_name = col_ref.split(".")[-1]
                for table in tables:
                    for col in table.columns:
                        if col.name == col_name and not col.index:
                            suggestions.append(
                                {
                                    "table": table.name,
                                    "column": col_name,
                                    "reason": f"Column '{col_name}' is used in WHERE clause",
                                }
                            )
        return suggestions


class DatabaseModule:
    """Standalone database module."""

    NAME = "database"
    DESCRIPTION = "Schema design, migrations, query optimization, ORM generation"

    def __init__(self) -> None:
        self.schema_designer = SchemaDesigner()
        self.migration_gen = MigrationGenerator()
        self.query_optimizer = QueryOptimizer()

    def design_and_generate(
        self,
        requirements: list[str],
        tags: list[str] | None = None,
        engine: DBEngine = DBEngine.POSTGRESQL,
    ) -> dict[str, Any]:
        tables = self.schema_designer.design_schema(requirements, tags)
        sql = self.schema_designer.to_sql(tables, engine)
        migration = self.migration_gen.create_migration("initial", tables)
        return {
            "tables": [{"name": t.name, "columns": len(t.columns)} for t in tables],
            "sql": sql,
            "migration_up": migration.up_sql,
            "migration_down": migration.down_sql,
        }

    def execute(self, task: Any) -> dict[str, Any]:
        tags = getattr(task, "tags", [])
        description = getattr(task, "description", "") if hasattr(task, "description") else ""
        requirements = [description] if description else tags
        engine = DBEngine.POSTGRESQL
        for tag in tags:
            try:
                engine = DBEngine(tag)
                break
            except ValueError:
                continue
        result = self.design_and_generate(requirements, tags, engine)
        return {
            "type": "database",
            "engine": engine.value,
            "tables": result["tables"],
            "sql_preview": result["sql"][:500],
            "_confidence": 0.85,
        }
