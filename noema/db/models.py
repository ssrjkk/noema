"""SQLAlchemy models for persistent storage."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB

from noema.db.engine import Base


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> datetime:
    return datetime.now(UTC)


# ─── Episodic Memory ────────────────────────────────────────────────────


class EpisodicMemoryRow(Base):
    __tablename__ = "episodic_memory"

    id = Column(String(12), primary_key=True, default=_uuid)
    timestamp = Column(Float, default=time.time, index=True)
    task_description = Column(Text, default="")
    solution_summary = Column(Text, default="")
    tech_stack = Column(String(500), default="")
    outcome = Column(String(20), default="", index=True)
    duration_seconds = Column(Float, default=0.0)
    error_message = Column(Text, default="")
    tags = Column(JSONB, default=list)
    context = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=_now)

    __table_args__ = (Index("ix_episodic_outcome_ts", "outcome", "timestamp"),)


# ─── Semantic Memory ───────────────────────────────────────────────────


class SemanticMemoryRow(Base):
    __tablename__ = "semantic_memory"

    id = Column(String(12), primary_key=True, default=_uuid)
    topic = Column(String(200), default="", index=True)
    fact = Column(Text, default="")
    confidence = Column(Float, default=0.0)
    source = Column(String(500), default="")
    use_count = Column(Integer, default=0)
    last_used = Column(Float, default=time.time)
    tags = Column(JSONB, default=list)
    created_at = Column(DateTime, default=_now)

    __table_args__ = (Index("ix_semantic_topic_confidence", "topic", "confidence"),)


# ─── Procedural Memory ────────────────────────────────────────────────


class ProceduralMemoryRow(Base):
    __tablename__ = "procedural_memory"

    id = Column(String(12), primary_key=True, default=_uuid)
    procedure_name = Column(String(200), default="", index=True)
    steps = Column(JSONB, default=list)
    success_rate = Column(Float, default=1.0)
    times_applied = Column(Integer, default=0)
    times_succeeded = Column(Integer, default=0)
    avg_duration = Column(Float, default=0.0)
    prerequisites = Column(JSONB, default=list)
    tags = Column(JSONB, default=list)
    created_at = Column(DateTime, default=_now)


# ─── Knowledge Entry ───────────────────────────────────────────────────


class KnowledgeEntryRow(Base):
    __tablename__ = "knowledge_entries"

    id = Column(String(12), primary_key=True, default=_uuid)
    title = Column(String(500), default="", index=True)
    content = Column(Text, default="")
    category = Column(String(100), default="", index=True)
    tags = Column(JSONB, default=list)
    source = Column(String(500), default="")
    confidence = Column(Float, default=0.8)
    use_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)


# ─── Feedback ──────────────────────────────────────────────────────────


class FeedbackRow(Base):
    __tablename__ = "feedback"

    id = Column(String(12), primary_key=True, default=_uuid)
    solution_id = Column(String(12), index=True, default="")
    rating = Column(Integer, default=0)  # 1-5
    comment = Column(Text, default="")
    tags = Column(JSONB, default=list)
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime, default=_now)


# ─── Evolution Log ─────────────────────────────────────────────────────


class EvolutionLogRow(Base):
    __tablename__ = "evolution_log"

    id = Column(String(12), primary_key=True, default=_uuid)
    patch_id = Column(String(12), index=True, default="")
    description = Column(Text, default="")
    file_path = Column(String(500), default="")
    status = Column(String(20), default="pending", index=True)  # pending, applied, rolled_back
    diff = Column(Text, default="")
    tests_passed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_now)
    applied_at = Column(DateTime, nullable=True)
