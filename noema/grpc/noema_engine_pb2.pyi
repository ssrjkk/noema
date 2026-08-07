from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ThinkRequest(_message.Message):
    __slots__ = (
        "task_id",
        "title",
        "description",
        "complexity",
        "tags",
        "tenant_id",
        "requirements",
        "context",
    )
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    COMPLEXITY_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    title: str
    description: str
    complexity: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    tenant_id: str
    requirements: _containers.RepeatedCompositeFieldContainer[Requirement]
    context: _struct_pb2.Struct
    def __init__(
        self,
        task_id: _Optional[str] = ...,
        title: _Optional[str] = ...,
        description: _Optional[str] = ...,
        complexity: _Optional[str] = ...,
        tags: _Optional[_Iterable[str]] = ...,
        tenant_id: _Optional[str] = ...,
        requirements: _Optional[_Iterable[_Union[Requirement, _Mapping]]] = ...,
        context: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
    ) -> None: ...

class Requirement(_message.Message):
    __slots__ = ("category", "description", "priority", "constraints")
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    category: str
    description: str
    priority: int
    constraints: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        category: _Optional[str] = ...,
        description: _Optional[str] = ...,
        priority: _Optional[int] = ...,
        constraints: _Optional[_Iterable[str]] = ...,
    ) -> None: ...

class ThinkResponse(_message.Message):
    __slots__ = (
        "solution_id",
        "task_id",
        "title",
        "summary",
        "quality",
        "confidence",
        "code_blocks_count",
        "duration_ms",
        "error",
    )
    SOLUTION_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    QUALITY_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    CODE_BLOCKS_COUNT_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    solution_id: str
    task_id: str
    title: str
    summary: str
    quality: str
    confidence: float
    code_blocks_count: int
    duration_ms: float
    error: str
    def __init__(
        self,
        solution_id: _Optional[str] = ...,
        task_id: _Optional[str] = ...,
        title: _Optional[str] = ...,
        summary: _Optional[str] = ...,
        quality: _Optional[str] = ...,
        confidence: _Optional[float] = ...,
        code_blocks_count: _Optional[int] = ...,
        duration_ms: _Optional[float] = ...,
        error: _Optional[str] = ...,
    ) -> None: ...

class ThinkStatus(_message.Message):
    __slots__ = ("stage", "status", "attempt", "progress", "message", "correlation_id")
    STAGE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CORRELATION_ID_FIELD_NUMBER: _ClassVar[int]
    stage: str
    status: str
    attempt: int
    progress: float
    message: str
    correlation_id: str
    def __init__(
        self,
        stage: _Optional[str] = ...,
        status: _Optional[str] = ...,
        attempt: _Optional[int] = ...,
        progress: _Optional[float] = ...,
        message: _Optional[str] = ...,
        correlation_id: _Optional[str] = ...,
    ) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("status", "version", "uptime_s")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    UPTIME_S_FIELD_NUMBER: _ClassVar[int]
    status: str
    version: str
    uptime_s: float
    def __init__(
        self,
        status: _Optional[str] = ...,
        version: _Optional[str] = ...,
        uptime_s: _Optional[float] = ...,
    ) -> None: ...

class MetricsRequest(_message.Message):
    __slots__ = ("tenant_id",)
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    tenant_id: str
    def __init__(self, tenant_id: _Optional[str] = ...) -> None: ...

class MetricsResponse(_message.Message):
    __slots__ = (
        "tasks_processed",
        "tasks_successful",
        "tasks_failed",
        "success_rate",
        "total_llm_calls",
        "total_refinements",
    )
    TASKS_PROCESSED_FIELD_NUMBER: _ClassVar[int]
    TASKS_SUCCESSFUL_FIELD_NUMBER: _ClassVar[int]
    TASKS_FAILED_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_RATE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_LLM_CALLS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_REFINEMENTS_FIELD_NUMBER: _ClassVar[int]
    tasks_processed: int
    tasks_successful: int
    tasks_failed: int
    success_rate: float
    total_llm_calls: int
    total_refinements: int
    def __init__(
        self,
        tasks_processed: _Optional[int] = ...,
        tasks_successful: _Optional[int] = ...,
        tasks_failed: _Optional[int] = ...,
        success_rate: _Optional[float] = ...,
        total_llm_calls: _Optional[int] = ...,
        total_refinements: _Optional[int] = ...,
    ) -> None: ...

class CancelRequest(_message.Message):
    __slots__ = ("task_id",)
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    def __init__(self, task_id: _Optional[str] = ...) -> None: ...

class CancelResponse(_message.Message):
    __slots__ = ("cancelled", "task_id", "status")
    CANCELLED_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    cancelled: bool
    task_id: str
    status: str
    def __init__(
        self,
        cancelled: _Optional[bool] = ...,
        task_id: _Optional[str] = ...,
        status: _Optional[str] = ...,
    ) -> None: ...

class EvolveRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class EvolveResponse(_message.Message):
    __slots__ = ("patches_generated", "patches_applied", "summary", "improvements")
    PATCHES_GENERATED_FIELD_NUMBER: _ClassVar[int]
    PATCHES_APPLIED_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    IMPROVEMENTS_FIELD_NUMBER: _ClassVar[int]
    patches_generated: int
    patches_applied: int
    summary: str
    improvements: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        patches_generated: _Optional[int] = ...,
        patches_applied: _Optional[int] = ...,
        summary: _Optional[str] = ...,
        improvements: _Optional[_Iterable[str]] = ...,
    ) -> None: ...
