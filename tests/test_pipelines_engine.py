"""Tests for noema/pipelines/engine.py."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from noema.pipelines.engine import (
    Pipeline,
    PipelineResult,
    PipelineStep,
    StepStatus,
    create_architecture_review_pipeline,
    create_fullstack_pipeline,
    create_quick_prototype_pipeline,
    create_security_audit_pipeline,
)


def _make_task():
    task = MagicMock()
    task.title = "Test task"
    task.description = "Test description"
    task.tags = []
    return task


class TestStepStatus:
    def test_status_values(self):
        assert StepStatus.PENDING == "pending"
        assert StepStatus.RUNNING == "running"
        assert StepStatus.COMPLETED == "completed"
        assert StepStatus.FAILED == "failed"
        assert StepStatus.SKIPPED == "skipped"


class TestPipelineStep:
    def test_defaults(self):
        step = PipelineStep(name="test")
        assert step.name == "test"
        assert step.kernel_name is None
        assert step.func is None
        assert step.phase == ""
        assert step.config == {}
        assert step.condition is None
        assert step.retry_count == 0
        assert step.timeout_seconds == 30.0
        assert step.status == StepStatus.PENDING
        assert step.result == {}
        assert step.error is None
        assert step.duration_ms == 0.0


class TestPipelineResult:
    def test_completed_and_failed_steps(self):
        steps = [
            PipelineStep(name="s1", status=StepStatus.COMPLETED),
            PipelineStep(name="s2", status=StepStatus.COMPLETED),
            PipelineStep(name="s3", status=StepStatus.FAILED),
            PipelineStep(name="s4", status=StepStatus.SKIPPED),
        ]
        result = PipelineResult(pipeline_name="test", steps=steps, final_output={})
        assert result.completed_steps == 2
        assert result.failed_steps == 1


class TestPipeline:
    def test_add_step_fluent(self):
        pipeline = Pipeline("test")
        result = pipeline.add_step("step1", kernel_name="k1")
        assert result is pipeline
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "step1"
        assert pipeline.steps[0].kernel_name == "k1"

    def test_on_step_complete_callback(self):
        pipeline = Pipeline("test")
        callback = MagicMock()
        result = pipeline.on_step_complete(callback)
        assert result is pipeline
        assert callback in pipeline._on_step_complete

    def test_on_error_callback(self):
        pipeline = Pipeline("test")
        callback = MagicMock()
        result = pipeline.on_error(callback)
        assert result is pipeline
        assert callback in pipeline._on_error

    async def test_execute_with_func(self):
        async def step_func(task, context, **kwargs):
            return {"result": "success"}

        pipeline = Pipeline("test")
        pipeline.add_step("step1", func=step_func)

        task = _make_task()
        result = await pipeline.execute(task)

        assert result.success is True
        assert result.steps[0].status == StepStatus.COMPLETED
        assert result.steps[0].result == {"result": "success"}
        assert result.completed_steps == 1

    async def test_execute_with_kernel(self):
        mock_kernel = AsyncMock()
        mock_kernel.execute = AsyncMock(return_value={"kernel_result": "ok"})

        mock_noema = MagicMock()
        mock_noema.kernels = {"test_kernel": mock_kernel}

        pipeline = Pipeline("test")
        pipeline.add_step("step1", kernel_name="test_kernel", phase="test")

        task = _make_task()
        result = await pipeline.execute(task, noema=mock_noema)

        assert result.success is True
        assert result.steps[0].status == StepStatus.COMPLETED
        mock_kernel.execute.assert_awaited_once()

    async def test_execute_kernel_not_found(self):
        mock_noema = MagicMock()
        mock_noema.kernels = {}

        pipeline = Pipeline("test")
        pipeline.add_step("step1", kernel_name="missing_kernel")

        task = _make_task()
        result = await pipeline.execute(task, noema=mock_noema)

        assert result.steps[0].result == {"error": "Kernel 'missing_kernel' not found"}

    async def test_execute_noop_step(self):
        pipeline = Pipeline("test")
        pipeline.add_step("step1")

        task = _make_task()
        result = await pipeline.execute(task)

        assert result.steps[0].status == StepStatus.COMPLETED
        assert result.steps[0].result == {"status": "noop"}

    async def test_execute_condition_skips_step(self):
        async def step_func(task, context, **kwargs):
            return {"result": "success"}

        pipeline = Pipeline("test")
        pipeline.add_step("step1", func=step_func, condition=lambda ctx: False)

        task = _make_task()
        result = await pipeline.execute(task)

        assert result.steps[0].status == StepStatus.SKIPPED
        assert result.success is True

    async def test_execute_condition_allows_step(self):
        async def step_func(task, context, **kwargs):
            return {"result": "success"}

        pipeline = Pipeline("test")
        pipeline.add_step("step1", func=step_func, condition=lambda ctx: True)

        task = _make_task()
        result = await pipeline.execute(task)

        assert result.steps[0].status == StepStatus.COMPLETED

    async def test_execute_timeout_no_retry(self):
        async def slow_func(task, context, **kwargs):
            await asyncio.sleep(10)
            return {"result": "success"}

        pipeline = Pipeline("test")
        pipeline.add_step("step1", func=slow_func, timeout=0.01)

        task = _make_task()
        result = await pipeline.execute(task)

        assert result.steps[0].status == StepStatus.FAILED
        assert "Timeout" in result.steps[0].error
        assert result.success is False

    async def test_execute_timeout_with_retry(self):
        call_count = 0

        async def flaky_func(task, context, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(10)
            return {"result": "success"}

        pipeline = Pipeline("test")
        pipeline.add_step("step1", func=flaky_func, timeout=0.01, retry_count=1)

        task = _make_task()
        result = await pipeline.execute(task)

        assert result.steps[0].status == StepStatus.COMPLETED
        assert call_count == 2

    async def test_execute_exception_no_retry(self):
        async def failing_func(task, context, **kwargs):
            raise ValueError("test error")

        pipeline = Pipeline("test")
        pipeline.add_step("step1", func=failing_func)

        task = _make_task()
        result = await pipeline.execute(task)

        assert result.steps[0].status == StepStatus.FAILED
        assert "test error" in result.steps[0].error
        assert result.success is False

    async def test_execute_exception_with_retry(self):
        call_count = 0

        async def flaky_func(task, context, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("temporary error")
            return {"result": "success"}

        pipeline = Pipeline("test")
        pipeline.add_step("step1", func=flaky_func, retry_count=1)

        task = _make_task()
        result = await pipeline.execute(task)

        assert result.steps[0].status == StepStatus.COMPLETED
        assert call_count == 2

    async def test_execute_calls_on_step_complete(self):
        async def step_func(task, context, **kwargs):
            return {"result": "success"}

        callback = AsyncMock()
        pipeline = Pipeline("test")
        pipeline.add_step("step1", func=step_func)
        pipeline.on_step_complete(callback)

        task = _make_task()
        result = await pipeline.execute(task)

        callback.assert_awaited_once()
        (completed_step,) = callback.call_args.args
        assert completed_step.name == "step1"
        assert completed_step.result == {"result": "success"}
        assert result.completed_steps == 1
        assert result.final_output["step1"] == {"result": "success"}

    async def test_execute_calls_on_error(self):
        async def failing_func(task, context, **kwargs):
            raise ValueError("test error")

        callback = AsyncMock()
        pipeline = Pipeline("test")
        pipeline.add_step("step1", func=failing_func)
        pipeline.on_error(callback)

        task = _make_task()
        result = await pipeline.execute(task)

        callback.assert_awaited_once()
        failed_step, error = callback.call_args.args
        assert failed_step.status is StepStatus.FAILED
        assert error == "test error"
        assert result.success is False
        assert result.failed_steps == 1

    async def test_execute_initial_context(self):
        async def step_func(task, context, **kwargs):
            return {"prev": context.get("prev_value")}

        pipeline = Pipeline("test")
        pipeline.add_step("step1", func=step_func)

        task = _make_task()
        result = await pipeline.execute(task, initial_context={"prev_value": 42})

        assert result.steps[0].result == {"prev": 42}

    async def test_execute_multiple_steps(self):
        async def step1_func(task, context, **kwargs):
            return {"step": 1}

        async def step2_func(task, context, **kwargs):
            return {"step": 2, "prev": context.get("step1")}

        pipeline = Pipeline("test")
        pipeline.add_step("step1", func=step1_func)
        pipeline.add_step("step2", func=step2_func)

        task = _make_task()
        result = await pipeline.execute(task)

        assert result.completed_steps == 2
        assert result.steps[1].result["prev"] == {"step": 1}
        assert result.total_duration_ms > 0


class TestFactoryPipelines:
    def test_create_fullstack_pipeline(self):
        pipeline = create_fullstack_pipeline()
        assert pipeline.name == "fullstack_generation"
        assert len(pipeline.steps) == 5
        assert pipeline.steps[0].name == "analysis"
        assert pipeline.steps[1].name == "architecture"
        assert pipeline.steps[2].name == "codegen"
        assert pipeline.steps[3].name == "optimization"
        assert pipeline.steps[4].name == "security"

    def test_create_quick_prototype_pipeline(self):
        pipeline = create_quick_prototype_pipeline()
        assert pipeline.name == "quick_prototype"
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].name == "analysis"
        assert pipeline.steps[0].timeout_seconds == 10.0
        assert pipeline.steps[1].name == "codegen"
        assert pipeline.steps[1].timeout_seconds == 15.0

    def test_create_security_audit_pipeline(self):
        pipeline = create_security_audit_pipeline()
        assert pipeline.name == "security_audit"
        assert len(pipeline.steps) == 3
        assert pipeline.steps[0].name == "analysis"
        assert pipeline.steps[1].name == "security"
        assert pipeline.steps[2].name == "optimization"

    def test_create_architecture_review_pipeline(self):
        pipeline = create_architecture_review_pipeline()
        assert pipeline.name == "architecture_review"
        assert len(pipeline.steps) == 3
        assert pipeline.steps[0].name == "analysis"
        assert pipeline.steps[1].name == "architecture"
        assert pipeline.steps[2].name == "security"
