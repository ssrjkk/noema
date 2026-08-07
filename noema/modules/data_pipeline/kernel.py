"""Data pipeline module — ETL, streaming, data processing."""

import re
from dataclasses import dataclass, field
from typing import Any, TypedDict


@dataclass
class PipelineStep:
    name: str
    type: str  # extract, transform, load, filter, aggregate, join
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingConfig:
    source: str = ""
    sink: str = ""
    transformation: str = ""
    window: str = "tumbling"


class PipelinePattern(TypedDict):
    description: str
    steps: list[PipelineStep]


BUILT_IN_PATTERNS: dict[str, PipelinePattern] = {
    "cdc": {
        "description": "Change Data Capture — track and propagate data changes",
        "steps": [
            PipelineStep(
                name="read_binlog",
                type="extract",
                config={"source": "mysql_binlog", "format": "debezium"},
            ),
            PipelineStep(
                name="parse_changes", type="transform", config={"operation": "parse_json"}
            ),
            PipelineStep(
                name="filter_operations", type="filter", config={"condition": "op IN (c,u,d)"}
            ),
            PipelineStep(
                name="enrich_metadata",
                type="transform",
                config={"add_fields": ["timestamp", "source_table"]},
            ),
            PipelineStep(
                name="write_to_sink", type="load", config={"sink": "kafka", "topic": "cdc_events"}
            ),
        ],
    },
    "deduplication": {
        "description": "Remove duplicate records based on key fields",
        "steps": [
            PipelineStep(name="read_source", type="extract", config={"source": "input_stream"}),
            PipelineStep(name="assign_key", type="transform", config={"key_fields": ["id"]}),
            PipelineStep(
                name="window_dedup",
                type="transform",
                config={"window": "5m", "dedup_strategy": "first"},
            ),
            PipelineStep(name="write_output", type="load", config={"sink": "output_stream"}),
        ],
    },
    "aggregation": {
        "description": "Aggregate data over time windows",
        "steps": [
            PipelineStep(name="read_stream", type="extract", config={"source": "input_stream"}),
            PipelineStep(name="key_by", type="transform", config={"key_field": "category"}),
            PipelineStep(
                name="window", type="transform", config={"window_type": "tumbling", "size": "1h"}
            ),
            PipelineStep(
                name="aggregate",
                type="aggregate",
                config={"functions": ["sum", "count", "avg"], "fields": ["amount"]},
            ),
            PipelineStep(name="write_results", type="load", config={"sink": "analytics_db"}),
        ],
    },
    "enrichment": {
        "description": "Enrich stream data with external lookups",
        "steps": [
            PipelineStep(name="read_stream", type="extract", config={"source": "events_stream"}),
            PipelineStep(
                name="lookup_user",
                type="join",
                config={"source": "user_db", "join_key": "user_id", "type": "left"},
            ),
            PipelineStep(
                name="lookup_product",
                type="join",
                config={"source": "product_catalog", "join_key": "product_id", "type": "left"},
            ),
            PipelineStep(name="flatten", type="transform", config={"operation": "flatten_nested"}),
            PipelineStep(name="write_enriched", type="load", config={"sink": "enriched_stream"}),
        ],
    },
}


class DataPipeline:
    def __init__(self, name: str = "pipeline") -> None:
        self.name = name
        self._steps: list[PipelineStep] = []

    def add_step(self, step: PipelineStep) -> "DataPipeline":
        self._steps.append(step)
        return self

    def validate(self) -> dict[str, Any]:
        errors = []
        warnings = []

        type_counts: dict[str, int] = {}
        for step in self._steps:
            type_counts[step.type] = type_counts.get(step.type, 0) + 1

        if type_counts.get("extract", 0) == 0:
            errors.append("Pipeline must have at least one extract step")
        if type_counts.get("load", 0) == 0:
            errors.append("Pipeline must have at least one load step")
        if type_counts.get("extract", 0) > 1:
            warnings.append("Multiple extract steps — ensure they are intentional")

        if self._steps:
            first = self._steps[0]
            if first.type != "extract":
                errors.append(f"First step should be 'extract', got '{first.type}'")
            last = self._steps[-1]
            if last.type != "load":
                warnings.append(f"Last step is '{last.type}', consider ending with a load step")

        for i, step in enumerate(self._steps):
            if not step.name:
                errors.append(f"Step {i} is missing a name")
            if step.type not in ("extract", "transform", "load", "filter", "aggregate", "join"):
                errors.append(f"Step '{step.name}' has unknown type '{step.type}'")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "step_count": len(self._steps),
            "type_distribution": type_counts,
        }

    def to_airflow_dag(self) -> str:
        dag_id = re.sub(r"[^a-z0-9_]", "_", self.name.lower())
        lines = [
            "from airflow import DAG",
            "from airflow.operators.python import PythonOperator",
            "from airflow.operators.bash import BashOperator",
            "from datetime import datetime, timedelta",
            "",
            "",
            "default_args = {",
            '    "owner": "data_engineering",',
            '    "depends_on_past": False,',
            '    "email_on_failure": True,',
            '    "email": ["alerts@example.com"],',
            '    "retries": 2,',
            '    "retry_delay": timedelta(minutes=5),',
            "}",
            "",
            "",
            "dag = DAG(",
            f'    "{dag_id}",',
            "    default_args=default_args,",
            f'    description="{self.name}",',
            '    schedule_interval="@daily",',
            "    start_date=datetime(2024, 1, 1),",
            "    catchup=False,",
            f'    tags=["{dag_id}"],',
            ")",
            "",
            "",
        ]

        task_ids = []
        for _i, step in enumerate(self._steps):
            task_id = re.sub(r"[^a-z0-9_]", "_", step.name.lower())
            task_ids.append(task_id)

            if step.type == "extract":
                lines.append(f"def extract_{task_id}():")
                lines.append(f'    """Extract from {step.config.get("source", "unknown")}"""')
                lines.append("    # Implementation for extraction")
                for k, v in step.config.items():
                    lines.append(f"    # {k} = {v}")
                lines.append("    pass")
                lines.append("")
                lines.append(f"{task_id} = PythonOperator(")
                lines.append(f'    task_id="extract_{task_id}",')
                lines.append(f"    python_callable=extract_{task_id},")
                lines.append("    dag=dag,")
                lines.append(")")
            elif step.type == "transform":
                lines.append(f"def transform_{task_id}():")
                lines.append(f'    """Transform: {step.config.get("operation", "custom")}"""')
                lines.append("    pass")
                lines.append("")
                lines.append(f"{task_id} = PythonOperator(")
                lines.append(f'    task_id="transform_{task_id}",')
                lines.append(f"    python_callable=transform_{task_id},")
                lines.append("    dag=dag,")
                lines.append(")")
            elif step.type == "filter":
                lines.append(f"def filter_{task_id}():")
                lines.append(f'    """Filter: {step.config.get("condition", "none")}"""')
                lines.append("    pass")
                lines.append("")
                lines.append(f"{task_id} = PythonOperator(")
                lines.append(f'    task_id="filter_{task_id}",')
                lines.append(f"    python_callable=filter_{task_id},")
                lines.append("    dag=dag,")
                lines.append(")")
            elif step.type == "aggregate":
                funcs = step.config.get("functions", [])
                lines.append(f"def aggregate_{task_id}():")
                lines.append(f'    """Aggregate: {", ".join(funcs)}"""')
                lines.append("    pass")
                lines.append("")
                lines.append(f"{task_id} = PythonOperator(")
                lines.append(f'    task_id="aggregate_{task_id}",')
                lines.append(f"    python_callable=aggregate_{task_id},")
                lines.append("    dag=dag,")
                lines.append(")")
            elif step.type == "join":
                lines.append(f"def join_{task_id}():")
                lines.append(f'    """Join with {step.config.get("source", "unknown")}"""')
                lines.append("    pass")
                lines.append("")
                lines.append(f"{task_id} = PythonOperator(")
                lines.append(f'    task_id="join_{task_id}",')
                lines.append(f"    python_callable=join_{task_id},")
                lines.append("    dag=dag,")
                lines.append(")")
            elif step.type == "load":
                lines.append(f"def load_{task_id}():")
                lines.append(f'    """Load to {step.config.get("sink", "unknown")}"""')
                lines.append("    pass")
                lines.append("")
                lines.append(f"{task_id} = PythonOperator(")
                lines.append(f'    task_id="load_{task_id}",')
                lines.append(f"    python_callable=load_{task_id},")
                lines.append("    dag=dag,")
                lines.append(")")
            lines.append("")

        lines.append("# Set task dependencies")
        for i in range(len(task_ids) - 1):
            lines.append(f"{task_ids[i]} >> {task_ids[i + 1]}")

        return "\n".join(lines)

    def to_dbt_model(self) -> str:
        re.sub(r"[^a-z0-9_]", "_", self.name.lower())
        lines = [
            "{{ config(materialized='table') }}",
            "",
        ]

        sources = []
        transforms = []
        for step in self._steps:
            if step.type == "extract":
                source = step.config.get("source", "unknown")
                sources.append(source)
            elif step.type == "transform":
                transforms.append(step.config)
            elif step.type == "filter":
                transforms.append({"type": "filter", **step.config})
            elif step.type == "aggregate":
                transforms.append({"type": "aggregate", **step.config})
            elif step.type == "join":
                transforms.append({"type": "join", **step.config})

        for src in sources:
            lines.append("WITH source AS (")
            lines.append(f"    SELECT * FROM {{{{ source('{src}') }}}}")
            lines.append("),")
            lines.append("")

        if transforms:
            lines.append("transformed AS (")
            lines.append("    SELECT * FROM source")
            for t in transforms:
                if t.get("type") == "filter" or "condition" in t:
                    cond = t.get("condition", "TRUE")
                    lines.append(f"    WHERE {cond}")
                elif t.get("type") == "join" or "join_key" in t:
                    join_src = t.get("source", "unknown")
                    join_key = t.get("join_key", "id")
                    lines.append(
                        f"    LEFT JOIN {{{{ source('{join_src}') }}}} ON source.{join_key} = {join_src}.{join_key}"
                    )
            lines.append(")")
            lines.append("")
            lines.append("SELECT * FROM transformed")
        else:
            lines.append("SELECT * FROM source")

        return "\n".join(lines)

    def to_code(self, language: str = "python") -> str:
        if language.lower() == "python":
            return self._to_python()
        elif language.lower() == "typescript":
            return self._to_typescript()
        elif language.lower() == "scala":
            return self._to_scala()
        return self._to_python()

    def _to_python(self) -> str:
        lines = [
            f'"""Pipeline: {self.name}"""',
            "import json",
            "from typing import Any, Dict, List",
            "",
            "",
            f"class {self.name.replace('-', '_').replace(' ', '_').title().replace('_', '')}Pipeline:",
            "    def __init__(self):",
            "        self.steps = []",
            "",
        ]

        for i, step in enumerate(self._steps):
            method_name = re.sub(r"[^a-z0-9_]", "_", step.name.lower())
            lines.append(f"    def {method_name}(self, data: Any) -> Any:")
            lines.append(f'        """Step {i + 1}: {step.type} — {step.name}"""')
            lines.append(f"        # Type: {step.type}")
            for k, v in step.config.items():
                lines.append(f"        # {k}: {v}")

            if step.type == "extract":
                lines.append("        # Read data from source")
                lines.append("        return data")
            elif step.type == "transform":
                lines.append("        # Apply transformation")
                lines.append("        return data")
            elif step.type == "filter":
                lines.append("        # Filter records")
                lines.append("        return data")
            elif step.type == "aggregate":
                lines.append("        # Aggregate data")
                lines.append("        return data")
            elif step.type == "join":
                lines.append("        # Join with external data")
                lines.append("        return data")
            elif step.type == "load":
                lines.append("        # Write data to sink")
                lines.append("        return data")
            lines.append("")

        lines.append("    def run(self, initial_data: Any = None) -> Any:")
        lines.append("        data = initial_data")
        step_methods = []
        for step in self._steps:
            method_name = re.sub(r"[^a-z0-9_]", "_", step.name.lower())
            step_methods.append(method_name)
        for m in step_methods:
            lines.append(f"        data = self.{m}(data)")
        lines.append("        return data")

        return "\n".join(lines)

    def _to_typescript(self) -> str:
        lines = [
            f"// Pipeline: {self.name}",
            "",
            "interface PipelineStep {",
            "  name: string;",
            "  type: string;",
            "  execute(data: any): any;",
            "}",
            "",
            f"class {self.name.replace('-', '').replace(' ', '').title()}Pipeline {{",
            "  private steps: PipelineStep[] = [];",
            "",
            "  addStep(step: PipelineStep): this {",
            "    this.steps.push(step);",
            "    return this;",
            "  }",
            "",
            "  async run(initialData: any = null): Promise<any> {",
            "    let data = initialData;",
            "    for (const step of this.steps) {",
            "      data = await step.execute(data);",
            "    }",
            "    return data;",
            "  }",
            "}",
        ]
        return "\n".join(lines)

    def _to_scala(self) -> str:
        lines = [
            f"// Pipeline: {self.name}",
            "import org.apache.spark.sql.DataFrame",
            "",
            f"object {self.name.replace('-', '').replace(' ', '').title()}Pipeline {{",
            "  def run(spark: org.apache.spark.sql.SparkSession): DataFrame = {",
            "    import spark.implicits._",
            "",
        ]
        for step in self._steps:
            lines.append(f"    // Step: {step.name} ({step.type})")
            if step.type == "extract":
                src = step.config.get("source", "input")
                lines.append(f'    var df = spark.read.format("parquet").load("{src}")')
            elif step.type == "load":
                sink = step.config.get("sink", "output")
                lines.append(f'    df.write.format("parquet").save("{sink}")')
            else:
                lines.append(f"    // Apply {step.type} logic")
        lines.append("    df")
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines)


class DataPipelineModule:
    NAME = "data_pipeline"
    DESCRIPTION = "Data pipeline management: ETL, streaming, data processing with Airflow, dbt, and code generation"

    def __init__(self) -> None:
        self.patterns = BUILT_IN_PATTERNS

    def execute(self, task: Any) -> dict[str, Any]:
        task_title = getattr(task, "title", "")
        task_tags = getattr(task, "tags", [])

        action = "build"
        if "validate" in str(task_title).lower() or "validate" in task_tags:
            action = "validate"
        elif "airflow" in str(task_title).lower() or "airflow" in task_tags:
            action = "airflow"
        elif "dbt" in str(task_title).lower() or "dbt" in task_tags:
            action = "dbt"
        elif "streaming" in str(task_title).lower() or "streaming" in task_tags:
            action = "streaming"
        elif "pattern" in str(task_title).lower() or "patterns" in task_tags:
            action = "patterns"

        metadata = {}
        if hasattr(task, "metadata"):
            metadata = task.metadata

        pipeline_name = metadata.get("name", "data_pipeline")
        steps_data = metadata.get("steps", [])
        pattern = metadata.get("pattern", "")
        language = metadata.get("language", "python")

        pipeline = DataPipeline(name=pipeline_name)

        if pattern and pattern in self.patterns:
            pattern_data = self.patterns[pattern]
            for step in pattern_data["steps"]:
                pipeline.add_step(step)
        else:
            for s in steps_data:
                pipeline.add_step(
                    PipelineStep(
                        name=s.get("name", "step"),
                        type=s.get("type", "transform"),
                        config=s.get("config", {}),
                    )
                )

        if not pipeline._steps:
            pipeline.add_step(
                PipelineStep(name="extract_data", type="extract", config={"source": "input"})
            )
            pipeline.add_step(
                PipelineStep(
                    name="transform_data", type="transform", config={"operation": "process"}
                )
            )
            pipeline.add_step(
                PipelineStep(name="load_data", type="load", config={"sink": "output"})
            )

        validation = pipeline.validate()

        if action == "validate":
            return {
                "action": "validate",
                "validation": validation,
                "_confidence": 0.90,
            }
        elif action == "airflow":
            dag_code = pipeline.to_airflow_dag()
            return {
                "action": "airflow_dag",
                "dag_code": dag_code,
                "_confidence": 0.85,
            }
        elif action == "dbt":
            dbt_code = pipeline.to_dbt_model()
            return {
                "action": "dbt_model",
                "model_code": dbt_code,
                "_confidence": 0.80,
            }
        elif action == "streaming":
            streaming = StreamingConfig(
                source=metadata.get("source", "kafka"),
                sink=metadata.get("sink", "kafka"),
                transformation=metadata.get("transformation", "passthrough"),
                window=metadata.get("window", "tumbling"),
            )
            return {
                "action": "streaming",
                "config": {
                    "source": streaming.source,
                    "sink": streaming.sink,
                    "transformation": streaming.transformation,
                    "window": streaming.window,
                },
                "_confidence": 0.80,
            }
        elif action == "patterns":
            return {
                "action": "patterns",
                "available_patterns": {
                    name: {
                        "description": p["description"],
                        "step_count": len(p["steps"]),
                        "steps": [{"name": s.name, "type": s.type} for s in p["steps"]],
                    }
                    for name, p in self.patterns.items()
                },
                "_confidence": 0.90,
            }
        else:
            code = pipeline.to_code(language)
            return {
                "action": "build",
                "pipeline_name": pipeline_name,
                "step_count": len(pipeline._steps),
                "validation": validation,
                "code": code,
                "_confidence": 0.85,
            }
